# the idea here is that we model parts of a fortran source code file
# - not sure if we will actually use this machinery, but to make an extremely
#   useful tool, we would need to match up parts of a file with the AST

from more_itertools import peekable
from enum import auto, Flag
import re
from typing import NamedTuple, Optional

from .f_chunk_parse import process_code_chunk, ChunkKind
from .f_chunk_parse import _CONTINUATION_LINE as _CONTINUE_PATTERN

class Origin(NamedTuple):
    lineno: int
    fname: Optional[str] = None

class SrcItem:
    pass

class WhitespaceLine(SrcItem):
    # represents an empty line
    def __init__(self, value = None, *, origin = None):
        self._value = value
        self.origin = origin

    @property
    def value(self):
        if self._value is None:
            return ''
        return self._value

    @property
    def lines(self): return (self.value,)

    def nlines(self): return 1

class PreprocKind(Flag):
    INCLUDE = auto()
    INCLUDE_phys_const = auto()
    INCLUDE_grackle_fortran_types = auto()
    INCLUDE_OMP = auto()
    IFDEF = auto()
    IFDEF_OPENMP = auto()
    ELSE = auto()
    ENDIF = auto()
    DEFINE = auto()

    @property
    def is_include(self):
        return str(self).startswith(str(self.__class__.INCLUDE))

    @property
    def is_ifdef(self):
        class_name = self.__class__.__name__
        return str(self).startswith(f"{class_name}.IFDEF")

assert PreprocKind.IFDEF_OPENMP.is_ifdef
assert not PreprocKind.DEFINE.is_ifdef

def _build_preproc_match_seq():

    def _include(kind, pattern):
        assert kind.is_include
        full_pattern = rf'^\s*#\s*include\s+"(?P<val>{pattern})"\s*$'
        return (kind, re.compile(full_pattern))

    # order sorta matters
    return (
        _include(PreprocKind.INCLUDE_phys_const, r"phys_const\.def"),
        _include(
            PreprocKind.INCLUDE_grackle_fortran_types,
            r"grackle_fortran_types\.def"
        ),
        _include(PreprocKind.INCLUDE_OMP, r"omp_lib\.h"),
        _include(PreprocKind.INCLUDE, r'[^"]+'),

        # ifdef lines
        (
            PreprocKind.IFDEF_OPENMP,
            re.compile(r"^\s*#\s*ifdef\s+(?P<val>_OPENMP)\s*$")
        ),
        (
            PreprocKind.IFDEF,
            re.compile(r"^\s*#\s*ifdef\s+(?P<val>[a-zA-Z0-9_]+)\s*$")
        ),

        # lines with trailing comments
        (PreprocKind.ELSE, re.compile(r"^\s*#\s*else\s*(/\*.*\*/\s*)*$")),
        (PreprocKind.ENDIF, re.compile(r"^\s*#\s*endif\s*(/\*.*\*/\s*)*$")),

        # define statement
        (PreprocKind.DEFINE, re.compile(r"^\s*#\s*define\s+(?P<val>.*)$"))
    )

_PREPROC_LINE_KINDS = _build_preproc_match_seq()

class PreprocessorDirective(SrcItem):
    # represents a pre-processor directive
    def __init__(self, value, *, origin = None):
        self.value = value
        self.origin = origin
        for kind, matcher in _PREPROC_LINE_KINDS:
            m = matcher.match(value)
            if m is not None:
                self.kind = kind
                self.kind_value = m.groupdict().get("val", None)
                break
        else:
            raise RuntimeError(
                f"Unable to match preprocessor line:\n  {value!r}"
            )

        if False:
            print(
                #'preprocessor line:',
                f'kind: {self.kind}, val: {self.kind_value!r}',
                #f'full line: {value!r}'
                sep='\n -> '
            )

    @property
    def lines(self): return (self.value,)

    def nlines(self): return 1


class Comment(SrcItem):
    # represents a single line
    def __init__(self, value, *, origin = None):
        self.value = value
        self.origin = origin

    @property
    def lines(self): return (self.value,)

    def nlines(self): return 1

class OMPDirective(SrcItem):
    # represents an openmp directive
    # -> when compound is True, this may contain multiple kinds of items
    def __init__(self, entries, compound = False, *, origin = None):
        self.entries = entries
        self.compound = compound
        self.origin = origin

    @property
    def lines(self):
        out = []
        for entry in self.entries:
            if isinstance(entry, SrcItem):
                out+=entry.lines
            else:
                out.append(entry)
        return out

    def nlines(self): return len(self.lines)


class Code(SrcItem):
    # represents actual logic
    def __init__(self, lines, origin = None):
        assert not isinstance(lines, str)
        assert len(lines) > 0
        assert isinstance(lines[0], str)
        self.lines = lines

        # the following check is not actually an error. But, it is a sign that
        # we will need to refactor, or (handle something manually)
        if len(lines) > 1:
            for line in lines:
                assert not isinstance(line, (OMPDirective, PreprocessorDirective))
        self.origin = origin

        kind, tokens, trailing_comment_start, has_label \
            = process_code_chunk(self.lines)

        self.kind = kind
        self.tokens = tokens
        self.trailing_comment_start = trailing_comment_start
        self.has_label = has_label

    def nlines(self): return len(self.lines)

    def maybe_trailing_comment(self):
        return self._maybe_trailing_comment

class LineProvider:
    # each entry is (lineno, content)
    def __init__(self, f, lineno_start = 0, strip_newline=True):
        if strip_newline:
            inner = (line[:-1] for line in f)
        else:
            inner = iter(f)
        self._iter = peekable(iter(enumerate(inner, start=lineno_start)))
        self._strip_newline = True

    @property
    def stripped_newline(self): return self._strip_newline

    def peek(self, *args, n_ahead = 1):
        # an argument can be passed for a default value
        return self._iter.peek(*args)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)

    def __bool__(self):
        try:
            self.peek()
            return True
        except StopIteration:
            return False


def _try_whitespace(line):
    if line == '' or line.isspace():
        return WhitespaceLine(line)
    return None


_OMP_START_STR = r"((\!omp\$)|(\!\$omp))"
_OMP_CONTINUE_STR = r"((\!omp\$\&)|(\!\$omp\&))"
_OMP_START_PATTERN = re.compile("^" + _OMP_START_STR, re.IGNORECASE)
_OMP_CONTINUE_PATTERN = re.compile("^" + _OMP_CONTINUE_STR, re.IGNORECASE)
_OMP_LEADING_WHITESPACE = re.compile(
    f"^\s+({_OMP_START_STR})|({_OMP_CONTINUE_STR})", re.IGNORECASE
)

def _try_omp_directive(line, provider):

    if _OMP_CONTINUE_PATTERN.match(line):
        raise RuntimeError(
            "Encountered seemingly isolated omp-continuation directive"
        )
    elif _OMP_LEADING_WHITESPACE.match(line):
        raise RuntimeError(
            "encountered an omp-directive with leading whitespece. not sure "
            "this is allowed"
        )
    elif _OMP_START_PATTERN.match(line):
        line_l = [line]
        while _OMP_CONTINUE_PATTERN.match(provider.peek((None, ""))[1]):
            line_l.append(next(provider)[1])
        return OMPDirective(line_l)
    return None

_COMMENT_PATTERN = re.compile(r"^[\*cCdD]|(^\s*\!)")
def _try_comment(line):
    # need to check this after openmp directive
    if line[0] == "!":
        assert _COMMENT_PATTERN.match(line) is not None
        return Comment(line)

    # https://docs.oracle.com/cd/E19957-01/805-4939/z40007332024/index.html
    return Comment(line) if _COMMENT_PATTERN.match(line) else None

_PREPROC_PATTERN = re.compile(r"^\s*\#")
def _try_preprocessor(line):
    return PreprocessorDirective(line) if _PREPROC_PATTERN.match(line) else None

def _inner_get_items(provider):
    assert provider.stripped_newline
    fname=None # we could definitely do better

    def _try_nonomp(line):
        for fn in [_try_whitespace, _try_comment, _try_preprocessor]:
            if item := fn(line):
                return item
        return None

    for lineno, line in provider:
        item = None
        if (item := _try_omp_directive(line, provider)):
            pass
        elif (item := _try_nonomp(line)):
            pass
        elif _CONTINUE_PATTERN.match(line):
            raise RuntimeError(
                f"something went wrong lineno {lineno}\n"
                f"-> {line!r}"
            )

        if item is not None:
            # we could do better with handling origin
            item.origin = Origin(lineno=lineno, fname=fname)
            yield lineno, item
        else:
            # finally we consider whether this could just be source code
            line_l = [line]

            # it's technichally possible to embed certain kinds of other
            # chunks (non-omp directives) within the source code
            cached_pairs = []

            while provider.peek((None, None))[1] is not None:
                next_lineno, next_line = provider.peek((None, ""))
                if _OMP_START_PATTERN.match(next_line):
                    break
                elif (tmp := _try_nonomp(next_line)) is not None:
                    cached_pairs.append((next_lineno, tmp))
                    next(provider)
                elif _CONTINUE_PATTERN.match(next_line) is not None:
                    for _, item in cached_pairs:
                        line_l.append(item)
                    cached_comment_pairs = []
                    line_l.append(next_line)
                    next(provider)
                else:
                    break
            item = Code(line_l, origin=Origin(lineno=lineno, fname=fname))
            yield lineno, item
            for lineno, item in cached_pairs:
                # we could do better with handling origin
                item.origin = Origin(lineno=lineno, fname=fname)
                yield lineno, item
            cached_comment_pairs = []

def get_items(provider):

    itr = _inner_get_items(provider)
    for lineno,item in itr:
        if not (isinstance(item, PreprocessorDirective) and
                (item.kind == PreprocKind.IFDEF_OPENMP)):
            yield lineno, item
        else:
            # here, we try to nicely group OpenMP compound directives
            cache = [(lineno,item)]
            properly_closed = False
            for i, (lineno,item) in enumerate(itr):
                cache.append((lineno,item))
                if isinstance(item, (Code, PreprocessorDirective)):
                    kind = getattr(item, "kind", None)
                    properly_closed = (kind is PreprocKind.ENDIF)
                    if kind is not PreprocKind.INCLUDE_OMP:
                        break

            if properly_closed:
                item = OMPDirective([p[1] for p in cache], compound = True)
                yield cache[0][0], item
            else:
                for pair in cache:
                    yield pair
            cache = []

# down below, we describe regions of a file.

class SrcRegion:
    def __init__(self, lineno_item_pairs, is_routine = False):
        assert len(lineno_item_pairs) > 0
        self.lineno_item_pairs = lineno_item_pairs
        self.is_routine = is_routine

class ItSrcRegion:
    def __init__(self, provider):
        self._started = False
        self._prologue = None
        self._item_itr = get_items(provider)
        self._cached_pair = None

    @property
    def prologue(self):
        """
        Retrieves the prologue region (it preceeds routine regions)
        """
        assert self._started
        return self._prologue

    def __iter__(self): return self

    def __next__(self):
        if self._cached_pair is None:
            cur_pairs = [next(self._item_itr)]
        else:
            cur_pairs = [self._cached_pair]
            self._cached_pair = None

        is_routine = isinstance(cur_pairs[0][1], Code)
        if is_routine:
            kind = cur_pairs[0][1].kind
            if kind != ChunkKind.SubroutineDecl:
                print(scan_chunk(item.lines))
                raise RuntimeError(
                    "Expected the first code chunk in a routine to be: "
                    f"{ChunkKind.SubroutineDecl!r}, not {kind!r}"
                )

            for pair in self._item_itr:
                cur_pairs.append(pair)
                _, item = pair
                if isinstance(item, Code):
                    if item.kind == ChunkKind.EndRoutine:
                        break
        else:
            for pair in self._item_itr:
                if isinstance(pair[1], Code):
                    self._cached_pair = pair
                    break
                cur_pairs.append(pair)

        region = SrcRegion(lineno_item_pairs=cur_pairs, is_routine=is_routine)

        if (not self._started) and (not is_routine):
            self._prologue = region
        elif self._started and not is_routine:
            for pair in cur_pairs:
                if not isinstance(pair[1], (Comment, WhitespaceLine)):
                    raise RuntimeError(
                        "it's a little surprising that a regions outside of "
                        "the file's prologue and routine-regions holds "
                        "non-comment or non-whitespace line(s)\n"
                        f" -> {pair[1].lines!r}"
                    )
        self._started = True

        return region




def get_source_regions(provider):
    return ItSrcRegion(provider)


if __name__ == '__main__':

    import os
    PREFIX = '/Users/mabruzzo/packages/c++/grackle/src/clib/'
    if True:
        fnames = [fname for fname in os.listdir(PREFIX) if fname.endswith('.F')]
    else:
        fnames = ['solve_rate_cool_g.F']
    for fname in fnames:
        print()
        print(fname)
        with open(os.path.join(PREFIX, fname), 'r') as f:
            provider = LineProvider(f)
            for lineno, item in get_items(provider):
                if isinstance(item, OMPDirective):
                    print(lineno, "OMPDIRECTIVE")
                elif isinstance(item, Code):
                    kind, tokens, trailing_comment_start, has_label \
                        = process_code_chunk(item.lines)
                    if kind == ChunkKind.Uncategorized:
                        print(lineno, has_label, [token.string for token in tokens])
                    else:
                        pass
                        #print(lineno, kind, has_label)
                else:
                    pass
                    #print(lineno, item.lines)