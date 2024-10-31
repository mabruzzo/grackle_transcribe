# the idea here is that we model parts of a fortran source code file
# - not sure if we will actually use this machinery, but to make an extremely
#   useful tool, we would need to match up parts of a file with the AST

from more_itertools import peekable
from enum import auto, Flag
from functools import partial
import re
from typing import NamedTuple, Optional

from .token import (
    Keyword,
    _CONTINUATION_LINE as _CONTINUE_PATTERN,
    process_code_chunk,
    scan_chunk,
    token_has_type
)

class Origin(NamedTuple):
    lineno: int
    fname: Optional[str] = None

class SrcItem:
    def __str__(self):
        klass = self.__class__.__name__
        tmp = [f'{self.__class__.__name__}(lines=[']
        lines = self.lines
        if len(lines) == 1:
            return tmp[0] + f"'{lines[0]}'])"
        for line in lines:
            tmp.append(f"  '{line}',")
        tmp.append('])')
        return '\n'.join(tmp)

class WhitespaceLines(SrcItem):
    # represents empty lines
    def __init__(self, lines, *, origin = None):
        self.lines = tuple(lines)
        self.origin = origin

    def nlines(self): return len(self.lines)

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
        (
            PreprocKind.DEFINE,
            re.compile(r"^\s*#\s*define\s+(?P<val>[a-zA-Z0-9_]+).*$")
        )
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
    # represents 1 or more comment lines
    def __init__(self, lines, *, origin = None):
        self.lines = tuple(lines)
        self.origin = origin

    def nlines(self): return len(self.lines)

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
    def __init__(self, entries, origin = None):
        assert not isinstance(entries, str)
        assert len(entries) > 0
        assert isinstance(entries[0], str)
        self.entries = entries

        # the following check is not actually an error. But, it is a sign that
        # we will need to refactor, or (handle something manually)
        if len(entries) > 1:
            for entry in entries:
                assert not isinstance(entry, (OMPDirective, PreprocessorDirective))
        self.origin = origin

        kind, tokens, trailing_comment_start, has_label \
            = process_code_chunk(self.entries)

        self.kind = kind
        self.tokens = tokens
        self.trailing_comment_start = tuple(trailing_comment_start)
        self.has_label = has_label

    def first_token_has_type(self, type_spec):
        return token_has_type(self.tokens[0], type_spec)

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

    def maybe_trailing_comment(self):
        return self._maybe_trailing_comment

class LineProvider:
    # each entry is (lineno, content)
    def __init__(self, f, lineno_start = 0, strip_newline=True, fname = None):
        self.fname = fname
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


_OMP_START_STR = r"((\!omp\$)|(\!\$omp))"
_OMP_CONTINUE_STR = r"((\!omp\$\&)|(\!\$omp\&))"
_OMP_START_PATTERN = re.compile("^" + _OMP_START_STR, re.IGNORECASE)
_OMP_CONTINUE_PATTERN = re.compile("^" + _OMP_CONTINUE_STR, re.IGNORECASE)
_OMP_LEADING_WHITESPACE = re.compile(
    f"^\s+({_OMP_START_STR})|({_OMP_CONTINUE_STR})", re.IGNORECASE
)
_ALL_OMP_PATTERNS = [
    _OMP_START_PATTERN, _OMP_CONTINUE_PATTERN, _OMP_LEADING_WHITESPACE
]
def _try_omp_directive(line, provider):
    assert provider is not None

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
def _is_comment_line(line):
    # https://docs.oracle.com/cd/E19957-01/805-4939/z40007332024/index.html
    if _COMMENT_PATTERN.match(line) is not None:
        return all(matcher.match(line) is None for matcher in _ALL_OMP_PATTERNS)
    return False

def _is_whitespace_line(line):
    return line == '' or line.isspace()

def _try_SrcItem_helper(line, provider, line_checker_fn, klass):

    if line_checker_fn(line):
        line_l = [line]
        if provider is not None:
            while True:
                try:
                    next_line = provider.peek()[1]
                except StopIteration:
                    break
                if line_checker_fn(next_line):
                    line_l.append(next(provider)[1])
                else:
                    break
        return klass(line_l)
    return None

_try_comment = partial(
    _try_SrcItem_helper, line_checker_fn=_is_comment_line, klass=Comment
)
_try_whitespace = partial(
    _try_SrcItem_helper,
    line_checker_fn=_is_whitespace_line,
    klass=WhitespaceLines
)

_PREPROC_PATTERN = re.compile(r"^\s*\#")
def _try_preprocessor(line):
    return PreprocessorDirective(line) if _PREPROC_PATTERN.match(line) else None

def _try_nonomp(line, provider):
    for fn in [_try_whitespace, _try_comment]:
        if item := fn(line, provider):
            return item
    if item := _try_preprocessor(line):
        return item
    return None

def _inner_get_items(provider):
    assert provider.stripped_newline
    fname=provider.fname

    for lineno, line in provider:
        item = None
        if (item := _try_omp_directive(line, provider)):
            pass
        elif (item := _try_nonomp(line, provider)):
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
                elif _try_nonomp(next_line, provider=None) is not None:
                    cached_pairs.append((
                        next_lineno,
                        _try_nonomp(next(provider)[1], provider=provider)
                    ))
                elif _CONTINUE_PATTERN.match(next_line) is not None:
                    for _, item in cached_pairs:
                        line_l.append(item)
                    cached_pairs = []
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
            if not cur_pairs[0][1].first_token_has_type(Keyword.SUBROUTINE):

                first_tok = cur_pairs[0][1].tokens[0]
                print(scan_chunk(cur_pairs[0][1].lines))
                raise RuntimeError(
                    "Expected the first token of the code chunk in a routine "
                    f"to the subroutine keyword, not {first_tok!r}"
                )

            for pair in self._item_itr:
                cur_pairs.append(pair)
                _, item = pair
                if isinstance(item, Code):
                    if item.first_token_has_type(Keyword.ENDROUTINE):
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
                if not isinstance(pair[1], (Comment, WhitespaceLines)):
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

