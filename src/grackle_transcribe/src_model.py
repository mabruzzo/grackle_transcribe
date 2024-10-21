# the idea here is that we model parts of a fortran source code file
# - not sure if we will actually use this machinery, but to make an extremely
#   useful tool, we would need to match up parts of a file with the AST

from more_itertools import peekable
from enum import auto, Enum
import re

from .f_chunk_parse import process_code_chunk, ChunkKind
from .f_chunk_parse import _CONTINUATION_LINE as _CONTINUE_PATTERN

class SrcItem:
    pass

class WhitespaceLine(SrcItem):
    # represents an empty line
    def __init__(self, value = None):
        self._value = value

    @property
    def value(self):
        if self._value is None:
            return ''
        return self._value

    @property
    def lines(self): return (self.value,)

    def nlines(self): return 1

class PreprocessorDirective(SrcItem):
    # represents a pre-processor directive
    def __init__(self, value):
        self.value = value

    @property
    def lines(self): return (self.value,)

    def nlines(self): return 1


class Comment(SrcItem):
    # represents a single line
    def __init__(self, value):
        self.value = value

    @property
    def lines(self): return (self.value,)

    def nlines(self): return 1

class OMPDirective(SrcItem):
    # represents an openmp directive
    def __init__(self, lines):
        self.lines = lines

    def nlines(self): return len(self.lines)


class Code(SrcItem):
    # represents actual logic
    def __init__(self, lines):
        assert not isinstance(lines, str)
        assert len(lines) > 0
        assert isinstance(lines[0], str)
        self.lines = lines

        # the following check is not actually an error. But, it is a sign that
        # we will need to refactor, or (handle something manually)
        if len(lines) > 1:
            for line in lines:
                assert not isinstance(line, (OMPDirective, PreprocessorDirective))

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

def get_items(provider):
    assert provider.stripped_newline

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
            item = Code(line_l)
            yield lineno, item
            for lineno, item in cached_pairs:
                yield lineno, item
            cached_comment_pairs = []

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
