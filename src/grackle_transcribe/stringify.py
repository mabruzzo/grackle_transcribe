import more_itertools

from .src_model import Code # for type annotation purposes
from .token import Token
from .syntax_unit import _iterate_tokens

from dataclasses import dataclass
from enum import auto, Enum
from itertools import chain, count
from typing import List, Optional

class TokenStringifyDirective(Enum):
    SKIP = auto() # skip over the token
    PRESERVE = auto()
    
SKIP_TOK = TokenStringifyDirective.SKIP
PRESERVE_TOK = TokenStringifyDirective.PRESERVE

class TokenAlternative(Enum):
    PREPEND_TO_NEXT = auto()
    APPEND_TO_NEXT = auto()


def is_split_across_lines(arg):
    # arg is an Expr, Stmt, ArgList, Token etc.
    #
    # this won't work in some cases (if we cheat make 1 multiline token)
    itr = _iterate_tokens(arg)
    first = next(itr)
    last = more_itertools.last(itr, first)
    return first.lineno != last.lineno

def _leading_origin_lineno_column(arg):
    if isinstance(arg, Token):
        tok = arg
    else:
        tok = next(_iterate_tokens(arg))
    return tok.lineno, tok.column

def _final_origin_stopping_column(arg):
    if isinstance(arg, Token):
        tok = arg
    else:
        tok = more_itertools.last(_iterate_tokens(arg))
    return len(tok.string) + tok.column

def _ntokens(arg):
    if isinstance(arg, Token):
        return 1
    return len(list(_iterate_tokens(arg)))

class _PairToStrProcessor:
    _prepend: Optional[str] = None
    _append: Optional[str] = None

    def __init__(self):
        self._prepend, self._append = None, None

    def pending_prepend(self): return self._prepend is not None
    def pending_append(self): return self._append is not None

    def process(self, orig, repl, dfltret=None):
        if isinstance(orig, TokenAlternative):
            if not isinstance(repl, str):
                raise ValueError(
                    "repl must be a string when the first argument is "
                    f"{orig}"
                )
            elif orig is TokenAlternative.PREPEND_TO_NEXT:
                self._prepend = (
                    repl if self._prepend is None else repl + self._prepend
                )
            else:
                self._append = (
                    repl if self._append is None else self._append + repl
                )
            return dfltret
        else:
            match repl:
                case TokenStringifyDirective.PRESERVE: s = orig.string
                case TokenStringifyDirective.SKIP: return dfltret
                case str(): s = repl
                case _: raise TypeError()

            if self._prepend is not None:
                s = self._prepend + s
                self._prepend = None

            if self._append is not None:
                s = s + self._append
                self._append = None
            return s


def concat_translated_pairs(pair_itr, delim = ' '):
    # this does what the builder does, but in a slightly more simple manner
    # - it is a little less careful and definitely is more crude
    # - we discard some extra information
    processor = _PairToStrProcessor()
    itr = (processor.process(orig, repl, None) for orig, repl in pair_itr)
    out = delim.join(s for s in itr if s is not None)
    if processor.pending_prepend() or processor.pending_append():
        raise RuntimeError()
    return out

@dataclass
class _DataChunk:
    origin_start_col: int # used to help with indents
    origin_stop_col: int  # used for trailing comments
    fragments: List[str]

class FormattedCodeEntryBuilder:
    """
    If we start out with a Code instance composed of N lines, we will produce
    N lines

    Aiming for algorithmically correct, but stupid.
    """

    _src: Code
    _maintain_indent: bool
    _tok_count: int
    _cur_origin_lineno: int
    _parts: List[_DataChunk]
    _processor: _PairToStrProcessor

    def __init__(self, src, *, maintain_indent = True):
        #assert isinstance(src, Code)
        self._src = src
        assert 0 < sum(
            isinstance(line, str) for line in src.entries
        )
        self._maintain_indent = maintain_indent

        self._tok_count = 0
        if src.has_label:
            raise RuntimeError()
        first_tok = src.tokens[0]
        self._cur_origin_lineno = first_tok.lineno
        self._parts = [
            _DataChunk(first_tok.column, first_tok.column, [])
        ]
        self._processor = _PairToStrProcessor()

    def _append_str(self, arg):
        assert isinstance(arg, str)
        self._parts[-1].fragments.append(arg)

    def put(self, arg, repl = TokenStringifyDirective.PRESERVE):
        """
        Parameters
        ----------
        arg
            The next ``Token`` to consider from the current line or object
            composed of the next ``Token`` instances (accessed through
            ``_iterate_tokens``). Alternatively, this can be a member of the
            the ``TokenAlternative`` enumeration
        repl
            Either a member of the ``TokenStringifyDirective`` enumeration OR
            a string to use in place of ``arg``.
        """
        if isinstance(arg, TokenAlternative):
            self._processor.process(arg, repl)
            return None
        elif isinstance(arg, Token):
            def mk_iter(arg): return iter([arg])
        else:
            mk_iter = _iterate_tokens
            assert not is_split_across_lines(arg)
            assert repl is not SKIP_TOK

        if not isinstance(repl, (str, TokenStringifyDirective)):
            raise TypeError(repl)

        arg_origin_lineno, arg_origin_col = _leading_origin_lineno_column(arg)

        if self._cur_origin_lineno > arg_origin_lineno:
            raise ValueError(
                "current token appears to come before prior token"
            )
        elif self._cur_origin_lineno == arg_origin_lineno:
            cur_origin_col = self._parts[-1].origin_stop_col
            if cur_origin_col > arg_origin_col:
                raise ValueError(
                    "current token appears to come before prior token"
                )
        else:
            cur_origin_col = arg_origin_col
            self._cur_origin_lineno = arg_origin_lineno
            self._parts.append(
                _DataChunk(cur_origin_col, cur_origin_col, [])
            )

        if repl is TokenStringifyDirective.PRESERVE:
            for tok in mk_iter(arg):
                # append any space since last origin col
                diff = tok.column - cur_origin_col
                if diff > 0: self._append_str(diff * ' ')

                # add the token string
                self._append_str(self._processor.process(tok, repl))
                self._tok_count += 1

                # get current origin col
                cur_origin_col = len(tok.string) + tok.column
        else:
            # add any space since last origin col
            diff = _leading_origin_lineno_column(arg)[1] - cur_origin_col
            if diff > 0: self._append_str(diff * ' ')

            # add the replacement-string
            if (s := self._processor.process(arg, repl, None)) is not None:
                self._append_str(s)
            self._tok_count += _ntokens(arg)

            # get the current origin col (totally independent of repl)
            cur_origin_col = _final_origin_stopping_column(arg)

        # record the current origin column for later
        self._parts[-1].origin_stop_col = cur_origin_col

    def append_semicolon(self):
        # the way this method is used, we can't use self._processor here with 
        # TokenAlternative.APPEND_TO_NEXT
        self._append_str(';')

    def build(self, trailing_comment_delim = '!'):
        assert self._tok_count == len(self._src.tokens)
        assert len(self._parts) == sum(
            isinstance(line, str) for line in self._src.entries
        )

        top_left_origin_col = self._parts[0].origin_start_col

        out = []
        part_index = 0
        itr = zip(self._src.entries, self._src.trailing_comment_start)
        for cur_entry, trailing_comment_start in itr:
            if not isinstance(cur_entry, str):
                assert trailing_comment_start is None # sanity check
                out.append(cur_entry)
                continue

            cur_part = self._parts[part_index]
            part_index += 1

            if self._maintain_indent:
                indent = [cur_entry[:cur_part.origin_start_col]]
            else:
                diff = cur_part.origin_start_col - top_left_origin_col
                indent = ['' if diff <= 0 else ' '*diff]

            if trailing_comment_start is None:
                suffix = []
            else:
                assert cur_entry[trailing_comment_start] == '!'
                diff = trailing_comment_start - cur_part.origin_stop_col
                assert diff >= 0
                suffix = [
                    '' if diff == 0 else diff * ' ',
                    trailing_comment_delim,
                    cur_entry[trailing_comment_start+1:]
                ]

            out.append(''.join(chain(indent, cur_part.fragments, suffix)))
        return out


