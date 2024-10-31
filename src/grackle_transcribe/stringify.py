import more_itertools

from .token import Token
from .parser import _iterate_tokens

from dataclasses import dataclass
from itertools import chain, count
from typing import List


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
        tok = next(iterate_tokens(arg))
    return tok.lineno, tok.column

def _final_origin_stopping_column(arg):
    if isinstance(arg, Token):
        tok = arg
    else:
        tok = more_itertools.last(iterate_tokens(arg))
    return

def _ntokens(arg):
    if isinstance(arg, Token):
        return 1
    return count(_iterate_tokens(arg))

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

    def _append_str(self, arg):
        assert isinstance(arg, str)
        self._parts[-1].fragments.append(arg)

    def put(self, arg, repl = None):
        if isinstance(arg, Token):
            def mk_iter(arg): iter([arg])
        else:
            mk_iter = _iterate_tokens
            assert not is_split_across_lines(arg)

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

        if isinstance(repl,str):
            assert '\n' not in repl
            # add any space since last origin col
            diff = _leading_origin_lineno_column(arg)[1] - cur_origin_col
            if diff > 0: self._append_str(diff * ' ')

            # add the replacement-string
            self._append_str(repl)
            self._tok_count += _ntokens(arg)

            # get the current origin col (totally independent of repl)
            cur_origin_col = _final_origin_stopping_column(arg)
        else:
            assert repl is None
            for tok in [arg]:
                # append any space since last origin col
                diff = tok.column - cur_origin_col
                if diff > 0: self._append_str(diff * ' ')

                # add the token string
                self._append_str(tok.string)
                self._tok_count += 1

                # get current origin col
                cur_origin_col = len(tok.string) + tok.column

        # record the current origin column for later
        self._parts[-1].origin_stop_col = cur_origin_col

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
                diff = cur_part.origin_start_col <= top_left_origin_col
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


