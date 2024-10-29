# this encodes logic for parsing a single chunk of fortran code

#class Matcher(Enum):
#    UNDEFINED = auto()
#

from enum import Enum, auto
from functools import partial
import re
from typing import Any, Callable, List, NamedTuple, Optional, Tuple, Union

def _attr_check(token, attr, matcher):
    attr_val = getattr(token, attr)
    if not isinstance(attr_val, str):
        return False
    return matcher.match(attr_val) is not None

def _isinstance_check(token, klass):
    return isinstance(token.type, klass)

class _ClassifyReq:
    # we only use regex for case insensitivity (we could get around this by
    # creating more types of tokens, but I think this is the simpler way to
    # do it)
    leading_tok_l: list[Union[Callable]]
    final_tok_str: Optional[re.Pattern] = None
    max_len: Optional[int]
    min_len: Optional[int]

    def __init__(
        self,
        leading_tok_l: list[Union[str, Tuple[Any,str]]],
        final_tok_str: Optional[str]= None,
        max_len: int = None,
        min_len: int = None
    ):
        assert len(leading_tok_l) > 0
        self.leading_tok_l = []

        for elem in leading_tok_l:

            if isinstance(elem, tuple):
                attr, expected = elem
            else:
                attr, expected = "string", elem

            if not isinstance(expected, str):
                assert attr == 'type'
                checker = partial(_isinstance_check, klass=expected)
            else:
                flags = re.IGNORECASE if attr == "string" else 0

                if expected[0] != '^':
                    expected = '^' + expected
                if expected[-1] != '$':
                    expected = expected +'$'
                matcher = re.compile(expected, flags)
                checker = partial(_attr_check, attr=attr, matcher=matcher)

            self.leading_tok_l.append(checker)

        if final_tok_str is None:
            self.final_tok_pattern = None
        else:
            self.final_tok_pattern = re.compile(final_tok_str, re.IGNORECASE)

        self.max_len = max_len
        self.min_len = min_len


    def tok_matches(self, tok_l):
        if len(self.leading_tok_l) > len(tok_l):
            return False
        for i, checker in enumerate(self.leading_tok_l):
            if not checker(tok_l[i]):
                return False

        if ((self.final_tok_pattern is not None) and
            (self.final_tok_pattern.match(tok_l[-1].string) is None)):
            #print("final token doesn't match!")
            return False
        elif ((self.max_len is not None) and
              (len(tok_l) > self.max_len)):
            #print("max length doesn't match!")
            return False
        elif ((self.min_len is not None) and
              (len(tok_l) < self.min_len)):
            #print("min length doesn't match!")
            return False

        return True

# the idea is to broadly characterize the chunk kind based on the first
# token(s) in a chunk and optionally the last token in a chunk
# -> this is very much heuristic based. We are not being rigorous.
# -> I think we can get away with this since Grackle doesn't use sophisticated
#    Fortran logic (other codes probably can't do this)
# -> Importantly, reconcilliation with `flang-new`'s AST will serve as a nice
#    sanity check!
class ChunkKind(Enum):
    Uncategorized = auto()
    SubroutineDecl = auto()
    EndRoutine = auto()
    ImplicitNone = auto()
    TypeSpec = auto()
    Parameter = auto()

    # apparently, Continue doesn't actually do anything
    # - it primarily exists to act as a dummy statement that can be labelled
    Continue = auto()

    GoTo = auto()
    Return = auto()
    Call = auto()

    # the write & format commands have some weird syntax!
    Write = auto()
    Format = auto()

    IfConstructStart = auto()
    IfSingleLine = auto()
    ElseIf = auto()
    Else = auto()
    EndIf = auto()

    DoWhileConstructStart = auto()
    DoConstructStart = auto()
    EndDo = auto()

# uncategorized will correspond to assignments and built-in procedures (like alloc, format, write)

class Type(Enum):
    """
    Represents a type.

    This has been customized to store the regex for each kind
    """

    def __new__(cls, value, regex_list):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.regex = re.compile(
            "(" + "|".join([f"({elem})" for elem in regex_list]) + ")",
            re.IGNORECASE
        )
        return obj

    f32 = (0, [r"real", r"real\*4"])
    f64 = (1, [r"real\*8"])
    i32 = (2, [r"integer", r"integer\*4"])
    i64 = (3, [r"integer\*8"])
    logical = (4, [r"logical"])
    mask_type = (5, ["mask_type"])
    gr_float = (6, ["r_prec"])

class _DummyMatcher:
    def match(self, *args, **kwargs):
        return None

class Operator(Enum):
    """
    Represents a binary or unary operation

    This has been customized to store the regex for each kind and to track
    whether it is a binary operation.
    """

    def __new__(cls, value, regex_list, is_binary_op):
        obj = object.__new__(cls)
        obj._value_ = value
        if isinstance(regex_list, _DummyMatcher):
            obj.regex = regex_list
        else:
            obj.regex = re.compile(
                "(" + "|".join([f"({elem})" for elem in regex_list]) + ")",
                re.IGNORECASE
            )
        obj._is_binary_op = is_binary_op
        return obj

    def always_binary_op(self):
        return isinstance(self._is_binary_op, bool) and self._is_binary_op

    def always_unary_op(self):
        return isinstance(self._is_binary_op, bool) and not self._is_binary_op

    def dependendent_operand_count(self):
        return self._is_binary_op is None

    CONCAT = (1, [r"//"], True) # R712 concat-op
    POW = (2, [r"\*\*"], True) # R708 power-op
    MULTIPLY = (3, [r"\*"], True) # R709 mult-op
    DIVIDE = (4, [r"/"], True) # R709 mult-op

    # the following 4 operations are not directly parsed (we need to convert an
    # existing token to these tokens depending on context)
    ADD = (5, _DummyMatcher(), True) # R710 add-op
    SUB = (6, _DummyMatcher(), True) # R710 add-op
    PLUS_UNARY = (7, _DummyMatcher(), False) # not sure if Fortran supports this
    MINUS_UNARY = (8, _DummyMatcher(), False) # F

    # rel-op: R714
    EQ = (9, [r"==", r"\.EQ\."], True)
    NE = (10, [r"/=", r"\.NE\."], True)
    LE = (11, [r"<=", r"\.LE\."], True)
    LT = (12, [r"<", r"\.LT\."], True)
    GE = (13, [r">=", r"\.GE\."], True)
    GT = (14, [r">", r"\.GT\."], True)

    NOT = (15, [r"\.NOT\."], False) # not-op R719
    AND = (16, [r"\.AND\."], True) # and-op R720
    OR = (17, [r"\.OR\."], True) # or-op R721
    EQV = (18, [r"\.EQV\."], True) # equiv-op R722
    NEQV = (19, [r"\.NEQV\."], True) # equiv-op R722
    DEFINED = (20, [r"\.[a-z]+\."], None) # defined op

_reqs = {
    ChunkKind.SubroutineDecl : _ClassifyReq(
        ["subroutine", ("type","arbitrary-name"), ("type", r"\(")]
    ),
    ChunkKind.EndRoutine : None, # it will be a full token match

    # declaration related chunks
    ChunkKind.ImplicitNone : None, # it will be a full token match
    ChunkKind.TypeSpec : _ClassifyReq([("type", Type)]),
    ChunkKind.Parameter : _ClassifyReq(
        ["parameter", ("type", r"\("), ("type", "arbitrary-name")],
    ),

    # miscellaneous routine-body
    ChunkKind.Continue: None, # it will be a full token match
    ChunkKind.GoTo: _ClassifyReq([("type", "goto")], min_len = 2),
    ChunkKind.Return: _ClassifyReq(["return"]),
    ChunkKind.Call: _ClassifyReq(
        ["call", ("type", "arbitrary-name"), ("type", r"\(")],
    ),

    ChunkKind.Write: _ClassifyReq(["write", ("type", r"\(")]),
    ChunkKind.Format: None, # it will be a full token match

    # if construct
    ChunkKind.IfConstructStart : _ClassifyReq(["if"], "then"),
    ChunkKind.ElseIf : _ClassifyReq([("type", "else if")], "then"),
    ChunkKind.Else : _ClassifyReq(["else"], max_len=1),
    ChunkKind.EndIf : None, # it will be a full token match

    # 1-line if-statement
    # it's REALLY important that this follows ChunkKind.IfConstructStart
    ChunkKind.IfSingleLine : _ClassifyReq(["if"]),

    # Do Statement Related
    ChunkKind.DoWhileConstructStart: _ClassifyReq(["do", "while"], min_len = 3),
    ChunkKind.DoConstructStart : _ClassifyReq(["do"], min_len = 2),
    ChunkKind.EndDo : None, # it will be a full token match 

}

    


def _get_chunk_kind(token_list):
    # assume that the label is already removed if there is one
    ntokens = len(token_list)
    if ntokens == 0:
        raise ValueError()
    elif (ntokens == 1) and isinstance(token_list[0].type, ChunkKind):
        return token_list[0].type

    for kind, req in _reqs.items():
        if req is None:
            continue
        elif req.tok_matches(token_list):
            return kind
    else:
        return ChunkKind.Uncategorized



class Token(NamedTuple):
    type: Union[str, ChunkKind]
    string: str
    lineno: int
    column: int



# this includes pdfs for various standard drafts
# https://github.com/kaby76/fortran?tab=readme-ov-file

_NAME_REGEX = r"[a-z][a-z\d_]*"
_KIND_PARAM_REGEX = f"(?:(?:{_NAME_REGEX})|(?:\\d+))"

# section 4.3.2.1 https://wg5-fortran.org/N1151-N1200/N1191.pdf
def _string_literal():
    opt_prefix = f"({_KIND_PARAM_REGEX}_)"
    single_quotation = r"(?:'(?:[^']|(?:''))*'(?!'))"
    double_quotation = single_quotation.replace("'", "\"")

    pattern = f"{opt_prefix}?({single_quotation}|{double_quotation})"
    assert re.match(pattern, "BOLD_'DON''T'",re.IGNORECASE) is not None
    return pattern

_REAL_PATTERN = (
    r'[-+]?(\d+(\.\d*)?|\.\d+)([ed][-+]?\d+)?'
    f'(_{_KIND_PARAM_REGEX})?'
)



def _make_token_map():
    all_inputs = [

        # the following choice is a catch-all solution
        # (if something comes along later with the same match, that later one gets preference)
        ("arbitrary-name", _NAME_REGEX),

    ]

    # need to put this before logical-literal
    all_inputs += [(e,e.regex) for e in Operator]

    # the following 2 tokens are meant to be used internally. We later replace
    # these with certain operator instances based on context
    all_inputs += [("INTERNAL+", r"\+"), ("INTERNAL-", r"\-")]

    all_inputs += [

        # literal tokens
        # ==============
        ('string-literal', _string_literal()),
        ('real-literal', _REAL_PATTERN),
        # order matters here, we always pick the second match of equal length
        ('integer-literal', f'[-+]?\\d+(_{_KIND_PARAM_REGEX})?'),
        ('logical-literal', r'\.((TRUE)|(FALSE))\.' + f'(_{_KIND_PARAM_REGEX})?'),
        # skip over non-base 10 integer-literals
    ]


    all_inputs += [

        ("assign", r"="),
        (";", r";"),

        # assorted
        # ========
        ("(", r"\("),
        (")", r"\)"),
        (",", r","),
        (":", ":"),
        ("::", r"::"),
        ("%", r"%"),
    ]

    all_inputs += [(e, e.regex) for e in Type]

    map_entries = []
    for name,pattern in all_inputs:
        if isinstance(pattern, str):
            matcher = re.compile(pattern, re.IGNORECASE)
        else:
            matcher = pattern
        map_entries.append((name, matcher, False))

    meta_inputs = [
        # desperation:
        (ChunkKind.Format, r"format[ \t]*(.*)", True),

        # these "tokens" are a little crude (I'm effectively combining tokens
        # together). They should only be used if all tokens on the line match
        #
        # honestly, it might be better to construct these from discrete-tokens
        # after the fact so that we properly handle line continuations
        ("use-iso_c_binding", r"use[ \t]+iso_c_binding", True),
        (ChunkKind.ImplicitNone, r"implicit[ \t]+none", True),
        (ChunkKind.Continue, r"continue", True),
        ("else-if", r"else[ \t]*if", False),
        ("goto", r"go[ \t]*to", False),
        (ChunkKind.EndIf, r"end[ \t]*if", True),
        (ChunkKind.EndDo, r"end[ \t]*do", True),
        (ChunkKind.EndRoutine, r"end", True)
    ]
    for name, pattern, expect_full_line_match in meta_inputs:
        map_entries.append(
            (name, re.compile(pattern, re.IGNORECASE), expect_full_line_match)
        )
    return map_entries


_TOKEN_MAP = _make_token_map()
_INTERNAL_TOKEN_TYPE_MAP = {
    'INTERNAL+' : (Operator.PLUS_UNARY, Operator.ADD),
    'INTERNAL-' : (Operator.MINUS_UNARY, Operator.SUB)
}

def _replace_internal_token_types(token_l, has_label = False):
    # replace all internal token types based on context

    for i, token in enumerate(token_l):
        try:
            unary_op, binary_op = _INTERNAL_TOKEN_TYPE_MAP[token]
        except KeyError:
            continue

        first_token = i == 0 or ((i==1) and has_label)
        if first_token:
            # technically this qualifies as unary, but I don't think there is
            # any reason anybody would actually write code like this. I think
            # it is probably indicative of some kind of error
            raise RuntimeError(
                f"we don't expect to encounter code where '{token.string}' is "
                "the first character in an expression"
            )
        prev_tok = token_l[i-1]

        if prev_tok.string == ';':
            raise RuntimeError("don't expect to see ';'")
        elif prev_tok.type in (
            'arbitrary-name', 'real-literal', 'integer-literal', ')'
        ):
            choice = binary_op
        elif prev_tok.type in (')', '=', ','):
            choice = unary_op
        else:
            raise ValueError("Encountered an unexpected scenario")

        token_l[i] = token._replace(type = choice)

# in fixed form, if the 6th column is occupied by anything other than a blank
# or 0 (including a !), then it is a continuation line!
_CONTINUATION_LINE = re.compile("^     [^ 0]")

# according to https://wg5-fortran.org/N1151-N1200/N1191.pdf
# -> the following special characters are "for operator symbols, bracketing,
#    and various forms of separating and delimiting other lexical tokens
# -> it sounds to me like when you have `<token1> <token2>`, you can separate
#    the whitespace IF either
#      - the last character of <token1> is a special character
#      - the first character of <token2> is a special character
# -> they generically say blank, but I included ' ' and '\t'
_SPECIAL_CHARACTER = "\t =+-*/(),.':!\"%&;<>?$"
assert len(_SPECIAL_CHARACTER) == 22

def scan_chunk(chunk_lines):
    if isinstance(chunk_lines,str):
        chunk_lines = [chunk_lines]

    tokens = []

    trailing_comment_start = []

    skip_regex = re.compile("[ \t]+")

    expect_full_match_token = False

    for lineno, line in enumerate(chunk_lines):
        pos = 0
        is_first_line = (lineno == 0)
        is_last_line_in_chunk = (lineno+1) == len(chunk_lines)
        if not isinstance(line, str):
            # this means hit a comment embedded in the chunk!
            assert lineno != 0
            assert not is_last_line_in_chunk
            continue
        elif not is_first_line:
            assert (pos == 0)
            m = _CONTINUATION_LINE.match(line, pos=0)
            assert m is not None
            pos = m.end()

        while pos < len(line):
            if (m := skip_regex.match(line, pos=pos)):
                pos = m.end()
                continue
            elif line[pos] == '!':
                assert line[pos-1].isspace()
                trailing_comment_start.append((lineno,pos))
                break

            best_match_type = None
            best_match = None
            best_match_requires_full_match = None
            for token_type, pattern, token_requires_full_match in _TOKEN_MAP:
                if (m := pattern.match(line, pos=pos)) is None:
                    continue

                if m.end() < len(line):
                    # since this isn't the last token on the line,
                    # lets check if it's a valid match
                    if ((m.group(0)[-1] not in _SPECIAL_CHARACTER) and
                        (line[m.end()] not in _SPECIAL_CHARACTER)):
                        continue

                if ((best_match is None) or
                    (len(m.group(0)) >= len(best_match.group(0)))):
                    best_match_type = token_type
                    best_match = m
                    best_match_requires_full_match = token_requires_full_match

            if best_match is None:
                raise RuntimeError(
                    f"struggling to find a match for: {line!r}\n"
                    f"-> currently at position {pos}\n"
                    f"-> the untokenized string segment is: {line[pos:]!r}"
                )

            tokens.append(Token(
                type=best_match_type,
                string=best_match.group(0),
                lineno=lineno,
                column=pos
            ))
            if best_match_requires_full_match:
                expect_full_match_token = True
            pos=best_match.end()

    has_label = (
        (len(tokens) > 1) and
        (tokens[0].type == "integer-literal") and
        (len(tokens[0].string) <= 5)
    )


    _replace_internal_token_types(tokens, has_label = has_label)
    assert not any(t in _INTERNAL_TOKEN_TYPE_MAP for t in tokens)

    if expect_full_match_token and len(tokens) != (1+has_label):
        raise ValueError(
            "there was a problem parsing line:\n"
            f"  {line!r}\n"
            "we expect there to be a single token on the line "
            "(excluding a label, if present). \n"
            f"-> has_label: {has_label}\n"
            f"-> tokens: {tokens}"
        )

    return tokens, trailing_comment_start, has_label


def process_code_chunk(chunk_lines):
    tokens, trailing_comment_start, has_label = scan_chunk(chunk_lines)
    kind = _get_chunk_kind(tokens[int(has_label):])
    return kind, tokens, trailing_comment_start, has_label

