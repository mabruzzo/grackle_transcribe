# this encodes logic for parsing a single chunk of fortran code

#class Matcher(Enum):
#    UNDEFINED = auto()
#

from .utils import index_non_space

from enum import Enum, auto
from functools import partial
import itertools
import pprint
import re
from typing import Any, Callable, List, NamedTuple, Optional, Tuple, Union

def _attr_check(token, attr, matcher):
    attr_val = getattr(token, attr)
    if not isinstance(attr_val, str):
        return False
    return matcher.match(attr_val) is not None

def token_has_type(tok, type_spec):
    if isinstance(type_spec, type):
        return isinstance(tok.type, type_spec)
    elif isinstance(type_spec, Enum):
        return tok.type == type_spec
    elif isinstance(type_spec, re.Pattern):
        if not isinstance(type_spec, str):
            return False
        return type_spec.match(tok.type) is not None
    return tok.type == type_spec

class _ClassifyReq:
    # we only use regex for case insensitivity (we could get around this by
    # creating more types of tokens, but I think this is the simpler way to
    # do it)
    leading_tok_l: list[Callable]
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
                checker = partial(token_has_type, type_spec=expected)
            else:
                flags = re.IGNORECASE if attr == "string" else 0

                if expected[0] != '^':
                    expected = '^' + expected
                if expected[-1] != '$':
                    expected = expected +'$'
                matcher = re.compile(expected, flags)
                if attr == 'type':
                    checker = partial(token_has_type, type_spec=matcher)
                else:
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



class _NullMatcher:
    def match(self, *args, **kwargs): return None

class _TokenEnum(Enum):
    """
    Generic base-class for enums that represent Token types.

    This supports tokens that are auto-numbered and are specified with a regex
    pattern
    """

    def __new__(cls, value, regex_list, *args):
        if isinstance(value, auto):
            value = len(cls.__members__) + 1

        obj = object.__new__(cls)
        obj._value_ = value
        if isinstance(regex_list, _NullMatcher):
            obj.regex = regex_list
        else:
            obj.regex = re.compile(
                "(" + "|".join([f"({elem})" for elem in regex_list]) + ")",
                re.IGNORECASE
            )
        return obj

    @property
    def req_full_line_match(self):
        return getattr(self, '_req_full_line_match', False)

    def __repr__(self):
        # we are following the advice of the docs and overriding repr to hide
        # the unimportant underlying value
        return '<%s.%s>' % (self.__class__.__name__, self.name)

# this includes pdfs for various standard drafts
# https://github.com/kaby76/fortran?tab=readme-ov-file

_NAME_REGEX = r"[a-z][a-z\d_]*"

class BuiltinFn(_TokenEnum):
    abs   = (auto(), [r'abs'])
    dabs  = (auto(), [r'dabs'])
    exp   = (auto(), [r"exp"])
    log   = (auto(), [r"log"])
    log10 = (auto(), [r"log10"])
    max   = (auto(), [r'max'])
    maxval = (auto(), [r'maxval'])
    min   = (auto(), [r'min'])
    mod   = (auto(), [r"mod"])
    int   = (auto(), [r"int"])
    sqrt  = (auto(), [r"sqrt"])

class BuiltinProcedure(_TokenEnum):
    allocate   = (auto(), [r'allocate'])
    deallocate = (auto(), [r'deallocate'])

class Type(_TokenEnum):
    """Represents a type."""

    f32       = (auto(), [r"real", r"real\*4"])
    f64       = (auto(), [r"real\*8"])
    i32       = (auto(), [r"integer", r"integer\*4"])
    i64       = (auto(), [r"integer\*8"])
    logical   = (auto(), [r"logical"])
    mask_type = (auto(), ["mask_type"])
    gr_float  = (auto(), ["r_prec"])


class Operator(_TokenEnum):
    """Represents a binary or unary operation
    """

    def __init__(self, value, regex_list, is_binary_op):
        self._is_binary_op = is_binary_op

    def is_always_binary_op(self):
        return isinstance(self._is_binary_op, bool) and self._is_binary_op

    def is_always_unary_op(self):
        return isinstance(self._is_binary_op, bool) and not self._is_binary_op

    def dependendent_operand_count(self):
        return self._is_binary_op is None


    # this has to go first so that it doesn't override other operators
    DEFINED     = (auto(), [r"\.[a-z]+\."], None) # defined op
    CONCAT      = (auto(), [r"//"], True) # R712 concat-op
    POW         = (auto(), [r"\*\*"], True) # R708 power-op
    MULTIPLY    = (auto(), [r"\*"], True) # R709 mult-op
    DIVIDE      = (auto(), [r"/"], True) # R709 mult-op

    # the following 4 operations are not directly parsed (we need to convert an
    # existing token to these tokens depending on context)
    ADD         = (auto(), _NullMatcher(), True) # R710 add-op
    SUB         = (auto(), _NullMatcher(), True) # R710 add-op
    PLUS_UNARY  = (auto(), _NullMatcher(), False) # does Fortran support this?
    MINUS_UNARY = (auto(), _NullMatcher(), False) # negates sign

    # rel-op: R714
    EQ          = (auto(), [r"==", r"\.EQ\."], True)
    NE          = (auto(), [r"/=", r"\.NE\."], True)
    LE          = (auto(), [r"<=", r"\.LE\."], True)
    LT          = (auto(), [r"<", r"\.LT\."], True)
    GE          = (auto(), [r">=", r"\.GE\."], True)
    GT          = (auto(), [r">", r"\.GT\."], True)

    NOT         = (auto(), [r"\.NOT\."], False) # not-op R719
    AND         = (auto(), [r"\.AND\."], True) # and-op R720
    OR          = (auto(), [r"\.OR\."], True) # or-op R721
    EQV         = (auto(), [r"\.EQV\."], True) # equiv-op R722
    NEQV        = (auto(), [r"\.NEQV\."], True) # equiv-op R722


# the following 2 Enums declare some "meta-tokens" that are effectively 2 or
# more tokens combined together. Some of them should only be used if they are
# the only token on a line.
#
# honestly, it might be better to construct these from discrete-tokens
# after the fact so that we properly handle line continuations

_WRITE_CONTROL_LIST_ARG = rf"([\*\d]|{_NAME_REGEX})"

class Keyword(_TokenEnum):
    def __init__(self, value, regex_list, require_full_line_match):
        self._req_full_line_match = require_full_line_match

    SUBROUTINE = (auto(), [r"subroutine"],    False)
    CALL       = (auto(), [r"call"],          False)
    PARAMETER  = (auto(), [r"parameter"],     False)
    THEN       = (auto(), [r"then"],          False)
    CONTINUE   = (auto(), [r"continue"],      True)
    # technically, this should be False, but for our purposes this is
    # probably ok
    RETURN     = (auto(), [r"return"],        True)
    CYCLE      = (auto(), [r"cycle"],         True)
    GOTO       = (auto(), [r"go[ \t]*to"],    False)
    IF         = (auto(), [r"if"],            False)
    ELSE       = (auto(), [r"else"],          True)
    ELSEIF     = (auto(), [r"else[ \t]*if"],  False)
    DO         = (auto(), [r"do"],            False)
    DOWHILE    = (auto(), [r"do[ \t]*while"], False)
    ENDIF      = (auto(), [r"end[ \t]*if"],   True)
    ENDDO      = (auto(), [r"end[ \t]*do"],   True)
    ENDROUTINE = (auto(), [r"end"],           True)

    WRITE = (auto(), [r"write"], False)

class Misc(_TokenEnum):
    """
    some of these shouldn't really be used... They mostly exist
    "so things work." We may want to revisit them in the near future...
    """
    def __init__(self, value, regex_list, require_full_line_match):
        self._req_full_line_match = require_full_line_match

    use_iso_c_binding = (auto(), [r"use[ \t]+iso_c_binding"], True)
    ImplicitNone      = (auto(), [r"implicit[ \t]+none"], True)

    # this exacts out of desperation to get things working...
    Format            = (auto(), [r"format[ \t]*(.*)"], True)

class Internal(_TokenEnum):
    # the following token-pair are used internally. We later replace
    # these with certain Operator enumerations based on context
    Minus = (auto(), [r"\-"])
    Plus = (auto(), [r"\+"])

_reqs = {
    ChunkKind.SubroutineDecl : _ClassifyReq(
        [("type", Keyword.SUBROUTINE),
         ("type","arbitrary-name"),
         ("type", r"\(")]
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
    ChunkKind.GoTo: _ClassifyReq([("type", Keyword.GOTO)], min_len = 2),
    ChunkKind.Return: _ClassifyReq(["return"]),
    ChunkKind.Call: _ClassifyReq(
        ["call", ("type", "arbitrary-name"), ("type", r"\(")],
    ),

    ChunkKind.Write: _ClassifyReq(["write", ("type", r"\(")]),
    ChunkKind.Format: None, # it will be a full token match

    # if construct
    ChunkKind.IfConstructStart : _ClassifyReq(["if"], "then"),
    ChunkKind.ElseIf : _ClassifyReq([("type", Keyword.ELSEIF)], "then"),
    ChunkKind.Else : _ClassifyReq([("type", Keyword.ELSE)], max_len=1),
    ChunkKind.EndIf : _ClassifyReq([("type", Keyword.ENDIF)], max_len=1),

    # 1-line if-statement
    # it's REALLY important that this follows ChunkKind.IfConstructStart
    ChunkKind.IfSingleLine : _ClassifyReq([("type", Keyword.IF)]),

    # Do Statement Related
    ChunkKind.DoWhileConstructStart: _ClassifyReq(
        [("type", Keyword.DOWHILE)], min_len = 2
    ),
    ChunkKind.DoConstructStart : _ClassifyReq(
        [("type", Keyword.DO)], min_len = 2
    ),
    ChunkKind.EndDo : _ClassifyReq([("type", Keyword.ENDDO)], max_len = 1),

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

class Literal(_TokenEnum):
    string = (auto(), [_string_literal()])

    # order matters for the next 2 entries (if multiple patterns match
    # equivalent the same string segment, we give precedence to the final match)
    real = (auto(), [_REAL_PATTERN])
    integer = (auto(), [f'[-+]?\\d+(_{_KIND_PARAM_REGEX})?'])

    logical = (auto(), [r'\.((TRUE)|(FALSE))\.' + f'(_{_KIND_PARAM_REGEX})?']) 
    # skip over non-base 10 integer-literals


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

def _prep_token_map(*args):
    # each arg is either a:
    # creates a sequence of tuples where each tuple has the form:
    #    `(name, regex-pattern, req_full_line_match)`
    # In more detail
    #   - `name` is a string or enumeration that is stored as a token's type
    #   - `regex-pattern` is an object that has a `.match` method (it's almost
    #     always an instance of `re.Pattern`).
    #   - `req_full_line_match` is a boolean. When `True`, a match to this
    #     token should produce an error unless it is the only token on the
    #     line (labels are obviously ignored)
    map_entries = []
    for elem in itertools.chain(*args):
        if isinstance(elem, tuple):
            name, pattern, req_full_line_match = elem
            matcher = re.compile(pattern, re.IGNORECASE)
        else:
            name, matcher = elem, elem.regex
            req_full_line_match = elem.req_full_line_match
        map_entries.append(
            (name, matcher, req_full_line_match)
        )
    return map_entries
    

class Tokenizer:

    # _token_map holds recipies for identifying tokens.
    #   -> a longer token gets precedence
    #   -> we break ties between types by picking the one later in the list

    _token_map = _prep_token_map(
        # the following choice is designed to match variables/constants. It
        # needs to go first so that it can be overridden.
        [("arbitrary-name", _NAME_REGEX, False)],

        # Operator comes before Literal since it conflicts with Literal.logical 
        iter(Operator),
        Literal,

        # We later replace these tokens with Operator enumerations based on 
        # context.
        # - if you move this before Literal, make sure you adjust the custom
        #   logic that deals with ambiguity of Internal.Minus and a negative
        #   number
        Internal,

        # "punctuation"
        [
            ("assign", r"=", False),
            (";", r";", False),
            ("(", r"\(", False),
            (")", r"\)", False),
            (",", r",", False),
            (":", ":", False),
            ("::", r"::", False),
            ("%", r"%", False),
        ],

        BuiltinFn,
        BuiltinProcedure,
        Type,
        Keyword, 
        Misc
    )

    def tokenize(self, chunk_entries):
        tokens = []
        trailing_comment_start = []
        expect_full_match_token = False

        for lineno, line in enumerate(chunk_entries):
            trailing_comment_start.append(None)
            pos = 0
            is_first_line = (lineno == 0)
            is_last_line_in_chunk = (lineno+1) == len(chunk_entries)
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
                next_nonspace = index_non_space(line, pos=pos, dflt=None)
                if next_nonspace != pos:
                    pos = len(line) if next_nonspace is None else next_nonspace
                    continue
                elif line[pos] == '!':
                    assert line[pos-1].isspace()
                    trailing_comment_start[-1] = pos
                    break

                best_match_type = None
                best_match = None
                best_match_requires_full_match = None
                for token_type, matcher, req_full_match in self._token_map:
                    if (m := matcher.match(line, pos=pos)) is None:
                        continue

                    if m.end() < len(line):
                        # since this isn't the last token on the line,
                        # lets check if it's a valid match
                        if ((m.group(0)[-1] not in _SPECIAL_CHARACTER) and
                            (line[m.end()] not in _SPECIAL_CHARACTER)):
                            continue

                    if (
                        isinstance(token_type, Internal) and
                        isinstance(best_match_type, Literal) and
                        (len(tokens) > 0)
                    ):
                        prev_token_type = prev_token_type = tokens[-1].type
                        override_match = (
                            (prev_token_type in [")", "arbitrary-name"]) or
                             isinstance(prev_token_type, Literal)
                        )
                    else:
                        override_match = (
                            (best_match is None) or
                            (len(m.group(0)) >= len(best_match.group(0)))
                        )

                    if override_match:
                        best_match_type = token_type
                        best_match = m
                        best_match_requires_full_match = req_full_match

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
            (tokens[0].type is Literal.integer) and
            (len(tokens[0].string) <= 5)
        )

        _replace_internal_token_types(tokens, has_label = has_label)
        if any(t.type in _INTERNAL_TOKEN_TYPE_MAP for t in tokens):
            raise AssertionError(
                "somehow an internal token was missed within \n"
                f"{compressed_concat_tokens(tokens)}"
            )

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
    


_INTERNAL_TOKEN_TYPE_MAP = {
    Internal.Plus : (Operator.PLUS_UNARY, Operator.ADD),
    Internal.Minus : (Operator.MINUS_UNARY, Operator.SUB)
}

def _replace_internal_token_types(token_l, has_label = False):
    # replace all internal token types based on context

    for i, token in enumerate(token_l):
        try:
            unary_op, binary_op = _INTERNAL_TOKEN_TYPE_MAP[token.type]
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
            'arbitrary-name', Literal.real, Literal.integer, ')'
        ):
            choice = binary_op
        elif prev_tok.string in ('(', '=', ','):
            choice = unary_op
        else:
            prior_tokens = f'\n  '.join(
                pprint.pformat(token_l[min(i-3, 0):i]).splitlines()
            )
            raise ValueError(
                "Encountered an unexpected scenario\n"
                "-> compressed-token-string:\n"
                f"  {compressed_concat_tokens(token_l)}\n"
                "-> preceeding tokens:\n"
                f"  {prior_tokens}\n"
                "-> current token:\n"
                f"  {token}\n"
                )

        token_l[i] = token._replace(type = choice)

    if any(t.type in _INTERNAL_TOKEN_TYPE_MAP for t in token_l):
        raise AssertionError(
            "somehow an internal token was missed within \n"
            f"{compressed_concat_tokens(token_l)}"
        )


def scan_chunk(chunk_lines):
    if isinstance(chunk_lines,str):
        chunk_lines = [chunk_lines]
    return Tokenizer().tokenize(chunk_lines)



        


def process_code_chunk(chunk_lines):
    tokens, trailing_comment_start, has_label = scan_chunk(chunk_lines)
    kind = _get_chunk_kind(tokens[int(has_label):])
    return kind, tokens, trailing_comment_start, has_label


def compressed_concat_tokens(token_itr):
    # concatenate tokens into a string with as few characters as possible
    token_itr = iter(token_itr)
    tmp = [next(token_itr).string]
    for tok in token_itr:
        if ((tmp[-1][-1] not in _SPECIAL_CHARACTER) and
            (tok.string[0] not in _SPECIAL_CHARACTER)):
            tmp.append(' ')
        tmp.append(tok.string)
    return ''.join(tmp)
