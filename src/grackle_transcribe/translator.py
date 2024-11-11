import more_itertools

from .identifiers import Constant

from .parser import (
    Expr,
    Stmt,
    AssocExpr,
    AddressOfExpr,
    ArrayAccess,
    CastExpr,
    UnaryOpExpr,
    NonPOWBinaryOpSeqExpr,
    POWOpExpr,
    FnEval,
    IdentifierExpr,
    LiteralExpr,
    ScalarAssignStmt,
    ArrayAssignStmt,
    Standard1TokenStmt,
    IfSingleLineStmt,
    IfConstructStartStmt,
    ElseIfStmt,
    CallStmt,
    DoStmt,
    DoWhileStmt,
    GoToStmt,
    ImpliedDoList,
    WriteStmt,
)

from .stringify import (
    TokenStringifyDirective, PRESERVE_TOK, SKIP_TOK,
    TokenAlternative, concat_translated_pairs,
)


from .token import (
    Token, Literal, Operator, Keyword, token_has_type, Type
)

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from itertools import chain
from typing import Any, List, NamedTuple, Optional, Tuple, Union

def _get_translated_label(label_tok):
    assert isinstance(label_tok, Token) and label_tok.type is Literal.integer
    return f"label_{label_tok.string}"

_TYPE_TRANSLATION_INFO = {
    Type.i32: ("int", "%d"),
    Type.i64: ("long long", "%lld"),
    Type.f32: ("float", "%g"),
    Type.f64: ("double", "%g"),
    Type.gr_float: ("gr_float", "%g"),
    Type.mask_type: ("gr_mask_type", "%d")
}

_TYPE_MAP = dict((k,v[0]) for k,v in _TYPE_TRANSLATION_INFO.items())

# we may want to revisit these to ensure consistency!
_custom_fns = {
    ('pow', 2) : 'std::pow',
    ('min', 2) : 'std::fmin',
    ('min', 3) : 'grackle::impl::fmin',
    ('min', 4) : 'grackle::impl::fmin',
    ('max', 2) : 'std::fmax',
    ('max', 3) : 'grackle::impl::fmax',
    ('log', 1) : 'std::log',
    ('exp', 1) : 'std::exp',
    ('abs', 1) : 'std::fabs',
    ('dabs', 1) : 'grackle::impl::dabs',
    ('sqrt', 1) : 'std::sqrt',
    ('mod', 2) : 'grackle::impl::mod'
}

# I don't love how I'm currently modelling types, but I don't know what the
# optimal way to do it is yet. leaving it for now...
class _CppTypeModifier(Enum):
    def __new__(cls, value, array_rank):
        if isinstance(value, auto):
            value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        obj.array_rank = array_rank
        return obj

    NONE = (auto(), None)
    scalar_pointer = (auto(), None)
    array_pointer_1D = (auto(), 1)
    vector = (auto(), 1)
    View2D = (auto(), 2)
    View3D = (auto(), 3)
    MACRO_CONST = (auto(), None)

class _CppType(NamedTuple):
    type : Type
    modifier : _CppTypeModifier

def _get_cpp_type(name, fortran_identifier_spec):
    # in the future, we should probably create an object that stores the type
    # (rather than always inferring it on the fly)

    fortran_var = fortran_identifier_spec[name]
    is_arg = fortran_identifier_spec.is_arg(name)
    if getattr(fortran_var, 'array_spec', None) is None:
        rank = None
    else:
        rank = fortran_var.array_spec.rank

    if isinstance(fortran_var, Constant):
        if fortran_var.is_macro:
            modifier = _CppTypeModifier.MACRO_CONST
        else:
            modifier = _CppTypeModifier.NONE
    elif (rank is None) and is_arg:
        modifier = _CppTypeModifier.scalar_pointer
    elif rank is None:
        modifier = _CppTypeModifier.NONE
    elif (rank == 1) and is_arg:
        modifier = _CppTypeModifier.array_pointer_1D
    elif rank == 1:
        modifier = _CppTypeModifier.vector
    elif fortran_var.array_spec.rank == 2:
        modifier = _CppTypeModifier.View2D
    elif fortran_var.array_spec.rank == 3:
        modifier = _CppTypeModifier.View3D
    else:
        raise RuntimeError("should be unreachable")

    assert rank == modifier.array_rank, "sanity check!"

    return _CppType(fortran_var.type, modifier)


# to do start using the following identifiers when passing querying the
# identifier string from _IdentifierModel

class IdentifierUsage(Enum):
    ScalarValue = auto()
    ScalarAddress = auto()
    ArrValue = auto() # for numpy-like operations
    ArrAddress = auto()
    ArrAccessValue = auto()
    ArrAccessAddress = auto()


class _IdentifierModel:
    # need to model the C++ data-type so we can properly provide .data
    def __init__(self, fortran_identifier_spec):
        self.fortran_identifier_spec = fortran_identifier_spec

    def fortran_identifier_props(self, name):
        return self.fortran_identifier_spec[name]

    def cpp_variable_name(
        self, fortran_identifier, identifier_usage, arr_ndim = None
    ):
        """
        This produces the string C++ variable name that corresponds to the
        Fortran variable name

        Parameters
        ----------
        fortran_identifier: str or IdentifierExpr
            name of the fortran variable
        identifier_usage
            describes how the identifier gets used
        """
        # I think we will want to get a lot more sophisticated, (and maybe
        # break this functionality out into a separate class/function) but to
        # start out we do something extremely simple
        # -> we might want to eventually return an intermediate class that
        #    the translator then uses to get a pointer address or access value
        #    based on context

        # we assume that the variable name is unchanged
        if isinstance(fortran_identifier, IdentifierExpr):
            var_name = fortran_identifier.token.string
        else:
            var_name = fortran_identifier

        cpptype = _get_cpp_type(var_name, self.fortran_identifier_spec)
        modifier = cpptype.modifier

        match identifier_usage:
            case IdentifierUsage.ScalarValue | IdentifierUsage.ScalarAddress:
                need_ptr = (identifier_usage == IdentifierUsage.ScalarAddress)
                if arr_ndim is not None:
                    raise ValueError(
                        "it makes no sense to specify arr_ndim"
                    )
                elif (
                    (modifier == _CppTypeModifier.NONE) or
                    (modifier == _CppTypeModifier.MACRO_CONST)
                ):
                    return f'&{var_name}' if need_ptr else var_name
                elif modifier == _CppTypeModifier.scalar_pointer:
                    return var_name if need_ptr else f'(*{var_name})'

            case IdentifierUsage.ArrAddress:
                if (arr_ndim is not None) and (arr_ndim != modifier.array_rank):
                    raise ValueError(
                        "the identifier doesn't have the expected rank"
                    )
                elif modifier == _CppTypeModifier.array_pointer_1D:
                    return var_name
                elif (
                    (modifier == _CppTypeModifier.vector) or
                    (modifier == _CppTypeModifier.View2D) or
                    (modifier == _CppTypeModifier.View3D)
                ):
                    return f'{var_name}.data()'

            case IdentifierUsage.ArrAccessValue:
                _valid_modifiers = (
                    _CppTypeModifier.array_pointer_1D,
                    _CppTypeModifier.vector,
                    _CppTypeModifier.View2D,
                    _CppTypeModifier.View3D
                )
                if (arr_ndim is not None) and (arr_ndim != modifier.array_rank):
                    raise ValueError(
                        "the identifier doesn't have the expected rank"
                    )
                elif modifier in _valid_modifiers:
                    return var_name

        raise NotImplementedError(
            "Something went very wrong! Can't handle:\n"
            f" -> identifier_usage: {identifier_usage}\n"
            f" -> modifier: {modifier}")




# here is the (WORKING) organizational plan:
# ==========================================
# we generally translate statements
# -> then this generically dispatches to translate expressions or particular
#    operators
#
# -> anytime an ArgList is involved we will require special handling.
#    -> an evaluation of a builtin function (FnEval) gets treated
#       separately from any other kind
#    -> for procedure calls, we need to take some care.
#       -> allocate/deallocate get special handling!
#       -> user-defined procedures will need more attention.
#          -> All expressions within an arg-list that are `IdentifierExpr`
#             and don't correspond to macros are easy to translate (they just
#             become pointers)
#          -> We will need to introduce temporary variables for EVERY other
#             expression
#               -> Question: for a procedure my_routine(x) that modifies x,
#                  how does fortran handle `x` in `call my_routine(real(x,8))`?
#                  Does it automagically propogate changes?
@dataclass(slots=True)
class Translation:
    """
    Represents a translation of an entity that spans 1 or more tokens.

    This exists to support our (annoying/questionable) goal of retaining the
    source code's strucutre.
    - This mandates that we need to map a single token to a single string or
      a token-collection to a single string
    - Previously, we had translation functions return lists of these pairs and
      we built up 1 long list. While this worked, it was undesirable for a few
      reasons:
       1. it was difficult to describe what we were returning
       2. we were tossing out information which made things difficult to debug
          (e.g. from just looking at a translation you may not know that it
          was a BinaryOpSeqExpr translation).
       3. It was cumbersome to modify the translation.

    Notes
    -----
    We may ultimately move away from this implementation if we come up with
    something better
    """
    # holds either a Token or anything we can apply _iterate_tokens to
    ref: Any
    # reminder, a value of None means that we retain the token's value
    val: Union[TokenStringifyDirective, str, List[Any]]
    prepend_append_pair: Tuple[Optional[str], Optional[str]] = (None, None)

    def __post_init__(self):
        if isinstance(self.val, list):
            assert all(isinstance(elem, Translation) for elem in self.val)
        elif not isinstance(self.val, (TokenStringifyDirective, str)):
            raise TypeError(
                "val should be list of Translations, a string, or a "
                "TokenStringifyDirective.\n"
                f" -> current value: {self.val}"
            )

        assert len(self.prepend_append_pair) == 2
        assert all(((e is None) or isinstance(e, str))
                   for e in self.prepend_append_pair)

    def __iter__(self):
        _islist = isinstance(self.val, list)
        itr = chain(*self.val) if _islist else iter([(self.ref, self.val)])
        
        prepend, append = self.prepend_append_pair

        if (prepend is None) and (append is None):
            yield from itr
            return None

        # holds the most recent (ref, val) pair with val != SKIP_TOK
        next_noskip = None
        # holds all pairs, with val==SKIP_TOK, since next_noskip
        subsequent_skip_pairs = []

        for pending in itr:  
            if pending[1] is SKIP_TOK:
                subsequent_skip_pairs.append(pending)
                continue

            if next_noskip is not None:
                yield next_noskip
            else:
                # since next_noskip is None, this means pending is the first
                # pair we encountered, where `pending[1] is not SKIP_TOK`
                if prepend is not None:
                    yield (TokenAlternative.PREPEND_TO_NEXT, prepend)

            yield from subsequent_skip_pairs
            subsequent_skip_pairs.clear()

            next_noskip = pending

        else: # executed once all pairs it itr consumed
            if next_noskip is None:
                # this means that a pair with pair[1] != SKIP_TOK wasn't ever
                # encountered
                raise RuntimeError(
                    "A Translation is unable to prepend/append any strings if "
                    "it only contains SKIP_TOK translations"
                )
            if append is not None:
                yield (TokenAlternative.APPEND_TO_NEXT, append)
            yield next_noskip
            yield from subsequent_skip_pairs

@dataclass
class _DummyInjectedExpr(Expr):
    # this is injected (for bookkeeping purposes)
    wrapped_l: List[Expr]

    def __post_init__(self):
        assert isinstance(self.wrapped_l, list)
        assert len(self.wrapped_l) > 0

    @property
    def src(self): return wrapped.src

    def iter_contents(self): yield from wrapped_l



def _binOperator_to_str(tok):
    assert token_has_type(tok, Operator)

    op_type = tok.type
    if not op_type.is_always_unary_op:
        raise NotImplementedError(
            "I suspect we may want to convert this to a runtime/value error "
            "and handle non-binary operators separately\n\n"
            "I have some vague concerns about concatentation issues, but "
            "I haven't fully thought it through yet (they may be unfounded)"
        )

    match op_type:
        case Operator.DEFINED: raise NotImplementedError()
        case Operator.CONCAT: raise NotImplementedError()
        case Operator.POW:
            raise NotImplementedError(
                "this case should never come up. It requires special "
                "intervention!"
            )
        case Operator.MULTIPLY: return "*"
        case Operator.DIVIDE: return "/"
        case Operator.ADD: return "+"
        case Operator.SUB: return "-"
        case Operator.PLUS_UNARY: raise NotImplementedError()
        case Operator.MINUS_UNARY: raise NotImplementedError()

        case Operator.EQ: return "=="
        case Operator.NE: return "!="
        case Operator.LE: return "<="
        case Operator.LT: return "<"
        case Operator.GE: return ">="
        case Operator.GT: return ">"
        
        case Operator.NOT: raise NotImplementedError()

        # at the time of writing, our strategy for concatenating converted
        # tokens is rudamentary (we simply base the minimum space between
        # tokens based on spaces between existing Fortran tokens). I don't
        # think this extra padding is necessary, but we put it in just in case
        case Operator.AND: return " && "
        case Operator.OR: return " || "
        case Operator.EQV: raise NotImplementedError()
        case Operator.NEQV: raise NotImplementedError()
        case _:
            raise RuntimeError(f"No handling for {op_type}")

def _strip_known_trailing_kind(tok):
    # to be used with integers and floats
    s = tok.string.upper()
    for known_kind in ("_DKIND", "_DIKIND", "_RKIND"):
        if s.endswith(known_kind):
            stripped = tok.string[:-len(known_kind)]
            kind = tok.string[(1-len(known_kind)):]
            assert '_' not in stripped
            return stripped, kind
    if '_' in s:
        raise NotImplementedError(f"{s} has a literal of unknown kind")
    return tok.string, None

def _Literal_to_str(tok):
    assert token_has_type(tok, Literal)
    # we may need to pass in a context object to help with interpretting some
    # literals

    match tok.type:
        case Literal.string:
            if tok.string[0] == '"' and tok.string[-1] == '"':
                bounding_quote = '"'
            elif tok.string[0] == "'" and tok.string[-1] == "'":
                bounding_quote = "'"
            else:
                # there is probably a _KIND suffix (this probably won't come up)
                raise NotImplementedError()

            str_val = tok.string[1:-1]
            if bounding_quote in str_val:
                # apply escaping rule: probably not necessary
                escaped_quote = bounding_quote*2
                n_escaped_quotes = str_val.count(escaped_quote)*2
                if (2*n_escaped_quotes) != str_val.count(bounding_quote):
                    raise RuntimeError("Something is very wrong!!!")
                str_val = bounding_quote.join(str_val.split(escaped_quote))
            return '"' + str_val.replace('"', r'\"') + '"'

        case Literal.integer:
            stripped, kind = _strip_known_trailing_kind(tok)
            if kind is not None:
                assert kind.upper() in ("DKIND", "DIKIND")
                return f'{stripped}LL'
            return stripped

        case Literal.real:
            stripped, kind = _strip_known_trailing_kind(tok)
            stripped = stripped.lower()

            if ('d' in stripped):
                tmp, prefer_double = stripped.replace('d', 'e'), True
            else:
                tmp, prefer_double = stripped, False

            if kind is None:
                prefix = ''
                suffix = '' if prefer_double else 'f'
            elif kind.upper() in ("DKIND", "DIKIND"):
                prefix, suffix = '', ''
            else:
                assert kind.upper() == 'RKIND'
                ctype_name = _TYPE_TRANSLATION_INFO[Type.gr_float][0]
                prefix, suffix = f'({ctype_name})(', ')'
            return f'{prefix}{tmp}{suffix}'

        case _:
            raise NotImplementedError()

def _NonPOWBinaryOpSeqExpr_translations(arg, identifier_model):
    """
    we could be a lot more careful here

    In particular, we could give some thought to dtype inference (to try to
    ensure consistency)
    """
    val = []
    for elem in arg.seq:
        if isinstance(elem, Expr):
            val.append(_translate_expr(elem, identifier_model))
        else:
            val.append(Translation(elem, _binOperator_to_str(elem)))
    return val

def _CastExpr_translations(arg, identifier_model, ret_desttype = False):
    name_tok = arg.name
    if name_tok.string.lower() == "real":
        mapping = {"4" : Type.f32,
                   "8" : Type.f64,
                   "rkind" : Type.gr_float,
                   "dkind" : Type.f64}
    elif name_tok.string.lower() == "int":
        mapping = {"4" : Type.i32, "8" : Type.i64}
    else:
        raise NotImplementedError()
    arg_l = arg.arg_l
    nargs = more_itertools.ilen(arg_l.get_args())

    if nargs != 2:
        raise NotImplementedError()

    input_expr, comma_tok, kind_expr = arg_l.seq
    kind_tok = kind_expr.token

    desttype = mapping[kind_tok.string.lower()]
    if ret_desttype:
        return desttype

    type_name = _TYPE_MAP[desttype]

    val = [
        Translation(name_tok, f'({type_name})'),
        Translation(arg_l.left, PRESERVE_TOK),
        _translate_expr(input_expr, identifier_model),
        Translation(comma_tok, SKIP_TOK),
        Translation(kind_tok, SKIP_TOK),
        Translation(arg_l.right, PRESERVE_TOK)
    ]
    return val

def _arglist_translation_and_count(arg_l, identifier_model,
                                   is_index_list=False):
    val = [None] # a placeholder
    nargs = more_itertools.ilen(arg_l.get_args())
    cur_arg_index = -1
    for elem in arg_l.seq:
        if isinstance(elem, Expr):
            cur_arg_index += 1
            translation = _translate_expr(elem, identifier_model)
            if is_index_list:
                # From a correctness standpoint, it might be better if we
                # injected an IndexArgExpr wrapper inside of the expression
                # hierarchy before translation (i.e. during parsing or in an
                # intermediate phase)
                suffix = "-1"
            else:
                suffix = ""

            # this is a really crude hack! (to override comma placement so that
            # it is always adjacent to the token)
            if ((cur_arg_index+1) < nargs):
                suffix += ","

            dummy = _DummyInjectedExpr([elem])
            val.append( Translation(dummy, [translation], (None, suffix)) )
        else:
            assert isinstance(elem, Token) and token_has_type(elem, ",")
            val.append(Translation(elem, SKIP_TOK))
    val.append(None) # another placeholder

    # replace the placeholders
    use_bracket = is_index_list and (nargs == 1)
    val[0]  = Translation(arg_l.left,  '[' if use_bracket else PRESERVE_TOK)
    val[-1] = Translation(arg_l.right, ']' if use_bracket else PRESERVE_TOK)
    return Translation(arg_l, val), nargs

def _translate_expr(arg, identifier_model):

    prepend_append_pair = (None, None)
    # the idea is that this translates the lowest level parts of a statement
    match arg:
        case LiteralExpr():
            val = _Literal_to_str(arg.token)
        case AddressOfExpr():
            if isinstance(arg.wrapped, IdentifierExpr):
                # the way we are getting usage here is slightly undesirable...
                # - we could be using this as a safety check since the parent
                #   context presumably knows what we are looking for
                tmp = identifier_model.fortran_identifier_props(arg.wrapped.token.string)
                if getattr(tmp, 'array_spec', None) is None:
                    usage = IdentifierUsage.ScalarAddress
                else:
                    usage = IdentifierUsage.ArrAddress
                val = identifier_model.cpp_variable_name(
                    arg.wrapped.token.string, usage
                )
            elif isinstance(arg.wrapped, ArrayAccess):
                arr_access_translation = _translate_expr(
                    arg.wrapped, identifier_model
                )
                val = [arr_access_translation]
                prepend_append_pair = ('&', None)
            else:
                # I think we will need to get very creative to work around this
                # case (we will need to inject new local variables)
                raise NotImplementedError() 
        case IdentifierExpr():
            val = identifier_model.cpp_variable_name(
                arg.token.string, IdentifierUsage.ScalarValue
            )
        case ArrayAccess():
            idx_trans, n_indices = _arglist_translation_and_count(
                arg.arg_l, identifier_model, is_index_list=True
            )
            identifier_str = identifier_model.cpp_variable_name(
                arg.array_name.token.string,
                IdentifierUsage.ArrAccessValue,
                arr_ndim=n_indices
            )
            val = [
                Translation(arg.array_name.token, identifier_str), idx_trans
            ]
        case AssocExpr():
            val = [
                Translation(arg.left, "("),
                _translate_expr(arg.expr, identifier_model),
                Translation(arg.right, ")")
            ]
        case CastExpr():
            val = _CastExpr_translations(arg, identifier_model)
        case UnaryOpExpr():
            if arg.op.type not in [Operator.PLUS_UNARY, Operator.MINUS_UNARY]:
                raise NotImplementedError()
            val = [
                Translation(arg.op, PRESERVE_TOK),
                _translate_expr(arg.operand, identifier_model)
            ]
        case POWOpExpr():
            val = [
                _translate_expr(arg.base, identifier_model),
                Translation(arg.pow_tok, ","),
                _translate_expr(arg.exponent, identifier_model)
            ]
            prepend_append_pair = (_custom_fns["pow", 2]+"(", ")")
        case NonPOWBinaryOpSeqExpr():
            val = _NonPOWBinaryOpSeqExpr_translations(arg, identifier_model)
        case FnEval():
            fn_name, arg_l = arg.fn_name, arg.arg_l
            nargs = more_itertools.ilen(arg_l.get_args())
            try:
                fn_name_trans = Translation(
                    fn_name, _custom_fns[fn_name.string, nargs]
                )
            except KeyError:
                raise NotImplementedError()
            arg_l_trans, _ = _arglist_translation_and_count(
                arg.arg_l, identifier_model, is_index_list=False
            )
            val = [fn_name_trans, arg_l_trans]
        case _:
            raise NotImplementedError()

    return Translation(arg, val, prepend_append_pair)

def _handle_write_stmt(stmt, identifier_model):
    if stmt.writes_to_stdout():
        fn = "printf"
    elif stmt.writes_to_stderr():
        fn = "eprintf"
    else:
        raise RuntimeError()


    if stmt.format_specifier() is not None:
        raise NotImplementedError()
    elif isinstance(stmt.output_list, ImpliedDoList):
        raise NotImplementedError()

    fmt_str_parts = []
    fn_args = []

    for i, elem in enumerate(stmt.output_list):
        if (i % 2) == 1:
            assert elem.string == ','
            continue

        if isinstance(elem, LiteralExpr):
            if token_has_type(elem.token, Literal.string):
                fmt_str_parts.append(elem.token.string[1:-1])
                continue
            raise NotImplementedError()
        elif isinstance(elem, (ArrayAccess, IdentifierExpr)):
            if isinstance(elem, IdentifierExpr):
                var_name = elem.token.string
            else:
                var_name = elem.array_name.token.string
            arg_type = _get_cpp_type(
                var_name, identifier_model.fortran_identifier_spec
            ).type

        elif isinstance(elem, CastExpr):
            arg_type = _CastExpr_translations(elem, identifier_model,
                                              ret_desttype=True)
        elif isinstance(elem, FnEval):
            # for now, we are just targetting a particular line of source code
            if elem.fn_name.string.lower() != 'abs':
                raise NotImplementedError()
            arg_iter = elem.arg_l.get_args()
            arg = next(arg_iter)
            if not isinstance(arg, NonPOWBinaryOpSeqExpr):
                raise NotImplementedError()
            elif not isinstance(arg.seq[0], LiteralExpr):
                raise NotImplementedError()
            elif arg.seq[0].token.string.lower() != "0.1_dkind":
                raise NotImplementedError()
            arg_type = Type.f64
        else:
            raise NotImplementedError()

        pair_itr = _translate_expr(elem, identifier_model)
        fn_arg = concat_translated_pairs(pair_itr)
        fn_args.append(fn_arg)

        formatter = _TYPE_TRANSLATION_INFO[arg_type][1]
        fmt_str_parts.append(formatter)

    fmt_str_literal = f'"{" ".join(fmt_str_parts)}\\n"'
    if len(fn_args) == 0:
        full_stmt = f'{fn}({fmt_str_literal});'
    else:
        # we make a crude attempt at line wrapping
        indent_len = len(fn) + 1
        len_approx = len(fmt_str_literal) + sum(len(e) for e in fn_args)
        if (indent_len + len_approx) > 66:
            sep = ',\n' + (' '* indent_len)
        else:
            sep = ', '
        full_stmt = f'{fn}({fmt_str_literal}{sep}{sep.join(fn_args)});'
    return full_stmt, False

def _translate_ScalarAssignStmt(stmt, identifier_model):
    t = Translation(
        stmt,
        [
            _translate_expr(stmt.lvalue, identifier_model),
            Translation(stmt.assign_tok, PRESERVE_TOK),
            _translate_expr(stmt.rvalue, identifier_model)
        ]
    )
    return t

def _translate_ArrayAssignStmt(stmt, identifier_model):
    # we are only going to handle small subsets of this
    lvalue, rvalue = stmt.lvalue, stmt.rvalue
    if not isinstance(lvalue,IdentifierExpr):
        raise NotImplementedError()
    l_var = identifier_model.fortran_identifier_props(lvalue.token.string)
    dst_arg = identifier_model.cpp_variable_name(
        l_var.name, IdentifierUsage.ArrAddress
    )

    if isinstance(rvalue, IdentifierExpr):
        r_var = identifier_model.fortran_identifier_props(rvalue.token.string)
        assert l_var.type == r_var.type
        assert l_var.array_spec.rank == r_var.array_spec.rank
        # we just assume that the shapes are the same (if they aren't, then the
        # input fortran code is wrong!)
        fn = 'std::memcpy'
        input_arg = identifier_model.cpp_variable_name(
            r_var.name, IdentifierUsage.ArrAddress
        )
    elif isinstance(rvalue, LiteralExpr):
        s = _Literal_to_str(rvalue.token)
        is_zero = (
            (s.lower().endswith('ll') and (int(s[:-2]) == 0)) or
            (s.lower().endswith('f') and (float(s[:-1]) == 0.0)) or
            (float(s) == 0.0)
        )
        if is_zero:
            fn = 'std::memset'
            input_arg = '0'
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()


    if l_var.array_spec.rank == 1:
        tmp = l_var.array_spec.axlens[0]
        n_elem_str = identifier_model.cpp_variable_name(
            tmp.token.string, IdentifierUsage.ScalarValue
        )
    else:
        raise NotImplementedError()

    cpp_type_name = _TYPE_TRANSLATION_INFO[l_var.type][0]
    count_arg = f'sizeof({cpp_type_name})*{n_elem_str}'

    return f'{fn}({dst_arg}, {input_arg}, {count_arg});', False


    

def _translate_GoToStmt(stmt, idenfitier_model):
    label_name = _get_translated_label(stmt.label_expr.token)
    t = Translation(
        stmt,
        [
            Translation(stmt.goto_tok, "goto"),
            Translation(stmt.label_expr, label_name),
        ],
    )
    return t

def _translate_stmt(stmt, identifier_model):
    """
    Returns
    -------
    out: string or list
        A string to completely replace the statement or a sequence of args to
        pass to a builder
    append_semicolon: bool
    """

    match stmt:
        case ArrayAssignStmt():
            return _translate_ArrayAssignStmt(stmt, identifier_model)
        case ScalarAssignStmt():
            out = list(_translate_ScalarAssignStmt(stmt, identifier_model))
            return out, True

        case GoToStmt():
            out = list(_translate_GoToStmt(stmt, identifier_model))
            return out, True

        case IfSingleLineStmt():
            if isinstance(stmt.consequent, ScalarAssignStmt):
                consequent_trans = _translate_ScalarAssignStmt(
                    stmt.consequent, identifier_model
                )
            elif isinstance(stmt.consequent, GoToStmt):
                consequent_trans = _translate_GoToStmt(
                    stmt.consequent, identifier_model
                )
            else: # ArrayAssignStmt or ...
                raise NotImplementedError()
            dummy = _DummyInjectedExpr([stmt])
            consequent_trans = Translation(
                dummy, [consequent_trans], ("{ ", "; }")
            )

            pairs = (
               [(stmt.if_tok, "if")] +
                list(_translate_expr(stmt.condition, identifier_model)) + 
                list(consequent_trans)
            )
            return pairs, False
        case IfConstructStartStmt():
            pairs = (
                [(stmt.if_tok, "if")] +
                list(_translate_expr(stmt.condition, identifier_model)) +
                [(stmt.then_tok, SKIP_TOK)]
            )
            return pairs, False
        case ElseIfStmt():
            pairs = (
                [(stmt.elseif_tok, "else if")] +
                list(_translate_expr(stmt.condition, identifier_model)) +
                [(stmt.then_tok, SKIP_TOK)]
            )
            return pairs, False

        case DoStmt():
            pairs = list(_translate_ScalarAssignStmt(
                stmt.init_stmt, identifier_model
            ))
            init_stmt = concat_translated_pairs(pairs)

            if isinstance(pairs[0][0], IdentifierExpr):
                itr_var = pairs[0][1]
            else:
                raise NotImplementedError()

            limit_expr = concat_translated_pairs(_translate_expr(
                stmt.limit_expr, identifier_model
            ))

            if stmt.has_increment_expr:
                raise NotImplementedError()
            else:
                if itr_var[0] == '*':
                    itr_expr = f'({itr_var})++'
                else:
                    itr_expr = f'{itr_var}++'
            out = f"for ({init_stmt}; {itr_var}<=({limit_expr}); {itr_var}++)"
            return out, False
        case DoWhileStmt():
            pairs = (
                [(stmt.dowhile_tok, "while")] +
                list(_translate_expr(stmt.condition, identifier_model))
            )
            return pairs, False
        case CallStmt():
            leading_part = [
                (stmt.call_tok, SKIP_TOK),
                (stmt.subroutine, f"FORTRAN_NAME({stmt.subroutine.string})"),
            ]
            trailing_part, _ = _arglist_translation_and_count(
                stmt.arg_l, identifier_model, is_index_list=False
            )
            return (leading_part + list(trailing_part)), True
        case WriteStmt():
            return _handle_write_stmt(stmt, identifier_model)
        case Standard1TokenStmt():
            match stmt.tok_type():
                case Keyword.CONTINUE:
                    raise RuntimeError(
                        "It is an error to parse a statement that just "
                        "consists of Fortran's `continue` keyword.\n"
                        "-> as I understand it, it is a dummy keyword that is "
                        "   used as a placeholder (to do nothing) on labeled "
                        "   lines that are goto destinations\n"
                        "-> Fortran's `cycle` keyword is more similar to C's "
                        "   `continue` keyword"
                    )
                case Keyword.ELSE:
                    return "else", False
                case Keyword.RETURN:
                    return "return;", False
                case _:
                    raise NotImplementedError()
        case _:
            raise NotImplementedError()


    tmp = _translate_helper(
        arg=arg,
        identifier_model=identifier_model,
        is_function_arg=False,
        is_array_access=False
    )

