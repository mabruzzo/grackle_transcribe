# this file defines types that represents Fortran syntax entities
# -> essentially they are groupings of tokens (namely statements & expressons)
# -> originally, these were defined in the same file as parsers

from .src_model import Code

from .token import (
    BuiltinFn,
    BuiltinProcedure,
    Keyword,
    Literal, Operator, Token, token_has_type, Type,
    compressed_concat_tokens
)

import dataclasses
from enum import Enum
from typing import Any, final, NamedTuple, Optional, Tuple, Union

class Stmt:
    # baseclass

    @property
    def item(self):
        return self.src # for backwards compatability

# Introduce the Concept of an expression/groups
# -> fundamentally, they are a way of grouping together tokens

class Expr:
    def __str__(self):
        return (
            f"<{self.__class__.__name__}: '{compressed_str_from_Expr(self)}'>"
        )


@dataclasses.dataclass(frozen=True)
class IdentifierExpr(Expr):
    token: Token

    def __post_init__(self): assert self.token.type == "arbitrary-name"
    def iter_contents(self): yield self.token

@dataclasses.dataclass(frozen=True)
class LiteralExpr(Expr):
    token: Token

    def __post_init__(self): assert isinstance(self.token.type, Literal)
    def iter_contents(self): yield self.token

@dataclasses.dataclass(frozen=True)
class ColonExpr(Expr):
    token: Token

    def __post_init__(self): assert self.token.string == ":"
    def iter_contents(self): yield self.token

@dataclasses.dataclass(frozen=True)
class AddressOfExpr(Expr):
    # put in by hand by us for calls to procedures (will make translation a
    # lot easier)
    wrapped: Expr

    def iter_contents(self): yield self.wrapped

@dataclasses.dataclass(frozen=True)
class UnaryOpExpr(Expr):
    op: Token
    operand: Expr

    def __post_init__(self): assert self.op.type.is_always_unary_op()

    def iter_contents(self):
        yield self.op
        yield self.operand

def _check_delim_sequence(seq, odd_index_check):
    if (len(seq) % 2) != 1:
        contents = '[\n' + '\n'.join(f'    {e},' for e in seq) + '\n]'
        raise ValueError(
            "seq must contain an odd number of entries. It "
            f"currently holds: {contents}"
        )
    for i, entry in enumerate(seq):
        is_even = ((i%2) == 0)
        if is_even and not isinstance(entry, Expr):
            raise TypeError(f"element {i} is not an expression")
        elif (not is_even) and not odd_index_check:
            raise ValueError(f"element {i} has an invalid value")

@dataclasses.dataclass(frozen=True)
class NonPOWBinaryOpSeqExpr(Expr):
    # the idea is that we don't want to deal with operator precedence
    seq: Tuple[Union[Expr,Token], ...]
    
    def __post_init__(self):
        _check_delim_sequence(self.seq, lambda e: e.is_always_binary_op())

    def iter_contents(self): yield from self.seq

@dataclasses.dataclass(frozen=True)
class POWOpExpr(Expr):
    # this has highest precedence of the operators (so its okay to deal with
    # it by itself
    base: Expr
    pow_tok: Token
    exponent: Expr

@dataclasses.dataclass(frozen=True)
class AssocExpr(Expr):
    left: Token
    expr: Expr
    right: Token

    def __post_init__(self):
        assert self.left.string == '(' and self.right.string == ')'
        assert isinstance(self.expr, Expr)

    def iter_contents(self):
        yield self.left
        yield self.expr
        yield self.right

@dataclasses.dataclass(frozen=True)
class ArgList:
    # this is NOT an expression, but it contains expressions!
    left: Token
    seq: Tuple[Union[Expr,Token], ...]
    right: Token

    def __post_init__(self):
        assert self.left.string == '(' and self.right.string == ')'
        _check_delim_sequence(self.seq, lambda e: e.string == ',')

    def n_args(self):
        # number of arguments in the argument list
        return sum(divmod(len(self.seq), 2))

    def iter_contents(self):
        yield self.left
        yield from self.seq
        yield self.right

    def get_args(self):
        for i, entry in enumerate(self.seq):
            if (i % 2) == 0:
                yield entry


@dataclasses.dataclass(frozen=True)
class FnEval(Expr):
    fn_name: Token
    arg_l: ArgList

    def __post_init__(self):
        assert isinstance(self.fn_name.type, BuiltinFn)
        assert isinstance(self.arg_l, ArgList)

    def iter_contents(self):
        yield self.fn_name
        yield self.arg_l

@dataclasses.dataclass(frozen=True)
class CastExpr(Expr):
    name: Token
    arg_l: ArgList

    def __post_init__(self):
        assert isinstance(self.name.type, (BuiltinFn, Type))
        assert isinstance(self.arg_l, ArgList)

    def iter_contents(self):
        yield self.name
        yield self.arg_l


@dataclasses.dataclass(frozen=True)
class ArrayAccess(Expr):
    array_name: IdentifierExpr
    arg_l: ArgList

    def iter_contents(self):
        yield self.array_name
        yield self.arg_l


## then we can create special statements

@dataclasses.dataclass(frozen=True)
class UncategorizedStmt(Stmt):
    src: Code
    ast: Any = None

    def iter_contents(self): yield from self.src.tokens


@dataclasses.dataclass(frozen=True)
class Standard1TokenStmt(Stmt):
    src: Code
    token: Token

    def __post_init__(self): assert self.token.type.req_full_line_match

    def iter_contents(self): yield self.token

    def tok_type(self): return self.token.type

@dataclasses.dataclass(frozen=True)
class ScalarAssignStmt(Stmt):
    # this is a standard assignment (C has the builtin equivalent)
    src: Code
    lvalue: Expr
    assign_tok: Token # the '=' token
    rvalue: Expr

    def iter_contents(self): 
        yield self.lvalue
        yield self.assign_tok
        yield self.rvalue

@dataclasses.dataclass(frozen=True)
class ArrayAssignStmt(Stmt):
    # this is the case where we are performing assignments on many array
    # elements at once
    src: Code
    lvalue: Expr
    assign_tok: Token # the '=' token
    rvalue: Expr

    def iter_contents(self): 
        yield self.lvalue
        yield self.assign_tok
        yield self.rvalue

@dataclasses.dataclass(frozen=True)
class GoToStmt(Stmt):
    src: Code
    goto_tok: Token
    label_expr: LiteralExpr

    def __post_init__(self):
        assert token_has_type(self.goto_tok, Keyword.GOTO)
        assert token_has_type(self.label_expr.token, Literal.integer)

    def iter_contents(self):
        yield self.goto_tok
        yield self.label_expr

@dataclasses.dataclass(frozen=True)
class CallStmt(Stmt):
    src: Code
    call_tok: Token
    subroutine: Token
    arg_l: ArgList

    def iter_contents(self):
        yield self.call_tok
        yield self.subroutine
        yield self.arg_l

@dataclasses.dataclass(frozen=True)
class BuiltinProcedureStmt(Stmt):
    src: Code
    procedure_tok: Token
    arg_l: ArgList

    def __post_init__(self):
        assert token_has_type(self.procedure_tok, BuiltinProcedure)

    def iter_contents(self):
        yield self.procedure_tok
        yield self.arg_l

@dataclasses.dataclass(frozen=True)
class IfSingleLineStmt(Stmt):
    src: Code
    if_tok: Token
    condition: AssocExpr
    consequent: Stmt

    def iter_contents(self):
        yield self.if_tok
        yield self.condition
        yield self.consequent

@dataclasses.dataclass(frozen=True)
class IfConstructStartStmt(Stmt):
    src: Code
    if_tok: Token
    condition: AssocExpr
    then_tok: Token

    def iter_contents(self):
        yield self.if_tok
        yield self.condition
        yield self.then_tok

@dataclasses.dataclass(frozen=True)
class ElseIfStmt(Stmt):
    src: Code
    elseif_tok: Token
    condition: AssocExpr
    then_tok: Token

    def iter_contents(self):
        yield self.elseif_tok
        yield self.condition
        yield self.then_tok

@dataclasses.dataclass(frozen=True)
class DoWhileStmt(Stmt):
    src: Code
    dowhile_tok: Token
    condition: AssocExpr

    def iter_contents(self):
        yield self.dowhile_tok
        yield self.condition

@dataclasses.dataclass(frozen=True)
class DoIndexList:
    # this is NOT an expression or statement, but it contains them!
    src: Code
    init_stmt: ScalarAssignStmt
    comma1_tok: Token
    limit_expr: Expr
    comma2_tok: Optional[Token]
    increment_expr: Optional[Token]

    def __post_init__(self):
        assert isinstance(self.init_stmt, ScalarAssignStmt)
        assert token_has_type(self.comma1_tok, ',')
        assert isinstance(self.limit_expr, Expr)
        if self.increment_expr is None:
            assert self.comma2_tok is None
        elif self.comma2_tok is None:
            assert self.increment_expr is None
        else:
            assert token_has_type(self.comma2_tok, ',')
            assert isinstance(self.increment_expr, Expr)

    def iter_contents(self):
        yield self.init_stmt
        yield self.comma1_tok
        yield self.limit_expr
        if self.comma2_tok is not None:
            yield self.comma2_tok
            yield self.increment_expr

    @property
    def has_increment_expr(self): return self.increment_expr is not None


@dataclasses.dataclass(frozen=True)
class ImpliedDoList:
    # this is NOT an expression or statement, but it contains them!
    #
    # this is not very generic either, but is probably good enough for each
    # relevant case within Grackle
    src: Code
    outer_l_tok: Token
    arr_name_tok: Token
    inner_l_tok: Token
    index_name_tok: Token
    inner_r_tok: Token
    first_comma: Token
    do_index_list: DoIndexList
    outer_r_tok: Token

    def __post_init__(self):
        assert self.outer_l_tok.string == '('
        assert token_has_type(self.arr_name_tok, "arbitrary-name")
        assert self.inner_l_tok.string == '('
        assert token_has_type(self.index_name_tok, "arbitrary-name")
        assert self.inner_r_tok.string == ')'
        assert isinstance(self.do_index_list, DoIndexList)
        assert self.outer_r_tok.string == ')'

    def iter_contents(self):
        yield self.outer_l_tok
        yield self.arr_name_tok
        yield self.inner_l_tok
        yield self.index_name_tok
        yield self.inner_r_tok
        yield self.first_comma
        yield self.do_index_list
        yield self.outer_r_tok

@dataclasses.dataclass(frozen=True)
class DoStmt(Stmt):
    src: Code
    do_tok: Token
    do_idx_list: DoIndexList

    def __post_init__(self):
        assert token_has_type(self.do_tok, Keyword.DO)
        assert isinstance(self.do_idx_list, DoIndexList)

    def iter_contents(self):
        yield self.do_tok
        yield self.do_idx_list

    # the following are for backwards compatability
    @property
    def has_increment_expr(self): return self.do_idx_list.has_increment_expr

    @property
    def init_stmt(self): return self.do_idx_list.init_stmt

    @property
    def comma1_tok(self): return self.do_idx_list.comma1_tok

    @property
    def limit_expr(self): return self.do_idx_list.limit_expr

    @property
    def comma2_tok(self): return self.do_idx_list.comma2_tok

    @property
    def increment_expr(self): return self.do_idx_list.increment_expr



@dataclasses.dataclass(frozen=True)
class WriteStmt(Stmt):
    # at some point in the future, we will flesh this out
    src: Code
    write_tok: Token
    # control-list tokens
    clist_left: Token
    clist_arg0: Token
    clist_comma: Token
    clist_arg1: Token
    clist_right: Token
    # output-list entries
    output_list: Union[ImpliedDoList, Tuple[Union[Expr,Token], ...]]

    def __post_init__(self):
        assert self.write_tok.type == Keyword.WRITE
        assert self.clist_left.string == '('
        if token_has_type(self.clist_arg0, Literal.integer):
            # other values theoretically allowed
            assert self.clist_arg0.string in ['0', '6']
        else:
            assert self.clist_arg0.string == '*'
            # could also store a variable-name, but that shouldn't come up
        assert self.clist_comma.string == ','
        assert (
            token_has_type(self.clist_arg1, Literal.string) or
            self.clist_arg1.string == '*'
        )
        assert self.clist_right.string == ')'
        if not isinstance(self.output_list, ImpliedDoList):
            _check_delim_sequence(self.output_list, lambda e: e.string == ',')

    def writes_to_stdout(self): return self.clist_arg0.string in ['0', '*']
    def writes_to_stderr(self): return self.clist_arg0.string == '6'

    def format_specifier(self):
        if self.clist_arg1.string == '*': return None
        return self.clist_arg1

    def iter_contents(self):
        yield self.write_tok
        yield self.clist_left
        yield self.clist_arg0
        yield self.clist_comma
        yield self.clist_arg1
        yield self.clist_right
        if isinstance(self.output_list, ImpliedDoList):
            yield self.output_list
        else:
            for elem in self.output_list:
                yield elem

# some helper functions for analysis!

def _default_itr_contents(obj, *, skipped_first_item_type=None):
    if dataclasses.is_dataclass(obj):
        fields = dataclasses.fields(obj)
        if skipped_first_item_type is not None:
            assert fields[0].type == skipped_first_item_type
            fields = fields[1:]
        for field in fields:
            out = getattr(obj, field.name)
            assert out is not None
            yield out
    elif isinstance(obj, NamedTuple):
        # temporary assumption
        assert skipped_first_item_type is None
        assert len(obj) > 0
        for key in obj:
            assert key is not None
            yield key
    else:
        raise RuntimeError(
            f"a default iter_contents can't be created for {obj!r} since it "
            "isn't a NamedTuple or a dataclass"
        )

def iter_contents(obj):
    # the Stmt and Expr classes don't need to implement an iter_contents method
    # unless it's a special case (like DoStmt)
    # - this also supports cases like ArgList
    if hasattr(obj,'iter_contents'):
        yield from obj.iter_contents()
    elif isinstance(obj, Stmt):
        yield from _default_itr_contents(obj, skipped_first_item_type=Code)
    elif isinstance(obj, Expr):
        yield from _default_itr_contents(obj)
    #elif isinstance
    else:
        raise TypeError(f'{obj} has unexpected type {obj.__class__.__name__}')

def iterate_true_contents(arg, predicate):
    # basically we yield all conents where predicate(elem) returns True
    # -> the idea is that we go down the stack until we hit True or we hit a
    #    token
    stack = [iter_contents(arg)]
    while len(stack) > 0:
        try:
            elem = next(stack[-1])
        except StopIteration:
            stack.pop()
        else:
            if predicate(elem):
                yield elem
            elif isinstance(elem, Token):
                pass
            else:
                stack.append(iter_contents(elem))

def _iterate_tokens(arg):
    def predicate(elem): return isinstance(elem, Token)
    yield from iterate_true_contents(arg, predicate)

def compressed_str_from_Expr(expr):
    return compressed_concat_tokens(_iterate_tokens(expr))

def stmt_has_single_tok_type(stmt, type_spec):
    assert isinstance(stmt, Stmt)
    itr = _iterate_tokens(stmt)
    first_tok = next(itr)
    try:
        next(itr)
        return False
    except StopIteration:
        return token_has_type(first_tok, type_spec)

@final
class _InvalidCls:
    # not subclassed type. Calling isinstance(arg, _InvalidCls)
    __slots__ = ()
    def __init__(*args,**kwargs): raise RuntimeError('unintializable')

class ControlConstructKind(Enum):
    # represents the different kinds of construct kinds and tracks properties

    MetaIfElse = (None, (), None)  # <-- needs to be manually handled
    IfElse = (IfConstructStartStmt, (ElseIfStmt, Keyword.ELSE), Keyword.ENDIF)
    DoLoop = (DoStmt, (), Keyword.ENDDO)
    DoWhileLoop = (DoWhileStmt, (), Keyword.ENDDO)

    def __new__(cls, start_stmt_cls, branch_stmt_props, close_tok_type):
        obj = object.__new__(cls)
        obj._value_ = len(cls.__members__) + 1

        if (close_tok_type is None) != (start_stmt_cls is None):
            raise ValueError(
                "close_tok_type and start_stmt must both be None or "
                "neither can be None"
            )
        elif start_stmt_cls is None:
            if len(branch_stmt_props) > 0:
                raise ValueError(
                    "branch_stmt_props must be empty when start_stmt_cls is "
                    "None"
                )
            start_stmt_cls = _InvalidCls

        assert issubclass(start_stmt_cls, (Stmt, _InvalidCls))
        for prop in branch_stmt_props:
            assert isinstance(prop, Keyword) or issubclass(prop,Stmt)
        assert close_tok_type is None or isinstance(close_tok_type, Keyword)

        obj._start_cls = start_stmt_cls
        obj._branch_props = branch_stmt_props
        obj._close_tok_type = close_tok_type
        return obj

    def kind(self): return self

    def is_branch_stmt(self, stmt):
        if not isinstance(stmt, Stmt):
            return False
        for p in self._branch_props:
            if isinstance(p,Keyword):
                if stmt_has_single_tok_type(stmt,p):
                    return True
            elif isinstance(stmt, p):
                return True
        return False

    def is_close_stmt(self, stmt):
        return (
            isinstance(stmt, Stmt) and
            (self._close_tok_type is not None) and
            stmt_has_single_tok_type(stmt,self._close_tok_type)
        )

    @classmethod
    def match_construct_start(cls, stmt):
        for construct_kind in cls:
            if isinstance(stmt, construct_kind._start_cls):
                return construct_kind
        return None

