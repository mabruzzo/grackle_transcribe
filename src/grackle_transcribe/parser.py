from more_itertools import peekable

from .identifiers import(
    ArrSpec, Constant, Variable, IdentifierSpec
)

from .src_model import (
    Code, PreprocessorDirective, SrcItem,
)

from .token import (
    BuiltinFn,
    BuiltinProcedure,
    Keyword,
    ChunkKind, Literal, Operator, Token, token_has_type, Type,
    compressed_concat_tokens
)

import dataclasses
from functools import partial
from enum import auto, Enum
from typing import Any, final, List, NamedTuple, Optional, Tuple, Union


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
#
# with the benefit of hindsight, I think the whole principle of ChunkKind was
# a mistake... I think we should have just parsed Stmts. We are kinda stuck
# with it for now...

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
        yield self.do_tok
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



def is_itr_exhausted(itr):
    return bool(itr) == False

class Parser:

    def __init__(self, identifier_ctx = None):
        self.identifier_ctx = identifier_ctx

    def _validate_identifier(self, identifier_name, require_array_dim=None):
        if self.identifier_ctx is None:
            return True
        return self.identifier_ctx.validate_identifier(
            identifier_name=identifier_name,
            require_array_dim=require_array_dim
        )


    def _match_ttype(self, type_spec, token_stream, *, consume=True,
                     require_match= False):
        try:
            tok = token_stream.peek()
        except StopIteration:
            if require_match:
                raise RuntimeError(
                    f"we expected a match to a token of type {type_spec}, but "
                    "there are no more tokens!"
                ) from None
            return None

        if token_has_type(tok, type_spec):
            if consume:
                return next(token_stream)
            return tok
        elif require_match:
            raise RuntimeError(
                f"the token, {tok}, does not have the expected type {type_spec}"
            )
        return None

    def _parse_delimited(self, token_stream, addressof=False):
        tmp = self._parse_expr(token_stream)
        if addressof:
            seq = [AddressOfExpr(tmp)]
        else:
            seq = [tmp]
        count = 1

        while self._match_ttype(',', token_stream, consume=False) is not None:
            seq.append(self._match_ttype(',', token_stream))
            tmp = self._parse_expr(token_stream)
            if addressof:
                tmp = AddressOfExpr(tmp)
            seq.append(tmp)
            count+=1

        return seq

    def _parse_arg_list(self, token_stream, addressof=False):
        l = self._match_ttype('(', token_stream, require_match=True)
        seq = self._parse_delimited(token_stream, addressof=addressof)
        r = self._match_ttype(')', token_stream, require_match=True)
        return ArgList(left=l, seq=seq, right=r)

    def _parse_assoc_expr(self, token_stream):
        l = self._match_ttype('(', token_stream, require_match=True)
        inner_expr = self._parse_expr(token_stream)
        r = self._match_ttype(')', token_stream, require_match=True)
        return AssocExpr(left = l, expr=inner_expr, right=r)

    def _parse_unchecked_identifier_expr(self,token_stream):
        tok = self._match_ttype(
            "arbitrary-name", token_stream, require_match=True
        )
        return IdentifierExpr(tok)

    def _parse_single_expr(self, token_stream):

        if tok := self._match_ttype(Literal, token_stream):
            return LiteralExpr(tok)

        elif tok := self._match_ttype(':', token_stream):
            return ColonExpr(tok)

        elif self._match_ttype('(', token_stream, consume=False) is not None:
            return self._parse_assoc_expr(token_stream)

        elif tok := self._match_ttype(Operator, token_stream):
            assert tok.type.is_always_unary_op()
            operand = self._parse_expr(token_stream)
            return UnaryOpExpr(op=tok, operand=operand)

        elif tok := self._match_ttype(Type, token_stream):
            assert tok.string.lower() == 'real'
            return CastExpr(
                name=tok, arg_l=self._parse_arg_list(token_stream)
            )

        elif tok := self._match_ttype(BuiltinFn, token_stream):
            if tok.type == BuiltinFn.int:
                return CastExpr(
                    name=tok, arg_l=self._parse_arg_list(token_stream)
                )

            return FnEval(
                fn_name=tok, arg_l=self._parse_arg_list(token_stream)
            )

        elif self._match_ttype(
            "arbitrary-name", token_stream, consume=False
        ) is not None:
            out = self._parse_unchecked_identifier_expr(token_stream)
            followed_by_arglist = self._match_ttype(
                '(', token_stream, consume=False
            ) is not None
            identifier_name = out.token.string
            if followed_by_arglist:
                out = ArrayAccess(out, arg_l=self._parse_arg_list(token_stream))
                nargs = len(list(out.arg_l.get_args()))
            else:
                nargs = None
            self._validate_identifier(identifier_name, require_array_dim=nargs)
            return out

        else:
            raise RuntimeError(f"NO IDEA HOW TO HANDLE THIS CASE: {tok}")

    def _parse_expr(self, token_stream, *, expected_type = None):
        expr_l = [self._parse_single_expr(token_stream)]
        while tok := self._match_ttype(Operator, token_stream):
            assert tok.type.is_always_binary_op()
            next_expr = self._parse_single_expr(token_stream)
            if token_has_type(tok, Operator.POW):
                tmp = POWOpExpr(
                    base=expr_l.pop(), pow_tok=tok, exponent=next_expr
                )
                expr_l.append(tmp)
            else:
                expr_l.append(tok)
                expr_l.append(next_expr)

        out = expr_l[0] if (len(expr_l) == 1) else NonPOWBinaryOpSeqExpr(expr_l)
        if expected_type is not None:
            assert isinstance(out, expected_type)
        return out


    def parse_variables_from_declaration(self, token_stream, *,
                                         common_var_kwargs = {}):
        # this currently won't support constants

        type_tok = self._match_ttype(Type, token_stream, require_match=True)
        if self._match_ttype(',', token_stream) is None:
            allocatable = False
            self._match_ttype('::', token_stream) # consume this if present
        else:
            attr = next(token_stream)
            assert attr.string.lower() == 'allocatable'
            allocatable = True
            self._match_ttype('::', token_stream, require_match=True)

        variables = []

        for tok in token_stream:
            if tok.type != "arbitrary-name":
                raise RuntimeError(
                    f"the {tok} token is not an arbitrary name"
                )
            followed_by_arglist = self._match_ttype(
                '(', token_stream, consume=False
            ) is not None
            if allocatable and not followed_by_arglist:
                raise RuntimeError(
                    "we expect all allocatable variables to be arrays"
                )
            elif followed_by_arglist:
                axlens = []
                arg_l = self._parse_arg_list(token_stream)
                _ARGLIST_EXPR_TYPES = (
                    IdentifierExpr,ArrayAccess,LiteralExpr,NonPOWBinaryOpSeqExpr
                )
                for arg in arg_l.get_args():
                    if isinstance(arg, ColonExpr):
                        assert allocatable
                        axlens.append(None)
                    elif isinstance(arg, _ARGLIST_EXPR_TYPES):
                        axlens.append(arg)
                    else:
                        raise RuntimeError(
                            "the argument list declaring the shape of "
                            f"{tok.string}, the argument list, {arg_l!s}, has "
                            f"an unexpected value {arg}"
                        )
                arr_spec = ArrSpec(axlens,allocatable)
            else:
                arr_spec = None
            variables.append(Variable(
                name=tok.string,
                type=type_tok.type,
                array_spec=arr_spec,
                variable_number_on_line=len(variables),
                **common_var_kwargs
            ))

            if bool(token_stream):
                # consume the next comma
                self._match_ttype(',', token_stream, require_match=True)
                # confirm that there is at least one more token
                assert bool(token_stream)
        return variables

    def parse_stmt(self, token_stream, *, sub_statement=False):
        # maybe src should be accessible through token_stream?

        #if not sub_statement:
        #    print(compressed_concat_tokens(token_stream.src.tokens))

        next_tok = token_stream.peek()
        out = None

        if next_tok.type == Keyword.IF:
            out = self.parse_if_stmt(token_stream)
        elif next_tok.type == Keyword.ELSEIF:
            out = self.parse_elseif_stmt(token_stream)
        elif next_tok.type == Keyword.DOWHILE:
            out = self.parse_dowhile_stmt(token_stream)
        elif next_tok.type == Keyword.DO:
            out = self.parse_do_stmt(token_stream)
        elif next_tok.type == Keyword.CALL:
            out = CallStmt(
                src=token_stream.src,
                call_tok=next(token_stream),
                subroutine=self._match_ttype(
                    "arbitrary-name", token_stream, require_match=True
                ),
                arg_l = self._parse_arg_list(token_stream, addressof=True)
            )
        elif next_tok.type == Keyword.WRITE:
            out = self.parse_write_stmt(token_stream)
        elif next_tok.type == Keyword.GOTO:
            out = GoToStmt(
                src=token_stream.src,
                goto_tok=next(token_stream),
                label_expr=self._parse_expr(
                    token_stream, expected_type=LiteralExpr
                )
            )
        elif isinstance(next_tok.type, BuiltinProcedure):
            out = BuiltinProcedureStmt(
                src=token_stream.src,
                procedure_tok=next(token_stream),
                arg_l= self._parse_arg_list(token_stream)
            )
        elif (
            next_tok.type != "arbitrary-name" and
            next_tok.type.req_full_line_match
        ):
            out = Standard1TokenStmt(src=token_stream.src,token=next(token_stream))
        else:

            # at this point, we should have exhausted all obvious statements.
            # So, now we try to parse assignment statement
            # -> in the future, we will eliminate this try-except and handle
            #    this far more gracefully
            if out is None:
                try:
                    out = self.try_parse_assignment_stmt(token_stream)
                except RuntimeError:
                    out = None

        if out is None: # in the future, this branch will become an error
            raise RuntimeError("Don't know how to parse statement")
        elif ((not is_itr_exhausted(token_stream)) and
              (not sub_statement) and
              (not isinstance(out, UncategorizedStmt))):
            next_token = token_stream.peek()
            raise RuntimeError(
                "something went wrong! we haven't exhausted token_stream\n"
                "-> all tokens:\n"
                f"    `{compressed_concat_tokens(token_stream.src.tokens)}`\n"
                "-> parsed tokens:\n"
                f"     `{compressed_str_from_Expr(out)}`\n"
                "-> next token:\n"
                f"      {next_token}\n"
                "-> cur Expr:\n"
                f"      {out}"
            )
        return out

    def try_parse_assignment_stmt(self, token_stream):
        # try to get the L-value and investigate its properties
        lval = self._parse_expr(token_stream)
        if isinstance(lval, ArrayAccess):
            # we may want to check for colons as array indices, that may
            # indicate an array access
            if any(isinstance(e, ColonExpr) for e in lval.arg_l.get_args()):
                is_array_op = True # modifying a whole slice
            else:
                is_array_op = False # modifing 1 scalar stored in the array
        elif isinstance(lval, IdentifierExpr):
            identifier = self.identifier_ctx[lval.token.string]
            if isinstance(identifier, Constant):
                return None
            elif identifier.array_spec is None:
                is_array_op = False # modifying a scalar variable
            else:
                is_array_op = True # modifying all contents of array?
        else:
            return None

        assign_tok = self._match_ttype("assign", token_stream)
        if assign_tok is None:
            return None
        rval = self._parse_expr(token_stream)
        
        klass = ArrayAssignStmt if is_array_op else ScalarAssignStmt
        return klass(
            src=token_stream.src, lvalue=lval, assign_tok=assign_tok, rvalue=rval
        )

    def parse_if_stmt(self, token_stream):
        if_tok = self._match_ttype(
            Keyword.IF, token_stream, require_match=True
        )
        condition = self._parse_assoc_expr(token_stream)

        if then_tok := self._match_ttype(Keyword.THEN, token_stream):
            if not is_itr_exhausted(token_stream):
                raise RuntimeError(
                    "another token seems to exist after `if (<cond>) then`"
                )
            return IfConstructStartStmt(
                src=token_stream.src,
                if_tok=if_tok,
                condition=condition,
                then_tok=then_tok
            )

        consequent = self.parse_stmt(token_stream, sub_statement=True)
        if isinstance(consequent, UncategorizedStmt):
            raise RuntimeError()
        else:
            return IfSingleLineStmt(
                src=token_stream.src,
                if_tok=if_tok,
                condition=condition,
                consequent=consequent
            )

    def parse_elseif_stmt(self, token_stream):
        return ElseIfStmt(
            src=token_stream.src,
            elseif_tok = self._match_ttype(
                Keyword.ELSEIF, token_stream, require_match=True
            ),
            condition = self._parse_assoc_expr(token_stream),
            then_tok = self._match_ttype(
                Keyword.THEN, token_stream, require_match=True
            )
        )

    def parse_dowhile_stmt(self, token_stream):
        return DoWhileStmt(
            src=token_stream.src,
            dowhile_tok=self._match_ttype(
                Keyword.DOWHILE, token_stream, require_match=True
            ),
            condition = self._parse_assoc_expr(token_stream),
        )

    def parse_do_index_list(self, token_stream, parsing_dostmt=False):

        init_stmt = self.parse_stmt(token_stream, sub_statement=True)
        assert isinstance(init_stmt, ScalarAssignStmt)
        comma1_tok=self._match_ttype(
            ",", token_stream, require_match=True
        )
        limit_expr = self._parse_expr(token_stream)
        if comma2_tok := self._match_ttype(",", token_stream):
            increment_expr = self._parse_expr(token_stream)
        elif parsing_dostmt and not is_itr_exhausted(token_stream):
            raise RuntimeError(
                "something went wrong parsing do-stmt. The next token "
                f"is {next(token_stream)}"
            )
        else:
            if parsing_dostmt:
                assert is_itr_exhausted(token_stream)
            increment_expr = None

        return DoIndexList(
            src=token_stream.src,
            init_stmt=init_stmt,
            comma1_tok=comma1_tok,
            limit_expr=limit_expr,
            comma2_tok=comma2_tok,
            increment_expr=increment_expr
        )

    def parse_do_stmt(self, token_stream):
        do_tok = self._match_ttype(
            Keyword.DO, token_stream, require_match=True
        )
        do_index_list = self.parse_do_index_list(
            token_stream, parsing_dostmt=True
        )
        return DoStmt(
            src=token_stream.src,
            do_tok=do_tok,
            do_idx_list=do_index_list
        )

    def parse_write_stmt(self, token_stream):
        kw = {'token_stream' : token_stream, 'require_match' : True}
        _pairs = [
            ('write_tok', Keyword.WRITE),
            ('clist_left', '('),
            ('clist_arg0', None),
            ('clist_comma', ','),
            ('clist_arg1', None),
            ('clist_right', ')')
        ]

        tokens = {}
        for name, expected in _pairs:
            if expected is None:
                tokens[name] = next(token_stream)
            else:
                tokens[name] = self._match_ttype(expected, **kw)

        if token_stream.peek().string != '(':
            output_list = self._parse_delimited(token_stream, addressof=False)
        else:
            list_toks = {
                'outer_l_tok'    : self._match_ttype('(', **kw),
                'arr_name_tok'   : self._match_ttype('arbitrary-name', **kw),
                'inner_l_tok'    : self._match_ttype('(', **kw),
                'index_name_tok' : self._match_ttype('arbitrary-name', **kw),
                'inner_r_tok'    : self._match_ttype(')', **kw),
                'first_comma'    : self._match_ttype(',', **kw),
            }
            do_index_list= self.parse_do_index_list(token_stream)
            list_toks['outer_r_tok'] = self._match_ttype(')', **kw)
            output_list = ImpliedDoList(
                src=token_stream.src, do_index_list=do_index_list, **list_toks
            )

        return WriteStmt(
            src=token_stream.src, output_list=output_list, **tokens
        )


def is_single_tok_stmt(stmt, tok_type):
    return (
        isinstance(stmt, Standard1TokenStmt) and
        token_has_type(stmt.token, tok_type)
    )

class TokenStream:
    def __init__(self, item):
        assert isinstance(item, Code)
        self._src = item
        slc = slice(1, None) if item.has_label else slice(None)
        self._peekable = peekable(item.tokens[slc])

    @property
    def src(self):
        return self._src

    def __iter__(self):
        return self

    def __bool__(self):
        return self._peekable.__bool__()

    def __next__(self):
        return next(self._peekable)

    def peek(self, *args):
        return self._peekable.peek(*args)

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
    else:
        raise TypeError()

def _iterate_tokens(arg):
    stack = [iter_contents(arg)]
    while len(stack) > 0:
        try:
            elem = next(stack[-1])
        except StopIteration:
            stack.pop()
        else:
            if isinstance(elem, Token):
                yield elem
            else:
                stack.append(iter_contents(elem))

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
