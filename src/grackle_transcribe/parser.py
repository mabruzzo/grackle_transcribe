from more_itertools import peekable

from .identifiers import(
    ArrSpec, Constant, Variable, IdentifierSpec
)

from .src_model import (
    Code, PreprocessorDirective, SrcItem, Origin
)

# I would like to avoid this blind importing, but is the only practical way
# to easily handle the reorganization (originally all contents of syntax_unit
# and this file were combined)
from .syntax_unit import *

from .token import (
    BuiltinFn,
    BuiltinProcedure,
    Keyword,
    Literal, Operator, Token, token_has_type, Type,
    compressed_concat_tokens
)

from enum import auto, Enum
from functools import partial
import itertools
from typing import Optional, Union, TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from .subroutine_sig import SubroutineSignature

def is_itr_exhausted(itr):
    return bool(itr) == False

class ArgConsistencyReq(Enum):
    NONE = auto()
    BASIC = auto()

class Parser:

    def __init__(
        self,
        identifier_ctx: IdentifierSpec = None,
        signature_registry: dict[str, 'SubroutineSignature'] = None,
        arg_consistency_req: Optional[ArgConsistencyReq] = None
    ):
        if signature_registry is not None:
            assert identifier_ctx is not None
        self.identifier_ctx = identifier_ctx
        self.signature_registry = signature_registry

        if arg_consistency_req is None and signature_registry is None:
            arg_consistency_req = ArgConsistencyReq.NONE
        elif arg_consistency_req is None:
            arg_consistency_req = ArgConsistencyReq.BASIC
        self.arg_consistency_req = arg_consistency_req

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

    def _parse_delimited(self, token_stream):
        tmp = self._parse_expr(token_stream)
        seq = [tmp]
        count = 1

        while self._match_ttype(',', token_stream, consume=False) is not None:
            seq.append(self._match_ttype(',', token_stream))
            tmp = self._parse_expr(token_stream)
            seq.append(tmp)
            count+=1

        return seq

    def _parse_arg_list(self, token_stream):
        l = self._match_ttype('(', token_stream, require_match=True)
        seq = self._parse_delimited(token_stream)
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
            out = self.parse_call_stmt(token_stream) 
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
                out = self.try_parse_assignment_stmt(token_stream)

                #try:
                #    out = self.try_parse_assignment_stmt(token_stream)
                #except RuntimeError:
                #    out = None

        if out is None: # in the future, this branch will become an error
            problematic_lines = '\n'.join(token_stream.src.lines)
            raise RuntimeError(
                f"Don't know how to parse statement\n  {problematic_lines}"
            )
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

    def parse_call_stmt(self, token_stream):
        call_tok = next(token_stream)
        subroutine = self._match_ttype(
            "arbitrary-name", token_stream, require_match=True
        )
        nominal_arg_l = self._parse_arg_list(token_stream)

        sig = None
        if self.signature_registry is not None:

            key = subroutine.string.casefold()
            try:
                sig = self.signature_registry[key]
            except KeyError:
                warnings.warn(
                    f"the called subroutine, {key!r}, doesn't have a known "
                    "signature"
                )

        if sig is None:
            sigref_it = itertools.repeat(None)
        else:
            assert_call_consistent_with_signature(
               sig=sig,
               call=nominal_arg_l,
               identifier_spec=self.identifier_ctx,
               origin=token_stream.src.origin,
               consistency_req=self.arg_consistency_req
            )
            sigref_it = iter(sig.arguments_iter)

        new_seq = []
        for e in nominal_arg_l.seq:
            if getattr(e, 'string', None) == ',':
                new_seq.append(e)
            else:
                new_seq.append(AddressOfExpr(e, next(sigref_it)))

        # if we ever want to support calls to functions that accept arguments
        # by value, we will need to replace AddressOfExpr with a more detailed
        # alternative based off the corresponding argument from the signature
        arg_l = ArgList(
            left=nominal_arg_l.left,
            seq=tuple(new_seq),
            right=nominal_arg_l.right
        )
        
        return CallStmt(
            src=token_stream.src,
            call_tok=call_tok,
            subroutine=subroutine,
            arg_l = arg_l
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
            output_list = self._parse_delimited(token_stream)
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


def assert_call_consistent_with_signature(
    sig: 'SubroutineSig',
    call: Union[CallStmt, ArgList],
    identifier_spec: IdentifierSpec,
    origin: Optional[Origin] = None,
    consistency_req: ArgConsistencyReq = ArgConsistencyReq.BASIC
):

    if isinstance(call, CallStmt):
        assert origin is None
        arg_l = call.arg_l
        expect_wrapped = True
    elif isinstance(call, ArgList):
        assert origin is not None
        arg_l = call
        expect_wrapped = False
    else:
        raise TypeError()

    fname = '<file-path>' if origin.fname is None else origin.fname
    msg_prefix = f"Call to {sig.name!r} @ {fname}:{origin.lineno}:"

    if arg_l.n_args() != sig.n_args():
        raise ValueError(
            f"{msg_prefix} passes {arg_l.n_args()} arguments, rather than the "
            f"expected {sig.n_args()} arguments"
        )

    literal_argpacks = []
    complex_argpacks = []
    def _filtered_argpack(sig, arg_l):
        argpack_it = enumerate( zip(sig.arguments_iter, arg_l.get_args()) )
        for arg_index, (sig_arg, call_arg) in argpack_it:
            if isinstance(call_arg, AddressOfExpr):
                # it's ok if expect_wrapped == False
                call_arg = call_arg.wrapped
            elif expect_wrapped:
                raise RuntimeError(
                    "Something went wrong. We expected the ArgList instance "
                    "to hold wrapper AddressOfExpr instances"
                )

            new_pack = (arg_index, (sig_arg, call_arg))
            if isinstance(call_arg, (ArrayAccess, IdentifierExpr)):
                yield new_pack
            else:
                if isinstance(call_arg, LiteralExpr):
                    literal_argpacks.append(new_pack)
                    descr = f'the literal, `{call_arg.token.string}`'
                else:
                    complex_argpacks.append(new_pack)
                    tmp = compressed_str_from_Expr(call_arg)
                    descr = f'a complex expression, `{tmp}`'
                warnings.warn(
                    f"{msg_prefix}\n"
                    f"-> argument number {arg_index+1}, {sig_arg.name!r}"
                    f"-> passed {descr}"
                )

    # let's find all pairs
    if consistency_req == ArgConsistencyReq.NONE:
        return None
    elif consistency_req != ArgConsistencyReq.BASIC:
        raise RuntimeError(
            "We have not implemented functionality to compare array shapes "
            "(this is significantly more difficult than just testing type and "
            "dimensionality!)"
        )
        # this would probably involve an iterative approach!
    else:
        for arg_index, (sig_arg, call_arg) in _filtered_argpack(sig, arg_l):
            sigarg_summary = (sig_arg.type, sig_arg.prop.rank)
            if isinstance(call_arg, IdentifierExpr):
                identifier = identifier_spec[call_arg.token.string]
                callarg_summary = (identifier.type, identifier.rank)
            else:
                assert isinstance(call_arg, ArrayAccess)
                identifier = identifier_spec[call_arg.array_name.token.string]
                callarg_summary = (identifier.type, None)

            if sigarg_summary != callarg_summary:
                descr = []
                for t,rank in [sigarg_summary, callarg_summary]:
                    if rank is None:
                        descr.append(f"a scalar of type {t}")
                    else:
                        descr.append(f"a {rank}D array of type {t}")
                callarg_str = compressed_str_from_Expr(call_arg)

                if (
                    (sigarg_summary[0] in [Type.gr_float, Type.f64]) and 
                    (callarg_summary[0] in [Type.gr_float, Type.f64]) and
                    (callarg_summary[1:] == sigarg_summary[1:])
                ):
                    warnings.warn(
                        f"{msg_prefix}\n"
                        f"-> gr_float issue @ argument # {arg_index+1}\n"
                        f"-> expect: `{sig_arg.name}`, {descr[0]}\n"
                        f"-> receive: `{callarg_str}`, {descr[1]}"
                    )
                    continue

                raise RuntimeError(
                    f"{msg_prefix}\n"
                    f"-> at argument # {arg_index+1}\n"
                    f"-> expect:  `{sig_arg.name}`, {descr[0]}\n"
                    f"-> receive: `{callarg_str}`, {descr[1]}"
                )
