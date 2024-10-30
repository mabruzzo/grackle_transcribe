from more_itertools import peekable

from .src_model import (
    Code,
    PreprocKind,
    PreprocessorDirective,
    SrcItem,
    SrcRegion,
)
from .f_ast_load import (
    AstCreateConfig,
    create_ast,
    load_AST_Nodes,
    NodeTraverser,
    FortranASTNode
)

from .f_chunk_parse import (
    BuiltinFn,
    BuiltinProcedure,
    ChunkKind, Literal, Operator, Token, token_has_type, Type,
    compressed_concat_tokens
)

from .subroutine_object import (
    _unwrap, _unwrap_child, _has_unwrap_nameseq, _extract_name
)

from dataclasses import dataclass
from enum import auto, Enum
from itertools import islice
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

# defining datatypes

class ArrSpec(NamedTuple):
    axlens: List[Optional[None]]
    allocatable: bool

    @property
    def rank(self):
        return len(self.axlens)

class Variable(NamedTuple):
    name: str
    type: Type
    decl_section_index: int
    variable_number_on_line: int
    array_spec: Optional[ArrSpec]

class Constant(NamedTuple):
    name: str
    type: Type
    is_macro: bool
    decl_section_index: int # the index of the object used for declarations
    # in the future, we might want to define a value when using a macro


class IdentifierSpec:
    _indentifiers: Dict[str, Tuple[Union[Variable,Constant],bool]]
    _n_args: int
    _first_const: int

    def __init__(self, arguments, variables, constants):
        self._identifiers = dict(
            [(elem.name, (elem, True)) for elem in arguments] +
            [(elem.name, (elem, False)) for elem in variables] +
            [(elem.name, (elem, False)) for elem in constants]
        )
        self._n_args = len(arguments)
        self._first_const = self._n_args + len(variables)
        assert len(self._identifiers) == (self._first_const + len(constants))


    def _is_kind(self, key, kind):
        try:
            val, is_arg = self._identifiers[key]
        except KeyError:
            return False
        if kind == "arg":
            return is_arg
        elif kind == "var":
            return (not is_arg) and isinstance(val, Variable)
        elif kind == "constant":
            return isinstance(val, Constant)
        raise RuntimeError("Internal error")

    def is_arg(self, key): return self._is_kind(key, 'arg')
    def is_var(self, key): return self._is_kind(key, 'var')
    def is_constant(self, key): return self._is_kind(key, 'constant')
    def __len__(self): return len(self._identifiers)
    def __getitem__(self, key): return self._identifiers[key][0]
    def __contains__(self, key): return key in self._identifiers
    def keys(self): return self._identifiers.keys()

    @property
    def arguments(self):
        return [
            v for v,_ in islice(self._identifiers.values(), self._n_args)
        ]

    @property
    def variables(self):
        return [
            v for v,_ in islice(
                self._identifiers.values(), self._n_args, self._first_const
            )
        ]

    @property
    def constants(self):
        return [
            v for v,_ in islice(
                self._identifiers.values(), self._first_const, None
            )
        ]

    def validate_identifier(self, identifier_name, require_array_dim=None):
        try:
            tmp = self[identifier_name]
        except KeyError:
            raise RuntimeError(
                f"{identifier_name} is not a known identifier. "
                "is it actually the name of a builtin function?"
            ) from None

        if require_array_dim is None:
            return True
        elif require_array_dim < 1:
            raise ValueError(
                "when specified, require_array_dim must be positive"
            )
        elif isinstance(tmp, Constant) or tmp.array_spec is None:
            raise RuntimeError(
                f"the fortran identifier {identifier_name} is not an array"
            )
        elif tmp.array_spec.rank != require_array_dim:
            raise RuntimeError(
                f"the fortran identifier {identifier_name} has "
                f"{tmp.array_spec.rank} dimensions (instead of "
                f"{require_array_dim} dimensions)."
            )
        return True


@dataclass(frozen=True)
class Declaration:
    """
    Represents a declaration of a single Constant or 1+ Variable(s)
    """
    identifier: Union[Constant, List[Variable]]
    src_item: Union[Code, List[SrcItem]]
    node: FortranASTNode

    def __post_init__(self):
        if isinstance(self.identifier, Constant):
            assert isinstance(self.src_item, Code)
        elif isinstance(self.src_item, list) and len(self.identifier) > 1:
            raise ValueError(
                "src_item can only be a list when we define a single Variable"
            )
        elif len(self.identifier) == 0:
            raise ValueError("a declaration can't define 0 variables")

    @property
    def defines_constant(self):
        return isinstance(self.identifer, Constant)

    @property
    def is_conditional_declaration(self):
        return isinstance(self.src_item,list)

class Stmt:
    # baseclass

    @property
    def item(self):
        return self.src # for backwards compatability

class ControlConstructKind(Enum):
    MetaIfElse = auto() # if preprocessor
    IfElse=auto()
    DoLoop=auto() # maybe differentiate between (a for vs while)

class _ConditionContentPair(NamedTuple):
    condition: Union[Stmt,PreprocessorDirective]
    content: List[Union[Stmt, SrcItem]]

@dataclass(frozen=True)
class ControlConstruct:
    """Represents if-statement, do-statement, ifdef-statment"""
    condition_contents_pairs: _ConditionContentPair
    end: Union[Stmt,PreprocessorDirective]

    def __post_init__(self):
        assert len(self.condition_contents_pairs) > 0
        is_code = isinstance(self.end,Code)
        for (condition,_) in self.condition_contents_pairs:
            assert isinstance(condition, Code) == is_code

    @property
    def n_branches(self):
        return len(self.condition_contents_pair)

class Declaration(NamedTuple):
    src: Union[Code, List[SrcItem]]
    ast: FortranASTNode
    identifiers: List[Union[Variable, Constant]]

    @property
    def is_precision_conditional(self):
        return not isinstance(self.src, Code)

class SubroutineEntity(NamedTuple):
    name: str
    identifiers: IdentifierSpec

    subroutine_stmt: Stmt

    # specifies any relevant include-directives
    prologue_directives: List[PreprocessorDirective]

    # specifies all entries related to declarations
    declaration_section: List[Union[Declaration,SrcItem]]

    # specifies all entries related to the actual implementation
    impl_section: List[Union[ControlConstruct,SrcItem]]

    endroutine_stmt: Stmt

    @property
    def arguments(self): return self.identifiers.arguments

    @property
    def variables(self): return self.identifiers.variables

    @property
    def constants(self): return self.identifiers.constants

# Introduce the Concept of an expression/groups
# -> fundamentally, they are a way of grouping together tokens

class Expr:
    def __str__(self):
        return (
            f"<{self.__class__.__name__}: '{compressed_str_from_Expr(self)}'>"
        )


def _iterate_expr_tokens(expr):

    stack = [expr.iter_contents()]
    while len(stack) > 0:
        try:
            elem = next(stack[-1])
        except StopIteration:
            stack.pop()
        else:
            if hasattr(elem, 'iter_contents'):
                stack.append(elem.iter_contents())
            else:
                yield elem

def compressed_str_from_Expr(expr):
    print(list(_iterate_expr_tokens(expr)))
    return compressed_concat_tokens(_iterate_expr_tokens(expr))

@dataclass(frozen=True)
class IdentifierExpr(Expr):
    token: Token

    def __post_init__(self): assert self.token.type == "arbitrary-name"
    def iter_contents(self): yield self.token

@dataclass(frozen=True)
class LiteralExpr(Expr):
    token: Token

    def __post_init__(self): assert isinstance(self.token.type, Literal)
    def iter_contents(self): yield self.token

@dataclass(frozen=True)
class ColonExpr(Expr):
    token: Token

    def __post_init__(self): assert self.token.string == ":"
    def iter_contents(self): yield self.token

@dataclass(frozen=True)
class UnaryOpExpr(Expr):
    op: Token
    operand: Expr

    def __post_init__(self): assert self.token.is_always_unary_op()

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

@dataclass(frozen=True)
class BinaryOpSeqExpr(Expr):
    # the idea is that we don't want to deal with operator precedence
    seq: Tuple[Union[Expr,Token], ...]
    
    def __post_init__(self):
        _check_delim_sequence(self.seq, lambda e: e.is_always_binary_op())

    def iter_contents(self): yield from self.seq

@dataclass(frozen=True)
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class FnEval(Expr):
    fn_name: Token
    arg_l: ArgList

    def __post_init__(self):
        assert isinstance(self.fn_name.type, BuiltinFn)
        assert isinstance(self.arg_l, ArgList)

    def iter_contents(self):
        yield self.fn_name
        yield self.arg_l

@dataclass(frozen=True)
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
@dataclass(frozen=True)
class UncategorizedStmt(Stmt):
    src: Code
    ast: Optional[FortranASTNode] = None

    def iter_contents(self): yield from self.src.tokens

@dataclass(frozen=True)
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

@dataclass(frozen=True)
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

@dataclass(frozen=True)
class CallStmt(Stmt):
    src: Code
    call_tok: Token
    subroutine: Token
    arg_l: ArgList

    def iter_contents(self):
        yield self.call_tok
        yield self.subroutine
        yield self.arg_l

@dataclass(frozen=True)
class IfSingleLineStmt(Stmt):
    src: Code
    if_tok: Token
    condition: AssocExpr
    consequent: Stmt

    def iter_contents(self):
        yield self.if_tok
        yield self.condition
        yield self.consequent

@dataclass(frozen=True)
class IfConstructStartStmt(Stmt):
    src: Code
    if_tok: Token
    condition: AssocExpr
    then_tok: Token

    def iter_contents(self):
        yield self.if_tok
        yield self.condition
        yield self.then_tok

#class DoWhileStartStmt: pass
#class DoStartStmt:pass

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

    def _parse_arg_list(self, token_stream):
        l = self._match_ttype('(', token_stream, require_match=True)
        seq = [self._parse_expr(token_stream)]
        while self._match_ttype(',', token_stream, consume=False) is not None:
            seq.append(self._match_ttype(',', token_stream))
            seq.append(self._parse_expr(token_stream))
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

        elif tok := self._match_ttype(BuiltinFn, token_stream):
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
            # we haven't handled casts yet!
            raise RuntimeError("NO IDEA HOW TO HANDLE THIS CASE")

    def _parse_expr(self, token_stream, *, expected_type = None):
        expr_l = [self._parse_single_expr(token_stream)]
        while tok := self._match_ttype(Operator, token_stream):
            assert tok.type.is_always_binary_op()
            expr_l.append(tok)
            expr_l.append(self._parse_single_expr(token_stream))

        out = expr_l[0] if (len(expr_l) == 1) else BinaryOpSeqExpr(expr_l)
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
                for arg in arg_l.get_args():
                    if isinstance(arg, ColonExpr):
                        assert allocatable
                        axlens.append(None)
                    elif isinstance(
                        arg,
                        (IdentifierExpr,ArrayAccess,LiteralExpr,BinaryOpSeqExpr)
                    ):
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

    def parse_stmt(self, token_stream, src, *,
                   sub_statement=False):
        # maybe src should be accessible through token_stream?

        #if not sub_statement:
        #    print(compressed_concat_tokens(src.tokens))

        next_tok = token_stream.peek()
        out = None

        if next_tok.string.lower() == 'if':
            out = self.parse_if_stmt(token_stream, src)
        elif next_tok.string.lower() == 'do':
            self.parse_do_stmt(token_stream, src)
        elif next_tok.string.lower() == 'call':
            out = CallStmt(
                src=src,
                call_tok=next(token_stream),
                subroutine=self._match_ttype(
                    "arbitrary-name", token_stream, require_match=True
                ),
                arg_l = self._parse_arg_list(token_stream)
            )
        elif next_tok.string.lower() == 'write':
            out = UncategorizedStmt(src=src)
        #elif ...
        #    ....
        else:

            # at this point, we should have exhausted all obvious statements.
            # So, now we try to parse assignment statement
            # -> in the future, we will eliminate this try-except and handle
            #    this far more gracefully
            if out is None:
                try:
                    out = self.try_parse_assignment_stmt(token_stream, src)
                except RuntimeError:
                    out = None

        if out is None: # in the future, this branch will become an error
            out = UncategorizedStmt(src=src)
        elif ((not is_itr_exhausted(token_stream)) and
              (not isinstance(out, UncategorizedStmt))):
            next_token = token_stream.peek()
            raise RuntimeError(
                "something went wrong! we haven't exhausted token_stream\n"
                "-> all tokens:\n"
                f"    `{compressed_concat_tokens(src.tokens)}`\n"
                "-> parsed tokens:\n"
                f"     `{compressed_str_from_Expr(out)}`\n"
                "-> next token:\n"
                f"      {next_token}\n"
                "-> cur Expr:\n"
                f"      {out}"
            )
        return out

    def try_parse_assignment_stmt(self, token_stream, src):
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

        assign_tok = self._match_ttype("=", token_stream)
        if assign_tok is None:
            return None
        rval = self._parse_expr(token_stream)
        
        klass = ArrayAssignStmt if is_array_op else ScalarAssignStmt
        return klass(
            src=src, lvalue=lval, assign_tok=assign_tok, rvalue=rval
        )


    def parse_if_stmt(self, token_stream, src):
        if_tok = next(token_stream)
        assert if_tok.string.lower() == 'if'
        condition = self._parse_assoc_expr(token_stream)
        post_condition_tok = next(token_stream)

        if post_condition_tok.string.lower() == 'then':
            if not is_itr_exhausted(token_stream):
                raise RuntimeError(
                    "another token seems to exist after `if (<cond>) then`"
                )
            return IfConstructStartStmt(
                src=src,
                if_tok=if_tok,
                condition=condition,
                then_tok=post_condition_tok
            )
        consequent = self.parse_stmt(token_stream, src, sub_statement=True)
        if isinstance(consequent, UncategorizedStmt):
            return UncategorizedStmt(src=src)
        else:
            return IfConstructStartStmt(
                src=src,
                if_tok=if_tok,
                condition=condition,
                consequent=consequent
            )
             

    def parse_do_stmt(self, token_stream, src):
        return UncategorizedStmt(src=src)





# machinery for coarser grained parsing!

class SubroutineAstNodes(NamedTuple):
    name: str
    arg_list: List[str]
    specification_node: FortranASTNode
    execution_node: FortranASTNode

    @classmethod
    def create(cls, node: FortranASTNode):
        # unwrap the root node. It could hold:
        #    Program -> ProgramUnit -> SubroutineSubprogram
        _parts = ["Program", "ProgramUnit", "SubroutineSubprogram"]
        if node.name == _parts[-1]:
            root_node = node
        else:
            _parts = _parts[_parts.index(node.name):]
            root_node = _unwrap(node, _parts)

        # https://github.com/llvm/llvm-project/blob/697d65ded678f405b637e769f1b0bcc755c8461a/flang/include/flang/Parser/parse-tree.h#L3282
        assert len(root_node.children) in [4,5]

        # extract name and arg names
        assert root_node.children[0].name == 'SubroutineStmt'
        # https://github.com/llvm/llvm-project/blob/697d65ded678f405b637e769f1b0bcc755c8461a/flang/include/flang/Parser/parse-tree.h#L3143
        first_child = root_node.children[0].children[0]
        if first_child.name != "Name":
            # it would be PrefixSpec, if anything
            raise RuntimeError("not currently equipped to handle this case")
        else:
            routine_name = _extract_name(first_child)

        # extract the arguments
        args = []
        for arg_node in root_node.children[0].children[1:]:
            if arg_node == "LanguageBindingSpec":
                raise RuntimeError("not currently equipped to handle this case")
            args.append(
                _extract_name(arg_node, wrapper_parent_name="DummyArg")
            )

        # extract specification
        specification_node = root_node.children[1]
        assert specification_node.name == 'SpecificationPart'

        # extract implementation
        execution_node = root_node.children[2]
        assert execution_node.name == "ExecutionPart"

        if len(root_node.children) == 5:
            assert root_node.children[3].name == 'InternalSubprogramPart'
            raise RuntimeError("NOT IMPLEMENTED")
        else:
            assert len(root_node.children) == 4
        # the last node ends the subrouting

        return cls(
            name=routine_name,
            arg_list=args,
            specification_node=specification_node,
            execution_node=execution_node
        )

def _build_ast(
        region: SrcRegion,
        prologue_directives: List[PreprocessorDirective],
        *,
        config: AstCreateConfig = AstCreateConfig()
    ):
    src_fname = "__excised_subroutine.F"
    dump_fname = "__subroutine_ast_double.txt"

    with open(src_fname, "w") as f:
        for elem in prologue_directives:
            for line in elem.lines:
                f.write(line)
                f.write('\n')
        f.write('\n\n')
        for _, item in region.lineno_item_pairs:
            for line in item.lines:
                if isinstance(line, str):
                    f.write(line)
                    f.write('\n')
                else:
                    for line in line.lines:
                        f.write(line)
                        f.write('\n')
    tmp = create_ast(
        src_fname,
        dump_fname,
        double_precision=True,
        grackle_src_fname=False
    )

    root_node_itr = load_AST_Nodes(dump_fname)
    return SubroutineAstNodes.create(next(root_node_itr))
                
def _unpack_declaration(
    node: FortranASTNode, is_const: bool = False, is_allocatable: bool = False
):
    nested_typedecl_seq = [
        "DeclarationConstruct", "SpecificationConstruct", "TypeDeclarationStmt"
    ]

    # unwrap the TypeDeclarationStmt node
    # https://github.com/llvm/llvm-project/blob/main/flang/include/flang/Parser/parse-tree.h#L1393
    decl = _unwrap(node, nested_typedecl_seq)

    # first child is always the type specification (we skip over this for
    # now, but we could always come back to it later)
    typespec = decl.children[0]
    assert typespec.name == "DeclarationTypeSpec"

    # handle attributes corresponding to Parameter or Allocatable
    if is_const and is_allocatable:
        raise ValueError("No support for a constant and allocatable decl")
    elif is_const or is_allocatable:
        attr = decl.children[1]
        assert attr.name == "AttrSpec"
        expected_attr = "Parameter" if is_const else "Allocatable"
        assert attr.children[0].name == expected_attr
        first_entitydecl_idx = 2
    else:
        first_entitydecl_idx = 1

    out = []
    for child_idx in range(first_entitydecl_idx, len(decl.children)):
        entitydecl = decl.children[child_idx]
        assert entitydecl.name == "EntityDecl"
        assert len(entitydecl.children) > 0
        name = _extract_name(entitydecl.children[0])
        out.append((name,child_idx))
    return out


def _get_typespec_attrs_from_item(item):
    assert item.kind == ChunkKind.TypeSpec
    if item.tokens[1].string == ',':
        attr = item.tokens[2].string.lower()
        if attr == "parameter":
            is_const, is_allocatable = True, False
        else:
            assert attr == "allocatable"
            is_const, is_allocatable = False, True
    else:
        is_const, is_allocatable = False, False
    return is_const, is_allocatable




def _include_directive_constants(item, decl_section_index):
    # this is for include directive whose only goal is to define constants
    # (to my knowledge this is all include directivges)

    if item.kind == PreprocKind.INCLUDE_grackle_fortran_types:
        # (name, type, is_macro, value)
        props = [
            ("MASK_KIND",  Type.i32,       False, 4),
            ("MASK_TRUE",  Type.mask_type, True,  1),
            ("MASK_FALSE", Type.mask_type, True,  0),
            ("tiny",       Type.gr_float,  True,  1e-20),
            ("huge",       Type.gr_float,  True,  1e+20),
            ("RKIND",      Type.i32,       False, None), #value = 4 or 8
            ("tiny8",      Type.f64,       True,  1e-40),
            ("huge8",      Type.f64,       True,  1e+40),
            ("DKIND",      Type.i32,       False, 8),
            ("DIKIND",     Type.i32,       False, 8),
        ]
    elif item.kind == PreprocKind.INCLUDE_phys_const:
        # some of the values when compiled in single precision are slightly
        # different (e.g. pi_val)
        props =[
            ('kboltz',    Type.gr_float, True, 1.3806504e-16),
            ('mass_h',    Type.gr_float, True, 1.67262171e-24),   
            ('mass_e',    Type.gr_float, True, 9.10938215e-28),
            ('pi_val',    Type.gr_float, True, 3.141592653589793e0),
            ('hplanck',   Type.gr_float, True, 6.6260693e-27),
            ('ev2erg',    Type.gr_float, True, 1.60217653e-12),
            ('c_light',   Type.gr_float, True, 2.99792458e10),
            ('GravConst', Type.gr_float, True, 6.67428e-8),
            ('sigma_sb',  Type.gr_float, True, 5.670373e-5),
            ('SolarMass', Type.gr_float, True, 1.9891e33),
            ('Mpc',       Type.gr_float, True, 3.0857e24),
            ('kpc',       Type.gr_float, True, 3.0857e21),
            ('pc',        Type.gr_float, True, 3.0857e18)
        ]
    elif (item.kind == PreprocKind.INCLUDE and
          item.kind_value == "dust_const.def"):
        props = [
            ("sSiM",     Type.f64, True, 2.34118e0),
            ("sFeM",     Type.f64, True, 7.95995e0),
            ("sMg2SiO4", Type.f64, True, 3.22133e0),
            ("sMgSiO3",  Type.f64, True, 3.20185e0),
            ("sFe3O4",   Type.f64, True, 5.25096e0),
            ("sAC",      Type.f64, True, 2.27949e0),
            ("sSiO2D",   Type.f64, True, 2.66235e0),
            ("sMgO",     Type.f64, True, 3.58157e0),
            ("sFeS",     Type.f64, True, 4.87265e0),
            ("sAl2O3",   Type.f64, True, 4.01610e0),
            ("sreforg",  Type.f64, True, 1.5e0),
            ("svolorg",  Type.f64, True, 1.0e0),
            ("sH2Oice",  Type.f64, True, 0.92e0),
        ]

    else:
        raise RuntimeError(
            "CAN'T HANDLE A GENERIC PREPOCESSOR DIRECTIVE!\n"
            f"  kind: {item.kind}\n"
            f"  contents: {item.lines}"
        )

    for name, type_spec, is_macro, _ in props:
        yield Constant(
            name=name, type=type_spec, is_macro=is_macro,
            decl_section_index=decl_section_index
        )

def process_declaration_section(prologue_directives, src_items,
                                subroutine_nodes):
    """
    Returns
    -------
    identifiers: dict
        ``identifier['arguments']`` is a list of ``Variable``s that represents
        details about each argument. ``identifer["variables"]`` is a list of
        ``Variable``s that represent details about each local variable. 
        ``identifier["constants"]`` is a list of ``Constants``
    entries: list
        A list of SrcItem and Declarations that corresponds to each SrcItem
        in the declaration section of the subroutine
    last_src_item_index: int
        The last index consumed from src_items
    """

    known_arg_list = subroutine_nodes.arg_list
    encountered_identifiers = set()
    identifiers = {
        "arguments" : [None for _ in known_arg_list],
        "variables" : [],
        "constants" : []
    }

    def _register_identifier(identifier):
        nonlocal known_arg_list, encountered_identifiers, identifiers
        if identifier.name in encountered_identifiers:
            raise RuntimeError(
                f"we already encountered an identifier called {identifer.name}"
            )
        encountered_identifiers.add(identifier.name)

        if isinstance(identifier, Constant):
            identifiers["constants"].append(identifier)
        else:
            try:
                index = known_arg_list.index(identifier.name.lower())
            except ValueError:
                identifiers["variables"].append(identifier)
            else:
                identifiers["arguments"][index] = identifier

    # go through the prologue directives
    for directive in prologue_directives:
        if directive.kind == PreprocKind.DEFINE:
            _register_identifier(Constant(
                name=directive.kind_value, type=None, is_macro=True,
                decl_section_index=None
            ))
        else:
            for const in _include_directive_constants(directive, None):
                _register_identifier(const)

    node_itr = peekable(subroutine_nodes.specification_node.children)
    def _has_another_node():
        return node_itr.peek(None) is not None

    parser = Parser()

    entries = []

    src_iter = iter(enumerate(src_items))

    for src_index, item in src_iter:
        index = len(entries)
        if not isinstance(item, (PreprocessorDirective, Code)):
            entries.append(item)
            continue

        elif ((item.kind == PreprocKind.IFDEF) and
              (item.kind_value == "GRACKLE_FLOAT_4")):
            item_seq = [item, next(src_iter)[1]]
            assert (True,False) == _get_typespec_attrs_from_item(item_seq[-1])
            item_seq.append(next(src_iter)[1])
            assert item_seq[-1].kind == PreprocKind.ELSE
            item_seq.append(next(src_iter)[1])
            assert (True,False) == _get_typespec_attrs_from_item(item_seq[-1])
            item_seq.append(next(src_iter)[1])
            assert item_seq[-1].kind == PreprocKind.ENDIF

            cur_node = next(node_itr)
            tmp = _unpack_declaration(cur_node, is_const=True)
            assert "tolerance" == tmp[0][0]

            identifier = Constant(
                name="tolerance", type=Type.gr_float, is_macro=False,
                decl_section_index=index
            )
            _register_identifier(identifier)
            entries.append(Declaration(
                src=item_seq, ast=cur_node, identifiers=[identifier]
            ))

        elif isinstance(item, PreprocessorDirective):
            # going to assume item is some kind of include directive!
            # -> the following call will raise an error if it isn't
            itr = _include_directive_constants(item, index)
            entries.append(item)

            _concrete_constants = []
            for constant in itr:
                if not constant.is_macro:
                    _concrete_constants.append(constant.name)
                _register_identifier(constant)

            # consume all PARAMETER nodes that corrrespond to this statement
            for name in _concrete_constants:
                tmp = _unpack_declaration(next(node_itr), is_const=True)
                assert len(tmp) == 1
                assert tmp[0][0].lower() == name.lower()

        elif item.kind == ChunkKind.ImplicitNone:
            assert node_itr.peek().name == "ImplicitPart"
            entries.append(UncategorizedStmt(src=item, ast=next(node_itr)))

        elif item.kind == ChunkKind.TypeSpec:
            is_const, is_allocatable = _get_typespec_attrs_from_item(item)

            cur_node = next(node_itr)

            # extract from the ast
            identifier_pairs = _unpack_declaration(
                cur_node, is_const=is_const, is_allocatable=is_allocatable
            )
            assert len(identifier_pairs) > 0

            # we can come back and make the checks a lot more robust!

            identifier_list = []
            if is_const:
                assert len(identifier_pairs) == 1
                name, _ = identifier_pairs[0]
                identifier_list.append(
                    Constant(
                        name=name, type=item, is_macro=False,
                        decl_section_index=index,
                    )
                )
            else:
                #print(compressed_concat_tokens(item.tokens))
                identifier_list = parser.parse_variables_from_declaration(
                    peekable(item.tokens),
                    common_var_kwargs = {'decl_section_index' : index}
                )
                for i, (name, _) in enumerate(identifier_pairs):
                    if name.lower() != identifier_list[i].name.lower():
                        print(name)
                        print(identifier_list[i])
                        raise AssertionError()
            for elem in identifier_list:
                _register_identifier(elem)

            entries.append(
                Declaration(src=item, ast=cur_node, identifiers=identifier_list)
            )
        else:
            print(item.kind)
            raise RuntimeError(
                "This should be unreachable. Current item corresponds to: \n"
                + '\n'.join(item.lines)
            )

        if not _has_another_node():
            if None in identifiers["arguments"]:
                index = identifiers["arguments"].index(None)

                raise RuntimeError(
                    "Something strange happened. We believe we are done with "
                    "parsing parameters, but we never encountered a type "
                    f"declaration for the {known_arg_list[index]} argument"
                )

            return IdentifierSpec(**identifiers), entries, src_index
    else:
        raise RuntimeError("something weird happended")


_STARTKINDS = (
    ChunkKind.IfConstructStart, ChunkKind.DoConstructStart, PreprocKind.IFDEF
)

class _LevelData:
    # the idea is that this is a temporary object used to collect src_items as
    # we move through (nested control constructs)
    def __init__(self, first_item):
        self.branch_content_pairs = [(first_item, [])]
        if first_item.kind == ChunkKind.IfConstructStart:
            pair = ([ChunkKind.ElseIf, ChunkKind.Else], ChunkKind.EndIf)
        elif first_item.kind in [
            ChunkKind.DoWhileConstructStart, ChunkKind.DoConstructStart
        ]:
            pair = ([], ChunkKind.EndDo)
        elif first_item.kind == PreprocKind.IFDEF:
            pair = ([PreprocKind.ELSE], PreprocKind.ENDIF)
        else:
            raise ValueError(
                f"first_item has invalid kind. It must be one of {_STARTKINDS}"
            )
        self.branch_kinds, self.level_end_kind = pair
        self.end_item = None

    def add_branch(self, item):
        assert item.kind in self.branch_kinds
        self.branch_content_pairs.append((item, []))

    def append_entry(self, item):
        self.branch_content_pairs[-1][1].append(item)

    def most_recent_item(self):
        # for debugging purposes
        if self.end_item is not None:
            return self.end_item
        last_branch_content_pair = self.branch_content_pairs[-1]
        if last_branch_content_pair[1] == []:
            return last_branch_content_pair[0]
        return last_branch_content_pair[1]


def _process_impl_items(first_item, src_items):
    # this deals with organizing items into (nested?) levels (if applicable)

    levels = []
    while True:
        nlevels = len(levels)
        if len(levels) == 0:
            item = first_item
        else:

            try:
                item = next(src_items)
            except StopIteration:
                raise RuntimeError(
                    "Something went wrong! We ran out of source items while "
                    f"we are {len(levels)} levels deep"
                ) from None

        if not isinstance(item, (PreprocessorDirective, Code)):
            # definitionally, we must be creating a level object
            levels[-1].append_entry(item)

        elif item.kind in _STARTKINDS:
            levels.append(_LevelData(item))

        elif (nlevels > 0) and (item.kind in levels[-1].branch_kinds):
            levels[-1].add_branch(item)

        elif (nlevels > 0) and (item.kind == levels[-1].level_end_kind):
            levels[-1].end_item = item
            if nlevels == 1:
                return levels[0] 
            tmp = levels.pop()
            levels[-1].append_entry(tmp)

        elif (nlevels > 0):
            levels[-1].append_entry(item)

        else:
            return item


def _reconcile_item_nodes(parser, item, defined_macros, node_itr):
    # this was originally written with the nominal intention of:
    # - converting _LevelData and items into a ControlConstruct and Stmt, where
    #   each statement holds reconciled AST nodes
    # - we never quite implemented logic for reconcilliation
    #
    # At this point, it still doesn't interact at all with AST nodes and we are
    # manually parsing statments.
    # - it would probably be better to handle parsing when we are creating the
    #   _LevelData instances (so we could move towards excising ChunkKind junk)
    # - we may move away from AST in general ...

    if isinstance(item, Code):
        # match up with node (this should be easy!)
        return parser.parse_stmt(
            token_stream=peekable(item.tokens), src=item
        )
    elif isinstance(item, _LevelData):
        # getting ast is a little tricky
        # -> we need to do some unwrapping based on the kind of loop
        # -> for a preprocessor ifdef, there may/may not be asts to examine
        ccpair_list = []
        level = item
        for branch_item, content_l in level.branch_content_pairs:
            if isinstance(branch_item, PreprocessorDirective):
                condition_stmt = branch_item
            else:
                condition_stmt = parser.parse_stmt(
                    token_stream=peekable(branch_item.tokens),
                    src=branch_item
                )
            bundle_list = []
            for entry in content_l:
                if not isinstance(entry, (Code, _LevelData)):
                    bundle_list.append(entry)
                else:
                    bundle_list.append(_reconcile_item_nodes(
                        parser, entry, defined_macros, node_itr
                    ))
            ccpair_list.append(_ConditionContentPair(
                condition_stmt, bundle_list
            ))
        if isinstance(level.end_item, PreprocessorDirective):
            end_stmt = level.end_item
        else:
            end_stmt = parser.parse_stmt(
                token_stream=peekable(level.end_item.tokens),
                src=level.end_item
            )
        return ControlConstruct(ccpair_list, end_stmt)

    else:
        raise TypeError("Unexpected item type")


def process_impl_section(identifiers, src_items,
                         subroutine_nodes):
    """
    Returns
    -------
    entries: list
        A list of ``SrcItem``, ``ControlConstruct``s, and ``Stmt``s that
        corresponds to each ``SrcItem`` in the implementation section of the 
        subroutine
    """

    # to start out, we will simply ignore the ast_nodes

    entries = []

    src_iter = iter(src_items)
    node_itr = peekable(subroutine_nodes.execution_node.children)

    defined_macros = [
        const.name for const in identifiers.constants if const.is_macro
    ]

    parser = Parser(identifiers)

    for item in src_iter:
        if not isinstance(item, (PreprocessorDirective, Code)):
            entries.append(item)
            continue
        elif item.kind == PreprocKind.DEFINE:
            entries.append(item)
            macro_name = item.kind_value
            defined_macros.append(macro_name)
            continue
        else:
            tmp_item = _process_impl_items(item, src_iter)
            entries.append(_reconcile_item_nodes(
                parser, tmp_item, defined_macros, node_itr
            ))

    return entries


def build_subroutine_entity(
        region: SrcRegion,
        prologue: Optional[SrcRegion]=None,
        *,
        config: AstCreateConfig = AstCreateConfig()
    ):
    if prologue is None:
        prologue_directives = []
    else:
        prologue_directives = [
            item for _, item in prologue.lineno_item_pairs
            if isinstance(item, PreprocessorDirective)
        ]

    src_items = [item for _,item in region.lineno_item_pairs]
    assert len(src_items) > 3
    assert src_items[0].kind == ChunkKind.SubroutineDecl
    subroutine_stmt = UncategorizedStmt(src=src_items[0], ast=None)
    assert src_items[-1].kind == ChunkKind.EndRoutine
    endroutine_stmt = UncategorizedStmt(src=src_items[-1], ast=None)

    src_items = src_items[1:-1]

    subroutine_nodes = _build_ast(region, prologue_directives, config=config)

    # step up and match up declarations between source items and ast nodes
    identifier_spec, declaration_entries, last_declaration_index = \
        process_declaration_section(
            prologue_directives=prologue_directives,
            src_items=src_items,
            subroutine_nodes=subroutine_nodes
        )

    # go through src_items[last_declaration_index+1:] and match up with
    # subroutine_nodes.execution_node
    impl_section = process_impl_section(
        identifier_spec, src_items[last_declaration_index+1:], subroutine_nodes
    )

    return SubroutineEntity(
        name = subroutine_nodes.name,
        identifiers = identifier_spec,
        subroutine_stmt = subroutine_stmt,
        prologue_directives = prologue_directives,
        declaration_section = declaration_entries,
        impl_section = impl_section,
        endroutine_stmt = endroutine_stmt
    )
