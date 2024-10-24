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
    ChunkKind, Type
)

from .subroutine_object import (
    _unwrap, _unwrap_child, _has_unwrap_nameseq, _extract_name
)

from dataclasses import dataclass
from enum import auto, Enum

from typing import Any, List, NamedTuple, Optional, Tuple, Union
import weakref

#__all__ = ["build_subroutine_entry"]



# it would be nice to differentiate between:
# -> arguments
# -> local variables
# -> named-constants
# -> special-constants (macros and named-constants whose value changes)
class Variable(NamedTuple):
    name: str

    #declaration_statement: 
    # index of declaration_stmt.children that corresponds to self
    child_index: int

    is_constant: bool

class ArrSpec(NamedTuple):
    axlens: List[Any]
    allocatable: bool

    @property
    def rank(self):
        return len(self.axlens)

class Variable(NamedTuple):
    name: str
    type: Type
    decl_section_index: int
    ast_child_index: int
    array_spec: Optional[ArrSpec]

class Constant(NamedTuple):
    name: str
    type: Type
    is_macro: bool
    decl_section_index: int # the index of the object used for declarations
    # in the future, we might want to define a value when using a macro

@dataclass(frozen=True)
class Declaration:
    # can either declare a single Constant or 1+ Variable(s)
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

@dataclass(frozen=True)
class Statement:
    item: Code
    node: FortranASTNode

class ControlConstructKind(Enum):
    MetaIfElse = auto() # if preprocessor
    IfElse=auto()
    DoLoop=auto() # maybe differentiate between (a for vs while)

@dataclass(frozen=True)
class ControlConstruct:
    """Represents if-statement, do-statement, ifdef-statment"""
    start_contents_pairs: List[Tuple[Union[Statement,PreprocessorDirective],List[Any]]]
    end: Union[Statement,PreprocessorDirective]

    def __post_init__(self):
        assert len(self.start_contents_pairs) > 0
        is_code = isinstance(end,Code)
        for (start,_) in self.start_contents_pair:
            assert isinstance(start, Code) == is_code


# it may make sense to use a class like the following instead of Code
# class Statement(NamedTuple):
#     src: Code
#     ast: Optional[FortranASTNode]
#
#     @property
#     def lines(self): return self.code.lines
#

class Declaration(NamedTuple):
    src: Union[Code, List[SrcItem]]
    ast: FortranASTNode
    identifiers: List[Union[Variable, Constant]]

    @property
    def is_precision_conditional(self):
        return not isinstance(self.src, Code)

class SubroutineEntity(NamedTuple):
    name: str
    arguments: List[Variable]
    variables: List[Variable]
    constants: List[Constant]

    # specifies any relevant include-directives
    prologue_directives: List[PreprocessorDirective]

    # specifies all entries related to declarations
    declaration_section: List[Union[Declaration,SrcItem]]

    # specifies all entries related to the actual implementation
    impl_section: List[Union[ControlConstruct,SrcItem]]

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
                index = known_arg_list.index(identifier.name)
            except ValueError:
                identifiers["variables"].append(identifier)
            else:
                identifiers["arguments"][index] = identifier

    node_itr = peekable(subroutine_nodes.specification_node.children)
    def _has_another_node():
        return node_itr.peek(None) is not None

    # we should go through the prologue directives and add any relevant constants
    entries = []

    src_iter = iter(enumerate(src_items))

    for src_index, item in src_iter:
        index = len(entries)
        if not isinstance(item, (PreprocessorDirective, Code)):
            entries.append(item)
            continue

        elif item.kind == PreprocKind.INCLUDE_grackle_fortran_types:
            entries.append(item)

            for name in ["RKIND", "DKIND", "DIKIND"]:
                tmp = _unpack_declaration(next(node_itr), is_const=True)
                assert len(tmp) == 1
                assert tmp[0][0].lower() == name.lower()

            _register_identifier(Constant(
                name="tiny", type=Type.gr_float, is_macro=True,
                decl_section_index=index #value = 1e-20
            ))
            _register_identifier(Constant(
                name="huge", type=Type.gr_float, is_macro=True,
                decl_section_index=index #value = 1e20
            ))
            _register_identifier(Constant(
                name="RKIND", type=Type.i32, is_macro=False,
                decl_section_index=index #value = 4 or 8
            ))
            _register_identifier(Constant(
                name="tiny8", type=Type.f64, is_macro=True,
                decl_section_index=index #value = 1e-40
            ))
            _register_identifier(Constant(
                name="huge8", type=Type.f64, is_macro=True,
                decl_section_index=index #value = 1e40
            ))
            _register_identifier(Constant(
                name="DKIND", type=Type.i32, is_macro=False,
                decl_section_index=index #value = 8
            ))
            _register_identifier(Constant(
                name="DIKIND", type=Type.i32, is_macro=False,
                decl_section_index=index #value = 8
            ))

        elif ((item.kind == PreprocKind.INCLUDE) and
              (item.kind_value == "dust_const.def")):
            entries.append(item)
            pairs = [
                ("sSiM", 2.34118e0), ("sFeM", 7.95995e0),
                ("sMg2SiO4",  3.22133e0), ("sMgSiO3", 3.20185e0),
                ("sFe3O4", 5.25096e0), ("sAC", 2.27949e0),
                ("sSiO2D", 2.66235e0), ("sMgO", 3.58157e0),
                ("sFeS", 4.87265e0), ("sAl2O3", 4.01610e0),
                ("sreforg", 1.5e0), ("svolorg", 1.0e0), ("sH2Oice", 0.92e0)]
            for name,_ in pairs:
                _register_identifier(Constant(
                    name=name, type=Type.f64, is_macro=True,
                    decl_section_index=index
                ))

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
            # generic preprocessor directive!
            raise RuntimeError(
                "CAN'T HANDLE A GENERIC PREPOCESSOR DIRECTIVE!\n"
                f"  kind: {item.kind}\n"
                f"  contents: {item.lines}"
            )

        elif item.kind == ChunkKind.ImplicitNone:
            assert node_itr.peek().name == "ImplicitPart"
            entries.append(Statement(item=item, node=next(node_itr)))

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
                identifiers['constants'].append(identifier_list[-1])
            else:
                tok_str_l = [tok.string.lower() for tok in item.tokens][1:]
                for name, ast_child_idx in identifier_pairs:
                    if name.lower() not in tok_str_l:
                        raise RuntimeError(f"can't find the {name} token")
                    assert name.lower() in tok_str_l
                    identifier_list.append(Variable(
                        name=name, type=item.tokens[0].type,
                        decl_section_index=index,
                        ast_child_index=ast_child_idx,
                        array_spec = None # come back to last part in future
                    ))
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
            return identifiers, entries, index
    else:
        raise RuntimeError("something weird happended")

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
    assert src_items[-1].kind == ChunkKind.EndRoutine
    src_items = src_items[1:-1]


    subroutine_nodes = _build_ast(region, prologue_directives, config=config)

    # step up and match up declarations between source items and ast nodes
    identifiers, declaration_entries, last_declaration_index = \
        process_declaration_section(
            prologue_directives=prologue_directives,
            src_items=src_items,
            subroutine_nodes=subroutine_nodes
        )

    # go through src_items[last_declaration_index+1:] and match up with
    # subroutine_nodes.execution_node

    return SubroutineEntity(
        name = subroutine_nodes.name,
        arguments = identifiers["arguments"],
        variables = identifiers["variables"],
        constants = identifiers["constants"],
        prologue_directives = prologue_directives,
        declaration_section = declaration_entries,
        impl_section = []
    )
