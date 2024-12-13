from more_itertools import peekable, prepend

from .f_ast_load import (
    AstCreateConfig,
    create_ast,
    load_AST_Nodes,
    NodeTraverser,
    FortranASTNode,
    _unwrap,
    _unwrap_child,
    _has_unwrap_nameseq,
    _extract_name
)

from .token import (
    BuiltinFn,
    BuiltinProcedure,
    Keyword, Misc,
    Literal, Operator, Token, token_has_type, Type,
    compressed_concat_tokens
)

from .identifiers import(
    ArrSpec, Constant, Variable, IdentifierSpec
)


from .src_model import (
    Code,
    PreprocKind,
    PreprocessorDirective,
    SrcItem,
    SrcRegion,
)

from .parser import (
    ControlConstructKind,
    Stmt,
    Parser,
    UncategorizedStmt,
    TokenStream,
    stmt_has_single_tok_type
)

from dataclasses import dataclass

from typing import List, NamedTuple, Optional, Tuple, Union

@dataclass(frozen=True)
class Declaration:
    """
    Represents a declaration of a single Constant or 1+ Variable(s)
    """
    src: Union[Code, List[SrcItem]]
    ast: FortranASTNode
    identifiers: Union[List[Variable], Constant, Variable]

    def __post_init__(self):
        if isinstance(self.identifiers, Constant):
            assert isinstance(self.src, Code)
        #elif isinstance(self.src, list) and len(self.identifiers) == 1:
        #    raise ValueError(
        #        "src can only be a list when we define multiple Variables"
        #    )
        elif len(self.identifiers) == 0:
            raise ValueError("a declaration can't define 0 variables")

    @property
    def defines_constant(self):
        if isinstance(self.identifiers, Constant):
            return True
        elif isinstance(self.identifiers, Variable):
            return False
        return any(isinstance(e, Constant) for e in self.identifiers)

    @property
    def is_precision_conditional(self):
        return not isinstance(self.src, Code)


class _ConditionContentPair(NamedTuple):
    condition: Union[Stmt,PreprocessorDirective]
    content: List[Union[Stmt, SrcItem]]

@dataclass(frozen=True)
class ControlConstruct:
    """Represents if-statement, do-statement, ifdef-statment"""
    condition_contents_pairs: Tuple[_ConditionContentPair, ...]
    end: Union[Stmt,PreprocessorDirective]
    kind: ControlConstructKind

    def __post_init__(self):
        assert len(self.condition_contents_pairs) > 0
        is_code = isinstance(self.end,Code)
        for (condition,_) in self.condition_contents_pairs:
            assert isinstance(condition, Code) == is_code

    @property
    def n_branches(self):
        return len(self.condition_contents_pair)

    @property
    def origin(self):
        first = self.condition_contents_pairs[0][0]
        if hasattr(first, "src"):
            return first.src.origin
        else:
            return first.origin



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

    @property
    def origin(self): return self.subroutine_stmt.src.origin



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
    assert token_has_type(item.tokens[0], Type)
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
            for count_i, name in enumerate(_concrete_constants):
                tmp = _unpack_declaration(next(node_itr), is_const=True)
                if len(tmp) != 1:
                    raise AssertionError(
                        "something is wrong! it seems like there wasn't a "
                        "declaration"
                    )
                elif tmp[0][0].lower() != name.lower():
                    raise AssertionError(
                        f"part {count_i+1} of the declaration to be for "
                        f"the {name} identifier. It seems to be for "
                        f"{tmp[0][0].lower()}."
                    )

        elif item.first_token_has_type(Misc.ImplicitNone):
            assert node_itr.peek().name == "ImplicitPart"
            entries.append(UncategorizedStmt(src=item, ast=next(node_itr)))

        elif item.first_token_has_type(Type):
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
                        name=name, type=item.tokens[0].type, is_macro=False,
                        decl_section_index=index,
                    )
                )
            else:
                #print(compressed_concat_tokens(item.tokens))
                identifier_list = parser.parse_variables_from_declaration(
                    TokenStream(item),
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

def _preproc_kind(arg, kind):
    return isinstance(arg, PreprocessorDirective) and arg.kind == kind

class _PreprocLevel:
    def kind(self): return ControlConstructKind.MetaIfElse
    def is_branch_stmt(self, arg): return _preproc_kind(arg, PreprocKind.ELSE)
    def is_close_stmt(self, arg): return _preproc_kind(arg, PreprocKind.ENDIF)

class _LevelProps(NamedTuple):
    # instances of this class are temporary objects used to collect information
    # while constructing ControlConstruct

    # each entry of pairs holds a branch followed by a list of contents
    pairs: List[Tuple[
        Union[PreprocessorDirective,Stmt],
        List[Union[Stmt,SrcItem,ControlConstruct]]
    ]]
    kind: Union[ControlConstructKind, _PreprocLevel]

def _process_impl_items(parser, first_item, src_items):
    # this deals with organizing items into (nested?) levels (if applicable)

    levels = []
    def _try_add_level_or_branch(stmt):
        new_level_kind = (
            _PreprocLevel() if _preproc_kind(stmt, PreprocKind.IFDEF) else
            ControlConstructKind.match_construct_start(stmt)
        )
        if new_level_kind is not None:
            levels.append(_LevelProps(pairs=[(stmt,[])], kind=new_level_kind))
            return True
        elif (len(levels) > 0) and levels[-1].kind.is_branch_stmt(stmt):
            levels[-1].pairs.append((stmt,[]))
            return True
        return False

    def _pop_full_level(close_stmt):
        if (len(levels) > 0) and levels[-1].kind.is_close_stmt(close_stmt):
            level = levels.pop()
            return ControlConstruct(
                tuple(_ConditionContentPair(*e) for e in level.pairs),
                end=close_stmt, kind=level.kind
            )
        return None

    for item in prepend(first_item, src_items):
        match item:
            case PreprocessorDirective() if _try_add_level_or_branch(item):
                continue
            case PreprocessorDirective() if entry := _pop_full_level(item):
                if len(levels) == 0:
                    return entry
                levels[-1].pairs[-1][1].append(entry)
            case Code():
                stmt = parser.parse_stmt(TokenStream(item))
                if _try_add_level_or_branch(stmt):
                    continue
                entry = _pop_full_level(stmt)
                if entry is None:
                    entry = stmt

                if len(levels) == 0:
                    return entry
                levels[-1].pairs[-1][1].append(entry)
            case SrcItem():
                # we must be in the process of making a level object
                assert len(levels) > 0
                levels[-1].pairs[-1][1].append(item)
            case _:
                raise TypeError(
                    f"Something went wildly wrong!\n"
                    f" -> current item's class: {item.__class__.__name__}\n"
                    " -> all types should be SrcItem subclasses\n"
                )
    # if we reach this point it is an error
    raise RuntimeError(
        "Something went wrong! We ran out of source items while "
        f"we are {len(levels)} levels deep"
    )

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
            entries.append(_process_impl_items(parser, item, src_iter))

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
    assert src_items[0].first_token_has_type(Keyword.SUBROUTINE)
    subroutine_stmt = UncategorizedStmt(src=src_items[0], ast=None)
    assert src_items[-1].first_token_has_type(Keyword.ENDROUTINE)
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
