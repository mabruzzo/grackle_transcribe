
from typing import Any, NamedTuple
from .f_ast_load import load_AST_Nodes, NodeTraverser

# everything is EXTREMELY UNOPTIMIZED

def _unwrap_child(node, *, parent_name = None, child_name = None):
    assert len(node.children) == 1
    if parent_name is not None:
        assert node.name == parent_name
    out = node.children[0]
    if child_name is not None:
        assert out.name == child_name
    return out

def _unwrap(node, name_sequence):
    # name_sequence[0] should match node.name
    # name_sequence[-1] should match the name of the returned node
    #
    # you can supply None as a wildcard
    assert not isinstance(name_sequence, str)

    n_names = len(name_sequence)
    assert n_names > 1
    try:
        out_node = node
        for i in range(n_names - 1):
            out_node = _unwrap_child(
                out_node,
                parent_name=name_sequence[i],
                child_name=name_sequence[i+1]
            )
    except AssertionError:
        last = node
        actual_names = []
        i = 0
        while i < n_names:
            actual_names.append(last.name)
            i+=1
            if len(last.children) == 0:
                break
            elif len(last.children) == 1:
                last = last.children[0]
                i+=1
            else:
                raise RuntimeError(
                    f"Exprected to unwrap {name_sequence}. Encountered "
                    f"{actual_names} and {actual_names[-1]} has multiple "
                    "children"
                ) from None
        raise RuntimeError(
            f"Exprected to unwrap {name_sequence}. Encountered {actual_names}"
        ) from None


    return out_node

def _has_unwrap_nameseq(node, name_sequence):
    try:
        tmp = _unwrap(node, name_sequence)
        return True
    except (RuntimeError, AssertionError):
        return False

def _extract_name(node, *, wrapper_parent_name = None):
    if wrapper_parent_name is None:
        name_node = node
    else:
        name_node = _unwrap_child(
            node, parent_name=wrapper_parent_name, child_name="Name"
        )
    assert name_node.source is not None
    assert len(name_node.children) == 0
    return name_node.source

def _SubroutineStmt_extract_name_and_args(node):
    # https://github.com/llvm/llvm-project/blob/697d65ded678f405b637e769f1b0bcc755c8461a/flang/include/flang/Parser/parse-tree.h#L3143
    first_child = node.children[0]
    if first_child.name != "Name":
        # it would be PrefixSpec, if any anything
        raise RuntimeError("not currently equipped to handle this case")
    else:
        routine_name = _extract_name(first_child)

    # extract the arguments
    args = []
    for arg_node in node.children[1:]:
        if arg_node == "LanguageBindingSpec":
            raise RuntimeError("not currently equipped to handle this case")
        args.append(
            _extract_name(arg_node, wrapper_parent_name="DummyArg")
        )

    return routine_name, args

class Variable(NamedTuple):
    name: str

    declaration_statement: Any
    # index of declaration_stmt.children that corresponds to self
    child_index: int

    is_constant: bool

def _find_variable_index(varlist, varname):
    # search in reverse order
    for index in range(len(varlist))[::-1]:
        if varlist[index].name == varname:
            return index
    raise ValueError(
        f"couldn't find variable in the specified list named {varname}"
    )

def _process_declarations(node, known_arg_names):
    # the order is probably meaningful
    arg_l, local_l = [], []
    known_vars = set()
    print(node.children[0].name)
    assert node.children[0].name == "ImplicitPart"

    nested_typedecl_seq = [
        "DeclarationConstruct", "SpecificationConstruct", "TypeDeclarationStmt"
    ]
    nested_constdef_seq = [
        "DeclarationConstruct", "SpecificationConstruct", "ParameterStmt", "NamedConstantDef"
    ]

    for decl in node.children[1:]:

        # first, we consider whether we are defining a NamedConstant's value
        # (previously, we would have declared it to look like a local var)
        if ((decl.name == nested_constdef_seq[-1]) or
            _has_unwrap_nameseq(decl, nested_constdef_seq)):
            if decl.name == nested_constdef_seq[-1]:
                constdef = decl
            else:
                constdef = _unwrap(decl, nested_constdef_seq)
            assert len(constdef.children) > 1
            name = _extract_name(
                constdef.children[0], wrapper_parent_name="NamedConstant"
            )
            index = _find_variable_index(local_l, name)
            local_l[index] = local_l[index]._replace(is_constant=True)

            continue

        # unwrap the TypeDeclarationStmt node
        # https://github.com/llvm/llvm-project/blob/main/flang/include/flang/Parser/parse-tree.h#L1393
        decl = _unwrap(decl, nested_typedecl_seq)

        # first child is always the type specification (we skip over this for
        # now, but we could always come back to it later)
        typespec = decl.children[0]
        assert typespec.name == "DeclarationTypeSpec"

        # optionally, we can have 1 or more AttrSpec
        if decl.children[1].name == "AttrSpec":
            attrspec = decl.children[1]
            if decl.children[2].name != "EntityDecl":
                raise RuntimeError("Encountered unhandled case!")
            elif (
                (len(attrspec.children) == 1) and 
                (attrspec.children[0].name == "Parameter") and
                (len(attrspec.children[0].children) == 0)
            ):
                # everything in the current type declaration is a constant
                constant_var = True
                first_entitydecl_idx = 2
            else:
                raise RuntimeError("unequipped to handle this case")
        else:
            constant_var = False
            first_entitydecl_idx = 1

        # now we go through the trailing variables
        for child_idx in range(first_entitydecl_idx, len(decl.children)):
            entitydecl = decl.children[child_idx]
            assert entitydecl.name == "EntityDecl"
            assert len(entitydecl.children) > 0
            name = _extract_name(entitydecl.children[0])

            assert name not in known_vars
            known_vars.add(name)

            var = Variable(
                name=name, declaration_statement=decl, child_index=child_idx,
                is_constant=constant_var
            )
            if name in known_arg_names:
                assert not constant_var
                arg_l.append(var)
            else:
                local_l.append(var)
    assert len(arg_l) == len(known_arg_names)
    return arg_l, local_l







class SubroutineEntity:
    def __init__(self, node):

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

        assert root_node.children[0].name == 'SubroutineStmt'
        self.name, known_args = _SubroutineStmt_extract_name_and_args(
            root_node.children[0]
        )
        assert root_node.children[1].name == 'SpecificationPart'
        self.SpecificationPart = root_node.children[1].name
        self.arguments, self.locals = _process_declarations(
            root_node.children[1], set(known_args)
        )
        assert root_node.children[2].name == 'ExecutionPart'
        self.ExecutionPart = root_node.children[2].name

        if len(root_node.children) == 5:
            assert root_node.children[3].name == 'InternalSubprogramPart'
            raise RuntimeError("NOT IMPLEMENTED")
        else:
            assert len(root_node.children) == 4
        self.EndSubroutineStmt = root_node.children[-1]


        


if __name__ == '__main__':
    root_node_itr = load_AST_Nodes(
        '/Users/mabruzzo/packages/c++/grackle/src/clib/log.txt'
    )
    for root_node in root_node_itr:
        print(f"\nstarting a new root node:")
        if True:
            subroutine = SubroutineEntity(root_node)
            args = [e.name for e in subroutine.arguments]
            local_vars = [
                e.name for e in subroutine.locals if not e.is_constant
            ]
            print(f'{subroutine.name}')
            print(f"-> args: {args}\n-> local-vars:{local_vars}")
        else:
            itr = NodeTraverser(root_node)
            for i, node in enumerate(itr):
                level = itr.cur_level()
                if node.source is None:
                    suffix = ''
                else:
                    suffix = f"src: `{node.source}`"
                print(f"{' '*level}{node.name}{suffix}")
