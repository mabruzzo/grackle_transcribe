
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
