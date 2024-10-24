# load in the AST dumped by LLVM's flang-new

# the AST is dumped by the Flang frontend: https://flang.llvm.org/docs/FlangDriver.html

# descriptions of the AST can be found here:
# https://github.com/llvm/llvm-project/blob/main/flang/include/flang/Parser/parse-tree.h

import dataclasses
import os
import re
import subprocess
from typing import List, NamedTuple, Optional


class DumpLineReader:
    """
    Iterates over lines. For each line, we return a 2-tuple holding
        1. number of indents and
        2. the chunk of line after removing the indents & trailing '\n'
    """
    def __init__(self, f, indent = "| "):
        self._itr = iter(f)
        self._next_pair = None
        self._indent, self._indent_len = indent, len(indent)

    def _get_next_pair(self, peek_op):
        if self._next_pair is None:
            try:
                line = next(self._itr)
            except StopIteration:
                pass # do nothing
            else:
                lstripped = line.lstrip(self._indent)
                if len(lstripped) == 0:
                    assert (line == '\n') and next(line_itr,None) is None
                else:
                    discard_len = len(line) - len(lstripped)
                    n_indents = discard_len // self._indent_len
                    assert (n_indents*self._indent_len) == discard_len
                    self._next_pair = (n_indents, lstripped.rstrip())
        out = self._next_pair
        if not peek_op:
            if out is None:
                raise StopIteration
            self._next_pair = None
        return out

    def peek_next_indent_count(self, dflt = None):
        pair = self._get_next_pair(True)
        return dflt if pair is None else pair[0]

    def __next__(self):
        return self._get_next_pair(False)

    def __iter__(self):
        return self

_NODE_NAMES = set()
_NODES_WITH_NULL_TRAILING = set()
_NODES_WITHOUT_NULL_TRAILING = set()


@dataclasses.dataclass
class FortranASTNode:
    name: str
    children: 'Tuple[FortranASTNode, ...]'
    source: Optional[str]

    @classmethod
    def create(cls, name, children, source, null_trailing_arrow):
        _NODE_NAMES.add(name)
        if null_trailing_arrow:
            _NODES_WITH_NULL_TRAILING.add(name)
        else:
            _NODES_WITHOUT_NULL_TRAILING.add(name)
        return cls(name=name, children=children, source=source)

_ARROW_PATTERN = re.compile(r"(?: -> )|(?: ->$)")
_SOURCE_PATTERN = re.compile(r"(?P<name>.*) = '(?P<src>[^']*)'$")

def _get_line_parts(line_chunk):
    # this expects where we trimmed indents and trailing white spaces
    #
    # generally you have 1 or mode node-names (separated by " -> ").
    # the last node name MAY be followed by an "extra descriptor". This
    # descriptor could be
    #  * ` ->` without anything else following it (this seems to
    #    indicate that the node denotes the end of some kind of code unit.
    #    For example, it could be a return statement or an "enddo" statement
    #  * ` = '<...>'` where <...> corresponds to an excerpt of source-code
    #    (where unnecessary spaces are removed)
    parts = _ARROW_PATTERN.split(line_chunk)

    null_trailing_arrow = parts[-1] == ""
    if null_trailing_arrow:
        del parts[-1]
    assert len(parts) > 0

    # now let's check if source code is provided
    m = _SOURCE_PATTERN.match(parts[-1])
    if m:
        source_code = m.group('src')
        parts[-1] = m.group('name')
    else:
        assert ' = ' not in parts[-1]
        source_code = None

    # sanity check!
    # (not sure if this is strictly true)
    assert not (null_trailing_arrow and (source_code is not None))
    return (parts, source_code, null_trailing_arrow)


def _build_node_hierarchy(reader, current_indent_depth):
    # it would be better if this weren't recursive (but it's probably ok)
    pair = next(reader)
    assert pair[0] == current_indent_depth
    # a line chunk may specify 1 or more "generations" of nodes
    cur_line_chunk = pair[1]

    # before we build any node(s) form cur_line_chunk, we retrieve all children
    # for the last node specified in cur_line_chunk
    children = []
    while reader.peek_next_indent_count() == (current_indent_depth+1):
        children.append(_build_node_hierarchy(reader, current_indent_depth+1))

    # retrieve all relevant details for nodes specified in cur_line_chunk
    node_names, source_code, null_trailing_arrow = _get_line_parts(
        cur_line_chunk
    )

    # build the node corresponding to the rightmost node in node_names
    # -> if node_names includes 1 or more names, the source_code and
    #    null_trailing_arrow variable only describe properties of this node
    highest_level_node = FortranASTNode.create(
        name=node_names[-1],
        children=tuple(children),
        source=source_code,
        null_trailing_arrow = null_trailing_arrow
    )

    # finally, we try to build any specified ancestors of this node (from
    # right to left)
    for node_name in node_names[-2::-1]:
        highest_level_node = FortranASTNode.create(
            name=node_name,
            children=(highest_level_node,),
            source=None,
            null_trailing_arrow = False
        )
    return highest_level_node



    


def load_AST_Nodes(path,):
    with open(path, 'r') as f:
        reader = DumpLineReader(f)
        while reader.peek_next_indent_count(-1) > -1:
            assert reader.peek_next_indent_count() == 0
            yield _build_node_hierarchy(reader, current_indent_depth=0)

class NodeTraverser:

    def __init__(self, root_node):
        # each stack index holds [node, last_child_index]
        self._stack = [[root_node, -2]]
        self._exhausted = False

    def cur_level(self):
        return len(self._stack) - 1

    def __iter__(self):
        return self

    def __next__(self):
        while len(self._stack) > 0:
            node, prev_child_index = self._stack[-1]
            cur_child_index = prev_child_index + 1
            if cur_child_index >= len(node.children):
                del self._stack[-1]
                continue
            self._stack[-1][1] = cur_child_index
            if cur_child_index == -1:
                return node
            else:
                self._stack.append(
                    [node.children[cur_child_index], -2]
                )
        else:
            raise StopIteration

_AUTOGEN_HEADER_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "autogen_headers"
))

class AstCreateConfig(NamedTuple):
    command: str = "flang-new"
    autogen_dir: str = _AUTOGEN_HEADER_DIR
    grackle_src_dir: str = "/Users/mabruzzo/packages/c++/grackle/src"

def create_ast(src_fname, out_fname, double_precision = True, macros = None, *,
               grackle_src_fname = True, config = AstCreateConfig()):

    clib_path = os.path.join(config.grackle_src_dir, "clib")
    if grackle_src_fname:
        src_path = os.path.join(clib_path, src_fname)
    else:
        src_path = src_fname
    assert os.path.isfile(src_path)

    autogen_dir_path = os.path.join(
        config.autogen_dir, "float8" if double_precision else "float4"
    )

    args = [
        config.command,
        "-fc1",
        "-fdebug-dump-parse-tree",
        "-cpp",
        f"-I{os.path.join(config.grackle_src_dir, 'include')}",
        f"-I{autogen_dir_path}"
    ]
    if not grackle_src_fname:
        args.append(f"-I{clib_path}")
    
    if macros is not None:
        for macro in macros:
            args.append(f"-D{macro}")
    args.append(src_path)

    with open(out_fname, "w") as f:
        subprocess.run(args, stdout=f, check=True)
    return out_fname
