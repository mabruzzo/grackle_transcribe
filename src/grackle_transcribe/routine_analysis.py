from dataclasses import dataclass

from .identifiers import (
    IdentifierSpec
)

from .parser import (
    IdentifierExpr,
    ArrayAccess,
    ArrayAssignStmt,
    ScalarAssignStmt,
    CallStmt,
    iterate_true_contents
)

from .writer import (
    EntryVisitor,
)

@dataclass
class VariableInfo:
    passed_to_subroutine: bool = False
    locally_used: bool = False # not just forwarded to subroutines
    #locally_mutated: bool = False

    # we might also want to record allocate/deallocate statements

def _iterate_var_names(grp):
    # grp is some kind of grouping of stmt, expressions or tokens
    def is_variable(arg):
        return isinstance(arg, (IdentifierExpr, ArrayAccess))

    stack = [iterate_true_contents(grp, is_variable)]
    while len(stack) > 0:
        try:
            elem = next(stack[-1])
        except StopIteration:
            stack.pop()
        else:
            if isinstance(elem, IdentifierExpr):
                yield elem.token.string
            elif isinstance(elem, ArrayAccess):
                yield elem.array_name.token.string
                stack.append(
                    iterate_true_contents(elem.arg_l, is_variable)
                )

def _extract_variable_props(grp, fortran_identifier_spec, info_map):
    pass

class VariableInspectionVisitor(EntryVisitor):
    fortran_identifier_spec: IdentifierSpec

    def __init__(self, fortran_identifier_spec):
        super().__init__(ignore_unknown=False)
        self.fortran_identifier_spec = fortran_identifier_spec
        self.info_map = {
            name.lower(): VariableInfo()
            for name in fortran_identifier_spec.keys()
        }

    def visit_WhitespaceLines(self, entry): pass
    def visit_Comment(self, entry): pass
    def visit_PreprocessorDirective(self, entry): pass
    def visit_OMPDirective(self, entry): pass
    def visit_Declaration(self, entry): pass

    def visit_Stmt(self, entry):
        if isinstance(entry, CallStmt):
            for var_name in _iterate_var_names(entry.arg_l):
                self.info_map[var_name.lower()].passed_to_subroutine = True
            # based on the position in the arg list and knowledge about the
            # subroutine, we could surmise whether the variable is mutated by
            # the subroutine
        else:
            for var_name in _iterate_var_names(entry):
                self.info_map[var_name.lower()].locally_used = True

            # todo: modify self.info_map[var_name].locally_mutated based on
            # the following
        
            if isinstance(entry, ScalarAssignStmt):
                if isinstance(entry.lvalue, IdentifierExpr):
                    pass
                elif isinstance(entry.lvalue, ArrayAccess):
                    pass
                else:
                    raise RuntimeError()
            elif isinstance(entry, ArrayAssignStmt):
                if isinstance(entry.lvalue, IdentifierExpr):
                    pass
                else:
                    raise RuntimeError()
            

    def visit_ControlConstruct(self, entry):
        for (condition, contents) in entry.condition_contents_pairs:
            self.dispatch_visit(condition)
            for content_entry in contents:
                self.dispatch_visit(content_entry)
        self.dispatch_visit(entry.end)

def analyze_routine(subroutine):
    vis = VariableInspectionVisitor(subroutine.identifiers)
    for entry in subroutine.impl_section:
        vis.dispatch_visit(entry)
    return vis.info_map


