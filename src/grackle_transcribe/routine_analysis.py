from dataclasses import dataclass
from enum import Enum, auto
from typing import Union, NamedTuple

from .identifiers import IdentifierSpec
from .lineno_search import LinenoSearchResult # purely for typing purposes
from .parser import (
    IdentifierExpr,
    ArrayAccess,
    ArrayAssignStmt,
    ScalarAssignStmt,
    CallStmt,
    iterate_true_contents,
    AddressOfExpr
)
from .subroutine_entity import SubroutineEntity
from .writer import (
    EntryVisitor,
)

#class _VarAccess(Enum):
#    NOT_ENCOUNTERED=auto()
#    UNKNOWN = auto()
#    READ_FROM = auto()
#    READ_THEN_MUTATE = auto()
#    MUTATE = auto()

@dataclass
class VariableInfo:
    passed_to_subroutine: bool = False
    locally_used: bool = False # not just forwarded to subroutines
    #var_access: Optional[_VarAcess]

    # we might also want to record allocate/deallocate statements

    @property
    def is_used(self):
        return self.passed_to_subroutine or self.locally_used

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
                yield elem.token.string, False
            elif isinstance(elem, ArrayAccess):
                yield elem.array_name.token.string, True
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
            for var_name, is_arr_access in _iterate_var_names(entry.arg_l):
                self.info_map[var_name.lower()].passed_to_subroutine = True

                # the following if-statement is required to properly handle
                # cases like passing a single element of `u` to
                # lookup_cool_rates0d inside of solve_rate_cool_g.
                if is_arr_access:
                    self.info_map[var_name.lower()].locally_used = True

            # based on the position in the arg list and knowledge about the
            # subroutine, we could surmise whether the variable is mutated by
            # the subroutine
        else:
            for var_name, _ in _iterate_var_names(entry):
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

def analyze_routine(target: Union[SubroutineEntity, LinenoSearchResult]):
    """
    extract information about variables used in a given subroutine or in a
    subsection of a subroutine

    Parameters
    ----------
    target : SubroutineEntity or LinenoSearchResult
        The subroutine entity that we are analyzing. Alternatively, this can
        specify the result of a line search that corresponds to a statement
        that is part of a governing statement of a control construct. In this
        case, we only analyze the control construct.

    Returns
    -------
    VariableInspectionVisitor

    """
    if isinstance(target, SubroutineEntity):
        subroutine = target
        entry_itr = subroutine.impl_section
    elif isinstance(target, LinenoSearchResult):

        subroutine = target.try_get_subroutine()
        if subroutine is None:
            raise ValueError(
                "the specified LinenoSearchResult does not specify a line in "
                "a subroutine"
            )
        elif not target.is_control_construct_stmt:
            raise ValueError(
                "lineno_spec doesn't correspond to a governing statement of a "
                "control construct"
            )
        # index -1 of target.src_entry_hierarchy is a governing statement
        # of the control construct that can be found at index -2
        entry_itr = [target.src_entry_hierarchy[-2]]

    else:
        # In the future, we should consider using duck-typing rather than
        # strict type-checking
        raise TypeError()


    vis = VariableInspectionVisitor(subroutine.identifiers)
        
    for entry in entry_itr:
        vis.dispatch_visit(entry)
    return vis.info_map

# the idea here is that we record the usage of all routines
# ---------------------------------------------------------
from .subroutine_sig import SubroutineArgRef

@dataclass(frozen=True, slots=True)
class SubroutineLocalVarRef:
    subroutine: str
    var: str

    def __init__(self, *, subroutine, var):
        assert isinstance(subroutine, str) and len(subroutine) > 0
        assert isinstance(var, str) and len(var) > 0
        object.__setattr__(self, 'subroutine', subroutine.casefold())
        object.__setattr__(self, 'var', var.casefold())

    def __str__(self):
        return f"{self.subroutine}${self.var}"

@dataclass(frozen = True, slots=True)
class ArbitraryAccess:
    wrapped: SubroutineArgRef | SubroutineLocalVarRef

    def __str__(self):
        return f"{self.wrapped!s}[ELEM]"

    @property
    def subroutine(self):
        return self.wrapped.subroutine

    @property
    def var(self):
        return self.wrapped.var

@dataclass
class VarLink:
    parts: list[SubroutineLocalVarRef | SubroutineArgRef | ArbitraryAccess]

    def __len__(self):
        return len(parts)
    def append(self, v):
        return self.parts.append(v)
    def __str__(self):
        return '|'.join(str(e) for e in self.parts)


@dataclass
class SubroutineCallInspectionVisitor(EntryVisitor):
    fortran_identifier_spec: IdentifierSpec
    subroutine_name: str
    call_props : dict[tuple[str,int],tuple[list,list]]

    def __init__(self, fortran_identifier_spec, subroutine_name):
        super().__init__(ignore_unknown=False)
        self.fortran_identifier_spec = fortran_identifier_spec
        self.subroutine_name = subroutine_name
        self.call_props = {}

    def visit_WhitespaceLines(self, entry): pass
    def visit_Comment(self, entry): pass
    def visit_PreprocessorDirective(self, entry): pass
    def visit_OMPDirective(self, entry): pass
    def visit_Declaration(self, entry): pass

    def visit_Stmt(self, entry):
        args = []
        if not isinstance(entry, CallStmt):
            return None

        def _get_local_ref(call_arg):
            if isinstance(call_arg, IdentifierExpr):
                name = call_arg.token.string
                elemof=False
            elif isinstance(call_arg, ArrayAccess):
                name = call_arg.array_name.token.string
                elemof=True
            else:
                return None

            if self.fortran_identifier_spec.is_arg(name):
                tmp = SubroutineArgRef(subroutine=self.subroutine_name, arg=name)
            else:
                tmp = SubroutineLocalVarRef(subroutine=self.subroutine_name, var=name)
            return ArbitraryAccess(tmp) if elemof else tmp

        simple_args = []
        nonsimple_sigargs = []
        for arg in entry.arg_l.get_args():
            assert isinstance(arg, AddressOfExpr)
            call_ref = _get_local_ref(arg.wrapped)
            arg_ref = arg.arg_name
            if call_ref is None:
                nonsimple_sigargs.append(arg_ref)
            else:
                simple_args.append(VarLink(parts=[call_ref,arg_ref]))
        key = (entry.subroutine.string.casefold(), entry.src.origin.lineno)
        self.call_props[key] = (simple_args, nonsimple_sigargs)

    def visit_ControlConstruct(self, entry):
        for (condition, contents) in entry.condition_contents_pairs:
            self.dispatch_visit(condition)
            for content_entry in contents:
                self.dispatch_visit(content_entry)
        self.dispatch_visit(entry.end)


def analyze_call_stmts(subroutine: SubroutineEntity):
    """
    extract information about variables used in a given subroutine or in a
    subsection of a subroutine

    Parameters
    ----------
    subroutine : SubroutineEntity
        The subroutine entity that we are analyzing.
    """
    assert isinstance(subroutine, SubroutineEntity)
    vis = SubroutineCallInspectionVisitor(
        fortran_identifier_spec=subroutine.identifiers,
        subroutine_name=subroutine.name
    )
    for entry in subroutine.impl_section:
        vis.dispatch_visit(entry)
    return vis.call_props
