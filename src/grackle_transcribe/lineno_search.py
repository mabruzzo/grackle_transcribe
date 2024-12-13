from dataclasses import dataclass

from .subroutine_entity import SubroutineEntity, ControlConstruct

def get_origin(arg):
    # retrieve the origin object from arg
    # -> arg is a SrcItem, a 
    if hasattr(arg, "src"):
        out = arg.src.origin
    else:
        out = arg.origin
    return out


@dataclass
class LinenoSearchResult:
    # a list of src-components, containing the lineno (ordered from components
    # spanning as large portions of the src-file down to small regions)
    src_entry_hierarchy: list
    # a value of True indicates that the lineno may come after the final
    # statement
    lineno_maybe_excluded: bool
    # indicates whether the line overlaps with the control construct statement
    is_control_construct_stmt: bool

    def __post_init__(self):
        assert len(self.src_entry_hierarchy) > 0
        assert sum(isinstance(e, SubroutineEntity)
                   for e in self.src_entry_hierarchy) <= 1

    def try_get_subroutine(self):
        for entry in self.src_entry_hierarchy:
            if isinstance(entry, SubroutineEntity):
                return entry
        return None


def _lineno_search(itr, lineno):
    # returns a pair
    # -> 1st element encapsulates a region of src-file containing the lineno
    # -> 2nd element is a boolean indicating whether we confidently know that
    #    the lineno is contained
    itr = iter(itr)
    cache = next(itr, None)
    if (cache is None) or (get_origin(cache).lineno > lineno):
        raise RuntimeError("we definitely can't find the lineno")
    while True:
        current_entry = cache
        if get_origin(current_entry).lineno == lineno:
            return current_entry, True
        cache = next(itr, None)
        if cache is None:
            return current_entry, False
        elif get_origin(cache).lineno > lineno:
            return current_entry, True

def lineno_search(itr, lineno):
    # search for sections of the source file (from an iterable) that contains
    # the specified lineno
    levels = []
    definitely_found = False
    is_control_construct_stmt = False

    while itr is not None:
        entry, confident = _lineno_search(itr, lineno)
        if confident and not isinstance(entry, SubroutineEntity):
            definitely_found = True
        levels.append(entry)

        isSubEnt = isinstance(entry, SubroutineEntity)
        isControlConstruct = isinstance(entry, ControlConstruct)

        if isSubEnt and get_origin(entry.endroutine_stmt).lineno <= lineno:
            itr = [entry.endroutine_stmt]
        elif isSubEnt and get_origin(entry.impl_section[0]).lineno <= lineno:
            itr = entry.impl_section
        elif isSubEnt:
            raise RuntimeError("lineno in a subroutine before impl_section")
        elif isControlConstruct and get_origin(entry.end).lineno <= lineno:
            itr = [entry.end]
        elif isControlConstruct:
            for condition, contents in entry.condition_contents_pairs[::-1]:
                if get_origin(contents[0]).lineno <= lineno:
                    itr = contents
                    break
                elif get_origin(condition).lineno <= lineno:
                    itr, is_control_construct_stmt = [condition], True
                    break
            else:
                raise RuntimeError("Something went wrong!")
        else:
            itr = None
    return LinenoSearchResult(
        src_entry_hierarchy=levels,
        lineno_maybe_excluded=not definitely_found,
        is_control_construct_stmt=is_control_construct_stmt
    )

