
from .token import Type

from itertools import islice
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

class ArrSpec(NamedTuple):
    axlens: List[Optional[None]]
    allocatable: bool

    @property
    def rank(self): return len(self.axlens)

class Variable(NamedTuple):
    name: str
    type: Type
    decl_section_index: int
    variable_number_on_line: int
    array_spec: Optional[ArrSpec]
    
    @property
    def is_array(self): return self.array_spec is not None

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


