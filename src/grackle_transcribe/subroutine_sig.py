from .syntax_unit import IdentifierExpr
from .token import Type
from .utils import caseless_streq

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict,Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .identifiers import IdentifierSpec, Variable


@dataclass(frozen=True, slots=True)
class SubroutineArgRef:
    subroutine: str
    arg: str

    def __init__(self, *, subroutine, arg):
        assert isinstance(subroutine, str) and len(subroutine) > 0
        assert isinstance(arg, str) and len(arg) > 0
        object.__setattr__(self, 'subroutine', subroutine.casefold())
        object.__setattr__(self, 'arg', arg.casefold())

    def __str__(self):
        return f'{self.subroutine}%{self.arg}'


class _UnknownAxLenType(Enum):
    UNKNOWN_AX_LEN = auto()
UNKNOWN_AX_LEN = _UnknownAxLenType.UNKNOWN_AX_LEN


@dataclass(frozen=True)
class ScalarArrayProp:
    # in the future, maybe we also store an int as an option
    _axlens : Tuple[Union[_UnknownAxLenType, SubroutineArgRef]]

    def __init__(self, _axlens):
        # not intended to be used directly
        assert isinstance(_axlens, tuple)
        for elem in _axlens:
            assert elem is UNKNOWN_AX_LEN or isinstance(elem, SubroutineArgRef)
        object.__setattr__(self, '_axlens', _axlens)

    @classmethod
    def Scalar(cls):
        return cls(())

    @classmethod
    def Array(cls, axlens, coerce_None=False):
        assert len(axlens) > 0
        if coerce_None:
            _axlens = tuple(
                (UNKNOWN_AX_LEN if e is None else e) for e in axlens
            )
        else:
            _axlens = tuple(axlens)
        return cls(_axlens)

    @classmethod
    def GenericArray(cls, rank:int):
        assert rank > 0
        return cls( tuple(UNKNOWN_AX_LEN for i in range(rank)) )

    @property
    def is_scalar(self): return self._axlens == ()

    @property
    def is_array(self): return not self.is_scalar

    @property
    def rank(self): return None if self._axlens == () else len(self._axlens)

    def axlen(self, index: int):
        if self._axlens == ():
            return None
        return self._axlens[index]

    def _fmt(self, repr: bool):
        if self.is_scalar:
            if repr:
                return f"{self.__class__.__name__}.Scalar()"
            return "Scalar"
        elif all(e is UNKNOWN_AX_LEN for e in self._axlens):
            rank = self.rank
            if repr:
                return f"{self.__class__.__name__}.GenericArray({rank})"
            return f"GenericArray{rank}D"
        else:
            tmp = ','.join(str(e) for e in self._axlens)
            if repr:
                return f"{self.__class__.__name__}.Array([{tmp}])"
            return f"Array({tmp})"

    def __repr__(self): return self._fmt(True)
    def __str__(self): return self._fmt(False)



@dataclass(frozen=True)
class ArgDescr:
    """Specifies an argument requirement in a function signature"""
    name: str
    type: Type
    prop: ScalarArrayProp

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__) and
            caseless_streq(self.name, other.name) and
            self.type == other.type and
            self.prop == other.prop
        )

@dataclass(frozen=True)
class SubroutineSignature:
    name: str
    _arg_map: Dict[str, ArgDescr]

    def __init__(self, name: str, arguments: Tuple[ArgDescr]):
        _args = {arg.name.casefold() : arg for arg in arguments}
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, '_arg_map', _args)

    @property
    def arguments(self): return tuple(self._arg_map.values())

    @property
    def arguments_iter(self): return self._arg_map.values()

    def n_args(self):
        return len(self._arg_map)

    def lookup_arg(self, key):
        key = key.casefold()
        return self._arg_map[key]

    def fmt_as_string(self, compact=False):
        if compact:
            indent, arg_sep, paren_pad = '', ', ', ''
        else:
            indent, arg_sep, paren_pad = '  ', ',\n', '\n'
        fmt = repr

        return ''.join([
            f"{self.__class__.__name__}({paren_pad}",
            f"{indent}name={fmt(self.name)}{arg_sep}",
            f"{indent}arguments=({paren_pad}{indent*2}",
            f"{arg_sep}{indent*2}".join(fmt(e) for e in self.arguments_iter()),
            f"{paren_pad}{indent})"
            f"{paren_pad})"
        ])
        

    def __repr__(self):
        return self.fmt_as_string(compact=True)

    @property
    def key(self): return self.name.casefold()

def build_subroutine_sig(name:str, identifier_spec: 'IdentifierSpec'):
    def _get_prop(var: 'Variable'):
        if var.is_array:
            axlens = []
            for axlen in var.array_spec.axlens:
                if isinstance(axlen, IdentifierExpr):
                    axlens.append(
                        SubroutineArgRef(
                            subroutine=name, arg=axlen.token.string
                        )
                    )
                else:
                    axlens.append(UNKNOWN_AX_LEN)
            return ScalarArrayProp.Array(axlens)
        return ScalarArrayProp.Scalar()
    arguments = tuple(
        ArgDescr(name=var.name, type=var.type, prop=_get_prop(var))
        for var in identifier_spec.arguments
    )
    return SubroutineSignature(name, arguments)


