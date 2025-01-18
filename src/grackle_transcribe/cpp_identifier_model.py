from dataclasses import dataclass
from enum import Enum, auto
from typing import Union, NamedTuple, Optional

from .clike_parse.tool import StructMemberVar, CStructTranscribeInfo
from .identifiers import Constant, Variable
from .parser import IdentifierExpr
from .routine_analysis import VariableInfo
from .token import Type

# First, we define logic that remaps names that may clash with C macros
# ---------------------------------------------------------------------
# This logic only really comes up for identifiers that represent physical
# constants (i.e. Fortran macros or Fortran parameter variables).


# The following map, maps MACRO identifiers from Fortran identifier to a
# corresponding C identifier
# -> there are a few special cases that we will expand on in comments below
# -> the vast majority of remapping addresses a common case where
#      - where the C layer and Fortran layer both define a constant using
#        exactly the same name and nominal value
#      - the C layer's version is defined as a double-precision literal
#      - the Fortran layer's version is defined as a floating point literal
#        with the same precision as gr_float
#    in these cases, we remap the Fortran identifier to the same name but with
#    a _grflt suffix.
_FORTRAN_MACRO_CONST_REMAP = {
    # the following is associated with different literal values depending on
    # the precision (neither value matches that of the C layer's `pi` macro)
    "pi_val" : "pi_fortran_val",
    # in the following 2 cases are kinda special:
    # - the `mass_e` and `mass_h` Fortran macros perfectly map to `me` and `mh`
    #   (before considering the literal's precision)
    # - Like many other Fortran constants, the literal's precision should match
    #   the precision of gr_float
    "mass_e" : "me_grflt",
    "mass_h" : "mh_grflt",
    # the following cases are not special at all
    "kboltz" : "kboltz_grflt",
    "hplanck" : "hplanck_grflt",
    "ev2erg" : "ev2erg_grflt",
    "sigma_sb" : "sigma_sb_grflt",
    "clight" : "clight_grflt",
    "GravConst" : "GravConst_grflt",
    "SolarMass" : "SolarMass_grflt",
    "Mpc" : "Mpc_grflt",
    "kpc" : "kpc_grflt",
    "pc" : "pc_grflt",
    # we define a custom version of huge and tiny since the C layer
    # theoretically allows arbitrary choices to be made
    "huge" : "huge_fortran_val",
    "tiny" : "tiny_fortran_val"
}

def _standard_renamed_constant_name(fortran_name, fortran_identifier):
    # there are some inconsistencies in the identifier and identifier naming
    # used in the existing C layer and Fortran layer.

    if fortran_identifier.is_macro:
        return _FORTRAN_MACRO_CONST_REMAP.get(fortran_name, None)
    else:
        lower_fortran_name = fortran_name.lower()
        if lower_fortran_name in ["me", "mh", "pi"]:
            # these names clash with a c-layer macro name
            out= f"{lower_fortran_name}_local_var"
            return out
        else:
            return None

# next, define typing infrastructure
# ----------------------------------

_TYPE_TRANSLATION_INFO = {
    Type.i32: ("int", "%d"),
    Type.i64: ("long long", "%lld"),
    Type.f32: ("float", "%g"),
    Type.f64: ("double", "%g"),
    Type.gr_float: ("gr_float", "%g"),
    Type.mask_type: ("gr_mask_type", "%d")
}

_TYPE_MAP = dict((k,v[0]) for k,v in _TYPE_TRANSLATION_INFO.items())

# I don't love how I'm currently modelling types, but I don't know what the
# optimal way to do it is yet. leaving it for now...
class _CppTypeModifier(Enum):
    def __new__(cls, value, array_rank, template):
        if isinstance(value, auto):
            value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        obj.array_rank = array_rank
        obj._template = template
        return obj

    def _fmt_type_str(self, ctype_str):
        return self._template.format(type=ctype_str)

    def is_pointer(self):
        return '_pointer_' in self.name.lower()
    def is_view(self):
        return self.name.lower().startswith('view')
    def is_vector(self):
        return self.name.lower().startswith('vector')

    NONE = (auto(), None, "{type}")
    scalar_pointer = (auto(), None, "{type}*")
    array_pointer_1D = (auto(), 1, "{type}*")
    vector = (auto(), 1, "std::vector<{type}>")
    # the following few types appear to be 1D, but they hold data for higher
    # dimensional arrays
    array_pointer_2Dalloc = (auto(), 2, "{type}*")
    array_pointer_3Dalloc = (auto(), 3, "{type}*")
    vector_2Dalloc = (auto(), 2, "std::vector<{type}>")
    vector_3Dalloc = (auto(), 3, "std::vector<{type}>")
    View2D = (auto(), 2, "grackle::impl::View<{type}**>")
    View3D = (auto(), 3, "grackle::impl::View<{type}***>")
    MACRO_CONST = (auto(), None, None)

class _CppType(NamedTuple):
    type : Type
    modifier : _CppTypeModifier

    def cpp_type_str(self):
        assert self.modifier is not _CppTypeModifier.MACRO_CONST
        partial_type_str = _TYPE_TRANSLATION_INFO[self.type][0]
        return self.modifier._fmt_type_str(partial_type_str)

def _mk_modifier_mapping():
    # constructs a dictionary encodes mapping between 
    #  - a 2-tuple indicating a Fortran variable's rank and whether it's an arg
    #  - to either:
    #    1. a (_CppTypeModifier, None)
    #       - In this case, the Fortran variable is represented
    #       by a single C++ variable for which this is the type-modifier
    #    2. a (_CppTypeModifier, _CppTypeModifier) pair
    #       - the Fortran variable is represented by 2 C++ variables
    #       - the 1st modifier corresponds to a variable holding a view used in
    #         the body of the function
    #       - the 2nd modifier corresponds to the variable wrapped by the first
    #         that is directly references underlying data (it is used as the
    #         argument or it performs the allocation)
    _MOD = _CppTypeModifier
    return {
        (None, True)  : (_MOD.scalar_pointer, None),
        (None, False) : (_MOD.NONE, None),
        (1, True)     : (_MOD.array_pointer_1D, None),
        (1, False)    : (_MOD.vector, None),
        (2, True)     : (_MOD.View2D, _MOD.array_pointer_2Dalloc),
        (2, False)    : (_MOD.View2D, _MOD.vector_2Dalloc),
        (3, True)     : (_MOD.View3D, _MOD.array_pointer_3Dalloc),
        (3, False)    : (_MOD.View3D, _MOD.vector_3Dalloc),
    }
_MODIFIER_MAP = _mk_modifier_mapping()



@dataclass
class CppIdentifierInfo:
    string: Optional[str]
    type: _CppType
    wrapped_key: Optional[str]
    # following attr can only be provided when string is None. In that case,
    # the identifer corresponds to a member of a struct
    struct_mem_arg: Optional[StructMemberVar] = None

    def __post_init__(self):
        if self.string is None:
            assert self.struct_mem_arg is not None
        elif self.struct_mem_arg is None:
            assert self.string is not None
        else:
            raise RuntimeError("string and struct_mem_arg can't both be None")

# now, move onto the guts of the Identifier Model
# -----------------------------------------------

def _prep_cpp_identifiers(
        name: str,
        is_arg : bool,
        fortran_identifier: Union[Variable, Constant],
        var_info: VariableInfo,
        struct_mem_arg: Optional[StructMemberVar] = None
    ):

    is_const = isinstance(fortran_identifier, Constant)
    if getattr(fortran_identifier, 'array_spec', None) is None:
        rank = None
    else:
        rank = fortran_identifier.array_spec.rank
        assert not is_const

    remapped_pr_name = None
    if (rank is None) or (rank == 1):
        # in this first case, we don't inject any wrapped variables
        if is_const:
            assert not is_arg
            assert struct_mem_arg is None
            if fortran_identifier.is_macro:
                pr_modifier = _CppTypeModifier.MACRO_CONST
            else:
                pr_modifier = _CppTypeModifier.NONE

            remapped_pr_name = _standard_renamed_constant_name(
                name, fortran_identifier
            )
        else:
            pr_modifier, dummy = _MODIFIER_MAP[rank, is_arg]
            assert dummy is None # sanity check!
        wrapped_key = None
        wrapped_pair = None

    else:
        pr_modifier, wr_modifier = _MODIFIER_MAP[rank, is_arg]
        if getattr(var_info, 'locally_used', True):
            wr_name, wr_struct_mem_arg = f'{name}_data_', None
            wrapped_key = wr_name.lower()
            if struct_mem_arg is not None:
                wr_name, wr_struct_mem_arg = None, struct_mem_arg
            wr_t = _CppType(fortran_identifier.type, wr_modifier)
            wrapped_pair = (
                wrapped_key,
                CppIdentifierInfo(
                    string=wr_name, type=wr_t, wrapped_key=None,
                    struct_mem_arg=struct_mem_arg
                )
            )
        else:
            pr_modifier = wr_modifier
            wrapped_key = None
            wrapped_pair = None

    pr_key = name.lower()
    if (struct_mem_arg is not None) and (wrapped_pair is None):
        assert remapped_pr_name is None
        pr_name, pr_struct_mem_arg = None, struct_mem_arg
    elif remapped_pr_name is not None:
        assert wrapped_pair is None
        pr_name, pr_struct_mem_arg = remapped_pr_name, None
    else:
        pr_name, pr_struct_mem_arg = name, None

    pr_t = _CppType(fortran_identifier.type, pr_modifier)
    primary_pair = (
        pr_key,
        CppIdentifierInfo(
            string=pr_name, type=pr_t, wrapped_key=wrapped_key,
            struct_mem_arg=pr_struct_mem_arg
        )
    )

    assert rank == primary_pair[1].type.modifier.array_rank, "sanity check"
    if wrapped_pair is not None:
        assert rank == wrapped_pair[1].type.modifier.array_rank, "sanity check"
    return primary_pair, wrapped_pair

_UNINITIALIZED = object()

def _build_identifier_map(
    fortran_identifier_spec,
    identifier_analysis_map,
    c_struct_transcribe_info: Optional[CStructTranscribeInfo]=None
):
    fortran_to_cpp_map = {}
    cpp_identifier_pairs = []

    fortran_arg_names = [
        arg.name.lower() for arg in fortran_identifier_spec.arguments
    ]
    cpp_arg_names = [_UNINITIALIZED for elem in fortran_arg_names]

    identifier_analysis_map_accesses = 0

    for name in fortran_identifier_spec.keys():
        # name is all lowercase, (that is the key), but fortran_var.name has
        # proper capitalization
        fortran_var = fortran_identifier_spec[name]
        try:
            arg_index = fortran_arg_names.index(name.lower())
        except ValueError:
            arg_index = None
        is_arg = (arg_index is not None)

        # if the identifier is an argument, and the c_struct_transcribe_info
        # object (if present) indicates that the argument is commonly called by
        # passing a struct member, we take some special steps (essentially we
        # assume that the struct is passed instead of this argument)
        if is_arg and c_struct_transcribe_info is not None:
            struct_mem_arg = c_struct_transcribe_info.try_get_structmembervar(
                arg_index
            )
            if not isinstance(struct_mem_arg, StructMemberVar):
                struct_mem_arg = None
        else:
            struct_mem_arg = None

        try:
            var_info = identifier_analysis_map[name]
            identifier_analysis_map_accesses+=1
        except KeyError:
            var_info = None
        primary_pair, wrapped_pair = _prep_cpp_identifiers(
            # use fortran_var.name for proper capitalization
            name=fortran_var.name,
            is_arg=is_arg,
            fortran_identifier=fortran_var,
            var_info=var_info,
            struct_mem_arg=struct_mem_arg
        )

        fortran_to_cpp_map[name] = primary_pair[0]
        cpp_identifier_pairs.append(primary_pair)
        if wrapped_pair is not None:
            cpp_identifier_pairs.append(wrapped_pair)

        if is_arg:
            if wrapped_pair is not None:
                arg_name = wrapped_pair[1].string # get proper capitalization
            else:
                arg_name = primary_pair[1].string # get proper capitalization
            cpp_arg_names[arg_index] = arg_name

    assert len(identifier_analysis_map) == identifier_analysis_map_accesses
    assert _UNINITIALIZED not in cpp_arg_names

    cpp_identifiers = dict(cpp_identifier_pairs)
    assert len(cpp_identifiers) == len(cpp_identifier_pairs)

    return fortran_to_cpp_map, cpp_identifiers

# to do start using the following identifiers when passing querying the
# identifier string from _IdentifierModel

class IdentifierUsage(Enum):
    ScalarValue = auto()
    ScalarAddress = auto()
    ArrValue = auto()          # for numpy-like operations
    ArrAddress = auto()        # in C, this is the pointer to the 1st value
    ArrAccessValue = auto()    # in C, this might be arr[idx]
    ArrAccessAddress = auto()  # in C, this might be &arr[idx]

class _IdentifierModel:
    # need to model the C++ data-type so we can properly provide .data
    def __init__(
        self,
        fortran_identifier_spec,
        identifier_analysis_map = None,
        c_struct_transcribe_info = None
    ):
        self.fortran_identifier_spec = fortran_identifier_spec
        if identifier_analysis_map is None:
            identifier_analysis_map = {}
            assert c_struct_transcribe_info is None
        fortran_to_cpp_map, cpp_identifiers = _build_identifier_map(
            fortran_identifier_spec,
            identifier_analysis_map,
            c_struct_transcribe_info=c_struct_transcribe_info
        )
        self.fortran_to_cpp_key = fortran_to_cpp_map
        self.cpp_identifiers = cpp_identifiers

    def _cpp_key(self, fortran_name):
        return self.fortran_to_cpp_key[fortran_name.lower()]

    def _cpp_identifier_from_fortran_name(self, fortran_name):
        key = self.fortran_to_cpp_key[fortran_name.lower()]
        return self.cpp_identifiers[key]

    def fortran_identifier_props(self, name):
        return self.fortran_identifier_spec[name]

    def get_cpp_type(self, name):
        tmp = self._cpp_identifier_from_fortran_name(name)
        return tmp.type.type

    def cpp_arglist_identifier(self, fortran_name):
        primary_idinfo = self._cpp_identifier_from_fortran_name(fortran_name)
        if primary_idinfo.wrapped_key is None:
            return primary_idinfo.string
        else:
            wrapped_idinfo = self.cpp_identifiers[primary_idinfo.wrapped_key]
            return wrapped_idinfo.string

    def cpp_variable_name(
        self, identifier, identifier_usage, arr_ndim = None,
        using_cpp_key = False
    ):
        """
        This produces the string C++ variable name that corresponds to the
        Fortran variable name

        Parameters
        ----------
        identifier: str or IdentifierExpr
            name of the fortran variable (unless using_cpp_key is True)
        identifier_usage
            describes how the identifier gets used
        """
        # I think we will want to get a lot more sophisticated, (and maybe
        # break this functionality out into a separate class/function) but to
        # start out we do something extremely simple
        # -> we might want to eventually return an intermediate class that
        #    the translator then uses to get a pointer address or access value
        #    based on context

        # we assume that the variable name is unchanged
        if using_cpp_key:
            assert isinstance(identifier,str)
            cpp_info = self.cpp_identifiers[identifier]
            var_name = cpp_info.string
            cpptype = cpp_info.type
            struct_mem_arg = cpp_info.struct_mem_arg
        else:
            if isinstance(identifier, IdentifierExpr):
                fortran_var_name = identifier.token.string
            else:
                fortran_var_name = identifier
            primary_cpp_info = self._cpp_identifier_from_fortran_name(
                fortran_var_name
            )
            var_name = primary_cpp_info.string
            cpptype = primary_cpp_info.type
            struct_mem_arg = primary_cpp_info.struct_mem_arg

        modifier = cpptype.modifier

        match identifier_usage:
            case IdentifierUsage.ScalarValue | IdentifierUsage.ScalarAddress:
                need_ptr = (identifier_usage == IdentifierUsage.ScalarAddress)
                if arr_ndim is not None:
                    raise ValueError(
                        "it makes no sense to specify arr_ndim"
                    )
                elif (
                    (modifier == _CppTypeModifier.NONE) or
                    (modifier == _CppTypeModifier.MACRO_CONST)
                ):
                    assert struct_mem_arg is None
                    return f'&{var_name}' if need_ptr else var_name
                elif modifier == _CppTypeModifier.scalar_pointer:
                    if struct_mem_arg is None:
                        return var_name if need_ptr else f'(*{var_name:s})'
                    else:
                        assert var_name is None
                        return struct_mem_arg.to_string(need_ptr)

            case IdentifierUsage.ArrAddress:
                if (arr_ndim is not None) and (arr_ndim != modifier.array_rank):
                    raise ValueError(
                        "the identifier doesn't have the expected rank"
                    )
                elif modifier.is_pointer():
                    if struct_mem_arg is not None:
                        assert var_name is None
                        return struct_mem_arg.to_string(True)
                    return var_name
                elif modifier.is_vector() or modifier.is_view():
                    assert struct_mem_arg is None
                    return f'{var_name:s}.data()'

            case IdentifierUsage.ArrAccessValue:
                _valid_modifiers = (
                    _CppTypeModifier.array_pointer_1D,
                    _CppTypeModifier.vector,
                    _CppTypeModifier.View2D,
                    _CppTypeModifier.View3D
                )
                if (arr_ndim is not None) and (arr_ndim != modifier.array_rank):
                    raise ValueError(
                        "the identifier doesn't have the expected rank"
                    )
                elif struct_mem_arg is not None:
                    if modifier is not _CppTypeModifier.array_pointer_1D:
                        raise NotImplementedError(
                            f"unexpected modifier, {modifier}, for the struct "
                            "member that was originally passed to the "
                            f"subroutine as {struct_mem_arg.original_arg_str}"
                        )
                    out = struct_mem_arg.to_string(True)
                    if out[0] == '&':
                        raise RuntimeError()
                    return out
                elif modifier in _valid_modifiers:
                    return var_name

        raise NotImplementedError(
            "Something went very wrong! Can't handle:\n"
            f" -> identifier_usage: {identifier_usage}\n"
            f" -> modifier: {modifier}")

class ArrInitSpec(NamedTuple):
    # used to provide array information when querying the tra
    fortran_name : str
    translated_axlens: tuple[Optional[str],...]

    def _joined_axlens(self, delim):
        assert None not in self.translated_axlens
        return delim.join(self.translated_axlens)

def _arr_idinfo_l(fortran_name, identifier_model):
    # returns a list of 1 or 2 CppIdentifierInfo instances for C++ identifiers
    # that are used to represent the fortran identifier named fortran_name
    #
    # there are 2 invariants
    # -> out[0] is never a view (it always owns its data)
    # -> when len(out) == 2, out[1] is ALWAYS a view
    primary_idinfo = identifier_model._cpp_identifier_from_fortran_name(
        fortran_name
    )
    if primary_idinfo.wrapped_key is None:
        out = [primary_idinfo]
    else:
        wrapped_idinfo = identifier_model.cpp_identifiers[
            primary_idinfo.wrapped_key
        ]
        out = [wrapped_idinfo, primary_idinfo]
        assert out[1].type.modifier.is_view() # check invariant
    assert not out[0].type.modifier.is_view() # check invariant
    return out

def _translate_deallocate(fortran_name, identifier_model):
    # technically we don't have to translate this (since we have destructors),
    # but, we'll do it anyway...
    idinfo_l = _arr_idinfo_l(fortran_name, identifier_model)[::-1]
    for idinfo in idinfo_l:
        if idinfo.type.modifier.is_view():
            suffix = f' = {idinfo.type.cpp_type_str()}();'
        else:
            suffix = '.clear();'
        yield idinfo.string + suffix

def _translate_allocatable_init(arr_init_spec, identifier_model, is_decl):
    tmp = identifier_model.fortran_identifier_props(
        arr_init_spec.fortran_name
    )
    assert getattr(getattr(tmp, 'array_spec', None), 'allocatable', False)

    idinfo_l = _arr_idinfo_l(arr_init_spec.fortran_name, identifier_model)
    assert idinfo_l[0].type.modifier.is_vector()
    out = []
    if is_decl:
        for idinfo in idinfo_l:
            typestr = idinfo.type.cpp_type_str()
            yield f'{typestr} {idinfo.string};'
    else:
        elem_count = arr_init_spec._joined_axlens(delim=' * ')
        yield f'{idinfo_l[0].string}.reserve({elem_count});'
        if len(idinfo_l) != 1:
            comma_delim = ', '
            constructor_args = (
                f'{idinfo_l[0].string}.data(){comma_delim}' +
                arr_init_spec._joined_axlens(delim=comma_delim)
            )
            typestr = idinfo_l[1].type.cpp_type_str()
            yield f'{idinfo_l[1].string} = {typestr}({constructor_args});'

def get_translated_declaration_lines(common_type,
                                     declared_scalar_fortran_names,
                                     declared_arrinitspec_l,
                                     identifier_model):

    if (
        len(declared_scalar_fortran_names) == 0 and
        len(declared_arrinitspec_l) == 0
    ):
        yield from ()
        return

    ctype = _TYPE_MAP[common_type]
    if len(declared_scalar_fortran_names) > 0: 
        yield f"{ctype} {', '.join(declared_scalar_fortran_names)};"

    for initspec in declared_arrinitspec_l:

        if None in initspec.translated_axlens:
            yield from _translate_allocatable_init(
                initspec, identifier_model, is_decl=True
            )
            continue
        fortran_name = initspec.fortran_name
        idinfo_l = _arr_idinfo_l(fortran_name, identifier_model)

        if identifier_model.fortran_identifier_spec.is_arg(fortran_name):
            # we can skip declaration of the first identifer
            # (this is true whether len(idinfo_l) is 1 or 2)
            idinfo_l = idinfo_l[1:]

        for idinfo in idinfo_l:
            modifier = idinfo.type.modifier
            assert modifier.array_rank == len(initspec.translated_axlens)
            if modifier.is_vector():
                constructor_args = initspec._joined_axlens(delim=' * ')
            elif modifier.is_view():
                assert isinstance(idinfo.wrapped_key, str)
                constructor_arg = identifier_model.cpp_variable_name(
                    idinfo.wrapped_key, 
                    IdentifierUsage.ArrAddress,
                    using_cpp_key=True
                )
                comma_delim = ', '
                constructor_args = (
                    constructor_arg + comma_delim +
                    initspec._joined_axlens(delim=comma_delim)
                )
            else:
                raise RuntimeError(modifier)
            typestr = idinfo.type.cpp_type_str()
            yield f'{typestr} {idinfo.string}({constructor_args});'
