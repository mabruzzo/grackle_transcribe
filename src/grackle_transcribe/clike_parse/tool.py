from dataclasses import dataclass
from enum import Enum
import itertools
import os
import re
from types import MappingProxyType
from typing import NamedTuple

class FuncCallLoc(NamedTuple):
    fname: str
    line_start: str
    line_stop: str

    def load_contents(self):
        with open(self.fname, 'r') as f:
            return ''.join(
                itertools.islice(iter(f), self.line_start, self.line_stop)
            )

def _parse_funcCallLoc(arg, dirname=None, argname="--fn-call-loc"):
    parts = arg.split(':')
    if len(parts) != 3 or any(len(e) == 0 for e in parts):
        raise ValueError(
            f"the {argname} argument expects a 3 part argument delimited by "
            "colons or `<path>:<start_lineno>:<stop_lineno>`. The provided "
            "value doesn't match this expectation!"
        )
    
    if dirname is None:
        fname = parts[0]
    else:
        fname = os.path.join(dirname, parts[0])
    
    if not os.path.isfile(fname):
        raise ValueError("There is no file called {fname}")

    start = int(parts[1])
    stop = int(parts[2])
    assert 0 <= start
    assert start < stop
    return FuncCallLoc(fname, start, stop)


class GrackleStruct(Enum):
    """
    Represents established grackle interface types

    The name (e.g. `GrackleStruct.CHEMISTRY_DATA.full_type_name`) is the name
    known to the c++ compiler.

    The implementation is loosely based on an example from python docs where we
    don't care about the integer values associated with each member
    """
    def __repr__(self):
        return '<%s.%s>' % (self.__class__.__name__, self.name)

    def __new__(cls, name, in_grackle_impl_namespace, prefer_pass_by_value):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        obj.base_type_name = name
        if in_grackle_impl_namespace:
            obj.full_type_name = f"grackle::impl::{name}"
        else:
            obj.full_type_name = name
        obj.prefer_pass_by_value = prefer_pass_by_value
        return obj


    CHEMISTRY_DATA = ("chemistry_data", False, False)
    CHEMISTRY_DATA_STORAGE = ("chemistry_data_storage", False, False)
    CODE_UNITS = ("code_units", False, False)
    GRACKLE_FIELD_DATA = ("grackle_field_data", False, False)
    PHOTO_RATE_STORAGE = ("photo_rate_storage", False, True)

    # newly-introduced internal data structures:
    INTERNAL_GR_UNITS = ("InternalGrUnits", False, True)
    INDEX_RANGE = ("IndexRange", False, True)
    GRAIN_SPECIES_COLLECTION = ("GrainSpeciesCollection", True, True)
    LOGT_LIN_INTERP_SCRATCH_BUF = ("LogTLinInterpScratchBuf", True, True)
    COOL_1D_MULTI_SCRATCH_BUF = ("Cool1DMultiScratchBuf", True, True)
    COOL_HEAT_SCRATCH_BUF = ("CoolHeatScratchBuf", True, True)
    SPECIES_COLLECTION = ("SpeciesCollection", True, True)
    COL_REC_RXN_RATE_COLLECTION = ("ColRecRxnRateCollection", True, True)
    PHOTO_RXN_RATE_COLLECTION = ("PhotoRxnRateCollection", True, True)
    CHEM_HEATING_RATES = ("ChemHeatingRates", True, True)
    TIME_DERIV_CONTEXT = ("time_deriv_0d::ContextPack", True, True)


class LocalStructVar(NamedTuple):
    var_name: str
    type: GrackleStruct
    is_ptr: bool

class FnCallInspectionConf(NamedTuple):
    fn_call_loc: FuncCallLoc
    local_struct_vars: list[LocalStructVar]

def _add_fnsig_simplifier_optgrp(parser):
    """
    Adds an argument group to a CLI-parser that is used to specify options
    related to reading a function call from a C/C++ section of the code.
    """
    fn_call_loc_opt, fn_call_loc_attrname = "--fn-call-loc", "fn_call_loc"

    arg_grp = parser.add_argument_group(
        title = "Fn Arg Simplifier Options",
        description = (
            "C/C++ calls to Fortran subroutines in Grackle often involve "
            "long argument lists where many struct members are passed as "
            f"separate arguments. The {fn_call_loc_opt} option can be used to "
            "specify where the subroutine being transcribed is called. The "
            "arguments can be used to specify the names of local variables "
            "corresponding that have different struct types. By using this "
            "information with the arg list, we can reduce the number of "
            "arguments in the transcribed routine. "
                    )
    )

    arg_grp.add_argument(
        fn_call_loc_opt,
        default = None,
        help = (
            "Takes 1 argument of the form "
            "`<path>:<lineno_start>:<lineno_stop>` that specfies the range of "
            "lines in a C/C++ file (at location <path>) where the subroutine "
            "that is being translated is called. See `Fn Arg Simplifier "
            "Options` for more info"
        )
    )

    _members = {
        f'{member.base_type_name}_ptr' : member for member in GrackleStruct
    }
    _member_l = "[" + ','.join(_members) + "]"

    arg_grp.add_argument(
        "--var-ptr-pairs",
        nargs='+',
        help=(
            "each argument passed to this option should have the form "
            "`<varname>,<type>` where <varname> is the name of a local "
            "variable and <type> is the corresponding type (known types "
            f"include {_member_l}). "
            "If a local variable corresponds to a stack-allocated instance of "
            "a struct, rather than a pointer, you should specify the "
            "variable name with a leading &. (Note you may need to escape & "
            "in many shells)."
        )
    )

    def _parse_args(args, dirname=None):
        l = []

        pair_l = args.var_ptr_pairs
        if pair_l is None:
            pair_l = []
        for pair_str in pair_l:
            pair = pair_str.split(',')
            assert len(pair) == 2
            var_name, type_str = pair
            struct_type = _members[type_str]
            if var_name[0] == '&':
                var_name, is_ptr = var_name[1:], False
            else:
                is_ptr = True
            l.append(LocalStructVar(var_name, struct_type, is_ptr))

        fn_call_loc_cliarg = getattr(args, fn_call_loc_attrname)
        if (fn_call_loc_cliarg is None):
            if len(l) > 0:
                raise ValueError(
                    f"When {fn_call_loc_opt} isn't given, you can't specify "
                    "any other Fn Arg Simplifier Option"
                )
            return None
        tmp = _parse_funcCallLoc(
            fn_call_loc_cliarg, dirname=dirname, argname=fn_call_loc_opt
        )
        return FnCallInspectionConf(tmp, l)
    return _parse_args

def _extract_args(start_pos, contents):
    #start_pos is the open parenthesis for contents

    def _nextdelim_slc(start_pos, contents):
        n_open_parens=0
        for ind in range(start_pos, len(contents)):
            match contents[ind]:
                case ',' if n_open_parens == 0:
                    return slice(ind, ind+1)
                case ')' if n_open_parens == 0:
                    if m := re.match(r'^\)\s*;', contents[ind:], re.MULTILINE):
                        return slice(ind, ind + m.end())
                    raise RuntimeError("should be unreachable")
                case ')': n_open_parens-=1
                case '(': n_open_parens+=1
                case ';': raise RuntimeError("should be unreachable")
                case _: pass
        return None

    assert contents[start_pos] == '('
    start_pos+=1
    out = []
    any_remaining = True
    while any_remaining:
        delim_slc = _nextdelim_slc(start_pos, contents)
        if delim_slc is None:
            raise ValueError("The contents never seem to end!")
        delim = contents[delim_slc]
        any_remaining = (delim[-1] != ';')
        cur_arg = contents[start_pos:delim_slc.start]
        start_pos = delim_slc.stop
        if (len(cur_arg) > 0) and not cur_arg.isspace():
            out.append(cur_arg.strip())
        elif ((len(out) == 0) and (not any_remaining)):
            break # this covers the case where there are 0 arguments
        else:
            raise RuntimeError("It appears that something went wrong")
    if contents[start_pos:].isspace() or len(contents) == start_pos:
        return out
    raise ValueError(
        "contents contains more than an arg list & a trailing semi-colon"
    )


class StructMemberExprDescr(NamedTuple):
    """
    A crude concept, but this essentially describes the relationship between a
    struct and some kind of member_str

    The term member_str refers to everything that isn't the top level struct.

    For concreteness, let's consider some examples:
      - the member_str is "my_integer" in `my_struct1.my_integer`
      - the member_str is "my_struct1.my_integer" in the following 2 cases:
        - my_struct2.my_struct1.my_integer
        - my_struct3->my_struct1.my_integer
      - the member_str is "my_struct1->my_integer" in the following 2 cases:
        - my_struct4.my_struct1->my_integer
        - my_struct4->my_struct1->my_integer
      - the member str is "data[2]" in `data_wrapper.data[2]`
    """
    struct_varname: str
    member_str: str
    full_expr_is_ptr: bool

    def fmt_string(self, struct_is_pointer, access_address):
        join = "->" if struct_is_pointer else "."
        if access_address:
            pre,suf = ('&', '') if not self.full_expr_is_ptr else ('','')
        else:
            pre,suf = ('', '') if not self.full_expr_is_ptr else ('*(', ')')
        return f'{pre}{self.struct_varname}{join}{self.member_str}{suf}'

class StructMemberVar(NamedTuple):
    """
    Represents an argument used in a C call to a subroutine that involves
    structs OR to represent the formatter if a variable used in subroutine
    being transcribed to C.
    """
    struct_is_pointer: bool
    descr: StructMemberExprDescr

    def to_string(self, access_address):
        """
        Returns expression to access the value or address of the struct member
        """
        return self.descr.fmt_string(
            struct_is_pointer=self.struct_is_pointer,
            access_address=access_address
        )

def _mk_matcher(local_struct_var):
    use_arrow = local_struct_var.is_ptr
    memjoin = r'\-\>' if use_arrow else r'\.'
    struct_varname = local_struct_var.var_name

    identifiertok_pat = r"[a-zA-Z_][a-zA-Z_\d\.]*"
    memstr_pat = identifiertok_pat
    index_pat = rf"(?:\d+|{identifiertok_pat}::{identifiertok_pat})"

    patterns = []

    _main_core_pattern = (
        rf"{struct_varname}\s*{memjoin}\s*"
        rf"(?P<mem_str>{memstr_pat}\s*(\[\s*{index_pat}\s*\])?)"
    )
    print(_main_core_pattern)

    _alt_core_pattern = (
        rf"{struct_varname}\s*{memjoin}\s*"
        rf"(?P<mem_str>{memstr_pat})\s*\+\s*(?P<ptr_offset>[0-6])"
    )

    l = [
        r"^&?\s*" + _main_core_pattern + "$",
        r"^&?\s*\(\s*" +  _main_core_pattern + "\s*\)$",
        r"^" + _alt_core_pattern + "$",
        r"^\(\s*" +  _alt_core_pattern + "\s*\)$"
    ]
    patterns = [re.compile(elem, re.MULTILINE) for elem in l]

    def _main_builder(s):
        for pattern in patterns:
            if (m := pattern.match(s)) is not None:
                d = m.groupdict()
                if "ptr_offset" in d:
                    member_str = m["mem_str"] + "[" + m["ptr_offset"] + "]"
                    arg_has_addressOf = True
                else:
                    member_str = d["mem_str"]
                    arg_has_addressOf = s.startswith('&')
                # Because this is being extracted from a call to a Fortran
                # Routine, in a dialect of Fortran where all arguments are
                # always passed by reference, the full expression specified by
                # s must ALWAYS specify the address of a scalar or the address
                # of the first element in an array (with an arbitrary number of
                # dimensions).
                # -> This let's us make the following inference:
                full_expr_is_ptr = not arg_has_addressOf

                struct_member_descr = StructMemberExprDescr(
                    struct_varname = struct_varname,
                    member_str = member_str,
                    full_expr_is_ptr = full_expr_is_ptr
                )
                return StructMemberVar(
                    struct_is_pointer = use_arrow,
                    descr = struct_member_descr
                )
    return _main_builder

def parse_fn_call(fn_call_loc, fn_name, local_struct_vars= []):
    """
    Produce a list of argument-details used in a call to a Fortran subroutine.

    Because we are calling a Fortran subroutine, we can cut some corners
    - it returns void
    - all args must be pointers
    """

    matchers = [
        (struct_var_id, _mk_matcher(struct_var))
        for struct_var_id, struct_var in enumerate(local_struct_vars)
    ]

    _allowed_first_patterns = [
        (r"^\s*FORTRAN_NAME\s*\(\s*" + fn_name + r"\s*\)\s*\("),
        (r"^\s*" + fn_name + r"\s*\(")
    ]
    m = None
    contents = fn_call_loc.load_contents()
    try:
        for pattern in _allowed_first_patterns:
            m = re.match(pattern, contents, flags=re.MULTILINE)
            if m is not None:
                break
        else:
            raise ValueError(
                "the specified contents don't appear to correspond to a call "
                f"the {fn_name} FORTRAN subroutine"
        )

        employed_struct_vars = set()

        results = []
        for arg in _extract_args(m.end()-1, contents):
            arg = arg.strip()
            for struct_var_id, matcher in matchers:
                if (tmp := matcher(arg)) is not None:
                    results.append(tmp)
                    employed_struct_vars.add(local_struct_vars[struct_var_id])
                    break
            else:
                print(f"not in any struct: {arg!r}")
                results.append(arg)
    except:
        raise RuntimeError(
            "Encountered an error with parsing function call from:\n"
            f" -> fname: {fn_call_loc.fname!r}\n"
            f" -> from line {fn_call_loc.line_start}\n"
            f" -> up to (but not including) line {fn_call_loc.line_stop}\n"
            f" -> {contents.splitlines(True)[0] + '...'!r}"
        )

    return (results, list(employed_struct_vars))

@dataclass(frozen=True)
class CStructTranscribeInfo:
    # contains information related to C structs to use during transcription
    orig_arg_l: list[str | StructMemberVar]
    _structvar_useptr_map: MappingProxyType[str, tuple[LocalStructVar,bool]]

    def get_structprop_useptr_pair(self, structvar_name):
        return self._structvar_useptr_map[structvar_name]

    def all_structvar_useptr_pairs(self):
        return list(self._structvar_useptr_map.values())

    def try_get_structmembervar(self, arg_index):
        tmp = self.orig_arg_l[arg_index]
        if not isinstance(tmp,StructMemberVar):
            return None
        _, use_ptr = self._structvar_useptr_map[tmp.descr.struct_varname]
        return StructMemberVar(use_ptr, tmp.descr)

def get_C_Struct_transcribe_info(fn_call_loc, fn_name, local_struct_vars= []):
    arg_l, employed_struct_vars = parse_fn_call(
        fn_call_loc=fn_call_loc,
        fn_name=fn_name,
        local_struct_vars=local_struct_vars
    )

    # in principle, we could make some different choices for whether or not to
    # use a pointer
    mapping = {
        struct_var.var_name : (
            struct_var, not struct_var.type.prefer_pass_by_value
        ) for struct_var in employed_struct_vars
    }
    return CStructTranscribeInfo(arg_l, MappingProxyType(mapping))

if __name__ == '__main__':

    fn_call_loc = _parse_funcCallLoc(
        "/Users/mabruzzo/packages/c++/grackle/src/clib/calculate_cooling_time.C:215:543"
    )
    parse_fn_call(
        fn_call_loc,
        fn_name = "cool_multi_time_g",
        local_struct_vars= [
            LocalStructVar(
                "my_units",
                GrackleStruct.CODE_UNITS,
                is_ptr = True
            ),
            LocalStructVar(
                "my_chemistry",
                GrackleStruct.CHEMISTRY_DATA,
                is_ptr = True
            ),
            LocalStructVar(
                "my_rates",
                GrackleStruct.CHEMISTRY_DATA_STORAGE,
                is_ptr = True
            ),
            LocalStructVar(
                "my_fields",
                GrackleStruct.GRACKLE_FIELD_DATA,
                is_ptr = True
            ),
            LocalStructVar(
                "my_uvb_rates",
                GrackleStruct.PHOTO_RATE_STORAGE,
                is_ptr = False
            )
        ]
    )
