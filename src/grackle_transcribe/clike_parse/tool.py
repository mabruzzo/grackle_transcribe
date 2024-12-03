from enum import Enum
import itertools
import os
import re
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

    The name (e.g. `GrackleStruct.CHEMISTRY_DATA.value`) is the name known to
    the c++ compiler.

    The implementation is loosely based on an example from python docs where we
    don't care about the integer values associated with each member
    """
    def __repr__(self):
        return '<%s.%s>' % (self.__class__.__name__, self.name)

    CHEMISTRY_DATA = "chemistry_data"
    CHEMISTRY_DATA_STORAGE = "chemistry_data_storage"
    CODE_UNITS = "code_units"
    GRACKLE_FIELD_DATA = "grackle_field_data"
    PHOTO_RATE_STORAGE = "photo_rate_storage"

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



    parser.add_argument(
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

    arg_grp = parser.add_argument_group(
        title = "Fn Arg Simplifier Options",
        description = (
            f"C/C++ calls to Fortran subroutines in Grackle often involve "
            "long argument lists where many struct members are passed as "
            "separate arguments. The {fn_call_loc_opt} option can be used to "
            "specify where the subroutine being transcribed is called. The "
            "arguments can be used to specify the names of local variables "
            "corresponding that have different struct types. By using this "
            "information with the arg list, we can reduce the number of "
            "arguments in the transcribed routine. "
            "If a local variable corresponds to a stack-allocated instance of "
            "a struct, rather than a pointer, you should specify the "
            "variable name with a leading &. (Note you may need to escape & "
            "in many shells)."
        )
    )

    for member in GrackleStruct:
        t = member.value
        arg_grp.add_argument(
            f"--{t}_ptr",
            help=f"name of local variable of type `{t}*`", default=None
        )

    def _parse_args(args, dirname=None):
        l = []
        for member in GrackleStruct:
            attr = f"{member.value}_ptr"
            var_name, is_ptr = getattr(args,attr), True
            if var_name is not None:
                if var_name[0] == '&':
                    var_name, is_ptr = var_name[1:], False
                l.append(LocalStructVar(var_name, member, is_ptr))

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


class StructMemberArg(NamedTuple):
    """
    Represents an argument used in a C call to a subroutine that involves
    structs.

    Using the `arg_str` value returns an equivalent c string. This will specify
    the address of a scalar or the address of the first element in an array
    (which may have 1 or more dimensions)
    """
    struct_varname: str
    member_str: str
    use_arrow: bool
    arg_has_addressOf: bool


    def _common(self, use_arrow=True):
        if use_arrow:
            return f'{self.struct_varname}->{self.member_str}'
        else:
            return f'{self.struct_var}.{self.member_str}'

    def original_arg_str(self):
        # this is equivalent to the original string
        if self.arg_has_addressOf:
            return '&'+self._common(use_arrow=self.use_arrow)
        return self._common(use_arrow=self.use_arrow)

    def accessexpr_in_fn(self, access_address):
        """
        Returns expression to access the value or address of the struct member 
        for use inside of a function, where `struct_arg` is the name of a
        function argument that passes the struct as a pointer.
        """
        if access_address:
            pre,suf = ('&', '') if self.arg_has_addressOf else ('','')
        else:
            pre,suf = ('', '') if self.arg_has_addressOf else ('*(', ')')
        return f'{pre}{self._common(use_arrow=True)}{suf}'


def _mk_matcher(local_struct_var):
    use_arrow = local_struct_var.is_ptr
    memjoin = r'\-\>' if use_arrow else r'\.'
    struct_varname = local_struct_var.var_name

    memstr_pat = r"[a-zA-Z_][a-zA-Z_\d\.]*"

    patterns = []

    _main_core_pattern = (
        rf"{struct_varname}\s*{memjoin}\s*"
        rf"(?P<mem_str>{memstr_pat}\s*(\[\s*\d+\s*\])?)"
    )

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
                return StructMemberArg(
                    struct_varname = struct_varname,
                    member_str = member_str,
                    use_arrow = use_arrow,
                    arg_has_addressOf=arg_has_addressOf
                )
    return _main_builder

class CFnCallArgListInfo(NamedTuple):
    arg_l: list[str | StructMemberArg]
    employed_struct_vars: list[LocalStructVar]

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
                results.append(arg)
    except:
        raise RuntimeError(
            "Encountered an error with parsing function call from:\n"
            f" -> fname: {fn_call_loc.fname!r}\n"
            f" -> from line {fn_call_loc.line_start}\n"
            f" -> up to (but not including) line {fn_call_loc.line_stop}\n"
            f" -> {contents.splitlines(True)[0] + '...'!r}"
        )


    return CFnCallArgListInfo(results, list(employed_struct_vars))

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
