from enum import Enum
import re

class _Dummy(Enum):
    _dummy=1
_DUMMY = _Dummy._dummy

_skip_regex = re.compile("[^ \t\n]")

def index_non_space(s, pos = _DUMMY, end = _DUMMY, *, dflt = _DUMMY):
    # somewhat inspired by str.index and str.find
    pos = 0 if pos is _DUMMY else pos
    end = len(s) if end is _DUMMY else end
    if (m := _skip_regex.search(s, pos=pos, endpos = end)) is not None:
        return m.start()
    elif dflt is _DUMMY:
        raise ValueError("non whitespace not found")
    return dflt

_bad_fnames = ('cool1d_cloudy_old_tables_g.F',
               'calc_grain_size_increment_1d.F',
               'gaussj_g.F')

def _valid_fortran_fname(fname):
    # this probably shouldn't be hardcoded, but we need it in multiple places
    return fname.endswith('.F') and fname not in _bad_fnames

def add_gracklesrc_opt(parser, src_file_opt, required = False):

    if hasattr(parser, "addoption"): #pytest parser
        adder = parser.addoption
    else: #argparse parser
        adder = parser.add_argument

    if src_file_opt:
        flag = '--grackle-src-file'
        help = (
            "Specifies the path to a Fortran source file in the Grackle "
            "repository."
        )
    else:
        flag = "--grackle-src-dir"
        help = (
            "Specifies the path to the src/clib subdirectory of the grackle "
            "repository."
        )

    adder(flag, action="store", required=required, help=help)


def add_gracklesrcdir_arg(parser, required = False):
    # TODO: remove this function
    # for compatability reasons, this wraps add_gracklesrc_opt
    add_gracklesrc_opt(parser, src_file_opt=False, required=required)

