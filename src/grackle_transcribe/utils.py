import more_itertools

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

def indented_fmt(items, indent = '  ', width = 80, subsequent_indent=None):
    # we can't use textwrap because we want to ensure that the type
    # declaration and argname are on the same line
    delim = ','
    delim_size = len(delim)

    if subsequent_indent is None:
        subsequent_indent = indent
    cur_indent = indent

    cur_buf, cur_buf_size = [], 0
    itr = more_itertools.peekable(items)
    for arg in itr:
        arg_len = len(arg)
        nominal_size = 1 + arg_len + delim_size
        if (len(cur_buf) != 0) and ((cur_buf_size + nominal_size) > width):
            yield ''.join(cur_buf)
            cur_indent = subsequent_indent
            cur_buf, cur_buf_size = [], 0

        if len(cur_buf) == 0:
            cur_buf.append(cur_indent)
            cur_buf_size += len(cur_indent)
            latest_chunk, latest_len = [arg], arg_len
        else:
            latest_chunk, latest_len = [' ', arg], 1 + arg_len

        if bool(itr): # not exhausted
            latest_chunk.append(delim)
            latest_len += delim_size
        cur_buf += latest_chunk
        cur_buf_size += latest_len
    if len(cur_buf) > 0:
        yield ''.join(cur_buf)

def caseless_streq(a: str, b: str):
    return a.casefold() == b.casefold()

