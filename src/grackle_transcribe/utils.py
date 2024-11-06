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

