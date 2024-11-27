import os
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--grackle-src-dir",
        action="store",
        help=(
            "Specifies the path to the src/clib subdirectory of the grackle "
            "repository.",
        )
    )

_bad_fnames = ('cool1d_cloudy_old_tables_g.F',
               'calc_grain_size_increment_1d.F',
               'gaussj_g.F')
def _valid_fortran_fname(fname):
    return fname.endswith('.F') and fname not in _bad_fnames

def pytest_generate_tests(metafunc):
    if "fortran_src_fname" in metafunc.fixturenames:
        PREFIX = metafunc.config.getoption("--grackle-src-dir")
        fnames = [
            os.path.join(PREFIX, f) for f in os.listdir(PREFIX) \
            if _valid_fortran_fname(f)
        ]
        metafunc.parametrize("fortran_src_fname", fnames)

