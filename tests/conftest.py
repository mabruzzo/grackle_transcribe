from grackle_transcribe.utils import (
    _valid_fortran_fname, add_gracklesrcdir_arg
)

import os
import pytest

def pytest_addoption(parser):
    add_gracklesrcdir_arg(parser)

def pytest_generate_tests(metafunc):
    if "fortran_src_fname" in metafunc.fixturenames:
        PREFIX = metafunc.config.getoption("--grackle-src-dir")
        fnames = [
            os.path.join(PREFIX, f) for f in os.listdir(PREFIX) \
            if _valid_fortran_fname(f)
        ]
        metafunc.parametrize("fortran_src_fname", fnames)

