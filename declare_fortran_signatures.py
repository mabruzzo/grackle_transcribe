#from grackle_transcribe.src_model import *
from grackle_transcribe.src_model import LineProvider, get_source_regions
from grackle_transcribe.subroutine_entity import build_subroutine_entity
from grackle_transcribe.translation_writer import c_like_fn_signature
from grackle_transcribe.utils import (
    _valid_fortran_fname, add_gracklesrcdir_arg
)

import argparse
import itertools
import os

parser = argparse.ArgumentParser()
add_gracklesrcdir_arg(parser, required=True)

_PROLOG = """\
#ifndef FORTRAN_FN_DECLARATIONS_HPP
#define FORTRAN_FN_DECLARATIONS_HPP

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include "grackle_macros.h" // FORTRAN_NAME
#include "grackle.h"        // gr_float
#include "phys_constants.h" // physical constants
#include <stdint.h>         // int32_t
typedef int32_t gr_mask_type;

#define MASK_TRUE 1
#define MASK_FALSE 0

"""

_EPILOG = """\

#ifdef __cplusplus
}  // extern "C"
#endif /* __cplusplus */

#endif /* FORTRAN_FN_DECLARATIONS_HPP */
"""

other_declarations = (
    """\
void FORTRAN_NAME(interpolate_1d_g)(
  double *input1, long long *gridDim, double *gridPar1, double *dgridPar1,
  long long *dataSize, double *dataField, double *value
)""",
    """\
void FORTRAN_NAME(interpolate_2d_g)(
  double *input1, double *input2, long long *gridDim, double *gridPar1,
  double *dgridPar1, double *gridPar2, double *dgridPar2, long long *dataSize,
  double *dataField, double *value
)""",
    """\
void FORTRAN_NAME(interpolate_3d_g)(
  double *input1, double *input2, double *input3, long long *gridDim,
  double *gridPar1, double *dgridPar1, double *gridPar2, double *dgridPar2,
  double *gridPar3, double *dgridPar3, long long *dataSize, double *dataField,
  double *value
)""",
    #skipping over interpolate_3dz_g
    """\
void FORTRAN_NAME(interpolate_4d_g)(
  double *input1, double *input2, double *input3, double *input4,
  long long *gridDim, double *gridPar1, double *dgridPar1, double *gridPar2,
  double *dgridPar2, double *gridPar3, double *dgridPar3, double *gridPar4,
  double *dgridPar4, long long *dataSize, double *dataField, double *value
)""",
    """\
void FORTRAN_NAME(interpolate_5d_g)(
  double *input1, double *input2, double *input3, double *input4,
  double *input5, long long *gridDim, double *gridPar1, double *dgridPar1,
  double *gridPar2, double *dgridPar2, double *gridPar3, double *dgridPar3,
  double *gridPar4, double *dgridPar4, double *gridPar5, double *dgridPar5,
  long long *dataSize, double *dataField, double *value
)""",

    """\
// it is plausible that I messed up manual transcription of the following
void FORTRAN_NAME(gaussj_g)(
    int* n, double* a, double* b, int* ierr
)"""
)

def declarations_iter(fnames, PREFIX):
    for fname in fnames:
        print(fname)
        with open(os.path.join(PREFIX, fname), 'r') as f:
            provider = LineProvider(f)
            it = get_source_regions(provider)
            for region in it:
                if not region.is_routine:
                    continue
                subroutine = build_subroutine_entity(region, it.prologue)
                print(f"{subroutine.name}, "
                      f"n_args: {len(subroutine.arguments)} "
                      f"n_local_vars: {len(subroutine.variables)}")
                yield c_like_fn_signature(
                    subroutine,
                    wrapped_by_fortran_name=True
                )[0]

def main(args):
    PREFIX = args.grackle_src_dir

    def _use_fname(fname):
        return _valid_fortran_fname(fname) and fname != 'interpolators_g.F'
    fnames = [fname for fname in os.listdir(PREFIX) if _use_fname(fname)]

    with open('fortran_func_decls.h', 'w') as f:
        f.write(_PROLOG)
        itr = itertools.chain(
            declarations_iter(fnames, PREFIX),
            other_declarations
        )
        for decl in itr:
            f.write(decl)
            f.write(';\n\n')
        f.write(_EPILOG)

if __name__ == '__main__':
    main(parser.parse_args())

