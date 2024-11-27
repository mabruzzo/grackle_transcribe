from grackle_transcribe.token import Type
from grackle_transcribe.src_model import *
from grackle_transcribe.subroutine_entity import build_subroutine_entity

from grackle_transcribe.writer import write_full_copy
from grackle_transcribe.translation_writer import (
    transcribe, c_like_fn_signature
)

import itertools
import filecmp
import sys

_PROLOG = """\
#ifndef FORTRAN_FN_DECLARATIONS_HPP
#define FORTRAN_FN_DECLARATIONS_HPP

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include "grackle_macros.h" // FORTRAN_NAME
#include "grackle.h"        // gr_float
#include <stdint.h>         // int32_t
typedef int32_t gr_mask_type;

#define MASK_TRUE 1
#define MASK_FALSE 0

// the following 2 declarations should be removed in the near future
// (they are simply a short-term stop-gap until I get the desired
// capitalization during translation)
#define mask_true 1
#define mask_false 0
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

def declarations_iter(fnames):
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
                )

if __name__ == '__main__':
    import os
    PREFIX = '/Users/mabruzzo/packages/c++/grackle/src/clib/'
    if True:
        skipped_fnames = [ 'cool1d_cloudy_old_tables_g.F',
                            'calc_grain_size_increment_1d.F',
                            'gaussj_g.F',
                            'interpolators_g.F' ]
        def _use_fname(fname):
            return fname.endswith('.F') and fname not in skipped_fnames

        fnames = [fname for fname in os.listdir(PREFIX) if _use_fname(fname)]

    with open('fortran_func_decls.h', 'w') as f:
        f.write(_PROLOG)
        itr = itertools.chain(
            declarations_iter(fnames),
            other_declarations
        )
        for decl in itr:
            f.write(decl)
            f.write(';\n\n')
        f.write(_EPILOG)
