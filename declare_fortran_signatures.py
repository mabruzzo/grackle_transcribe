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
#include <stdint.h>         // int32_t
typedef int32_t gr_mask_type;

#define MASK_TRUE 1
#define MASK_FALSE 0

// defining constants
// ------------------
// -> these constants are faithful to phys_consts.def
// -> we have noted where the variables deviate from phys_constants.h
// -> we can't directly include phys_constants.h because it defines mh, me, and pi
//    as macro-names and some of the transcribed routines use those same names as the
//    names of const-variables

// the following 2 constants have the same values as mh and me in phys_constants.h
#define mass_h 1.67262171e-24
#define mass_e 9.10938215e-28

// `pi` macro in phys_constants.h has different precisions from both branches
#ifdef GRACKLE_FLOAT_4
  #define pi_val 3.14159265f
#else
  #define pi_val 3.141592653589793
#endif

// the following were ripped out of phys_constants.h (they are entirely consistent
// with the values in phys_consts.def
#define kboltz    1.3806504e-16
#define hplanck   6.6260693e-27
#define ev2erg    1.60217653e-12
#define sigma_sb  5.670373e-5
#define clight    2.99792458e10
#define GravConst 6.67428e-8
#define SolarMass 1.9891e33
#define Mpc       3.0857e24
#define kpc       3.0857e21
#define pc        3.0857e18

// constants from grackle_fortran_types.def
// -> `tiny` and `huge` are properly defined by grackle_macros.h

#define tiny8 1.e-40
#define huge8 1.e+40


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
                )

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

