from grackle_transcribe.signature_registry import build_signature_registry
from grackle_transcribe.translation_writer import c_like_fn_signature
from grackle_transcribe.utils import add_gracklesrcdir_arg

import argparse

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


def main(args):
    sig_registry = build_signature_registry(src_dir=args.grackle_src_dir,
                                            verbose=True)

    with open('fortran_func_decls.h', 'w') as f:
        f.write(_PROLOG)
        for sig in sig_registry.values():
            decl = c_like_fn_signature(
                sig, wrapped_by_fortran_name=True
            )[0]
            f.write(decl)
            f.write(';\n\n')
        f.write(_EPILOG)

if __name__ == '__main__':
    main(parser.parse_args())

