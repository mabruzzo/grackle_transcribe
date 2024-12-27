from grackle_transcribe.signature_registry import build_signature_registry
from grackle_transcribe.src_model import LineProvider, get_source_regions
from grackle_transcribe.subroutine_entity import build_subroutine_entity
from grackle_transcribe.translation_writer import transcribe
from grackle_transcribe.utils import (
    _valid_fortran_fname, add_gracklesrc_opt
)
from grackle_transcribe.clike_parse.tool import _add_fnsig_simplifier_optgrp

import argparse
import os
import sys

def main(args):
    fn = args.fn
    return fn(args)

parser = argparse.ArgumentParser(prog='transcriber',)
subparsers = parser.add_subparsers(required=True)

# convert subcommand
# ------------------
parser_convert = subparsers.add_parser(
    'convert',
    help="actually convert the specified file"
)
add_gracklesrc_opt(parser_convert, True, required = True)
extract_fncall_inspect_conf = _add_fnsig_simplifier_optgrp(parser_convert)
parser_convert.add_argument(
    '--use-C-linkage',
    action=argparse.BooleanOptionalAction,
    required=True,
    help="whether the transcribed function has C linkage"
)
parser_convert.add_argument(
    '--preparse-signatures',
    action='store_true',
    help=(
        "when True, we preparse a registry of subroutine signatures, which "
        "can help us identify errors in the original fortran code."
    )
)

def main_convert(args):
    """
    Reads a file ``<path/to/grackle/src/clib>/<stem>.F`` and transcribes the
    first subroutine in the file. Then we write out:
       - ``./<stem>-cpp.C``
       - ``./<stem>-cpp.h``

    We choose to add the ``-cpp.C`` suffix to the output rather than just
    ``.C`` to maintain compatibility with grackle's old build-system.
      - consider the case where ``<stem>.F`` holds multiple subroutines.
        We will still need to continue compiling some of those subroutines
        to ensure that the transcribed subroutine works.
      - under the classic build-system, ``<stem>.F`` and ``<stem>.C`` would
        produce object files with the same names.
    """

    in_fname = args.grackle_src_file
    assert _valid_fortran_fname(in_fname)
    use_C_linkage = args.use_C_linkage

    fncall_inspect_conf = extract_fncall_inspect_conf(args)

    stem = os.path.splitext(os.path.basename(in_fname))[0]
    # we add the -cpp.C suffix instead of just .C in order to avoid clashes
    # with object files in the classic build system
    # - for example solve_rate_ch
    out_fname = f'{stem}-cpp.C'
    out_header_fname = f'{stem}-cpp.h'

    if args.preparse_signatures:
        print("Building subroutine signature registry")
        signature_registry = build_signature_registry(
            src_dir=os.path.dirname(in_fname),
            verbose = True
        )
    else:
        signature_registry = None

    with open(in_fname, 'r') as f:
        provider = LineProvider(f)
        it = get_source_regions(provider)
        for region in it:
            if not region.is_routine:
                continue
            with open(out_fname, "w") as out_f:
                transcribe(
                    in_fname, out_f,
                    extern_header_fname = out_header_fname,
                    use_C_linkage=True,
                    fncall_inspect_conf=fncall_inspect_conf,
                    signature_registry=signature_registry
                )
            return 0

parser_convert.set_defaults(fn=main_convert)

# analyze subcommand
# ------------------
from grackle_transcribe.signature_registry import build_signature_registry
from grackle_transcribe.translation_writer import c_like_fn_signature
from grackle_transcribe.utils import _valid_fortran_fname
from grackle_transcribe.src_model import get_source_regions, LineProvider

parser_analyze = subparsers.add_parser(
    "analyze",
    help="check if existing fortran code has any obvious issues"
)
add_gracklesrc_opt(parser_analyze, False, required=True)

def main_analyze(args):
    print("parsing the registry:")
    print()
    sig_registry = build_signature_registry(
        src_dir=args.grackle_src_dir, verbose=True
    )
    print()
    print("Reparsing all routines and cross-checking subroutine calls against "
          "the registry")

    with os.scandir(args.grackle_src_dir) as it:
        for entry in it:
            if entry.name in ['lookup_cool_rates0d.F']:
                # lookup_cool_rates0d.F does a number of sketchy things
                # (in terms of passing scalars into subroutines)
                print(f"skipping {entry.name}")
                continue
            elif _valid_fortran_fname(entry.name):
                print("reading file:", entry.name)
                with open(entry.path, 'r') as f:
                    provider = LineProvider(f, fname = entry.path)
                    it = get_source_regions(provider)
                    for region in it:
                        if not region.is_routine:
                            continue
                        subroutine = build_subroutine_entity(
                            region,
                            it.prologue,
                            signature_registry=sig_registry
                        )
                        print('  |-->', subroutine.name)

parser_analyze.set_defaults(fn=main_analyze)

# declare_fortran_signatures subcommand
# -------------------------------------
parser_declarations = subparsers.add_parser(
    "declarations",
    help=(
        "create a C header of function declarations for each Fortran routine"
    )
)
add_gracklesrc_opt(parser_declarations, False, required=True)

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

def main_declarations(args):
    sig_registry = build_signature_registry(
        src_dir=args.grackle_src_dir, verbose=True
    )

    with open('fortran_func_decls.h', 'w') as f:
        f.write(_PROLOG)
        for sig in sig_registry.values():
            decl = c_like_fn_signature(
                sig, wrapped_by_fortran_name=True
            )[0]
            f.write(decl)
            f.write(';\n\n')
        f.write(_EPILOG)

parser_declarations.set_defaults(fn=main_declarations)

if __name__ == '__main__':
    sys.exit(main(parser.parse_args()))
