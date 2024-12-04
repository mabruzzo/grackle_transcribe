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

parser = argparse.ArgumentParser(prog='transcriber',)
add_gracklesrc_opt(parser, True, required = True)
extract_fncall_inspect_conf = _add_fnsig_simplifier_optgrp(parser)
parser.add_argument(
    '--use-C-linkage',
    action=argparse.BooleanOptionalAction,
    required=True,
    help="whether the transcribed function has C linkage"
)


def main(args):
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

    if True:

        with open(in_fname, 'r') as f:
            provider = LineProvider(f)
            it = get_source_regions(provider)
            for region in it:
                if not region.is_routine:
                    continue
                subroutine = build_subroutine_entity(region, it.prologue)
                print(f"{subroutine.name}, "
                      f"n_args: {len(subroutine.arguments)} "
                      f"n_local_vars: {len(subroutine.variables)}")
                with open(out_fname, "w") as out_f:
                    transcribe(
                        in_fname, out_f,
                        extern_header_fname = out_header_fname,
                        use_C_linkage=True,
                        fncall_inspect_conf=fncall_inspect_conf
                    )
                return 0


if __name__ == '__main__':
    sys.exit(main(parser.parse_args()))
