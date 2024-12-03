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
    in_fname = args.grackle_src_file
    assert _valid_fortran_fname(in_fname)
    use_C_linkage = args.use_C_linkage

    fncall_inspect_conf = extract_fncall_inspect_conf(args)

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
                with open("my-result.C", "w") as out_f:
                    transcribe(
                        in_fname, out_f,
                        extern_header_fname = "my-result-decl.h",
                        use_C_linkage=True,
                        fncall_inspect_conf=fncall_inspect_conf
                    )
                return 0


if __name__ == '__main__':
    sys.exit(main(parser.parse_args()))
