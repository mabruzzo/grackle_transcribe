from grackle_transcribe.src_model import LineProvider, get_source_regions
from grackle_transcribe.subroutine_entity import build_subroutine_entity
from grackle_transcribe.translation_writer import transcribe
from grackle_transcribe.utils import (
    _valid_fortran_fname, add_gracklesrcdir_arg
)

import argparse
import os
import sys

parser = argparse.ArgumentParser(prog='transcriber',)
add_gracklesrcdir_arg(parser, required = True)
parser.add_argument(
    "--fname",
    help="the basename of the fortran file that you want to transcribe",
    required=True
)

def main(args):
    PREFIX = args.grackle_src_dir
    fname = args.fname

    assert _valid_fortran_fname(fname)

    if True:
        in_fname = os.path.join(PREFIX, fname)

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
                with open("my-result.C", "w") as out_f:
                    transcribe(
                        in_fname, out_f,
                        extern_header_fname = "my-result-decl.h",
                        use_C_linkage=True
                    )
                return 0


if __name__ == '__main__':
    sys.exit(main(parser.parse_args()))
