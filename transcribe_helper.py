from grackle_transcribe.token import Type
from grackle_transcribe.src_model import *
from grackle_transcribe.subroutine_entity import build_subroutine_entity

from grackle_transcribe.writer import write_full_copy
from grackle_transcribe.translation_writer import transcribe

import filecmp
import sys


if __name__ == '__main__':
    import os
    PREFIX = '/Users/mabruzzo/packages/c++/grackle/src/clib/'
    if False:
        fnames = [fname for fname in os.listdir(PREFIX) if fname.endswith('.F')]
    else:
        fnames = ['solve_rate_cool_g.F', 'cool_multi_time_g.F']
        fnames = fnames[::-1]
    for fname in fnames:
        if fname in [ 'cool1d_cloudy_old_tables_g.F',
                      'calc_grain_size_increment_1d.F',
                      'gaussj_g.F']:
            continue
        print()
        print(fname)
        in_fname = os.path.join(PREFIX, fname)
        out_fname = os.path.join('.', 'copies', fname)
        print(
            "making a copy:",
            f"  -> in_fname: {in_fname}",
            f"  -> out_fname: {out_fname}",
            sep = '\n'
        )
        write_full_copy(in_fname, out_fname)

        if filecmp.cmp(in_fname, out_fname, shallow = False):
            print("FILES ARE EQUAL!")
        else:
            raise RuntimeError("the copied file isn't the same!")

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

                sys.exit(0)
