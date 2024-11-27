# this file was originally used for converting logical type to MASK_KIND
# -> a lot about the codebase has changed since then (there may be some kinks)
# -> more importantly, the employed strategies probably aren't very robust

from grackle_transcribe.token import (
    process_code_chunk, Type,
)
from grackle_transcribe.src_model import *
from grackle_transcribe.f_ast_load import create_ast
from grackle_transcribe.subroutine_entity import build_subroutine_entity

from grackle_transcribe.writer import write_full_copy, replace_logical_with_mask

import filecmp
import sys



if True:

    import os
    PREFIX = '/Users/mabruzzo/packages/c++/grackle/src/clib/'
    if False:
        fnames = [fname for fname in os.listdir(PREFIX) if fname.endswith('.F')]
    else:
        
        #complete: ['lookup_cool_rates0d.F',
        # 'solve_rate_cool_g.F',
        # 'cool_multi_time_g.F',
        # 'calc_temp_cloudy_g.F',
        # 'calc_tdust_3d_g.F',
        # 'calc_tdust_1d_g.F',
        # 'calc_grain_size_increment_1d.F', # manually converted!
        # ]
        fnames = ['cool1d_multi_g.F']
    for fname in fnames:
        # we ignore the following cases because the files are either temperamental OR we don't want to convert them
        if fname in [ 'cool1d_cloudy_old_tables_g.F',
                      'calc_grain_size_increment_1d.F',
                      'gaussj_g.F', 'interpolators_g.F']:
            continue
        in_fname = os.path.join(PREFIX, fname)
        #out_fname = os.path.join('.', 'copies', fname)
        print(f'\n{in_fname}')
        out_fname = in_fname
        #print(
        #    "transforming file:",
        #    f"  -> in_fname: {in_fname}",
        #    f"  -> out_fname: {out_fname}",
        #    sep = '\n'
        #)
        replace_logical_with_mask(in_fname, out_fname)

elif False:

    import os
    PREFIX = '/Users/mabruzzo/packages/c++/grackle/src/clib/'
    if False:
        fnames = [fname for fname in os.listdir(PREFIX) if fname.endswith('.F')]
    else:
        fnames = ['solve_rate_cool_g.F', 'cool1d_multi_g.F']
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
