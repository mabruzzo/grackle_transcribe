from grackle_transcribe.f_chunk_parse import (
    process_code_chunk, ChunkKind, Type,
    _CONTINUATION_LINE as _CONTINUE_PATTERN
)
from grackle_transcribe.src_model import *
from grackle_transcribe.f_ast_load import create_ast
from grackle_transcribe.subroutine_entity import build_subroutine_entity

from grackle_transcribe.writer import write_full_copy

import filecmp
import sys


def c_like_fn_signature(subroutine):
    type_map = {Type.i32: "int", Type.i64: "long long",
                Type.f32: "float", Type.f64: "double",
                Type.logical: 'bool',
                Type.gr_float: "gr_float"}
    arg_list = (
        f"  {type_map[arg.type]}* {arg.name}"
        for arg in subroutine.arguments
    )
    rslt = f"void {subroutine.name}(\n" + ',\n'.join(arg_list) + '\n)'
    return rslt



if False:
    create_ast('solve_rate_cool_g.F', 'dummy2.txt', double_precision = False)

else:

    import os
    PREFIX = '/Users/mabruzzo/packages/c++/grackle/src/clib/'
    if True:
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
        
        #with open(os.path.join(PREFIX, fname), 'r') as f:
        #    provider = LineProvider(f)
        #    it = get_source_regions(provider)
        #    for region in it:
        #        if not region.is_routine:
        #            continue
        #        subroutine = build_subroutine_entity(region, it.prologue)
        #        #print(c_like_fn_signature(subroutine),'\n')
        #        print(f" -- {subroutine.name}")
        #        #sys.exit(0)
        if False:
            """
            for region in it:
                for lineno, item in region.lineno_item_pairs:
                    if isinstance(item, OMPDirective):
                        print(lineno, "OMPDIRECTIVE", item.lines)
                        #pass
                    elif isinstance(item, Code):
                        kind, tokens, trailing_comment_start, has_label \
                            = process_code_chunk(item.lines)
                        if kind == ChunkKind.Uncategorized:
                            #print(lineno, has_label, [token.string for token in tokens])
                            pass
                        else:
                            #print(lineno, kind, has_label)
                            pass
                    else:
                        pass
                        #print(lineno, item.lines)
            """
