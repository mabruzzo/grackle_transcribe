# the idea here is to test the roundtrip of reading a file from grackle and
# then writing it back out

from grackle_transcribe.writer import write_full_copy
import os
import filecmp

def test_roundtrip(fortran_src_fname, tmp_path):

    basename = os.path.basename(fortran_src_fname)
    in_fname = fortran_src_fname
    out_fname = os.path.join(tmp_path, basename)
    print(
        "making a copy:",
        f"  -> in_fname: {in_fname}",
        f"  -> out_fname: {out_fname}",
        sep = '\n'
    )
    write_full_copy(in_fname, out_fname)
    if not filecmp.cmp(in_fname, out_fname, shallow = False):
        raise AssertionError(f"the copy of {basename} isn't the same!")
