"""
This is an interactive script intended to help replace (i-1) indexing

This is invoked in 2 steps! (and probably requires a little manual cleanup)
"""

import argparse
import functools
import os
import re
import shutil
import tempfile


def _find_and_replace(in_f, out_f, pattern_repl_pairs,
                      full_line_replacer_funcs=[],
                      announce_changes=True, dryrun=False):
    """
    This function does all the heavy lifting!

    Theoretically, it would be really nice to limit this operation to tokens
    outside of prepreocessor-directives, comments, string-literals, and
    character-literals. But, given the anticipated lifetime of this script,
    it probably isn't worth the effort!
    """
    modified_line_count, total_change_count = 0, 0

    _replacers = tuple(
        functools.partial(re.compile(pattern).subn, repl=repl)
        for pattern, repl in pattern_repl_pairs
    )
    def _get_output_line(line):
        for fn in full_line_replacer_funcs:
            modified_line = fn(line)
            if modified_line is not None:
                return modified_line, 1
        for replacer in _replacers:
            modified_line, change_count = replacer(string=line)
            if change_count > 0:
                return modified_line, change_count
        return line, 0

    for line in in_f:
        outline, change_count = _get_output_line(line)
        if change_count > 0:
            modified_line_count+=1
            total_change_count += change_count
        if not dryrun:
            out_f.write(outline)

    if announce_changes:
        print(
            f"{total_change_count} changes on {modified_line_count} lines"
        )
    return total_change_count, modified_line_count



def replace_i_with_ip1(in_f, out_f):
    # we are using negative lookahead and negative lookbehind to make sure that
    # i is not part of a variable
    pattern_repl_pairs = [ (r"(?<!\w)i(?!\w)", "ip1") ]
    _find_and_replace(in_f, out_f, pattern_repl_pairs)

def convert_to_0based_i(in_f, out_f):

    full_line_pattern = re.compile(
        r"for \(\s*(int\s+)?ip1 = idx_range\.i_start \+ 1; ip1\<\=\(idx_range\.i_end \+ 1\); ip1\+\+\)\s*\{"
    )
    _nominal_full_line_repl = (
        'for (int i = idx_range.i_start; i < idx_range.i_stop; i++) {\n'
    )

    def try_replace_full_line(line):
        m = full_line_pattern.match(line.strip())
        if m is None:
            return None
        indent = ' ' * (len(line) - len(line.lstrip())) # <- very crude!
        return indent + _nominal_full_line_repl


    # we are using negative lookahead and negative lookbehind to make sure that
    # `ip1` is not part of a variable and `1` is not part of a literal
    pattern_repl_pairs = [
        (r"(?<!\w)ip1\s*\-\s*1(?!\w)", "i")
    ]
    _, modified_line_count = _find_and_replace(
        in_f, out_f, pattern_repl_pairs, [try_replace_full_line]
    )

    # here we prompt some manual action
    if modified_line_count == 0:
        print("Did you remember to complete the first step?")
    else:
        # get the number of lines that still have the ip1 variable
        cur_offset = out_f.tell()
        out_f.seek(0)
        _, lines_with_ip1 = _find_and_replace(
            out_f, None, pattern_repl_pairs=[(r"(?<!\w)ip1(?!\w)", "1p1")],
            full_line_replacer_funcs=[], announce_changes=False, dryrun=True
        )
        out_f.seek(cur_offset)
        if lines_with_ip1 == 0:
            print("There are no remaining occurences of ip1")
        else:
            print(
                f"There are {lines_with_ip1} lines with ip1. You need to "
                "manually clean these up!"
            )


_CHOICES = {'replace-i-with-ip1' : replace_i_with_ip1,
            'convert-to-0based-i' : convert_to_0based_i}
parser = argparse.ArgumentParser(
    prog='index_replacer',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=f"""\
Tool to help replace occurences of `i-1`.

This assumes that we are using a variable called idx_range, which is an of the
instance of the `IndexHelper` type. This tool knows nothing about strings or
comments (you may need to fix that yourself)

This must be invoked twice to perform {len(_CHOICES)} separate actions. See the
end of the help message for more details about invoking them.""",
    epilog=f"""\
The `{list(_CHOICES)[0]}` action must be performed first. This is a simplistic
search-and-replace step that replaces every occurence of the `i` variable
in a C/C++ file with `ip1` (this stands for "i plus 1"). You should make sure
everything compiles when you do this and that all tests pass (you may want to
make this an isolated commit).

The `{list(_CHOICES)[1]}` action is performed 2nd. It replaces the for-loops 
over `ip1` with a for-loop over `i` (where `i` now uses 0-based indexing). It
also replaces all occurences of `ip1-1` with `i`.
"""
)

parser.add_argument("action", choices = _CHOICES, help="specify an action")
parser.add_argument("--input", required=True, help="the input file")
exclusive_group = parser.add_mutually_exclusive_group(required=True)
exclusive_group.add_argument(
    "--inplace",
    const=None,
    dest="output",
    action="store_const",
    help="modify the input file inplace"
)
exclusive_group.add_argument("--output", help="output file")

def main(args):
    fn = _CHOICES[args.action]
    with tempfile.NamedTemporaryFile(mode='w+') as tmp_outfile:
        with open(args.input, 'r') as infile:
            fn(infile, tmp_outfile)
        tmp_outfile.flush()
        dst = args.input if args.output is None else args.output
        shutil.copyfile(src=tmp_outfile.name, dst=dst)

if __name__ == '__main__':
    main(parser.parse_args())

