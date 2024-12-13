# a script that we can use to query information about a section of a fortran
# file

from grackle_transcribe.lineno_search import lineno_search, get_origin
from grackle_transcribe.src_model import LineProvider, get_source_regions
from grackle_transcribe.subroutine_entity import build_subroutine_entity
from grackle_transcribe.utils import add_gracklesrc_opt
from grackle_transcribe.writer import write_extracted_region_as_fn
import more_itertools

import argparse
from itertools import chain

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(required=True)
_cmd_map = {
    'examine' : 'Examine the region corresponding to lineno',
    'extract' : (
        'Extract the region corresponding to lineno into a new routine. '
        'This is highly imperfect, but it gets us most of the way to the '
        'desired result'
    )
}
for cmd, cmd_help in _cmd_map.items():
    subcommand_parser = subparsers.add_parser(cmd, help=cmd_help)
    subcommand_parser.set_defaults(cmd=cmd)
    add_gracklesrc_opt(subcommand_parser, True, required = True)
    subcommand_parser.add_argument(
        "--lineno", type=int, required=True, help = "location being queried"
    )
    if cmd != 'examine':
        subcommand_parser.add_argument(
            "-o", "--output", required=True, help="output file"
        )
        subcommand_parser.add_argument(
            "--routine-name", required=True, help="name of extracted routine"
        )

def _my_itr(f):
    provider = LineProvider(f)
    it = get_source_regions(provider)
    for region in it:
        if region.is_routine:
            yield build_subroutine_entity(region, it.prologue)
        else:
            yield region

def _report_search_rslt(rslt, lineno):
    
    levels = rslt.src_entry_hierarchy
    closest_match = levels[-1]
    if hasattr(closest_match, 'src'):
        lines = closest_match.src.lines
    else:
        lines = closest_match.lines

    subroutine = rslt.try_get_subroutine()
    if subroutine is None:
        print(f"lineno {lineno} is not found within a subroutine")
        print("the associated chunk of source code is: ")
        print(levels[-1])
        return

    print(f"lineno {lineno} is within the `{subroutine.name}` subroutine")

    print(f"the following statement is on the specified lineno, {lineno}:")
    for line in lines:
        print('>', line, sep='')

    if rslt.is_control_construct_stmt:
        cfc = levels[-2]
        l = [get_origin(c).lineno for c,_ in cfc.condition_contents_pairs]
        l.append(get_origin(cfc.end).lineno)

        print(f"""\
This statement is part of a statement governing a control flow contstruct.
-> the complete list of governing statements can be found on lines:
   {l!r}""")
        print(
            "The control-flow construct ends on lineno "
            f"{levels[-2].end.src.origin.lineno}"
        )

def main(args):

    in_fname = args.grackle_src_file
    lineno = args.lineno

    with open(in_fname, 'r') as f:
        itr = _my_itr(f)
        rslt = lineno_search(itr, lineno)
        _report_search_rslt(rslt, lineno)

        if args.cmd == 'extract':
            write_extracted_region_as_fn(
                rslt, args.routine_name, args.output
            )
        




if __name__ == '__main__':
    main(parser.parse_args())
