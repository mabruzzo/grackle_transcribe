import more_itertools

from dataclasses import dataclass
from functools import partialmethod
from itertools import cycle
import os
import re
import textwrap


from .clike_parse.tool import get_C_Struct_transcribe_info
from .cpp_identifier_model import (
    IdentifierUsage, _IdentifierModel, _TYPE_MAP,
    ArrInitSpec, get_translated_declaration_lines
)
from .identifiers import Constant
from .parser import Parser, TokenStream
from .routine_analysis import analyze_routine
from .src_model import (
    SrcItem,
    WhitespaceLines,
    Code,
    Comment,
    PreprocKind,
    PreprocessorDirective,
    OMPDirectiveKind,
    OMPDirective,
    LineProvider,
    get_source_regions
)
from .stringify import FormattedCodeEntryBuilder, concat_translated_pairs
from .subroutine_entity import (
    Declaration, build_subroutine_entity, ControlConstruct, SubroutineEntity
)
from .syntax_unit import (
     ArrayAccess, IdentifierExpr, LiteralExpr, Standard1TokenStmt, Stmt,
     UncategorizedStmt, _iterate_tokens, ControlConstructKind, CallStmt
)
from .translator import (
    _translate_stmt, _get_translated_label, _translate_expr
)
from .token import Type, Keyword, Misc, token_has_type
from .utils import index_non_space, indented_fmt
from .writer import (
    EntryVisitor,
    _Consumer,
    _passthrough_SrcItem
)

def _fmt_function(routine_name, indent, arg_l,
                  wrapped_by_fortran_name,
                  compressed_args = True,
                  is_function_call = False):
    if not compressed_args:
        arg_list_str = indent + f',\n {indent}'.join(arg_l)
    else:
        arg_list_str = '\n'.join(indented_fmt(arg_l, indent = indent))

    if wrapped_by_fortran_name:
        routine_name = f'FORTRAN_NAME({routine_name})'
    else:
        routine_name = routine_name

    if is_function_call:
        prefix, suffix = f"{routine_name}(", ");"
    else:
        prefix, suffix = f"void {routine_name}(", ")"
    return '\n'.join([prefix, arg_list_str, suffix])


# the following function probably does a little too much (maybe we should
# refactor in the future?)
def c_like_fn_signature(subroutine, identifier_model = None,
                        wrapped_by_fortran_name = False,
                        c_struct_transcribe_info = None):
    """
    Come up with a C-like function signature for a given subroutine

    Parameters
    ----------
    subroutine : SubroutineSignature or SubroutineEntity
        Holds baseline information about the parsed Fortran routine
    identifier_model : optional
        Holds information mapping the Fortran variables to C++ variables. When
        specified, this may slightly change argument names
    wrapped_by_fortran_name: bool, optional
        Whether the function name should be wrapped by the FORTRAN_NAME macro
        to achieve proper name-mangling
    c_struct_transcribe_info: CStructTranscribeInfo, optional
        This can be the object returned by
        `clike_parse.tool.get_C_Struct_transcribe_info`,
        which specifies information about what structs to be used when
        transcribing a function signature. When specified, we try to reduce
        the number of arguments passed into this function and pass in structs
        instead.

    Returns
    -------
    signature: str
        The translated function signature
    local_fn_call: str or None
        This is None unless c_struct_transcribe_info is provided. In this case,
        this should be the new local function call.
    n_args: int
        The number of arguments
    """

    def _arglist_entry(arg):
        if identifier_model is None:
            modify = arg.prop.is_array and (arg.prop.rank > 1)
            arg_name = f"{arg.name}_data_ptr" if modify else arg.name
        else:
            arg_name = identifier_model.cpp_arglist_identifier(arg.name)
        return f"{_TYPE_MAP[arg.type]}* {arg_name}"

    if isinstance(subroutine, SubroutineEntity):
        subroutine = subroutine.subroutine_signature()

    indent = '  '

    arg_list = []
    if c_struct_transcribe_info is None:
        for i, arg in enumerate(subroutine.arguments_iter):
            arg_list.append(_arglist_entry(arg))
        local_args = None
    else:
        assert not wrapped_by_fortran_name
        local_args = []
        n_args = len(subroutine.arguments)
        assert n_args == len(c_struct_transcribe_info.orig_arg_l)
        # step build up arg_list & local_args using the arguments directly
        # taken from the Fortran subroutine (skipping over all arguments that
        # can be accessed from structs)
        for arg, local_arg in zip(subroutine.arguments,
                                  c_struct_transcribe_info.orig_arg_l):
            if isinstance(local_arg, str): # preserve the argument
                arg_list.append(_arglist_entry(arg))
                local_args.append(local_arg)
            else:
                pass # we skip over the argument, since we choose to pass in
                     # the struct
        # add any structs as arguments
        l = sorted(
            c_struct_transcribe_info.all_structvar_useptr_pairs(),
            key=lambda e: e[0].type.value
        )
        for struct_var, use_ptr in l:
            tmp = "*" if use_ptr else ""
            arg_list.append(
                f'{struct_var.type.full_type_name}{tmp} {struct_var.var_name}'
            )
            if struct_var.is_ptr == use_ptr:
                local_args.append(struct_var.var_name)
            elif struct_var.is_ptr:
                local_args.append(f"*{struct_var.var_name}")
            else:
                local_args.append(f'&{struct_var.var_name}')

    signature = _fmt_function(
        subroutine.name, indent=indent, arg_l=arg_list,
        wrapped_by_fortran_name = wrapped_by_fortran_name,
        compressed_args = True
    )
    if local_args is None:
        local_fn_call = None
    else:
        local_fn_call = _fmt_function(
            subroutine.name, indent=indent, arg_l=local_args,
            wrapped_by_fortran_name=wrapped_by_fortran_name,
            compressed_args=True, is_function_call=True
        )
    return signature, local_fn_call, len(arg_list)

def _get_constant_decl_init(decl, line, identifier_model):
    # This function cuts a lot of corners. To do this more properly, we would
    # probably need to do a lot of refactoring
    # - I think a bunch of this logic is essentially duplicated (we probably
    #   want to move this inside of a Parser method for the sake of
    #   consistency)

    # decl is a Declaration object
    # - in some cases, entry may consist of conditional branches that are
    #   used to define a constant in different cases
    # - line will always correspond to a Code instance
    assert isinstance(decl, Declaration)
    assert isinstance(line, Code)
    assert len(decl.identifiers) == 1
    assert isinstance(decl.identifiers[0], Constant)

    parser = Parser(identifier_model.fortran_identifier_spec)
    tok_stream = TokenStream(line)
    assert token_has_type(next(tok_stream), Type)
    assert next(tok_stream).string == ','
    assert token_has_type(next(tok_stream), Keyword.PARAMETER)
    assert next(tok_stream).string == '::'
    identifier_tok = next(tok_stream)
    assert token_has_type(identifier_tok, 'arbitrary-name')
    assert identifier_tok.string.lower() == decl.identifiers[0].name.lower()
    assert next(tok_stream).string == '='

    var_name = identifier_model.cpp_variable_name(
        identifier_tok.string, IdentifierUsage.ScalarValue
    )
    expr = parser._parse_single_expr(tok_stream)
    translation_pairs = _translate_expr(expr, identifier_model)
    assignment_rhs = concat_translated_pairs(translation_pairs)

    ctype = _TYPE_MAP[decl.identifiers[0].type]
    return f'const {ctype} {var_name} = {assignment_rhs};'

_TODO_PREFIX = '//_//'

class CppTranslator(EntryVisitor):

    def __init__(self, consume_callback, identifier_model):
        super().__init__(ignore_unknown=False)
        self.consumer = _Consumer(consume_callback)
        def unaddressed_consume(arg):
            consume_callback(f"  {_TODO_PREFIX} PORT: {arg}")
        self.unaddressed_consumer = _Consumer(unaddressed_consume)
        self.identifier_model = identifier_model
        self.block_level=1

    def _write_translated_line(self, line):
        indent = '  ' * self.block_level
        self.consumer.consume(f'{indent}{line}')

    def _passthrough_SrcItem(self, entry, unaddressed = True):
        if unaddressed:
            _passthrough_SrcItem(self.unaddressed_consumer, entry)
        else:
            _passthrough_SrcItem(self.consumer, entry)

    def _cpp_variable_name(self, *args, **kwargs):
        return self.identifier_model.cpp_variable_name(*args, **kwargs)

    def visit_WhitespaceLines(self, entry):
        self._passthrough_SrcItem(entry, unaddressed = False)

    def visit_Comment(self, entry):

        if len(entry.lines) == 1:
            stripped_line = entry.lines[0].strip()

            self._write_translated_line('// ' + (stripped_line[1:]).strip())
        else:
            prefix = cycle(['//', '//-'])

            def _write(prefix, lines):
                lines = textwrap.indent(
                    textwrap.dedent('\n'.join(lines)), f'{prefix} '
                ).splitlines()
                for line in lines:
                    self._write_translated_line(line.rstrip())

            itr = iter(entry.lines)
            chunk = []
            escape_prefix = None # holds the prefix up through the comment char
            for line in itr:
                cur_escape_prefix_stop = 1+index_non_space(line)
                clipped_line = line[cur_escape_prefix_stop:]
                if escape_prefix == line[:cur_escape_prefix_stop]:
                    chunk.append(clipped_line)
                else:
                    if len(chunk) > 0:
                        _write(next(prefix), chunk)
                    chunk = [clipped_line]
                    escape_prefix = line[:cur_escape_prefix_stop]
            if len(chunk) > 0:
                _write(next(prefix), chunk)

    def visit_PreprocessorDirective(self, entry):
        if entry.kind in [
            PreprocKind.IFDEF, PreprocKind.ELSE, PreprocKind.ENDIF, PreprocKind.DEFINE
        ]:
            self._passthrough_SrcItem(entry, unaddressed = False)
        elif entry.kind == PreprocKind.INCLUDE_grackle_fortran_types:
            return None
        else:
            self._passthrough_SrcItem(entry)

    def visit_OMPDirective(self, entry):
        if entry.kind is OMPDirectiveKind.INCLUDE_OMP:
            pass # don't write anything!
        elif entry.kind is OMPDirectiveKind.END_PARALLEL_DO:
            self.block_level-=1
            self._write_translated_line('}  // OMP_PRAGMA("omp parallel")')

        elif entry.kind is OMPDirectiveKind.CRITICAL:
            self._write_translated_line('OMP_PRAGMA_CRITICAL')
            self._write_translated_line('{')
            self.block_level+=1
        elif entry.kind is OMPDirectiveKind.END_CRITICAL:
            self.block_level-=1
            self._write_translated_line('}')
        else:
            parallel_do = entry.kind is OMPDirectiveKind.PARALLEL_DO
            if not parallel_do:
                parallel_do = 'parallel do' in '\n'.join(entry.lines)

            if parallel_do:
                # we aren't going to try to automatically handle the private
                # clause, but incrementing the block level essentially provides
                # room for us to manually create copies that were previously 
                # in the private clause
                self._passthrough_SrcItem(entry)

                self._write_translated_line(
                    f'{_TODO_PREFIX} TODO_USE: OMP_PRAGMA("omp parallel")'
                )
                self._write_translated_line('{')
                self.block_level+=1
                self._write_translated_line(
                    f'{_TODO_PREFIX} TODO: move relevant variable declarations to '
                    'here to replace OMP private'
                )
                self._write_translated_line(
                    f'{_TODO_PREFIX} TODO_USE: OMP_PRAGMA("omp for")'
                )
            else:
                self._passthrough_SrcItem(entry)






    def visit_Declaration(self, entry):
        identifier_spec = self.identifier_model.fortran_identifier_spec
        if isinstance(entry.src, (list, tuple)):
            # this is the weird conditional parameter scenario
            for elem in entry.src:
                if isinstance(elem, Code):
                    translated_line = _get_constant_decl_init(
                        decl=entry,
                        line=elem,
                        identifier_model=self.identifier_model
                    )
                    self._write_translated_line(translated_line)
                else:
                    self.dispatch_visit(elem)
            return None
        elif isinstance(entry.src, PreprocessorDirective):
            raise RuntimeError()
        elif (
            (len(entry.identifiers) == 1) and
            isinstance(entry.identifiers[0], Constant)
        ):
            translated_line = _get_constant_decl_init(
                decl=entry,
                line=entry.src,
                identifier_model=self.identifier_model
            )
            self._write_translated_line(translated_line)
            return None


        def _handle_arr(identifier, is_arg):
            arrspec = identifier.array_spec
            if arrspec.allocatable: assert not is_arg

            def _expr_tostring(expr):
                if isinstance(expr, LiteralExpr):
                    return expr.token.string
                elif isinstance(expr, IdentifierExpr):
                    return self._cpp_variable_name(
                        expr, IdentifierUsage.ScalarValue
                    )
                return None

            axlens = []
            problem_case = None
            for elem in arrspec.axlens:
                if arrspec.allocatable:
                    axlens.append(None)
                elif (axlen := _expr_tostring(elem)) is not None:
                    axlens.append(axlen)
                elif isinstance(elem, ArrayAccess):
                    # in this case, the current axis length is stored inside
                    # of a separate array. 
                    inner_idx_l = list(elem.arg_l.get_args())
                    inner_idx_str = _expr_tostring(inner_idx_l[0])
                    #print(
                    #    f"HANDLING ARRAY DECLARATION for {identifier.name}\n"
                    #    " -> current axlen is an array access\n"
                    #    f" -> number of indices: {len(inner_idx_l)}\n"
                    #    f" -> translated index: {inner_idx_str!r}"
                    #)

                    if (len(inner_idx_l) != 1) or (inner_idx_str is None):
                        problem_case = elem
                        break
                    inner_array_name = self._cpp_variable_name(
                        elem.array_name, IdentifierUsage.ArrAccessValue
                    )
                    axlens.append(f"{inner_array_name}[{inner_idx_str}-1]")
                else:
                    problem_case = elem
            if problem_case is not None:
                raise RuntimeError(
                    "not currently equipped to translate the length of an "
                    "array-shape in the declaration of {identifier.name}:\n"
                    f" -> axlen has type {problem_case.__class__.__name__}\n"
                    " -> The full representation of the identifier is"
                    f"{identifier}"
                )
            return ArrInitSpec(identifier.name,tuple(axlens))

        scalar_decls = []
        declared_arrinitspec_l = []
        for identifier in entry.identifiers:
            rank = getattr(getattr(identifier,'array_spec',None), 'rank', None)
            is_arg = identifier_spec.is_arg(identifier.name)
            if (rank is not None) and ((not is_arg) or rank > 1):
                declared_arrinitspec_l.append(_handle_arr(identifier, is_arg))
            elif is_arg:
                continue
            else:
                scalar_decls.append(identifier.name)

        if (scalar_decls == [] and declared_arrinitspec_l == []):
            self._write_translated_line(
                '// -- removed line (previously just declared arg types) -- '
            )
        else:
            itr = get_translated_declaration_lines(
                entry.identifiers[0].type,
                scalar_decls,
                declared_arrinitspec_l,
                self.identifier_model
            )
            for line in itr:
                self._write_translated_line(line)
        return None

    def _visit_Stmt(self, entry):
        # I have no idea what this will ultimately look like, but we need to
        # start somewhere!
        translation_rslt, append_semicolon = _translate_stmt(
            entry, self.identifier_model
        )

        if isinstance(translation_rslt, str):
            # we would have already appended the `;` if we wanted it
            assert not append_semicolon
        return translation_rslt, append_semicolon
                    
    def visit_Stmt(self, entry, prepend_brace=False, append_brace=False):

        # here is the idea: we should explicitly try to:
        # -> write the label
        # -> use spacing from the tokens
        # -> make sure to print trailing comments


        try:
            if entry.src.has_label:
                assert not prepend_brace
                assert not append_brace
                if len(entry.src.tokens) != 2:
                    raise NotImplementedError()
                elif entry.src.tokens[1].type != Keyword.CONTINUE:
                    raise NotImplementedError()
                label_str = _get_translated_label(entry.src.tokens[0])
                self.consumer.consume(f'{label_str}:')
                return
            elif (
                isinstance(entry, UncategorizedStmt) and
                (len(entry.src.tokens) == 1) and
                token_has_type(entry.src.tokens[0], Misc.ImplicitNone)
            ):
                # we totally omit this line!
                return

            translation_rslt, append_semicolon = self._visit_Stmt(entry)

            if (prepend_brace or append_brace) and append_semicolon:
                raise RuntimeError("SOMETHING IS VERY WRONG!")
            elif isinstance(translation_rslt, str):
                if prepend_brace:
                    translation_rslt =  "} " + translation_rslt
                if append_brace:
                    translation_rslt += " {"
                for line in translation_rslt.splitlines():
                    self._write_translated_line(line)
                return True
            else:
                builder = FormattedCodeEntryBuilder(
                    entry.src,maintain_indent=False
                )
                for pair in translation_rslt:
                    builder.put(*pair)
                if append_semicolon:
                    builder.append_semicolon()
                chunk_l = builder.build(trailing_comment_delim="//")
                if prepend_brace:
                    chunk_l[0] = "} " + chunk_l[0]
                final_trailing_comment = (
                    entry.src.trailing_comment_start[-1] is not None
                )
                if append_brace and not final_trailing_comment:
                    chunk_l[-1] += " {"

                chunk_l[0] = chunk_l[0] # crude workaround
                for chunk in chunk_l:
                    if isinstance(chunk, str):
                        self._write_translated_line(chunk)
                    else:
                        print(chunk)
                        self.dispatch_visit(chunk)
                if append_brace and final_trailing_comment:
                    self._write_translated_line('{')

        except NotImplementedError:
            if prepend_brace:
                self._write_translated_line('}')
            self._passthrough_SrcItem(entry.item)
            if append_brace:
                self._write_translated_line('{')

        

    def visit_ControlConstruct(self, entry):
        non_meta_construct = (
            isinstance(entry.kind, ControlConstructKind) and
            entry.kind != ControlConstructKind.MetaIfElse
        )

        itr = enumerate(entry.condition_contents_pairs)
        for i, (condition, contents) in itr:
            if non_meta_construct:
                if (i > 0):
                    self.block_level-=1
                self.visit_Stmt(
                    condition, prepend_brace=(i > 0), append_brace=True
                )
                self.block_level+=1
            else:
                self.dispatch_visit(condition)

            for content_entry in contents:
                self.dispatch_visit(content_entry)

        if non_meta_construct:
            self.block_level-=1
            self._write_translated_line('}')
            # we explicitly skip the endif/enddo statement in entry.end
        else:
            self.dispatch_visit(entry.end)

def _header_guard_name(fname):
    basename = os.path.basename(fname)
    return re.sub(r"[-\.\s]", "_", "my_file-cpp.h").upper()

def _common_prolog_text(fname, fn_name):
    return f"""\
//===----------------------------------------------------------------------===//
//
// See the LICENSE file for license and copyright information
// SPDX-License-Identifier: NCSA AND BSD-3-Clause
//
//===----------------------------------------------------------------------===//
///
/// @file
/// Declares signature of {fn_name}
///
//===----------------------------------------------------------------------===//

// This file was initially generated automatically during conversion of the
// {fn_name} function from FORTRAN to C++

"""

_IMPLEMENTATION_HEADERS = """\
#include <cstdio>
#include <vector>

#include "grackle.h"
#include "fortran_func_decls.h"
#include "utils-cpp.hpp"
"""

def _is_header_fname(fname):
    _, ext = os.path.splitext(fname)
    return ext == '.h' or ext == '.hpp'

@dataclass(frozen=True)
class BoilerPlateWriter:
    """
    Helps write boilerplate prologs and epilogs to a C++ file source file

    (This is only a dataclass for mutability purposes)
    """
    prolog_parts: list[str]
    epilog_parts: list[str]

    def _write(self, f, write_prolog):
        parts = self.prolog_parts if write_prolog else self.epilog_parts
        for part in parts:
            f.write(part)

    write_prolog = partialmethod(_write, write_prolog=True)
    write_epilog = partialmethod(_write, write_prolog=False)

    @classmethod
    def implementation_file(cls, fname:str, use_C_linkage: bool, fn_name: str,
                            extern_header_fname = None):
        inline_header = _is_header_fname(fname)
        if not inline_header:
            assert os.path.splitext(fname)[1] in ['.C', '.cpp']

        prolog, epilog = [], []
        prolog.append(_common_prolog_text(fname, fn_name))

        if inline_header:
            header_guard_name = _header_guard_name(fname)
            assert not use_C_linkage
            assert extern_header_fname is None
            prolog.append(f"""\
#ifndef {header_guard_name}
#define {header_guard_name}

{_IMPLEMENTATION_HEADERS}""")

            epilog.append(f"""
#endif /* {self.header_guard_name} */
""")
        else:
            prolog.append(_IMPLEMENTATION_HEADERS)
            if extern_header_fname is not None:
                prolog.append(f'\n#include "{extern_header_fname}"\n\n')

            if use_C_linkage:
                prolog.append("""\
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

""")
                epilog.append("""
#ifdef __cplusplus
}  // extern "C"
#endif /* __cplusplus */
""")

        return cls(prolog_parts = prolog, epilog_parts = epilog)

    @classmethod
    def declaration_hdr(cls, fname: str, use_C_linkage: bool, fn_name: str):
        header_guard_name = _header_guard_name(fname)
        prolog, epilog = [], []
        prolog.append(_common_prolog_text(fname, fn_name))
        prolog.append(f"""\
#ifndef {header_guard_name}
#define {header_guard_name}

#include "grackle.h"             // gr_float
#include "fortran_func_decls.h"  // gr_mask_int

""")
        if use_C_linkage:
            prolog.append("""\
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
// the following function can be called from C or C++

""")
            epilog.append("""
#ifdef __cplusplus
}  // extern "C"
#endif /* __cplusplus */
""")
        epilog.append(f"""
#endif /* {header_guard_name} */
""")
        return cls(prolog_parts = prolog, epilog_parts = epilog)

def _write_declaration_header(fname, signature, use_C_linkage, fn_name):
    writer = BoilerPlateWriter.declaration_hdr(
        fname=fname, use_C_linkage=use_C_linkage, fn_name = fn_name
    )
    with open(fname, 'w') as f:
        writer.write_prolog(f)
        f.write(signature)
        f.write(';\n')
        writer.write_epilog(f)

def transcribe(in_fname, out_f, extern_header_fname = None,
               use_C_linkage = True, fncall_inspect_conf=None,
               signature_registry = None):
    """
    This does the heavy lifting of transcribing the first routine in
    ``in_fname`` and writing it to ``out_f``.

    If extern_header_fname is provided, we will also write a header file that
    can be used to declare this functionality.
    """
    out_fname = out_f.name
    def writer(arg):
        out_f.write(f'{arg}\n')

    with open(in_fname, "r") as in_f:
        provider = LineProvider(in_f, fname = in_fname)
        it = get_source_regions(provider)

        for region in it:
            if not region.is_routine:
                continue
            subroutine = build_subroutine_entity(region, it.prologue)
            props = analyze_routine(subroutine)

            if fncall_inspect_conf is None:
                c_struct_transcribe_info = None
            else:
                print("Parsing subroutine call-site from C/C++ routine")
                c_struct_transcribe_info = get_C_Struct_transcribe_info(
                    fn_call_loc=fncall_inspect_conf.fn_call_loc,
                    fn_name=subroutine.name.lower(),
                    local_struct_vars=fncall_inspect_conf.local_struct_vars
                )

            identifier_model = _IdentifierModel(
                subroutine.identifiers,
                identifier_analysis_map=props,
                c_struct_transcribe_info=c_struct_transcribe_info
            )

            translator = CppTranslator(
                writer, identifier_model = identifier_model
            )

            signature, local_fn_call_str, n_args = c_like_fn_signature(
                subroutine,
                identifier_model,
                c_struct_transcribe_info=c_struct_transcribe_info
            )

            print()
            print("Translated signature:")
            print(signature)
            print()
            if local_fn_call_str is not None:
                if n_args != len(subroutine.arguments):
                    print(
                        "The number of arguments has changed in the "
                        f"transcription from {len(subroutine.arguments)} to "
                        f"{n_args}"
                    )
                else:
                    print("The args may have changed during transcription")
                print("The function call at the specified location in the C "
                      "file should look like:")
                print(local_fn_call_str)
            #raise RuntimeError("EARLY EXIT")

            if extern_header_fname is not None:
                _write_declaration_header(
                    fname=extern_header_fname,
                    signature=signature,
                    use_C_linkage=use_C_linkage,
                    fn_name=subroutine.name
                )

            boiler_plate_writer = BoilerPlateWriter.implementation_file(
                fname=out_fname,
                use_C_linkage=use_C_linkage,
                fn_name = subroutine.name,
                extern_header_fname = extern_header_fname
            )

            boiler_plate_writer.write_prolog(out_f)
            writer(signature)
            writer("{")

            for entry in subroutine.declaration_section:
                translator.dispatch_visit(entry)
            
            for entry in subroutine.impl_section:
                translator.dispatch_visit(entry)


            writer("}")
            break
        boiler_plate_writer.write_epilog(out_f)
        return



