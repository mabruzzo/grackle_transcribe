import more_itertools

from itertools import cycle
import textwrap

from .cpp_identifier_model import (
    IdentifierUsage, _IdentifierModel, _TYPE_MAP,
    ArrInitSpec, get_translated_declaration_lines
)
from .identifiers import Constant
from .parser import (
     IdentifierExpr, LiteralExpr, Standard1TokenStmt, Stmt, UncategorizedStmt,
     _iterate_tokens, ControlConstructKind, CallStmt,
     Parser, TokenStream
)
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
    Declaration, build_subroutine_entity, ControlConstruct
)
from .translator import (
    _translate_stmt, _get_translated_label, _translate_expr
)
from .token import Type, Keyword, Misc, token_has_type
from .utils import index_non_space
from .writer import (
    EntryVisitor,
    _Consumer,
    _passthrough_SrcItem
)

def _arglines(arg_list, indent = '  '):

    # we can't use textwrap because we want to ensure that the type
    # declaration and argname are on the same line
    width = 80
    delim = ','
    indent_size, delim_size = len(indent), len(delim)

    cur_buf, cur_buf_size = [], 0
    itr = more_itertools.peekable(arg_list)
    for arg in itr:
        arg_len = len(arg)
        nominal_size = 1 + arg_len + delim_size
        if (len(cur_buf) != 0) and ((cur_buf_size + nominal_size) > width):
            yield ''.join(cur_buf)
            cur_buf, cur_buf_size = [], 0

        if len(cur_buf) == 0:
            cur_buf.append(indent)
            cur_buf_size += indent_size
            latest_chunk, latest_len = [arg], arg_len
        else:
            latest_chunk, latest_len = [' ', arg], 1 + arg_len

        if bool(itr): # not exhausted
            latest_chunk.append(delim)
            latest_len += delim_size
        cur_buf += latest_chunk
        cur_buf_size += latest_len
    if len(cur_buf) > 0:
        yield ''.join(cur_buf)


def c_like_fn_signature(subroutine, identifier_model = None,
                        wrapped_by_fortran_name = False):
    if identifier_model is None:
        def _get_argname(arg):
            modify = (arg.array_spec is not None) and (arg.array_spec.rank > 1)
            return f"{arg.name}_data_ptr" if modify else arg.name
    else:
        def _get_argname(arg):
            return identifier_model.cpp_arglist_identifier(arg.name)

    arg_list = []
    for arg in subroutine.arguments:
        arg_name = _get_argname(arg)
        arg_list.append(f"{_TYPE_MAP[arg.type]}* {arg_name}")


    indent = '  '
    if False:
        arg_list_str = indent + f',\n {indent}'.join(arg_list)
    else:
        arg_list_str = '\n'.join(_arglines(arg_list, indent = indent))

    if wrapped_by_fortran_name:
        routine_name = f'FORTRAN_NAME({subroutine.name})'
    else:
        routine_name = subroutine.name

    rslt = '\n'.join([f"void {routine_name}(", arg_list_str, ')'])
    return rslt


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
        elif entry.kind is OMPDirectiveKind.CRITICAL:
            self._write_translated_line('OMP_PRAGMA_CRITICAL')
            self._write_translated_line('{')
            self.block_level+=1
        elif entry.kind is OMPDirectiveKind.END_CRITICAL:
            self.block_level-=1
            self._write_translated_line('}')
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

            axlens = []
            for elem in arrspec.axlens:
                if arrspec.allocatable:
                    axlens.append(None)
                elif isinstance(elem, LiteralExpr):
                    axlens.append(elem.token.string)
                elif isinstance(elem, IdentifierExpr):
                    axlens.append(self._cpp_variable_name(
                        elem, IdentifierUsage.ScalarValue
                    ))
                else:
                    raise RuntimeError("not equipped to handle this case yet")
            return ArrInitSpec(identifier.name,tuple(axlens))

        scalar_decls = []
        declared_arrinitspec_l = []
        for identifier in entry.identifiers:
            rank = getattr(getattr(identifier,'array_spec',None), 'rank', None)
            is_arg = identifier_spec.is_arg(identifier.name)
            if (rank is not None) and rank > 1:
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

def _write_declaration_header(fname, signature, use_C_linkage):
    # writes out a header file whose sole purpose is to provide a declaration
    # that can be included in C files
    with open(fname, 'w') as f:
        f.write("""\
#include "grackle.h"             // gr_float
#include "fortran_func_decls.h"  // gr_mask_int

""")
        if use_C_linkage:
            f.write("""\
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

""")
        f.write(signature)
        f.write(';\n')
        if use_C_linkage:
            f.write("""\
#ifdef __cplusplus
}  // extern "C"
#endif /* __cplusplus */
""")

def transcribe(in_fname, out_f, prolog = None, epilog = None,
               extern_header_fname = None, use_C_linkage = True):
    """
    This does the heavy lifting of transcribing the first routine in
    ``in_fname`` and writing it to ``out_f``.

    If extern_header_fname is provided, we will also write a header file that
    can be used to declare this functionality.
    """

    with open(in_fname, "r") as in_f:
        provider = LineProvider(in_f, fname = in_fname)
        it = get_source_regions(provider)

        if prolog is not None:
            out_f.write(prolog)

        # handle outputs
        def writer(arg):
            out_f.write(f'{arg}\n')

        out_f.write("""\
#include <cstdio>
#include <vector>

#include "grackle.h"
#include "fortran_func_decls.h"
#include "utils-cpp.hpp"
""")
        if extern_header_fname is not None:
            writer(f'#include "{extern_header_fname}"')

        writer("")
        writer("//_// TODO: ADD ANY OTHER INCLUDE DIRECTIVES")
        writer("")

        if use_C_linkage:
            writer('extern "C" {')

        for region in it:
            if not region.is_routine:
                continue
            subroutine = build_subroutine_entity(region, it.prologue)

            # todo: in the future, pass props into _IdentifierModel constructor
            props = analyze_routine(subroutine)
            identifier_model = _IdentifierModel(
                subroutine.identifiers,
                #identifier_analysis_map=props
            )

            translator = CppTranslator(
                writer, identifier_model = identifier_model
            )

            signature = c_like_fn_signature(subroutine, identifier_model)
            if extern_header_fname is not None:
                _write_declaration_header(
                    extern_header_fname, signature, use_C_linkage
                )


            writer(signature)
            writer("{")

            for entry in subroutine.declaration_section:
                translator.dispatch_visit(entry)
            
            for entry in subroutine.impl_section:
                translator.dispatch_visit(entry)


            writer("}")
            break
        if use_C_linkage:
            writer("")
            writer('}  // extern "C"')

        if epilog is not None:
            writer("")
            out_f.write(epilog)
        return



