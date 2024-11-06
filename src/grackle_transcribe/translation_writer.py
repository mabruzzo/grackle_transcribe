from itertools import cycle
import textwrap

from .identifiers import Constant
from .parser import (
     IdentifierExpr, LiteralExpr, Standard1TokenStmt, Stmt, UncategorizedStmt,
     _iterate_tokens, ControlConstructKind, CallStmt
)
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
from .stringify import FormattedCodeEntryBuilder
from .subroutine_entity import (
    Declaration,
    build_subroutine_entity,
    ControlConstruct
)
from .translator import _translate_stmt, _IdentifierModel, IdentifierUsage, _TYPE_MAP
from .token import Type
from .utils import index_non_space
from .writer import (
    EntryVisitor,
    _Consumer,
    _passthrough_SrcItem
)





def c_like_fn_signature(subroutine):
    arg_list = []
    for arg in subroutine.arguments:
        if (arg.array_spec is not None) and (arg.array_spec.rank > 1):
            arg_name = f"{arg.name}_data_ptr"
        else:
            arg_name = arg.name
        arg_list.append(f"  {_TYPE_MAP[arg.type]}* {arg_name}")
    rslt = f"void {subroutine.name}(\n" + ',\n'.join(arg_list) + '\n)'
    return rslt

_TODO_PREFIX = '//_//'

class CppTranslator(EntryVisitor):

    def __init__(self, consume_callback, identifier_spec):
        super().__init__(ignore_unknown=False)
        self.consumer = _Consumer(consume_callback)
        def unaddressed_consume(arg):
            consume_callback(f"  {_TODO_PREFIX} PORT: {arg}")
        self.unaddressed_consumer = _Consumer(unaddressed_consume)
        self.identifier_model = _IdentifierModel(identifier_spec)
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
        if isinstance(entry.src, (list, tuple)):
            # this is the weird conditional parameter scenario
            for elem in entry.src:
                self._passthrough_SrcItem(elem)
            return None
        elif isinstance(entry.src, PreprocessorDirective):
            self._passthrough_SrcItem(elem.src)
            return None

        def _handle_arr(identifier, is_arg):
            arrspec = identifier.array_spec
            ctype = _TYPE_MAP[identifier.type]
            vec_type = f'std::vector<{ctype}>'
            view_type = f'grackle::View<{ctype}{"*"*arrspec.rank}>'

            primary_name = identifier.name
            if arrspec.rank == 1:
                _data_name = primary_name
            elif is_arg:
                _data_name = f'{identifier.name}_data_ptr'
            else:
                _data_name = f'{identifier.name}_data_vec_'

            if identifier.array_spec.allocatable:
                assert not is_arg
                out = [f'{vec_type} {_data_name};']
                if arrspec.rank > 1:
                    out.append(f'{view_type} {primary_name};')
                return out

            axlens = []
            for elem in arrspec.axlens:
                if isinstance(elem, LiteralExpr):
                    axlens.append(elem.token.string)
                elif isinstance(elem, IdentifierExpr):
                    axlens.append(
                        self._cpp_variable_name(elem, IdentifierUsage.ScalarValue)
                    )
                else:
                    raise RuntimeError("not equipped to handle this case yet")

            assert len(axlens) == arrspec.rank
            vec_size = '*'.join(axlens)

            out = [] if is_arg else [f'{vec_type} {_data_name}({vec_size});']

            if arrspec.rank == 1:
                assert not is_arg
            else:
                ptr = _data_name if is_arg else f'{_data_name}.data()'
                out.append(
                    f'{view_type} {primary_name}({ptr}, ' +
                    ','.join(axlens) +
                    ');'
                )
            return out

        identifier_spec = self.identifier_model.fortran_identifier_spec
        scalar_decls = []
        array_decl_lines = []
        for identifier in entry.identifiers:
            if isinstance(identifier, Constant):
                self._passthrough_SrcItem(entry.src)
                return None
            elif identifier_spec.is_arg(identifier.name):
                if identifier.array_spec is None:
                    continue
                elif identifier.array_spec.rank == 1:
                    continue
                else:
                    array_decl_lines += _handle_arr(identifier, True)
            elif identifier.array_spec is None:
                scalar_decls.append(identifier)
            else:
                array_decl_lines += _handle_arr(identifier, False)

        if (scalar_decls == [] and array_decl_lines == []):
            self._write_translated_line(
                '// -- removed line (previously just declared arg types) -- '
            )
            return None


        ctype = _TYPE_MAP[entry.identifiers[0].type]
        if len(scalar_decls) > 0:
            self._write_translated_line(
                f"{ctype} {', '.join(v.name for v in scalar_decls)};"
            )

        for line in array_decl_lines:
            self._write_translated_line(line)
        return None

    def _visit_Stmt(self, entry):

        # I have no idea what this will ultimately look like, but we need to
        # start somewhere!
        if entry.src.has_label:
            # I think all of these should be Keyword.CONTINUE lines that we
            # will translate here
            raise NotImplementedError()
        else:

            try:
                translation_rslt, append_semicolon = _translate_stmt(
                    entry, self.identifier_model
                )

            except NotImplementedError:
                raise
            else:
                if isinstance(translation_rslt, str):
                    # we would have already appended the `;` if we wanted it
                    assert not append_semicolon
                    if len(entry.src.entries) != 1:
                        raise NotImplementedError()
                return translation_rslt, append_semicolon
                    
    def visit_Stmt(self, entry, prepend_brace=False, append_brace=False):

        # here is the idea: we should explicitly try to:
        # -> write the label
        # -> use spacing from the tokens
        # -> make sure to print trailing comments

        print_vals = [entry.__class__.__name__]
        if isinstance(entry, Standard1TokenStmt):
            print_vals.append(entry.tok_type())
        elif isinstance(entry, UncategorizedStmt):
            print_vals.append(entry.src.lines)

        try:
            translation_rslt, append_semicolon = self._visit_Stmt(entry)

            if (prepend_brace or append_brace) and append_semicolon:
                raise RuntimeError("SOMETHING IS VERY WRONG!")
            elif isinstance(translation_rslt, str):
                if prepend_brace:
                    translation_rslt =  "} " + translation_rslt
                if append_brace:
                    translation_rslt += " {"
                self._write_translated_line(translation_rslt)
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


def transcribe(in_fname, out_f):

    temp_fname = './__my_dummy_file'
    with open(in_fname, "r") as in_f:
        provider = LineProvider(in_f, fname = in_fname)
        it = get_source_regions(provider)

        # handle outputs
        def writer(arg):
            out_f.write(f'{arg}\n')

        writer("//_// TODO: ADD INCLUDE DIRECTIVES")
        writer("")


        for region in it:
            if not region.is_routine:
                continue
            subroutine = build_subroutine_entity(region, it.prologue)


            translator = CppTranslator(
                writer, identifier_spec = subroutine.identifiers
            )

            #translator.dispatch_visit(subroutine.subroutine_stmt)
            writer(c_like_fn_signature(subroutine))
            writer("{")

            for entry in subroutine.declaration_section:
                translator.dispatch_visit(entry)
            
            writer("")
            writer("  //_// TODO: TRANSLATE IMPLEMENTATION")
            writer("")

            for entry in subroutine.impl_section:
                translator.dispatch_visit(entry)


            writer("}")
            return



