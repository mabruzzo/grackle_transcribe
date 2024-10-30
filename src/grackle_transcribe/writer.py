# we are loosely inspired by sphinx and docutils
from contextlib import ExitStack
import re
import shutil

from .subroutine_entity import (
    Declaration,
    Stmt,
    ControlConstruct,
    build_subroutine_entity,
    Constant
)
from .subroutine_entity import (
    IdentifierExpr, LiteralExpr
)

from .src_model import (
    SrcItem,
    WhitespaceLines,
    Code,
    Comment,
    PreprocessorDirective,
    OMPDirective,
    LineProvider,
    get_source_regions
)


from .f_chunk_parse import Type, ChunkKind, Literal, _NAME_REGEX

# an entry is one of the following types:
# -> we explicitly forbid it from being a Code instance (although other types
#    can contain Code instances)
_ENTRY_TYPES = (
    Declaration, Stmt, ControlConstruct,
    WhitespaceLines, Comment, PreprocessorDirective, OMPDirective
)


class EntryVisitor:
    def __init__(self, ignore_unknown = False):
        self._ignore_unknown = ignore_unknown

    def unknown_visit(self, entry):
        if not self._ignore_unknown:
            raise RuntimeError(
                f"no implementation for visiting {entry.__class__.__name__!r}"
            )

    def unknown_depart(self, entry):
        if not self._ignore_unknown:
            raise RuntimeError(
                f"no implementation for departing {entry.__class__.__name__!r}"
            )

    def dispatch_visit(self, entry):
        # we should probably remove this isinstance check since it kinda defeats
        # the purpose of the visitor pattern
        assert isinstance(entry, _ENTRY_TYPES)
        name = entry.__class__.__name__
        if isinstance(entry, Stmt):
            name = 'Stmt'
        method = getattr(self, f'visit_{name}', self.unknown_visit)
        return method(entry)

    def dispatch_depart(self, entry):
        # we should probably remove this isinstance check since it kinda defeats
        # the purpose of the visitor pattern
        assert isinstance(entry, _ENTRY_TYPES)
        name = entry.__class__.__name__
        if isinstance(entry, Stmt):
            name = 'Stmt'
        method = getattr(self, f'depart_{name}', self.unknown_depart)
        return method(entry)


class _Consumer:
    def __init__(self, fn):
        self.fn = fn
    def consume(self, arg):
        fn = self.fn
        fn(arg)

def _passthrough_SrcItem(consumer,entry):
    if not isinstance(entry, SrcItem):
        raise TypeError(f"entry can't have type {entry.__class__.__name__}")
    lines = entry.lines
    for line in lines:
        if isinstance(line, str):
            consumer.consume(line)
        elif isinstance(line, SrcItem):
            passthrough_SrcItem(consumer, line)
        else:
            raise RuntimeError()


class UnchangedTranslator(EntryVisitor):

    def __init__(self, consume_callback):
        super().__init__(ignore_unknown=False)
        self.consumer = _Consumer(consume_callback)

    def visit_WhitespaceLines(self, entry):
        _passthrough_SrcItem(self.consumer, entry)

    def visit_Comment(self, entry):
        _passthrough_SrcItem(self.consumer, entry)

    def visit_PreprocessorDirective(self, entry):
        _passthrough_SrcItem(self.consumer, entry)

    def visit_OMPDirective(self, entry):
        _passthrough_SrcItem(self.consumer, entry)

    def visit_Declaration(self, entry):
        if isinstance(entry.src, (list, tuple)):
            # this is the weird conditional parameter scenario
            for elem in entry.src:
                _passthrough_SrcItem(self.consumer,elem)
        else:
            _passthrough_SrcItem(self.consumer, entry.src)

    def visit_Stmt(self, entry):
        _passthrough_SrcItem(self.consumer,entry.item)

    def visit_ControlConstruct(self, entry):
        for (condition, contents) in entry.condition_contents_pairs:
            self.dispatch_visit(condition)
            for content_entry in contents:
                self.dispatch_visit(content_entry)
        self.dispatch_visit(entry.end)

def write_full_copy(in_fname, out_fname):
    with ExitStack() as stack:
        # handle inputs
        in_f = stack.enter_context(open(in_fname, "r"))
        provider = LineProvider(in_f)
        it = get_source_regions(provider)

        # handle outputs
        out_f = stack.enter_context(open(out_fname, "w"))
        def writer(arg):
            out_f.write(f'{arg}\n')
        translator = UnchangedTranslator(writer)

        for region in it:
            if region.is_routine:
                subroutine = build_subroutine_entity(region, it.prologue)
                translator.dispatch_visit(subroutine.subroutine_stmt)
                for entry in subroutine.declaration_section:
                    translator.dispatch_visit(entry)
                for entry in subroutine.impl_section:
                    translator.dispatch_visit(entry)
                translator.dispatch_visit(subroutine.endroutine_stmt)
            else:
                for _, entry in region.lineno_item_pairs:
                    translator.dispatch_visit(entry)


class ReplaceLogicalTranslator(EntryVisitor):

    def __init__(self, consume_callback):
        super().__init__(ignore_unknown=False)
        self.consumer = _Consumer(consume_callback)
        self.logical_varnames = []

    def _passthrough_SrcItem(self, entry):
        #pass # for now, do nothing
        _passthrough_SrcItem(self.consumer, entry)

    def visit_WhitespaceLines(self, entry):
        self._passthrough_SrcItem(entry)

    def visit_Comment(self, entry):
        self._passthrough_SrcItem(entry)

    def visit_PreprocessorDirective(self, entry):
        self._passthrough_SrcItem(entry)

    def visit_OMPDirective(self, entry):
        self._passthrough_SrcItem(entry)

    def visit_Declaration(self, entry):
        if isinstance(entry.src, (list, tuple)):
            # this is the weird conditional parameter scenario
            for elem in entry.src:
                self._passthrough_SrcItem(elem)
        else:
            self._passthrough_SrcItem(entry.src)

    def visit_Stmt(self, entry):
        item = entry.item

        logical_tokens = list(filter(
            lambda tok: (tok.type == Literal.logical or
                         tok.string.lower() in self.logical_varnames),
            item.tokens
        ))

        if len(logical_tokens) == 0 or item.kind == ChunkKind.SubroutineDecl:
            self._passthrough_SrcItem(item)
            return
        elif item.kind == ChunkKind.Call and all(
            tok.type == "arbitrary-name" for tok in logical_tokens
        ):
            self._passthrough_SrcItem(item)
            return
        elif item.has_label:
            raise RuntimeError()
        elif item.trailing_comment_start != []:
            raise RuntimeError()

        if item.nlines()==1 and any(tok.string == '=' for tok in item.tokens):

            #pattern = rf"^\s+{_NAME_REGEX}(\(i\))?\s*=\s*\.(TRUE|FALSE)\.\s*$"
            if re.match(
                rf"^\s+{_NAME_REGEX}(\(i\))?\s*=\s*\.(TRUE|FALSE)\.\s*$",
                item.lines[0], re.IGNORECASE
            ):
                def repl(matchobj):
                    if matchobj.group(0).lower() == '.true.':
                        return "MASK_TRUE"
                    return "MASK_FALSE"
                #print("matched:", item.lines)
                replacement = re.sub(
                    '\.(TRUE|FALSE)\.', repl, item.lines[0], flags=re.IGNORECASE
                )
                self.consumer.consume(replacement)
                return
            elif m := re.match(
                rf"^\s+(?P<L>{_NAME_REGEX})(\(i\))?\s*=\s*(?P<R>{_NAME_REGEX})(\(i\))?\s*$",
                item.lines[0]
            ):
                assert m.group("L") in self.logical_varnames
                assert m.group("R") in self.logical_varnames
                self._passthrough_SrcItem(item)
                return
            else:
                pass # deal with this case at the end!

        elif m := re.match(
            rf"^\s+if\s*\(\s*(?P<var>{_NAME_REGEX})\s*\)",
            item.lines[0],
            re.IGNORECASE
        ):
            assert m.group("var") in self.logical_varnames
            for i,line in enumerate(item.lines):
                if i == 0:
                    line = re.sub(
                        m.group("var"), f'{m.group("var")} .ne. MASK_FALSE',
                        line, count=1, flags=re.IGNORECASE
                    )
                    #print(f"replaced {item.lines[0]!r} with {line!r}")
                self.consumer.consume(line)
            return

        elif m := re.match(
            rf"^\s+if\s*\(\s*(?P<var>{_NAME_REGEX})\(i\)\s*\)",
            item.lines[0],
            re.IGNORECASE
        ):
            assert m.group("var") in self.logical_varnames
            tmp = m.group("var") + r"\(i\)"
            for i,line in enumerate(item.lines):
                if i == 0:
                    line = re.sub(
                        tmp, f'{m.group("var")}(i) .ne. MASK_FALSE',
                        line, count=1, flags=re.IGNORECASE
                    )
                    #print(f"replaced {item.lines[0]!r} with {line!r}")
                self.consumer.consume(line)
            return

        print(f"can't handle:\n-> {item.origin}\n-> {item!s}")
        self._passthrough_SrcItem(item)


    def visit_ControlConstruct(self, entry):
        for (condition, contents) in entry.condition_contents_pairs:
            self.dispatch_visit(condition)
            for content_entry in contents:
                self.dispatch_visit(content_entry)
        self.dispatch_visit(entry.end)



def replace_logical_with_mask(in_fname, out_fname):

    temp_fname = './__my_dummy_file'
    with ExitStack() as stack:
        # handle inputs
        in_f = stack.enter_context(open(in_fname, "r"))
        provider = LineProvider(in_f, fname = in_fname)
        it = get_source_regions(provider)


        # handle outputs
        out_f = stack.enter_context(open(temp_fname, "w"))
        def writer(arg):
            out_f.write(f'{arg}\n')
        translator = ReplaceLogicalTranslator(writer)

        for region in it:
            if region.is_routine:
                subroutine = build_subroutine_entity(region, it.prologue)

                logical_varnames = []
                for arg in subroutine.arguments + subroutine.variables:
                    if arg.type == Type.logical:
                        logical_varnames.append(arg.name.lower())

                print(subroutine.name)
                translator.logical_varnames = logical_varnames


                translator.dispatch_visit(subroutine.subroutine_stmt)
                for entry in subroutine.declaration_section:
                    translator.dispatch_visit(entry)
                for entry in subroutine.impl_section:
                    translator.dispatch_visit(entry)
                translator.dispatch_visit(subroutine.endroutine_stmt)
            else:
                for _, entry in region.lineno_item_pairs:
                    translator.dispatch_visit(entry)
    shutil.copyfile(src=temp_fname, dst=out_fname)

# ========================================================================


_TYPE_MAP = {
    Type.i32: "int",
    Type.i64: "long long",
    Type.f32: "float",
    Type.f64: "double",
    Type.mask_type: "gr_mask_type",
    Type.gr_float: "gr_float"
}

def c_like_fn_signature(subroutine):
    arg_list = (
        f"  {_TYPE_MAP[arg.type]}* {arg.name}"
        for arg in subroutine.arguments
    )
    rslt = f"void {subroutine.name}(\n" + ',\n'.join(arg_list) + '\n)'
    return rslt


_TODO_PREFIX = '//_//'
import textwrap

# ok, so when we do this for real, we will need to be a little more careful
# with c++ variables.
# -> We should probably create a class where we map the fortran name to a C++
#    variable or a member of a C++ struct

class CppTranslator(EntryVisitor):

    def __init__(self, consume_callback, identifier_spec):
        super().__init__(ignore_unknown=False)
        self.needs_dealloc = []
        self.consumer = _Consumer(consume_callback)
        def unaddressed_consume(arg):
            consume_callback(f"  {_TODO_PREFIX} PORT: {arg}")
        self.unaddressed_consumer = _Consumer(unaddressed_consume)
        self.identifier_spec = identifier_spec
        self.block_level=1

    def _write_translated_line(self, line):
        indent = '  ' * self.block_level
        self.consumer.consume(f'{indent}{line}')

    def _passthrough_SrcItem(self, entry, unaddressed = True):
        if unaddressed:
            _passthrough_SrcItem(self.unaddressed_consumer, entry)
        else:
            _passthrough_SrcItem(self.consumer, entry)

    def _cpp_variable_name(self, fortran_identifier, as_ptr = False):
        """
        This produces the string C++ variable name that corresponds to the
        Fortran variable name

        Parameters
        ----------
        fortran_identifier: str or IdentifierExpr
            name of the fortran variable
        as_ptr: bool
            When False, the string accesses the value of the variable. When
            True, the string specifies the address of the value
        """

        # I think we will want to get a lot more sophisticated, (and maybe
        # break this functionality out into a separate class/function) but to
        # start out we do something extremely simple
        # -> we might want to eventually return an intermediate class that
        #    the translator then uses to get a pointer address or access value
        #    based on context

        # we assume that the variable name is unchanged
        if isinstance(fortran_identifier, IdentifierExpr):
            var_name = fortran_identifier.token.string
        else:
            var_name = fortran_identifier

        if self.identifier_spec.is_arg(var_name):
            return var_name if as_ptr else f'(*{var_name})'
        elif ((self.identifier_spec.is_var(var_name)) or
              (not self.identifier_spec[var_name].is_macro)):
            return f'(&{var_name})' if as_ptr else var_name
        else: # it is a macro constant
            # NOTE: I vaguely recall that we may need to remap macro names
            # between Fortran and C++, but that's a problem for the future
            assert self.identifier_spec[var_name].is_macro
            if as_ptr:
                raise ValueError(
                    "It doesn't make any sense to return a pointer to a macro "
                    "that expands to a constant"
                )
            return var_name


    def visit_WhitespaceLines(self, entry):
        self._passthrough_SrcItem(entry, unaddressed = False)

    def visit_Comment(self, entry):
        if len(entry.lines) == 1:
            line = entry.lines[0]
            self._write_translated_line('//' + line.strip()[1:])
        else:
            for line in entry.lines:
                self._write_translated_line('//' + line.strip()[1:])

            #lines = textwrap.indent(
            #    textwrap.dedent('\n'.join(entry.lines)), '// '
            #).splitlines()
            #for line in lines:
            #    self._write_translated_line(line)

    def visit_PreprocessorDirective(self, entry):
        self._passthrough_SrcItem(entry)

    def visit_OMPDirective(self, entry):
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


        scalar_decls, array_decls = [], []
        for identifier in entry.identifiers:
            if isinstance(identifier, Constant):
                self._passthrough_SrcItem(entry.src)
                return None
            elif self.identifier_spec.is_arg(identifier.name):
                continue
            elif identifier.array_spec is None:
                scalar_decls.append(identifier)
            else:
                array_decls.append(identifier)

        ctype = _TYPE_MAP[entry.identifiers[0].type]

        if array_decls == [] and scalar_decls == []:
            self._write_translated_line(
                '// -- removed line (previously just declared arg types) -- '
            )
            return None

        if len(scalar_decls) > 0:
            self._write_translated_line(
                f"{ctype} {', '.join(v.name for v in scalar_decls)};"
            )

        for var in array_decls:
            axlens = var.array_spec.axlens 
            if (None in axlens) or var.array_spec.allocatable:
                self._write_translated_line(
                    f'{ctype}* {var.name}=NULL;'
                )
                # we don't need to worry about appending this to
                # self.needs_dealloc, deallocation should be handled within the
                # function body
                continue

            factors = []
            for elem in axlens:
                if isinstance(elem, LiteralExpr):
                    factors.append(elem.token.string)
                elif isinstance(elem, IdentifierExpr):
                    factors.append(self._cpp_variable_name(elem, as_ptr=False))
                else:
                    raise RuntimeError("not equipped to handle this case yet")
            self._write_translated_line(
                f"{ctype}* {var.name} = "
                f"({ctype}*)malloc({'*'.join(factors)}*sizeof({ctype}));"
            )
            self.needs_dealloc.append(var.name)

        return None

    def visit_Stmt(self, entry):
        print(entry.__class__.__name__)
        self._passthrough_SrcItem(entry.item)

    def visit_ControlConstruct(self, entry):
        for (condition, contents) in entry.condition_contents_pairs:
            self.dispatch_visit(condition)
            for content_entry in contents:
                self.dispatch_visit(content_entry)
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


            writer("  // handle deallocation!")
            for elem in translator.needs_dealloc:
                writer(f"  free({elem});")
            writer("}")
            return


