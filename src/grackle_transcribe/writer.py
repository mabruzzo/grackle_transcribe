# we are loosely inspired by sphinx and docutils
from contextlib import ExitStack
import re
import shutil

from .identifiers import Variable, Constant

from .subroutine_entity import (
    Declaration,
    build_subroutine_entity,
    ControlConstruct
)
from .src_model import (
    SrcItem,
    WhitespaceLines,
    Code,
    Comment,
    PreprocKind,
    PreprocessorDirective,
    OMPDirective,
    LineProvider,
    get_source_regions
)
from .syntax_unit import (
     IdentifierExpr, ArrayAccess, LiteralExpr, Standard1TokenStmt, Stmt, 
     UncategorizedStmt, _iterate_tokens, ControlConstructKind,
     compressed_str_from_Expr
)
from .stringify import FormattedCodeEntryBuilder
from .token import (
    token_has_type, Keyword, Literal, _NAME_REGEX, Type
)
from .utils import indented_fmt

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
    if isinstance(entry, str):
        consumer.consume(entry)
        return
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

_TYPE_MAP_FORTRAN_STR = {
    Type.f32 : "real*4", Type.f64 : "real*8",
    Type.i32 : "integer", Type.i64 : "integer*8",
    Type.logical : "logical", Type.mask_type : "MASK_TYPE",
    Type.gr_float : "R_PREC"
}

def _declare_variable_subset(identifier_l, line_len = 71):
    type_str = _TYPE_MAP_FORTRAN_STR[identifier_l[0].type]
    indent = f"      {type_str} "
    subsequent_indent = "     &" + (" " * (len(type_str) + 1))

    entries = []
    for identifier in identifier_l:
        assert isinstance(identifier, Variable)
        arr_spec = identifier.array_spec
        if arr_spec is None:
            entries.append(identifier.name)
        else:
            assert not arr_spec.allocatable
            shape = []
            for axlen in arr_spec.axlens:
                assert axlen is not None
                if isinstance(axlen, ArrayAccess):
                    assert all(
                        isinstance(e, LiteralExpr)
                        for e in axlen.arg_l.get_args()
                    )
                shape.append(compressed_str_from_Expr(axlen))
            entries.append(f"{identifier.name}({','.join(shape)})")
    #print(entries)

    return '\n'.join(indented_fmt(
        entries, indent = indent, width = line_len,
        subsequent_indent=subsequent_indent
    ))

class UnchangedTranslator(EntryVisitor):
    # we do allow skipping over declarations

    def __init__(self, consume_callback, skip_identifier_declaration = None):
        super().__init__(ignore_unknown=False)
        self.consumer = _Consumer(consume_callback)
        self.skip_identifier_declaration = skip_identifier_declaration

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
        elif self.skip_identifier_declaration is None:
            _passthrough_SrcItem(self.consumer, entry.src)
        else:
            if isinstance(entry.identifiers, (Constant, Variable)):
                iden_l = [entry.identifiers]
            else:
                iden_l = entry.identifiers
            used_iden_l = [
                e for e in iden_l \
                if e.name.lower() not in self.skip_identifier_declaration
            ]
            if len(used_iden_l) == 0:
                return None
            elif entry.defines_constant or len(iden_l) == len(used_iden_l):
                _passthrough_SrcItem(self.consumer, entry.src)
            else:
                consumer = self.consumer
                consumer.consume(
                    _declare_variable_subset(used_iden_l, line_len = 71)
                )

    def visit_Stmt(self, entry):
        if entry.item.has_label:
            _passthrough_SrcItem(self.consumer,entry.src)
        else:
            builder = FormattedCodeEntryBuilder(entry.src)
            for tok in entry.src.tokens:
                builder.put(tok)
            new_entries = builder.build()

            assert len(new_entries) == len(entry.src.entries)
            for i, ref in enumerate(entry.src.entries):
                if isinstance(new_entries[i], str):
                    if new_entries[i].rstrip() != ref.rstrip():
                        raise RuntimeError(
                            "MISMATCH: \n"
                            f" -> new: {new_entries[i].strip()!r}\n"
                            f" -> ref: {ref.strip()!r}\n"
                        )

            if False and (new_entries != entry.src.entries):
                def _pprint(l):
                    return (
                        '[\n' +
                        '        ' +
                        ',\n       '.join(repr(e) for e in l ) +
                        '\n     ]'
                    )
                raise RuntimeError(
                    "Reconstructed entries:\n"
                    f" -> {_pprint(new_entries)}\n"
                    "Original:\n"
                    f" -> {_pprint(entry.src.entries)}"
                )
            if True:
                _passthrough_SrcItem(self.consumer,entry.src)
            else:
                for chunk in new_entries:
                    _passthrough_SrcItem(self.consumer,chunk)

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

def write_extracted_region_as_fn(lineno_search, routine_name,
                                 out_fname):
    from grackle_transcribe.routine_analysis import analyze_routine

    info_map = analyze_routine(lineno_search)

    subroutine = lineno_search.try_get_subroutine()
    assert subroutine is not None
    assert lineno_search.is_control_construct_stmt
    control_construct = lineno_search.src_entry_hierarchy[-2]

    # figure the list of arguments and unused identifiers
    fortran_identifier_spec = subroutine.identifiers
    extra_required_identifiers = set()
    for varname, varinfo in info_map.items():
        key = varname.lower()
        array_spec = getattr(fortran_identifier_spec[key], "array_spec", None)
        if array_spec is None:
            continue
        for axlen in array_spec.axlens:
            if isinstance(axlen, ArrayAccess):
                extra_required_identifiers.add(
                    axlen.array_name.token.string.lower()
                )
                all_inner_literals = all(
                    isinstance(e, LiteralExpr) for e in axlen.arg_l.get_args()
                )
                if not all_inner_literals:
                    raise RuntimeError(
                        "our simplifying assumption that inner array-accesses "
                        "in array-shape declarations use literals is wrong"
                    )
            elif isinstance(axlen, IdentifierExpr):
                extra_required_identifiers.add(axlen.token.string.lower())
            else:
                assert (axlen is None) or isinstance(axlen, LiteralExpr)

    arg_names, unused_identifiers = [], []
    for key in fortran_identifier_spec.keys():
        if getattr(fortran_identifier_spec[key], 'is_macro', False):
            continue
        varinfo = info_map[key]
        if (not varinfo.is_used) and (key not in extra_required_identifiers):
            unused_identifiers.append(key)
        elif not fortran_identifier_spec.is_constant(key):
            arg_names.append(fortran_identifier_spec[key].name)
    routine_sig = f"{routine_name}({', '.join(arg_names)})"

    _nominal_indent = ' '*6

    _indent =f'{_nominal_indent}subroutine {routine_name}(' 
    _subsequent_indent = '     &                '
    routine_sig = '\n'.join(indented_fmt(
        arg_names, indent = _indent, width = 71,
        subsequent_indent=_subsequent_indent
    )) + f'\n{_subsequent_indent[:-2]})'



    #print(len(unused_identifiers))
    #print("\n\nunused identifiers:")
    #print("  ", unused_identifiers)


    with open(out_fname, "w") as out_f:
        out_f.write(routine_sig)
        out_f.write('\n')

        def writer(arg):
            out_f.write(f'{arg}\n')
        translator = UnchangedTranslator(
            writer,
            skip_identifier_declaration=set(unused_identifiers)
        )
        for entry in subroutine.declaration_section:
            translator.dispatch_visit(entry)

        out_f.write('\n')
        out_f.write('! The following was extracted from another subroutine\n')
        translator.dispatch_visit(control_construct)
        translator.dispatch_visit(subroutine.endroutine_stmt)


# The following logic is all pretty old. I'm not sure it even works any more!

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

        if (len(logical_tokens) == 0 or
            token_has_type(item.tokens, Keyword.SUBROUTINE)):
            self._passthrough_SrcItem(item)
            return
        elif (
            token_has_type(item.tokens[0], Keyword.CALL) and
            all(tok.type == "arbitrary-name" for tok in logical_tokens)
        ):
            self._passthrough_SrcItem(item)
            return
        elif item.has_label:
            raise RuntimeError()
        elif item.trailing_comment_start != []:
            raise RuntimeError()

        if item.nlines()==1 and any(tok.string == '=' for tok in item.tokens):

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
