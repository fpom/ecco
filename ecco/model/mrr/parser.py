import re
import ast
import subprocess
import warnings

from typing import NoReturn
from functools import reduce
from itertools import chain
from operator import or_

from .mrrparse import Indenter, Transformer, ParseError, Token, v_args
from .mrrmodparse import Lark_StandAlone as LarkModelParser
from .mrrpatparse import Lark_StandAlone as LarkPatternParser

#
# cpp
#

_line_dir = re.compile(r'^#\s+([0-9]+)\s+"([^"]+)"[\s\d]*$')


def cpp(text, **var):
    out = subprocess.run(
        ["cpp", "--traditional", *(f"-D{k}={v}" for k, v in var.items())],
        input=text,
        encoding="utf-8",
        capture_output=True,
    ).stdout
    lines = []
    skip = True
    for line in out.splitlines():
        if match := _line_dir.match(line):
            if match.group(2) == "<stdin>":
                num = int(match.group(1))
                while len(lines) < num - 1:
                    lines.append("\n")
                skip = not num
        elif skip:
            pass
        else:
            lines.append(line.split("#", 1)[0].rstrip() + "\n")
    return "".join(lines)


#
# aux elements
#


def _error(line, column, message, errcls=ParseError) -> NoReturn:
    if column is not None:
        err = errcls(f"[{line}:{column}] {message}")
    else:
        err = errcls(f"[{line}] {message}")
    setattr(err, "line", line)
    setattr(err, "column", column)
    raise err


def _assert(test, line, column, message, errcls=ParseError):
    if not test:
        _error(line, column, message, errcls)


def _int(tok, min=None, max=None, message="invalid int litteral"):
    try:
        val = int(tok.value)
        if min is not None:
            _assert(val >= min, tok.line, tok.column, message)
        if max is not None:
            _assert(val <= max, tok.line, tok.column, message)
        return val
    except ParseError:
        raise
    except Exception:
        _error(tok.line, tok.column, message)


class AST(dict):
    def __init__(self, *largs, **kwargs):
        super().__init__(*largs, **kwargs)
        for k in self:
            if hasattr(self.__class__, k):
                warnings.warn(f"AST items {k} not available as attribute", stacklevel=2)

    def __getattr__(self, attr):
        return self.get(attr)

    def __or__(self, other):
        return self.__class__(dict(self) | dict(other))


#
# Lark handlers
#


class MRRIndenter(Indenter):  # type: ignore
    NL_type = "_NL"  # type: ignore
    OPEN_PAREN_types = []  # type: ignore
    CLOSE_PAREN_types = []  # type: ignore
    INDENT_type = "_INDENT"  # type: ignore
    DEDENT_type = "_DEDENT"  # type: ignore
    tab_len = 4  # type: ignore


@v_args(inline=True)
class MRRTrans(Transformer):
    def clocks(self, first, *rest):
        d = dict(first)
        for r in rest:
            d.update(r)
        return d

    def clockdecl(self, name, desc):
        return {name.value: desc.value}

    def model(self, *args):
        k2i = {a.kind: a[a.kind] for a in args if a is not None and a.kind != "model"}
        return AST(
            kind="model",
            **k2i,
            locations=tuple(a for a in args if a is not None and a.kind == "model"),
        )

    def variables(self, *args):
        return AST(kind="variables", variables=args)

    def constraints(self, *args):
        return AST(kind="constraints", constraints=args)

    def rules(self, *args):
        return AST(kind="rules", rules=args)

    def location(self, name, size, model):
        return model | AST(
            line=name.line,
            name=name.value,
            shape=None if size is None else _int(size, 1, None, "invalid array size"),
        )

    def vardecl_short(self, name, sign, desc):
        return AST(
            kind="decl",
            line=name.line,
            name=name.value,
            domain={0, 1},
            init={"+": {1}, "-": {0}, "*": {0, 1}}.get(sign.value),
            desc=desc.value.strip(),
        )

    def vardecl_long(self, typ, name, init, desc):
        return (
            typ
            | (init or AST(init=typ.domain if typ.clock is None else {-1}))
            | AST(
                kind="decl",
                line=name.line,
                name=name.value,
                desc=desc.value.strip(),
            )
        )

    def type(self, typ, shape=None):
        return typ | {"shape": None if shape is None else _int(shape)}

    def type_bool(self):
        return AST(domain={0, 1})

    def type_clock(self, clock, start, end):
        return AST(clock=clock.value) | self.type_interval(start, end, {-1})

    def type_interval(self, start, end, more=set()):
        return AST(domain=set(range(_int(start), _int(end) + 1)) | more)

    def init(self, init):
        return init

    def init_seq(self, *args):
        return AST(
            kind="init",
            line=args[0].line,
            init=tuple(None if a is None else a.init for a in args),
        )

    def init_int(self, min, *args):
        if len(args) >= 1 and isinstance(args[0], Token):
            max, *args = args
            init = set(range(_int(min), _int(max) + 1))
        else:
            init = {_int(min)}
        return AST(
            kind="init",
            line=min.line,
            init=None
            if any(a is None for a in args)
            else reduce(or_, (a.init for a in args), init),
        )

    def init_all(self):
        return None

    def actdecl(self, *args):
        if args and isinstance(args[0], Token):
            tags, *args = args
        else:
            tags = AST(value="")
        _assert(
            all(a.kind in ("expr", "assign", "quant") for a in args),
            args[0].line,
            None,
            "unexpected item",
        )
        return AST(
            kind="act",
            line=args[0].line,
            tags=tuple(ts for t in (tags.value or "").split(",") if (ts := t.strip())),
            guard=tuple(a for a in args if a.kind == "expr"),
            assign=tuple(a for a in args if a.kind == "assign"),
            quant=reduce(or_, (a.quant for a in args if a.kind == "quand"), {}),
        )

    def condition_long(self, left, op, right):
        line, column = left.line, left.column
        if isinstance(left, Token):
            column -= len(left.value) - 1
            left = _int(left)
        if isinstance(right, Token):
            right = _int(right)
        return AST(kind="expr", line=line, column=column, left=left, op=op, right=right)

    def condition_short(self, var, op):
        values = {"+": 1, "-": 0, "*": None}
        _assert(
            op.value in values,
            var.line,
            var.column + 1 - len(op.value),
            f"invalid condition {op.value!r}",
        )
        return AST(
            kind="expr",
            line=var.line,
            column=var.column,
            left=var,
            op="==",
            right=values[op.value],
        )

    def assignment_long(self, target, assign, value):
        if isinstance(value, Token):
            value = None if value.value == "*" else _int(value)
        if (op := assign.value[0]) in ("+", "-"):
            value = AST(
                kind="expr",
                line=target.line,
                column=target.column,
                left=target,
                op=op,
                right=value,
            )
        return AST(
            kind="assign",
            line=target.line,
            column=target.column,
            target=target,
            value=value,
        )

    def assignment_short(self, target, op):
        if op.value in ("++", "--"):
            return AST(
                kind="assign",
                line=target.line,
                column=target.column,
                target=target,
                value=AST(
                    kind="expr",
                    line=target.line,
                    column=target.columns,
                    left=target,
                    op=op.value[0],
                    right=1,
                ),
            )
        elif op.value in ("+", "-", "*"):
            return AST(
                kind="assign",
                line=target.line,
                column=target.column,
                target=target,
                value={"+": 1, "-": 0, "*": None}[op.value],
            )
        elif op.value == "~":
            return AST(
                kind="assign",
                line=target.line,
                column=target.column,
                target=target,
                value=AST(
                    kind="expr",
                    line=target.line,
                    column=target.columns,
                    left=1,
                    op="-",
                    right=target,
                ),
            )
        else:
            _error(op.line, op.column, "invalid assignment {op.value!r}")

    def var(self, left, mark=None, right=None):
        if mark is None:
            return left
        elif mark == "@":
            assert right is not None
            return left | AST(locrel="inner", locname=right.name, locidx=right.index)
        else:
            assert mark == "." and right is not None
            return right | AST(
                line=left.line,
                column=left.column,
                locrel="inner",
                locname=left.name,
                locidx=left.index,
            )

    def mark(self, mark):
        return mark.value

    def up_var(self, var):
        return var | {"locrel": "outer"}

    def name_idx(self, name, index=None):
        return name | {"index": index}

    def name(self, name):
        return AST(
            kind="var",
            line=name.line,
            column=name.column,
            name=ast.literal_eval(name.value)
            if name.value[0] in ("'", '"')
            else name.value,
        )

    def index(self, idx, op=None, inc=None):
        return AST(
            kind="expr",
            line=idx.line,
            column=idx.column,
            left=int(idx.value) if idx.value.isdecimal() else idx.value,
            op=None if op is None else op.value,
            right=None if inc is None else int(inc.value),
        )

    def quantifier(self, op, *args):
        q = {}
        for arg in args:
            if isinstance(arg, AST):
                _assert(
                    not any(v in q for v in arg.quant),
                    arg.line,
                    arg.column,
                    "multiple quantification",
                )
                q.update(arg.quant)
            else:
                _assert(
                    arg.value not in q, arg.line, arg.column, "multiple quantification"
                )
                q[arg.value] = op.value
        return AST(kind="quant", line=op.line, column=op.column, quant=q)


@v_args(inline=True)
class ModelTrans(MRRTrans):
    def start(self, *args):
        *clocks, model = args
        return model | {"clocks": clocks[0] if clocks else {}}


@v_args(inline=True)
class PatternTrans(MRRTrans):
    def start(self, *rules):
        line = None
        for obj in rules:
            line = getattr(obj, "line", None)
            if line is not None:
                break
        return AST(
            kind="pattern",
            line=line,
            variables=(),
            locations=(),
            constraints=(),
            rules=rules,
        )


#
# main parsers
#


class ModelParser:
    Parser = LarkModelParser
    Transformer = ModelTrans

    @classmethod
    def parse_src(cls, src, *defs, **vardef):
        for d in defs:
            try:
                v, t = d.split("=", 1)
                vardef[v] = t
            except Exception:
                vardef[d] = None
        parser = cls.Parser(transformer=cls.Transformer(), postlex=MRRIndenter())
        return parser.parse(cpp(src, **vardef))

    @classmethod
    def parse(cls, path, *defs, **vardef):
        return cls.parse_src(open(path).read(), *defs, **vardef)


class PatternParser(ModelParser):
    Parser = LarkPatternParser
    Transformer = PatternTrans
