import re
import ast
import subprocess
import warnings

from typing import Iterator, Self
from functools import reduce
from operator import or_
from collections import defaultdict

from .mrrparse import Indenter, Transformer, UnexpectedInput, Token, v_args
from .mrrmodparse import Lark_StandAlone as LarkModelParser
from .mrrpatparse import Lark_StandAlone as LarkPatternParser

#
# cpp
#

_line_dir = re.compile(r'^#\s+([0-9]+)\s+"([^"]+)"[\s\d]*$')


def cpp(text: str, **var: str) -> str:
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


class ParseError(Exception):
    def __init__(self, message, line, column=None, src=None):
        if column is None:
            msg = f"[{line}] {message}"
        else:
            msg = f"[{line}:{column}] {message}"
        if src is not None:
            msg += f"\n{src}"
            if column is not None:
                msg += f"\n{' ' * (column - 1)}^^^"
        super().__init__(msg)
        self.msg = message
        self.lin = line
        self.col = column
        self.src = src

    @classmethod
    def ensure(cls, test, message, line, column=None, src=None):
        if not test:
            raise cls(message, line, column, src)


def _int(tok, min=None, max=None, message="invalid int litteral"):
    try:
        val = int(tok.value)
    except Exception:
        raise ParseError(message, tok.line, tok.column)
    else:
        if min is None and tok.type == "NAT":
            min = 0
        if min is not None:
            ParseError.ensure(val >= min, message, tok.line, tok.column)
        if max is not None:
            ParseError.ensure(val <= max, message, tok.line, tok.column)
        return val


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

    def error(self, message, src=None):
        raise ParseError(message, self.line, self.column, src)

    def ensure(self, check, message, src=None):
        ParseError.ensure(check, message, self.line, self.column, src)

    def search(self, match) -> Iterator[Self]:
        if all(self.get(k) == v for k, v in match.items()):
            yield self
        cls = self.__class__
        for val in self.values():
            if isinstance(val, cls):
                yield from val.search(match)
            elif isinstance(val, (tuple, list)):
                for v in val:
                    if isinstance(v, cls):
                        yield from v.search(match)
            elif isinstance(val, dict):
                for v in val.values():
                    if isinstance(v, cls):
                        yield from v.search(match)

    def sub(self, match, repl):
        if all(self.get(k) == v for k, v in match.items()):
            return self.__class__({k: repl.get(k, v) for k, v in self.items()})
        else:
            new, cls = {}, self.__class__
            for key, val in self.items():
                if isinstance(val, cls):
                    new[key] = val.sub(match, repl)
                elif isinstance(val, (tuple, list)):
                    new[key] = type(val)(
                        v.sub(match, repl) if isinstance(v, cls) else v for v in val
                    )
                elif isinstance(val, dict):
                    new[key] = {
                        k: v.sub(match, repl) if isinstance(v, cls) else v
                        for k, v in val.items()
                    }
                else:
                    new[key] = val
            return cls(new)


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

    def location(self, name, *args):
        if len(args) == 1:
            shape, model = None, args[0]
        else:
            shape, model = args
        return model | AST(
            line=name.line,
            name=name.value,
            shape=None if shape is None else _int(shape, 1, None, "invalid array size"),
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
        if init is None:
            init = AST(init=typ.domain if typ.clock is None else {-1})
        init.ensure(
            not isinstance(init.init, tuple) or len(init.init) == typ.shape,
            "wrong array init length",
        )
        init.init.intersection_update(typ.domain)
        return (
            typ
            | init
            | AST(
                kind="decl",
                line=name.line,
                name=name.value,
                desc=desc.value.strip(),
            )
        )

    def type(self, typ, shape=None):
        return typ | {
            "shape": None
            if shape is None
            else _int(shape, min=1, message="invalid array size")
        }

    def type_bool(self):
        return AST(domain={0, 1})

    def type_clock(self, clock, start, end):
        return AST(clock=clock.value) | self.type_interval(start, end, {-1})

    def type_interval(self, start, end, more=set()):
        return AST(domain=set(range(_int(start), _int(end, 1) + 1)) | more)

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
        parts = defaultdict(list)
        for a in args:
            parts[a.kind].append(a)
        return AST(
            kind="act",
            line=args[0].line,
            tags=parts["tags"][0].tags if parts["tags"] else (),
            guard=tuple(parts["expr"]),
            assign=tuple(parts["assign"]),
            quant=parts["quant"][0].quant if parts["quant"] else {},
        )

    def tags(self, *args):
        return AST(kind="tags", tags=tuple(a.value for a in args))

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
        ParseError.ensure(
            op.value in values,
            f"invalid condition {op.value!r}",
            var.line,
            var.column + 1 - len(op.value),
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
            raise ParseError(f"invalid assignment {op.value!r}", op.line, op.column)

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
                ParseError.ensure(
                    "multiple quantification",
                    not any(v in q for v in arg.quant),
                    arg.line,
                    arg.column,
                )
                q.update(arg.quant)
            else:
                ParseError.ensure(
                    "multiple quantification", arg.value not in q, arg.line, arg.column
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
            except ValueError:
                vardef[d] = None
            else:
                vardef[v] = t
        sourcelines = src.splitlines()
        parser = cls.Parser(transformer=cls.Transformer(), postlex=MRRIndenter())
        try:
            ast = parser.parse(cpp(src, **vardef))
        except UnexpectedInput as err:
            try:
                srcline = sourcelines[err.line - 1]
            except Exception:
                srcline = None
            raise ParseError(
                "unexpected input",
                err.line,
                err.column,
                srcline,
            ).with_traceback(err.__traceback__)
        except ParseError as err:
            try:
                srcline = sourcelines[err.lin - 1]
            except (TypeError, IndexError):
                srcline = None
            raise ParseError(err.msg, err.lin, err.col, srcline).with_traceback(
                err.__traceback__
            )
        except Exception as err:
            raise ParseError(str(err), 0).with_traceback(err.__traceback__)
        ast["source"] = sourcelines
        return ast

    @classmethod
    def parse(cls, path, *defs, **vardef):
        return cls.parse_src(open(path).read(), *defs, **vardef)


class PatternParser(ModelParser):
    Parser = LarkPatternParser
    Transformer = PatternTrans
