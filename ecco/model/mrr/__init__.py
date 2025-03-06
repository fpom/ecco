import re
import operator

from typing import Mapping, Self, Any, Iterator
from pathlib import Path
from datetime import datetime
from itertools import product, chain

from rich.text import Text  # pyright: ignore

from .parser import ModelParser, PatternParser, AST, ParseError
from ..record import Record, Printer
from .. import (
    Model as _Model,
    Variable as _Variable,
    Expression as _Expression,
    Action as _Action,
    ActionKind,
)


def _mrr_var(var):
    if any(c in "[.]" for c in str(var)):
        return Text.assemble('"', var, '"', style="blue")
    else:
        return Text.assemble(var, style="blue")


_mrr_styles = {
    ("var",): _mrr_var,
    ("decl", "domain"): "magenta",
    ("decl", "domain", "clock"): "bold",
    ("decl", "init", "+"): "green bold",
    ("decl", "init", "-"): "red bold",
    ("decl", "init", "*"): "yellow bold",
    ("decl", "var"): "bold",
    ("expr", "op"): "",
    ("expr", "+"): "green",
    ("expr", "-"): "red",
    ("action", "op", ">>"): Text(" >> ", "bold red"),
    ("action", "tag"): "cyan",
    ("action", "assign", "+"): "green",
    ("action", "assign", "-"): "red",
    ("action", "assign", "*"): "yellow",
    ("comment",): "dim",
    ("header",): "bold red",
}


class Variable(_Variable):
    def __txt__(self, styles={}, context={}):  # pyright: ignore
        prn = Printer(_mrr_styles | styles)
        with prn.decl:
            if self.domain == {0, 1}:
                with prn.init:
                    if self.init == {0, 1} or self.init == {-1}:
                        init = prn("*")
                    elif self.init == {0}:
                        init = prn("-")
                    else:
                        init = prn("+")
                return (prn / "")(
                    prn(self.name, "var"),
                    init,
                    prn(f": {self.comment}", "comment"),
                )
            else:
                with prn.domain:
                    if self.clock is None:
                        domain = prn(f"{{{min(self.domain)}..{max(self.domain)}}}")
                    else:
                        domain = (prn / "")(
                            "{",
                            prn(self.clock, "clock"),
                            f": {min(self.domain - {-1})}..{max(self.domain)}",
                            "}",
                        )
                with prn.init:
                    if self.init == self.domain or self.init == {-1}:
                        init = prn("*")
                    elif len(self.init) == 1:
                        init = prn(list(self.init)[0])
                    else:
                        init = prn(f"{min(self.init)}..{max(self.init)}")
                return prn[
                    domain,
                    " ",
                    prn(self.name, "var"),
                    " ",
                    prn("=", "op"),
                    " ",
                    init,
                    prn(f": {self.comment}", "comment"),
                ]


class Expression(_Expression):
    def __txt__(self, styles={}, context={}):  # pyright: ignore
        prn = Printer(_mrr_styles | styles)
        dom = context.get("domains", {})
        with prn.expr:
            if len(self.vars) == 1 and dom.get(var := list(self.vars)[0]) == {0, 1}:
                if self.const == -1:
                    return prn[prn(var, "var"), prn("+")]
                else:
                    return prn[prn(var, "var"), prn("-")]
            s = []
            for i, (v, c) in enumerate(self.coeffs.items()):
                p = prn(v, "var")
                if i and c > 0:
                    s.append(prn("+", "op"))
                if c == 1:
                    s.append(p)
                elif c == -1:
                    s.extend([prn("-", "op"), p])
                else:
                    s.extend([prn(c), p])
            if self.op is None:
                if self.const > 0:
                    if s:
                        return prn[*s, prn("+", "op"), prn(self.const)]
                    else:
                        return prn(self.const)
                elif self.const < 0:
                    return prn[*s, prn(self.const)]
                elif not s:
                    return prn(self.const)
                else:
                    return prn[*s]
            elif not s:
                return prn[prn(self.const), prn(self.op, "op"), prn(0)]
            else:
                return prn[*s, prn(self.op, "op"), prn(-self.const)]


class Action(_Action):
    def __txt__(self, styles={}, context={}):  # pyright: ignore
        prn = Printer(_mrr_styles | styles)
        dom = context.get("domains", {})
        with prn.action:
            if self.tags:
                with prn.tag:
                    tags = [prn("["), *(prn(t) for t in self.tags), prn("]")]
            else:
                tags = []
            with prn.assign:
                assign = []
                for v, x in self.assign.items():
                    if not x.vars and dom.get(v) == {0, 1}:
                        assign.append(prn[prn(v, "var"), prn("+" if x.const else "-")])
                    elif not x.vars and x.const == -1:
                        assign.append(prn[prn(v, "var"), prn("*")])
                    elif {v} == x.vars:
                        if x.const == 1:
                            assign.append(prn[prn(v, "var"), prn("++", "op")])
                        elif x.const == -1:
                            assign.append(prn[prn(v, "var"), prn("--", "op")])
                        else:
                            s = "-" if x.const < 0 else "+"
                            c = -x.const if x.const < 0 else x.const
                            assign.append(
                                prn[prn(v, "var"), prn(f"{s}=", "op"), prn(c)]
                            )
                    else:
                        assign.append(prn[prn(v, "var"), prn("=", "op"), prn(x)])
            return prn[
                *tags,
                " " if self.tags else "",
                (prn / ", ")(g.__txt__(styles, context) for g in self.guard),
                prn(">>", "op"),
                (prn / ", ")(assign),
                "  ",
                prn(f"# {self.name}", "comment"),
            ]


class Model(_Model):
    def _txt_context(self, context):
        dom = {v.name: v.domain for v in self.variables}
        if "domains" in context:
            ctx = dict(context)
            ctx["domains"] |= dom
        else:
            ctx = context | {"domains": dom}
        return ctx

    def __txt__(self, styles={}, context={}):  # pyright: ignore
        prn = Printer(_mrr_styles | styles)
        ctx = self._txt_context(context)
        lines = [prn("# === flattened model ===", "comment")]
        if self.meta:
            lines.append(prn("# meta information:", "comment"))
            for key, val in self.meta.items():
                if isinstance(val, (Path, str, int, bool, float)):
                    txt = f"{val}"
                elif isinstance(val, datetime):
                    txt = val.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    txt = repr(val)
                lines.append(prn(f"#  - {key}: {txt}", "comment"))
        if self.variables:
            lines.append(prn("variables:", "header"))
        lines.extend(prn["  ", v.__txt__(styles, ctx)] for v in self.variables)
        for attr in ["constraints", "rules"]:
            if actions := getattr(self, attr):
                lines.append(prn(f"{attr}:", "header"))
                lines.extend(prn["  ", a.__txt__(styles, ctx)] for a in actions)
        return (prn / "\n")(lines)

    @classmethod
    def from_source(cls, _source: str, **_vardefs: str) -> "Model":
        return AST2Model.from_source(_source, **_vardefs)

    @classmethod
    def from_path(cls, _path: str | Path, **_vardefs: str) -> "Model":
        if not isinstance(_path, Path):
            _path = Path(_path)
        return AST2Model.from_path(_path, **_vardefs)


class Pattern(Model):
    def __txt__(self, styles={}, context={}):  # pyright: ignore
        prn = Printer(_mrr_styles | styles)
        lines = [prn("# === pattern ===", "comment")]
        if self.meta:
            lines.append(prn("# meta information:", "comment"))
            for key, val in self.meta.items():
                if isinstance(val, (Path, str, int, bool, float)):
                    txt = f"{val}"
                elif isinstance(val, datetime):
                    txt = val.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    txt = repr(val)
                lines.append(prn(f"#  - {key}: {txt}", "comment"))
            lines.extend(a.__txt__(styles, context) for a in self.actions)
        return (prn / "\n")(lines)

    @classmethod
    def from_source(cls, _source: str, **_vardefs: str) -> "Model":
        return AST2Pattern.from_source(_source, **_vardefs)

    @classmethod
    def from_path(cls, _path: str | Path, **_vardefs: str) -> "Model":
        if not isinstance(_path, Path):
            _path = Path(_path)
        return AST2Pattern.from_path(_path, **_vardefs)


#
# AST to Model/Pattern
#


class AST2Model:
    LarkParser = ModelParser
    ModelClass = Model

    _binops = {
        None: (lambda i, _: i),
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "==": operator.eq,
        "!=": operator.ne,
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
    }

    @classmethod
    def check(cls, loc, locname=None, up={}, down={}, clocks=None):
        if clocks is None:
            clocks = set(loc.clocks or ())
        _up = {}
        for decl in loc.variables or ():
            decl.ensure(
                decl.clock is None or decl.clock in clocks,
                f"clock {decl.clock} not declared",
            )
            decl.ensure(decl.name not in _up, "variable already declared locally")
            _up[decl.name] = decl
        for var in down.values():
            cls._check_var(var, _up, f"{locname!r}")
        _loc_down = {}
        _loc_shape = {}
        for sub in loc.locations or ():
            _loc_down[sub.name] = {}
            _loc_shape[sub.name] = sub.shape
        for act in chain(loc.constraints or (), loc.rules or ()):
            for var in act.search({"kind": "var"}):  # pyright: ignore
                if var.locrel is None:
                    cls._check_var(var, _up, None)
                elif var.locrel == "outer":
                    var.ensure(locname is not None, "no outer location")
                    cls._check_var(var, up, "outer location")
                elif var.locrel == "inner":
                    var.ensure(
                        var.locname in _loc_down,
                        f"inner location {var.locname!r} not declared",
                    )
                    var.ensure(
                        var.locidx is None or _loc_shape[var.locname] is not None,
                        "single location cannot be indexed",
                    )
                    var.ensure(
                        _loc_shape[var.locname] is None or var.locidx is not None,
                        "array location should be indexed",
                    )
                    _loc_down[var.locname].setdefault(var.name, var)
                else:
                    assert False, "should not occur, this is a bug"
        for sub in loc.locations or ():
            cls.check(sub, sub.name, _up, _loc_down[sub.name], clocks)

    @classmethod
    def _check_var(cls, var, scope, locname):
        where = "locally" if locname is None else f"in {locname}"
        var.ensure(var.name in scope, f"variable not declared {where}")
        var.ensure(
            var.index is None or scope[var.name].shape is not None,
            "scalar variable cannot be indexed",
        )
        var.ensure(
            scope[var.name].shape is None or var.index is not None,
            "array variable should be indexed",
        )

    @classmethod
    def from_source(cls, _source: str, **_vardefs: str) -> Model:
        ast = cls.LarkParser.parse_src(_source, **_vardefs)
        return cls.from_ast(ast)

    @classmethod
    def from_path(cls, _path: Path, **_vardefs: str) -> Model:
        ast = cls.LarkParser.parse(_path, **_vardefs)
        mod = cls.from_ast(
            ast,
            path=_path,
            timestamp=datetime.fromtimestamp(_path.stat().st_mtime),
        )
        return mod

    @classmethod
    def from_ast(cls, ast, **meta) -> Model:
        try:
            cls.check(ast)
            loader = cls()
            loader._load_loc(ast)
            loader._make_ticks()
        except ParseError as err:
            try:
                srcline = ast.source[err.lin - 1]
            except Exception:
                srcline = None
            raise ParseError(err.msg, err.lin, err.col, srcline)
        return cls.ModelClass(
            variables=tuple(loader.variables),
            actions=tuple(loader.actions),
            meta=meta | {"locations": loader.locations},
        )

    def __init__(self):
        self.variables: list[Variable] = []
        self.actions: list[Action] = []
        self.clocks: dict[str, list[Variable]] = {}
        self.shapes: dict[str, int | None] = {}
        self.locations: dict[str, dict] = {}

    def _load_loc(self, loc, path=[], selfidx=None):
        tree = self.locations
        for p in path:
            tree = tree[re.sub(r"\[\d*\]", "[]", p)]
        for sub in loc.locations or ():
            tree[sub.name if sub.shape is None else f"{sub.name}[]"] = {}
        # load clocks
        self.clocks.update({c: [] for c in loc.clocks or ()})
        # load local variables
        for var in loc.variables:
            self.shapes[".".join([*path, var.name])] = var.shape
            if var.shape is not None:
                for i in range(var.shape):
                    self.variables.append(
                        self._load_decl(var, [*path, f"{var.name}[{i}]"], i)
                    )
            else:
                self.variables.append(self._load_decl(var, [*path, var.name]))
            if var.clock is not None:
                self.clocks[var.clock].extend(self.variables[-(var.shape or 1) :])
        # load sub-locations (before actions to resolve nested variables)
        for sub in loc.locations:
            self.shapes[".".join([*path, sub.name])] = sub.shape
            if sub.shape is not None:
                for i in range(sub.shape):
                    self._load_loc(sub, [*path, f"{sub.name}[{i}]"], i)
            else:
                self._load_loc(sub, [*path, sub.name], None)
        # load local actions
        for i, act in enumerate(loc.constraints or (), start=1):
            self.actions.extend(
                self._load_act(act, [*path, f"C{i}"], ActionKind.cons, selfidx)
            )
        for i, act in enumerate(loc.rules or (), start=1):
            self.actions.extend(
                self._load_act(act, [*path, f"R{i}"], ActionKind.rule, selfidx)
            )

    def _load_decl(self, var, path, idx=None):
        if isinstance(var.init, tuple):
            assert idx is not None and 0 <= idx < len(var.init)
            init = var.init[idx]
        else:
            assert isinstance(var.init, set)
            init = var.init
        return Variable(
            name=".".join(path),
            domain=frozenset(var.domain),
            init=frozenset(init),
            comment=var.desc,
            clock=var.clock,
        )

    def _load_act(self, act, path, kind, selfidx=None) -> Iterator[Action]:
        actname = ".".join(path)
        path = path[:-1]
        for new, idx in self._expand_any(act, path):
            if selfidx is not None:
                idx["self"] = selfidx
            yield Action(
                name=actname
                + (
                    f"[{','.join(f'{k}={v}' for k, v in sorted(idx.items()))}]"
                    if idx
                    else ""
                ),
                guard=tuple(self._load_expr(g, path, idx) for g in new.guard or ()),
                assign={
                    self._load_var(a.target, path, idx): Expression(-1)
                    if a.value is None
                    else self._load_expr(a.value, path, idx)
                    for a in new.assign or ()
                },
                tags=new.tags or frozenset(),
                kind=kind,
            )

    def _expand_any(self, act, path) -> Iterator[tuple[AST, dict[str, int]]]:
        doms = self._load_doms(act, path)
        if not act.quant:
            yield self._expand_all(act, doms), {}
        else:
            quant = [v for v, q in act.quant.items() if q == "any"]
            for combi in product(*(doms[v] for v in quant)):
                yield self._expand_all(act, doms), dict(zip(quant, combi))

    def _load_doms(self, act, path):
        doms = {}
        for var in act.search({"kind": "var"}):
            if var.index is not None:
                idx = var.index.left
                vfn = ".".join([*path, var.name])
                if idx == "self":
                    continue
                elif idx not in act.quant:
                    var.index.error(f"unquantified index {idx!r}")
                elif vfn not in self.shapes:
                    var.error(f"undeclared variable {var.name!r}")
                inc = self._binops[var.index.op](0, var.index.right)
                dom = set(range(inc, inc + self.shapes[vfn]))
                if idx not in doms:
                    doms[idx] = dom
                else:
                    doms[idx].intersection_update(dom)
            if var.locidx is not None:
                assert var.locrel == "inner"
                idx = var.locidx.left
                lfn = ".".join([*path, var.locname])
                if idx == "self":
                    continue
                elif idx not in act.quant:
                    var.locidx.error(f"unquantified index {idx!r}")
                elif lfn not in self.shapes:
                    var.error(f"undeclared location {var.name!r}")
                inc = self._binops[var.locidx.op](0, var.locidx.right)
                dom = set(range(inc, inc + self.shapes[lfn]))
                if idx not in doms:
                    doms[idx] = dom
                else:
                    doms[idx].intersection_update(dom)
        return doms

    def _expand_all(self, act, doms) -> AST:
        quant = [v for v, q in act.quant.items() if q == "all"]
        expand = {"guard": [], "assign": []}
        for attr, side in expand.items():
            for a in getattr(act, attr):
                new = [a]
                for v in quant:
                    if any(True for _ in a.search({"kind": "expr", "left": v})) or any(
                        True for _ in a.search({"kind": "expr", "right": v})
                    ):
                        new = [
                            x.sub({"kind": "expr", "left": v}, {"left": d}).sub(
                                {"kind": "expr", "right": v}, {"right": d}
                            )
                            for x in new
                            for d in doms[v]
                        ]
                side.extend(new)
        return act | expand

    def _load_var(self, var, path, idxvals) -> str:
        if var.index is None:
            name = var.name
        else:
            idx = self._load_idx(var.index, idxvals)
            name = f"{var.name}[{idx}]"
        if var.locrel is None:
            path = path + [name]
        elif var.locrel == "inner":
            if var.locidx is None:
                path = path + [var.locname, name]
            else:
                idx = self._load_idx(var.locidx, idxvals)
                path = path + [f"{var.locname}[{idx}]", name]
        elif var.locrel == "outer":
            path = path[:-1] + [name]
        return ".".join(path)

    def _load_idx(self, idx, idxvals):
        if isinstance(idx, int):
            return idx
        elif isinstance(idx, str):
            if (val := idxvals.get(idx)) is not None:
                return val
        elif idx.kind == "expr":
            if idx.op is None:
                return self._load_idx(idx.left, idxvals)
            else:
                return self._binops[idx.op](
                    self._load_idx(idx.left, idxvals),
                    self._load_idx(idx.right, idxvals),
                )

    def _load_expr(self, expr, path, idxvals) -> Expression:
        if isinstance(expr, int):
            return Expression(expr)
        elif expr.kind == "var":
            return Expression(0, {self._load_var(expr, path, idxvals): 1})
        elif expr.op is None:
            return self._load_expr(expr.left, path, idxvals)
        else:
            return self._binops[expr.op](
                self._load_expr(expr.left, path, idxvals),
                self._load_expr(expr.right, path, idxvals),
            )

    def _make_ticks(self):
        for clock, variables in self.clocks.items():
            name = f"{clock}.tick"
            guard = []
            assign = {}
            for var in variables:
                guard.append(Expression.v(var.name) < max(var.domain))
                assign[var.name] = Expression.v(var.name) + 1
            self.actions.append(
                Action(name, tuple(guard), assign, frozenset(), ActionKind.tick)
            )


class AST2Pattern(AST2Model):
    LarkParser = PatternParser
    ModelClass = Pattern

    @classmethod
    def check(cls, loc, locname=None, up={}, down={}, clocks=None):
        pass

    def _make_ticks(self):
        pass
