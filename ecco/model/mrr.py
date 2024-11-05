import operator

from typing import Optional
from pathlib import Path

import z3

from .z3model import Model, Action, Expression, Variable, ActionKind
from ..mrr.parser import parse_src, VarDecl, Action as ActDecl, BinOp, VarUse, Location


class Spec2Model:
    def __init__(self) -> None:
        self.variables: list[Variable] = []
        self.actions: list[Action] = []
        self.clocks: dict[str, list[Variable]] = {}

    @classmethod
    def _load_spec(
        cls, spec, path: Optional[str | Path] = None, source: Optional[str] = None
    ):
        loader = cls()
        loader._load_Location(spec, [], None)
        loader._make_ticks()
        return Model(
            tuple(loader.variables),
            tuple(loader.actions),
            str(path or "<string>"),
            source,
        )

    @classmethod
    def from_source(cls, source: str, *defs: str, path: Optional[str | Path] = None):
        return cls._load_spec(parse_src(source, *defs), path, source)

    @classmethod
    def from_path(cls, path: str | Path, *defs: str) -> Model:
        return cls.from_source(Path(path).read_text(), *defs, path=path)

    def _load_Location(self, loc, path, sidx):
        # load clocks
        self.clocks.update({c: [] for c in loc.clocks})
        # load local variables
        for var in loc.variables:
            if var.size is not None:
                for i in range(var.size):
                    self.variables.append(
                        self._load_VarDecl(var, path + [f"{var.name}[{i}]"], idx=i)
                    )
            else:
                self.variables.append(self._load_VarDecl(var, path + [var.name]))
            if var.clock is not None:
                self.clocks[var.clock].extend(self.variables[-(var.size or 1) :])
        # load local actions
        for i, act in enumerate(loc.constraints, start=1):
            self.actions.extend(
                self._load_ActDecl(act, path + [f"C{i}"], ActionKind.cons, sidx)
            )
        for i, act in enumerate(loc.rules, start=1):
            self.actions.extend(
                self._load_ActDecl(act, path + [f"R{i}"], ActionKind.rule, sidx)
            )
        # load sub-locations
        for sub in loc.locations:
            assert sub.name is not None
            if sub.size is not None:
                for i in range(sub.size):
                    self._load_Location(sub, path + [f"{sub.name}[{i}]"], i)
            else:
                self._load_Location(sub, path + [sub.name], None)

    def _load_VarDecl(self, var: VarDecl, path: list[str], idx: Optional[int] = None):
        if var.init is None:
            init = {-1}
        elif idx is not None and isinstance(var.init, tuple):
            init = var.init[idx]
        else:
            assert isinstance(var.init, set)
            init = var.init
        return Variable(
            name=".".join(path),
            domain=frozenset(var.domain if var.clock is None else var.domain | {-1}),
            init=frozenset(init),
            comment=var.description,
        )

    def _load_ActDecl(
        self, act: ActDecl, path: list[str], kind: ActionKind, sidx: Optional[int]
    ):
        for new, vars in act.expand_any():
            yield self._load_ActDecl_expanded(new, path, kind, sidx, vars)

    def _load_ActDecl_expanded(
        self,
        new: ActDecl,
        path: list[str],
        kind: ActionKind,
        sidx: Optional[int],
        vars: dict[str, int],
    ):
        guard = []
        assign: dict[str, Expression] = {}
        for g in new.left:
            z, v = self._load_expr(g, path[:-1], sidx)
            guard.append(Expression(z, frozenset(v)))
        for ass in new.right:
            t = self._load_VarUse(ass.target, path[:-1], sidx)
            z, v = self._load_expr(ass.value, path[:-1], sidx)
            assign[t] = Expression(z, frozenset(v))
        if vars:
            index = f"[{','.join(f'{k}={v}' for k, v in vars.items())}]"
            return Action(
                ".".join(path) + index,
                tuple(guard),
                assign,
                frozenset(new.tags),
                kind,
            )
        else:
            return Action(
                ".".join(path), tuple(guard), assign, frozenset(new.tags), kind
            )

    _ops = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "<": operator.lt,
        ">": operator.gt,
        "<=": operator.le,
        ">=": operator.ge,
        "==": operator.eq,
        "!=": operator.ne,
    }

    def _load_expr(self, expr, path, sidx) -> tuple[z3.ExprRef, set[str]]:
        if isinstance(expr, BinOp):
            le, lv = self._load_expr(expr.left, path, sidx)
            re, rv = self._load_expr(expr.right, path, sidx)
            return self._ops[expr.op](le, re), lv | rv
        elif isinstance(expr, VarUse):
            v = self._load_VarUse(expr, path, sidx)
            return z3.Int(v), {v}
        elif expr is None:
            return z3.IntVal(-1), set()
        else:
            assert isinstance(expr, int)
            return z3.IntVal(expr), set()

    def _load_VarUse(self, expr, path, sidx):
        if expr.index is None:
            name = expr.name
        else:
            name = f"{expr.name}[{sidx if expr.index == 'self' else expr.index}]"
        if expr.locrel is None:
            path = path + [name]
        elif expr.locrel == "inner":
            if expr.locidx is None:
                path = path + [expr.locname, name]
            else:
                path = path + [f"{expr.locname}[{expr.locidx}]", name]
        elif expr.locrel == "outer":
            path = path[:-1] + [name]
        return ".".join(path)

    def _make_ticks(self):
        for clock, variables in self.clocks.items():
            name = f"{clock}.tick"
            guard = []
            assign = {}
            for var in variables:
                z = z3.Int(var.name)
                g = z < max(var.domain)
                guard.append(Expression(g, frozenset({var.name})))
                assign[var.name] = z3.If(z == -1, z3.IntVal(-1), z + 1)
            self.actions.append(
                Action(name, tuple(guard), assign, frozenset(), ActionKind.tick)
            )


class Patt2Model(Spec2Model):
    @classmethod
    def from_source(cls, source: str, *defs: str, path: Optional[str | Path] = None):
        spec = Location(
            line=1,
            variables=(),
            constraints=(),
            rules=parse_src(source, *defs, pattern=True),
            locations=(),
            name=None,
            size=None,
            clocks={},
        )
        return cls._load_spec(spec, path, source)

    def _load_ActDecl(
        self, act: ActDecl, path: list[str], kind: ActionKind, sidx: Optional[int]
    ):
        yield self._load_ActDecl_expanded(act, path, kind, sidx, {})
