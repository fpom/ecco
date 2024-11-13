import operator

from typing import Optional, Mapping, Self, Any
from pathlib import Path

from rich.text import Text

from .parser import ModelParser, PatternParser
from ..record import Record, Printer
from .. import (
    Model as _Model,
    Variable as _Variable,
    Expression as _Expression,
    Action as _Action,
    ActionKind,
)

_mrr_styles = {
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
    def __txt__(self, styles={}):
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
    def __txt__(self, styles={}):
        prn = Printer(_mrr_styles | styles)
        with prn.expr:
            if (
                len(self.vars) == 1
                and (d := self._mod.v.get(var := list(self.vars)[0]))
                and d.domain == {0, 1}
            ):
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
    def __txt__(self, styles={}):
        prn = Printer(_mrr_styles | styles)
        with prn.action:
            if self.tags:
                with prn.tag:
                    tags = [prn("["), *(prn(t) for t in self.tags), prn("]")]
            else:
                tags = []
            with prn.assign:
                assign = []
                for v, x in self.assign.items():
                    if not x.vars and (d := self._mod.v.get(v)) and d.domain == {0, 1}:
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
                        assign.append(
                            prn[prn(v, "var"), prn("=", "op"), x.__txt__(styles)]
                        )
            return prn[
                *tags,
                " " if self.tags else "",
                (prn / ", ")(g.__txt__(styles) for g in self.guard),
                prn(">>", "op"),
                (prn / ", ")(assign),
                "  ",
                prn(f"# {self.name}", "comment"),
            ]


class Model(_Model):
    def __txt__(self, styles={}):
        prn = Printer(_mrr_styles | styles)
        lines = [
            prn(f"# loaded from {str(self.path)!r}", "comment"),
            prn("variables:", "header"),
        ]
        lines.extend(prn["  ", v.__txt__(styles)] for v in self.variables)
        for attr in ["constraints", "rules"]:
            if actions := getattr(self, attr):
                lines.append(prn(f"{attr}:", "header"))
                lines.extend(prn["  ", a.__txt__(styles)] for a in actions)
        return (prn / "\n")(lines)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], **impl: type[Record]) -> Self:
        impl = {
            "Model": Model,
            "Action": Action,
            "Expression": Expression,
            "Variable": Variable,
        } | impl
        return super().from_dict(data, **impl)

    @classmethod
    def from_spec(
        cls, spec, path: Optional[str | Path] = None, source: Optional[str] = None
    ) -> "Model":
        return Spec2Model.from_spec(spec, path, source)

    @classmethod
    def from_source(
        cls, source: str, *defs: str, path: Optional[str | Path] = None
    ) -> "Model":
        return Spec2Model.from_source(source, *defs, path=path)

    @classmethod
    def from_path(cls, path: str | Path, *defs: str) -> "Model":
        return Spec2Model.from_path(path, *defs)


# class Spec2Model:
#     # FIXME: this class should be replaced by direct parsing
#     def __init__(self) -> None:
#         self.variables: list[Variable] = []
#         self.actions: list[Action] = []
#         self.clocks: dict[str, list[Variable]] = {}
#
#     @classmethod
#     def from_spec(
#         cls, spec, path: Optional[str | Path] = None, source: Optional[str] = None
#     ) -> Model:
#         loader = cls()
#         loader._load_Location(spec, [], None)
#         loader._make_ticks()
#         return Model(
#             tuple(loader.variables),
#             tuple(loader.actions),
#             str(path or "<string>"),
#             source,
#         )
#
#     @classmethod
#     def from_source(
#         cls, source: str, *defs: str, path: Optional[str | Path] = None
#     ) -> Model:
#         return cls.from_spec(parse_src(source, *defs), path, source)
#
#     @classmethod
#     def from_path(cls, path: str | Path, *defs: str) -> Model:
#         return cls.from_source(Path(path).read_text(), *defs, path=path)
#
#     def _load_Location(self, loc, path, sidx):
#         # load clocks
#         self.clocks.update({c: [] for c in loc.clocks})
#         # load local variables
#         for var in loc.variables:
#             if var.size is not None:
#                 for i in range(var.size):
#                     self.variables.append(
#                         self._load_VarDecl(var, path + [f"{var.name}[{i}]"], idx=i)
#                     )
#             else:
#                 self.variables.append(self._load_VarDecl(var, path + [var.name]))
#             if var.clock is not None:
#                 self.clocks[var.clock].extend(self.variables[-(var.size or 1) :])
#         # load local actions
#         for i, act in enumerate(loc.constraints, start=1):
#             self.actions.extend(
#                 self._load_ActDecl(act, path + [f"C{i}"], ActionKind.cons, sidx)
#             )
#         for i, act in enumerate(loc.rules, start=1):
#             self.actions.extend(
#                 self._load_ActDecl(act, path + [f"R{i}"], ActionKind.rule, sidx)
#             )
#         # load sub-locations
#         for sub in loc.locations:
#             assert sub.name is not None
#             if sub.size is not None:
#                 for i in range(sub.size):
#                     self._load_Location(sub, path + [f"{sub.name}[{i}]"], i)
#             else:
#                 self._load_Location(sub, path + [sub.name], None)
#
#     def _load_VarDecl(self, var: VarDecl, path: list[str], idx: Optional[int] = None):
#         if var.init is None:
#             init = {-1}
#         elif idx is not None and isinstance(var.init, tuple):
#             init = var.init[idx]
#         else:
#             assert isinstance(var.init, set)
#             init = var.init
#         return Variable(
#             name=".".join(path),
#             domain=frozenset(var.domain if var.clock is None else var.domain | {-1}),
#             init=frozenset(init),
#             comment=var.description,
#             clock=var.clock,
#         )
#
#     def _load_ActDecl(
#         self, act: ActDecl, path: list[str], kind: ActionKind, sidx: Optional[int]
#     ):
#         for new, vars in act.expand_any():
#             yield self._load_ActDecl_expanded(new, path, kind, sidx, vars)
#
#     def _load_ActDecl_expanded(
#         self,
#         new: ActDecl,
#         path: list[str],
#         kind: ActionKind,
#         sidx: Optional[int],
#         vars: dict[str, int],
#     ):
#         guard = []
#         assign: dict[str, Expression] = {}
#         for g in new.left:
#             guard.append(self._load_expr(g, path[:-1], sidx))
#         for ass in new.right:
#             t = self._load_VarUse(ass.target, path[:-1], sidx)
#             assign[t] = self._load_expr(ass.value, path[:-1], sidx)
#         if vars:
#             index = f"[{','.join(f'{k}={v}' for k, v in vars.items())}]"
#             return Action(
#                 ".".join(path) + index,
#                 tuple(guard),
#                 assign,
#                 frozenset(new.tags),
#                 kind,
#             )
#         else:
#             return Action(
#                 ".".join(path), tuple(guard), assign, frozenset(new.tags), kind
#             )
#
#     _ops = {
#         "+": operator.add,
#         "-": operator.sub,
#         "*": operator.mul,
#         "<": operator.lt,
#         ">": operator.gt,
#         "<=": operator.le,
#         ">=": operator.ge,
#         "==": operator.eq,
#         "!=": operator.ne,
#     }
#
#     def _load_expr(self, expr, path, sidx) -> Expression:
#         if isinstance(expr, BinOp):
#             le = self._load_expr(expr.left, path, sidx)
#             re = self._load_expr(expr.right, path, sidx)
#             return self._ops[expr.op](le, re)
#         elif isinstance(expr, VarUse):
#             return Expression.v(self._load_VarUse(expr, path, sidx))
#         elif expr is None:
#             return Expression(-1)
#         else:
#             return Expression(int(expr))
#
#     def _load_VarUse(self, expr, path, sidx):
#         if expr.index is None:
#             name = expr.name
#         else:
#             name = f"{expr.name}[{sidx if expr.index == 'self' else expr.index}]"
#         if expr.locrel is None:
#             path = path + [name]
#         elif expr.locrel == "inner":
#             if expr.locidx is None:
#                 path = path + [expr.locname, name]
#             else:
#                 path = path + [f"{expr.locname}[{expr.locidx}]", name]
#         elif expr.locrel == "outer":
#             path = path[:-1] + [name]
#         return ".".join(path)
#
#     def _make_ticks(self):
#         for clock, variables in self.clocks.items():
#             name = f"{clock}.tick"
#             guard = []
#             assign = {}
#             for var in variables:
#                 guard.append(Expression.v(var.name) < max(var.domain))
#                 assign[var.name] = Expression.v(var.name) + 1
#             self.actions.append(
#                 Action(name, tuple(guard), assign, frozenset(), ActionKind.tick)
#             )
#
#
# class Patt2Model(Spec2Model):
#     @classmethod
#     def from_source(cls, source: str, *defs: str, path: Optional[str | Path] = None):
#         spec = Location(
#             line=1,
#             variables=(),
#             constraints=(),
#             rules=parse_src(source, *defs, pattern=True),
#             locations=(),
#             name=None,
#             size=None,
#             clocks={},
#         )
#         return cls.from_spec(spec, path, source)
#
#     def _load_ActDecl(
#         self, act: ActDecl, path: list[str], kind: ActionKind, sidx: Optional[int]
#     ):
#         yield self._load_ActDecl_expanded(act, path, kind, sidx, {})
