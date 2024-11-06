from dataclasses import dataclass
from functools import cache, cached_property, reduce
from operator import or_
from typing import Self, Optional, Mapping

from . import Model, Action, Variable, ActionKind
from . import Expression as _Expression

import z3


@cache
def zvars(zexpr) -> frozenset[str]:
    match (decl := zexpr.decl()).kind():
        case z3.Z3_OP_ANUM:
            return frozenset()
        case z3.Z3_OP_UNINTERPRETED:
            return frozenset([decl.name()])
        case _:
            return reduce(or_, (zvars(c) for c in zexpr.children()), frozenset())


class _zvdict(dict):
    def __missing__(self, key):
        v = self[key] = z3.Int(key)
        return v


@dataclass(frozen=True)
class Expression(_Expression):
    zexpr: z3.ExprRef
    src: Optional[str] = None

    @classmethod
    def compile(cls, src):
        zexpr = eval(src, _zvdict())
        if isinstance(zexpr, int):
            zexpr = z3.IntVal(zexpr)
        elif isinstance(zexpr, bool):
            zexpr = z3.BoolVal(zexpr)
        return cls(zexpr, src)

    @cached_property
    def vars(self) -> frozenset[str]:
        return zvars(self.zexpr)

    @cached_property
    def neg(self) -> Self:
        zexpr = z3.Not(self.zexpr)
        assert isinstance(zexpr, z3.BoolRef)
        return self.__class__(zexpr)

    @cached_property
    def true(self) -> bool:
        solver = z3.Solver()
        solver.add(z3.ForAll([z3.Int(v) for v in self.vars], self.zexpr))
        return solver.check() == z3.sat

    def sub(self, assign: Mapping[str, Self | int]) -> Self:
        zexpr = z3.substitute(
            self.zexpr,
            *(
                (z3.Int(v), z3.IntVal(x) if isinstance(x, int) else x.zexpr)
                for v, x in assign.items()
            ),
        )
        return self.__class__(zexpr)

    @classmethod
    def solver(cls, dom, *exprs: Self) -> z3.Solver:
        solver = z3.Solver()
        solver.add(*(z3.Or(*(z3.Int(v) == n for n in d)) for v, d in dom.items()))
        solver.add(*(x.zexpr for x in exprs))
        return solver

    @classmethod
    def sat(cls, dom: Mapping[str, frozenset[int]], *exprs: Self) -> bool:
        return cls.solver(dom, *exprs).check() == z3.sat

    @cached_property
    def op(self) -> str:
        name = self.zexpr.decl().name()
        if name == "distinct":
            return "!="
        else:
            return name

    @cached_property
    def bool(self) -> bool:
        return bool(z3.simplify(self.zexpr))

    @cached_property
    def int(self) -> int:
        return z3.simplify(self.zexpr).as_long()

    def __str__(self) -> str:
        return str(self.zexpr)


__all__ = ["Model", "Action", "Expression", "Variable", "ActionKind"]
