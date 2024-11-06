import re

from dataclasses import dataclass
from functools import cached_property
from typing import Self, Iterable, Mapping

from unidecode import unidecode

from . import Model, Action, Variable, ActionKind, Expression
from ..lrr.st import Parser


@dataclass(frozen=True, eq=True, order=True)
class Expr(Expression):
    sign: bool
    name: str | None = None

    def __str__(self):
        sign = "+" if self.sign else "-"
        if self.name is None:
            return sign
        else:
            return f"{self.name}{sign}"

    @cached_property
    def vars(self) -> frozenset[str]:
        if self.name is None:
            return frozenset()
        else:
            return frozenset({self.name})

    def sub(self, assign: Mapping[str, Self]) -> Self:
        if self.name is None or self.name not in assign:
            return self
        else:
            return self.__class__(assign[self.name].sign == self.sign)

    @cached_property
    def neg(self):
        return self.__class__(not self.sign, self.name)

    @cached_property
    def true(self):
        return self.name is None and self.sign

    @classmethod
    def sat(
        cls,
        dom: Mapping[str, frozenset],
        all: Iterable[Self] = (),
        iff: Iterable[tuple[Self, Self]] = (),
    ) -> bool:
        iff_mem: dict[Self, Self] = {}
        for s, t in iff:
            if s in iff_mem:
                if iff_mem[s] != t:
                    return False
            else:
                iff_mem[s] = t
                iff_mem[s.neg] = t.neg
        all_mem: set[Self] = set()
        for x in all:
            if x.name is None:
                if not x.sign:
                    return False
            elif x.neg in all_mem:
                return False
            else:
                all_mem.add(x)
        return True


class Mod(Model):
    @classmethod
    def from_dict(cls, data, **impl):
        impl.setdefault("Model", cls)
        impl.setdefault("Expression", Expr)
        return super().from_dict(data, **impl)


def print_model(model):
    if model.variables:
        print("variables:")
        for v in model.variables:
            if v.init == v.domain:
                init = "*"
            elif v.init == {0}:
                init = "-"
            else:
                init = "+"
            print(f"  {v.name}{init}: {v.comment}")
    for k, d in [("constraints", model.c), ("rules", model.r)]:
        if d:
            print(f"{k}:")
        for a in d.values():
            print(
                f"  [{','.join(a.tags)}]" if a.tags else " ",
                ", ".join(str(g) for g in a.guard),
                ">>",
                ", ".join(str(Expr(e.sign, v)) for v, e in a.assign.items()),
                f" # {a.name}",
            )


def to_model(spec):
    variables = {}
    for s in spec.meta:
        variables[s.state.name] = Variable(
            s.state.name,
            frozenset({0, 2}),
            frozenset({int(s.state.sign)}),
            f"{s.kind}: {s.description}",
        )
    actions = []
    for kind, acts in (
        (ActionKind.cons, spec.constraints),
        (ActionKind.rule, spec.rules),
    ):
        tpl = f"{kind[0].upper()}{{}}"
        for a in acts:
            actions.append(
                Action(
                    tpl.format(a.num),
                    tuple(Expr(s.sign, s.name) for s in a.left),
                    {s.name: Expr(s.sign) for s in a.right},
                    frozenset(),
                    kind,
                )
            )
    return Mod(tuple(variables.values()), tuple(actions))


_const = re.compile(r"^\s*constraints\s*:\s*(#.*)?$")
_rules = re.compile(r"^\s*rules\s*:\s*(#.*)?$")


def load_model(path):
    actions = []
    kind = ActionKind.rule
    with open(path, encoding="utf-8", errors="replace") as f:
        src = unidecode(f.read())
    top_parser = Parser(path, src)
    parse = top_parser._parser.parse
    for line in src.splitlines():
        if _rules.match(line):
            kind = ActionKind.rule
        elif _const.match(line):
            kind = ActionKind.cons
        elif ln := line.split("#", 1)[0].strip():
            left, right, _ = parse(ln + "\n", "rule", semantics=top_parser)
            actions.append(
                Action(
                    f"R{len(actions)+1}",
                    tuple(Expr(s.sign, s.name) for s in left),
                    {s.name: Expr(s.sign) for s in right},
                    frozenset(),
                    kind,
                )
            )
    variables = {Variable(v, {0, 1}, {0}, "...") for a in actions for v in a.vars}
    return Mod(tuple(variables), tuple(actions))
