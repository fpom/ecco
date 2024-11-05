import re

from dataclasses import dataclass, fields
from enum import StrEnum
from typing import (
    Iterator,
    Any,
    Self,
    Union,
    Optional,
    get_origin,
    get_args,
)
from abc import ABC, abstractmethod
from inspect import isclass
from functools import reduce, cache, cached_property
from operator import or_

import pandas as pd

from frozendict import frozendict


_re_digits = re.compile(r"(\d+)")


def _human_sortkey(text):
    """Key to sort strings in natural order."""
    return tuple(int(m) if m.isdigit() else m.lower() for m in _re_digits.split(text))


class ActionKind(StrEnum):
    """The different kinds of actions.

    Attributes:
        cons: A constraint (action with a higher priority).
        rule: A rule (action with a lower-priority).
        tick: A clock tick (automatically added action with rule-priority to model clocks).
    """

    cons = "cons"
    rule = "rule"
    tick = "tick"


class InlineError(Exception):
    pass


@dataclass(frozen=True, eq=True, order=True)
class Record:
    """Base class for model elements."""

    @classmethod
    def _fields(cls) -> Iterator[tuple[str, bool, type | None, type]]:
        """Enumerate all the fields of a `Record`.

        Yields:
            4-tuples whose items are:

            - `name`: the name of the field
            - `optional`: `True` if the field is optional, `False` otherwise
            - `type`: if not `None`, the field is a container of class `type`
            - `cls`: the type of the field
        """
        for field in fields(cls):
            opt = False
            typ = field.type
            if isclass(typ):
                yield field.name, opt, None, typ
            else:
                if get_origin(typ) is Union:
                    opt = True
                    typ = get_args(typ)[0]
                if (org := get_origin(typ)) in (tuple, frozenset, dict):
                    yield field.name, opt, org, get_args(typ)[0]
                else:
                    yield field.name, opt, None, object

    def __post_init__(self):
        for name, _, typ, _ in self._fields():
            if (val := getattr(self, name)) is None:
                pass
            elif typ is dict:
                self.__dict__[name] = frozendict(val)
            elif typ in (tuple, frozenset) and not isinstance(val, typ):
                self.__dict__[name] = typ(val)

    @classmethod
    def from_dict(cls, data: dict[str, Any], **impl: type["Record"]) -> Self:
        """Build a new `Record` instance from its `dict` serialiation.

        Args:
            data: A `dict` holding a previously serialized `Record`.
            impl: Concrete implementation of `Record` classes, in particular of `Expression`.

        Returns:
            A `Record` instance.

        Raises:
            ValueError: If some fields are missing in `data`.
        """
        args = {}
        for name, opt, cont, typ in cls._fields():
            val = data.get(name, None)
            if val is None:
                if not opt:
                    raise ValueError(f"field '{cls.__name__}.{name}' cannot be empty")
                continue
            if issubclass(typ, Record):
                typ = impl.get(typ.__name__, typ)
                if cont is None:
                    args[name] = typ.from_dict(val, **impl)
                else:
                    args[name] = cont(typ.from_dict(v, **impl) for v in val)
            else:
                if cont is None:
                    args[name] = val
                else:
                    args[name] = cont(val)
        return cls(**args)

    def to_dict(self) -> dict[str, object]:
        """Serialize a `Record` into a `dict`.

        Returns:
            A `dict` holding all the data from the `Record`.
        """
        d = {}
        for name, opt, cont, typ in self._fields():
            val = getattr(self, name)
            if val is None and opt:
                continue
            if not issubclass(typ, Record):
                if cont is None:
                    d[name] = val
                else:
                    d[name] = list(val)
            elif cont is None:
                d[name] = val.to_dict()
            else:
                d[name] = [v.to_dict() for v in val]
        return d

    def copy(self, **repl) -> Self:
        """Copy a `Record`, replacing some of its fields.

        Args:
            repl: Fields to be replaced, given by name and value.

        Returns:
            A new `Record` instance.
        """
        args = {}
        for name, _, cont, typ in self._fields():
            val = repl.get(name, getattr(self, name, None))
            if issubclass(typ, Record):
                if cont is None:
                    args[name] = val.copy()
                else:
                    args[name] = cont(v.copy() for v in val)
            else:
                if cont is None:
                    args[name] = val
                else:
                    args[name] = cont(val)
        return self.__class__(**args)


@dataclass(frozen=True, eq=True, order=True)
class Variable(Record):
    """A variable of the model.

    Attributes:
        name: The name of the variable.
        domain: The set of values of its domain.
        init: The set of initial values.
        comment: Textual description of the variable.
    """

    name: str
    domain: frozenset[int]
    init: frozenset[int]
    comment: str


@dataclass(frozen=True, eq=True, order=True)
class Expression(Record, ABC):
    """Base class for expressions, with useless implementation."""

    @cached_property
    @abstractmethod
    def vars(self) -> frozenset[str]:
        """Names of all the variables involved in the expression."""

    @abstractmethod
    def sub(self, assign: dict[str, Self]) -> Self: ...

    @cached_property
    @abstractmethod
    def neg(self) -> Self: ...

    @cached_property
    @abstractmethod
    def true(self) -> bool: ...

    @classmethod
    @abstractmethod
    def sat(cls, dom: dict[str, frozenset[int]], *exprs: Self) -> bool: ...

    @cached_property
    @abstractmethod
    def op(self) -> str: ...


@dataclass(frozen=True, eq=True, order=True)
class Action(Record):
    """An action of the model.

    Attributes:
        name: A model-wise unique name for the action.
        guard: Expressions that must all evaluate to `True` to allow firing on a given state.
        assign: Map target variables to expressions they are assigned to onto firing.
        tags: Arbitrary tags that decorate the action.
        kind: Kind of action (constraint, rule, or clock tick).
    """

    name: str
    guard: tuple[Expression, ...]
    assign: dict[str, Expression]
    tags: frozenset[str]
    kind: ActionKind

    def __repr__(self):
        return f"<Action {self.name!r}>"

    def inline(self, cons: Self, mod: "Model"):
        if self.kind != ActionKind.rule:
            raise InlineError("only rule can inline constraints")
        if cons.kind != ActionKind.cons:
            raise InlineError("only constraints can be inlined")
        dom = {x: mod.v[x].domain for x in cons.test_vars | self.vars}
        new = {n for g in cons.guard if not (n := g.sub(self.assign)).true}
        self_guard = set(self.guard)
        if mod.expr_cls.sat(dom, *(guard := new | self_guard)):
            yield self.__class__(
                f"{self.name}+{cons.name}",
                tuple(guard),
                self.assign | cons.assign,
                self.tags | cons.tags,
                self.kind,
            )
        for i, n in enumerate(new, start=1):
            if n.true:
                continue
            if mod.expr_cls.sat(
                dom, *(guard := (self_guard | (set() if (m := n.neg).true else {m})))
            ):
                yield self.__class__(
                    f"{self.name}/{cons.name}:{i}",
                    tuple(guard),
                    self.assign,
                    self.tags,
                    self.kind,
                )

    @cached_property
    def test_vars(self) -> frozenset[str]:
        return reduce(or_, (g.vars for g in self.guard), frozenset())

    @cached_property
    def read_vars(self) -> frozenset[str]:
        return reduce(or_, (a.vars for a in self.assign.values()), frozenset())

    @cached_property
    def get_vars(self) -> frozenset[str]:
        return self.test_vars | self.read_vars

    @cached_property
    def set_vars(self) -> frozenset[str]:
        return frozenset(self.assign)

    @cached_property
    def vars(self) -> frozenset[str]:
        """Names of all the variables involved in the action."""
        return self.get_vars | self.set_vars


@dataclass(frozen=True, eq=True, order=True)
class Model(Record):
    """A complete model.

    Attributes:
        variables: The variables of the model.
        actions: The actions of the model.
        path: The name of the source file from which the model was built.
        source: The source code from which the model was built.
    """

    variables: tuple[Variable, ...]
    actions: tuple[Action, ...]
    path: Optional[str] = None
    source: Optional[str] = None

    @cached_property
    def vars(self) -> frozenset[str]:
        return frozenset(v.name for v in self.variables)

    @cached_property
    def constraints(self) -> tuple[Action, ...]:
        """The constrains in the model."""
        return tuple(a for a in self.actions if a.kind == ActionKind.cons)

    @cached_property
    def rules(self) -> tuple[Action, ...]:
        """The rules in the model."""
        return tuple(a for a in self.actions if a.kind == ActionKind.rule)

    @cached_property
    def ticks(self) -> tuple[Action, ...]:
        """The clock ticks in the model."""
        return tuple(a for a in self.actions if a.kind == ActionKind.tick)

    @cached_property
    def expr_cls(self) -> type[Expression]:
        return self.actions[0].guard[0].__class__

    @cached_property
    def v(self) -> dict[str, Variable]:
        return {v.name: v for v in self.variables}

    @cached_property
    def a(self) -> dict[str, Action]:
        return {a.name: a for a in self.actions}

    @cached_property
    def c(self) -> dict[str, Action]:
        return {a.name: a for a in self.actions if a.kind == ActionKind.cons}

    @cached_property
    def r(self) -> dict[str, Action]:
        return {a.name: a for a in self.actions if a.kind == ActionKind.rule}

    @cached_property
    def t(self) -> dict[str, Action]:
        return {a.name: a for a in self.actions if a.kind == ActionKind.tick}

    def _getarg_vars(self, variables):
        if variables is not None:
            _vars = list(set(variables) & self.vars)
        else:
            _vars = list(self.vars)
        if not _vars:
            raise ValueError(f"invalid argument {variables=}")
        _vars.sort(key=_human_sortkey)
        return _vars

    def _getarg_kind(self, kind):
        aks = set(ActionKind)
        if kind is None:
            _kind = aks
        elif isinstance(kind, (ActionKind, str)):
            _kind = {kind} & aks
        else:
            _kind = set(kind) & aks
        if not _kind:
            raise ValueError(f"invalid argument {kind=}")
        return _kind

    def charact(self, variables=None, kind=None):
        _vars = self._getarg_vars(variables)
        _kind = self._getarg_kind(kind)
        # counts occurrences
        vmap = {v.name: v for v in self.variables}
        vidx = {v: i for i, v in enumerate(_vars)}
        ta, ra, wa = [[0] * len(_vars) for _ in range(3)]
        for act in (a for a in self.actions if a.kind in _kind):
            for exp in act.guard:
                for var in exp.vars:
                    ta[vidx[var]] += 1
            for var, expr in act.assign.items():
                wa[vidx[var]] += 1
                for v in expr.vars:
                    ra[vidx[v]] += 1
        # build dataframe
        counts = pd.DataFrame(
            index=_vars,
            data={
                "init": [vmap[v].init for v in _vars],
                "test": ta,
                "read": ra,
                "set": wa,
            },
        )
        counts["get"] = counts["test"] + counts["read"]
        counts["use"] = counts["get"] + counts["set"]
        return counts

    def ecograph(
        self, variables=None, kind=None, read=True
    ) -> dict[tuple[str, str], set[str]]:
        _vars = set(self._getarg_vars(variables))
        _kind = self._getarg_kind(kind)
        edges: dict[tuple[str, str], set[str]] = {}
        for act in (a for a in self.actions if a.kind in _kind):
            for src in _vars & (act.get_vars if read else act.test_vars):
                for tgt in _vars & act.set_vars:
                    edges.setdefault((src, tgt), set())
                    edges[src, tgt].add(act.name)
        return edges

    def ecohyper(
        self, variables=None, kind=None, read=True, broad=False
    ) -> dict[str, tuple[set[str], set[Expression], dict[str, Expression]]]:
        _vars = set(self._getarg_vars(variables))
        _kind = self._getarg_kind(kind)
        edges = {}
        for act in (a for a in self.actions if a.kind in _kind):
            if src := _vars & (act.get_vars if read else act.test_vars):
                edges[act.name] = (
                    src,
                    {
                        expr
                        for expr in act.guard
                        if (expr.vars & _vars if broad else expr.vars <= _vars)
                    },
                    {
                        var: expr
                        for var, expr in act.assign.items()
                        if broad or var in _vars
                    },
                )
        return edges

    def inline(self, *constraints) -> Self:
        old, new = list(self.rules + self.ticks), []
        for c in constraints if constraints else self.constraints:
            for act in old:
                new.extend(act.inline(c, self))
            old, new = new, []
        return self.__class__(variables=self.variables, actions=tuple(old))

    def match(self, pattern) -> Iterator:
        # work around circular import
        from .match import Match

        yield from Match.match(pattern, self)
