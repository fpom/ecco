import re
import operator

from dataclasses import dataclass, fields, field
from enum import StrEnum
from typing import (
    Iterator,
    Any,
    Self,
    Union,
    Optional,
    Mapping,
    Iterable,
    get_origin,
    get_args,
)
from collections import defaultdict
from inspect import isclass
from functools import reduce, cached_property, cache

import pandas as pd
import sympy as sp
import z3

from frozendict import frozendict

_re_digits = re.compile(r"(\d+)")


class tested_cached_property(cached_property):
    def __init__(self, func):
        super().__init__(func)
        self.__module__ = func.__module__


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
                if (org := get_origin(typ)) in (tuple, frozenset, dict, Mapping):
                    yield field.name, opt, org, get_args(typ)[0]
                else:
                    yield field.name, opt, None, object

    def __post_init__(self):
        for name, _, typ, _ in self._fields():
            if (val := getattr(self, name)) is None:
                pass
            elif isinstance(val, dict):
                self.__dict__[name] = frozendict(val)
            elif typ in (tuple, frozenset) and not isinstance(val, typ):
                self.__dict__[name] = typ(val)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], **impl: type["Record"]) -> Self:
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
    clock: Optional[str] = None


_SPOPS = {
    None: sp.Add,
    "<": sp.StrictLessThan,
    "<=": sp.LessThan,
    "==": sp.Eq,
    "!=": sp.Ne,
    ">": sp.StrictGreaterThan,
    ">=": sp.GreaterThan,
}

_Z3OPS = {
    None: z3.Sum,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.le,
}

_PYOPS = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.le,
}
_NEGOP = {
    "<": ">=",
    "<=": ">",
    "==": "!=",
    "!=": "==",
    ">": "<=",
    ">=": "<",
}


@dataclass(frozen=True)
class Expression(Record):
    """`bool`- or `int`-valuated expressions.

    `int`-valuated expressions are a linear combination of variables, plus a constant.
    `bool`-valuated expressions are `int`-valuated expressions compared to zero.

    Attributes:
        const: Constant `int` value in the combination.
        coeffs: Maps each variable name to its coefficient in the combination.
        domains: Provide the domain of each variable in `coeffs`.
        op: if not `None`, comparator of the combination to zero.
    """

    const: int
    coeffs: Mapping[str, int] = field(default_factory=dict)
    op: Optional[str] = None

    def __post_init__(self):
        self.__dict__["coeffs"] = frozendict(
            {k: v for k, v in self.coeffs.items() if v != 0}
        )
        if self.op not in (None, "<", "<=", "==", "!=", ">=", ">"):
            raise ValueError(f"invalid operator {self.op!r}")

    def __str__(self) -> str:
        """
        >>> print(Expression(3, {'a': 1, 'b': -1, 'c': 2, 'd': -2}))
        a-b+2c-2d+3
        >>> print(Expression(3, {'a': 1, 'b': -1, 'c': 2, 'd': -2}, ">"))
        a-b+2c-2d > -3
        >>> print(Expression(3, {}, ">"))
        3 > 0
        """
        s = []
        for i, (v, c) in enumerate(self.coeffs.items()):
            if i and c > 0:
                s.append("+")
            if c == 1:
                s.append(v)
            elif c == -1:
                s.append(f"-{v}")
            else:
                s.append(f"{c}{v}")
        s = "".join(s)
        if self.op is None:
            if self.const > 0:
                return f"{s}+{self.const}"
            elif self.const < 0:
                return f"{s}{self.const}"
            elif not s:
                return f"{self.const}"
            else:
                return s
        elif not s:
            return f"{self.const} {self.op} 0"
        else:
            return f"{s} {self.op} {-self.const}"

    @classmethod
    def v(cls, var: str):
        """Create an integer expression consisting of only a variable.

        >>> print(Expression.v("t[0].a") > 0)
        t[0].a > 0
        """
        return cls(0, {var: 1})

    @classmethod
    def c(cls, expr):
        """Create an expression from a Python expression.

        **Warning:** this method uses `eval` and should not be called on untrusted data.
        Doing so, would be a major security risk as it could expose to the execution of arbitrary Python code, including malicious code.

        >>> print(Expression.c("a + 2*b - c == 4"))
        a+2b-c == 4
        """

        class vd(dict):
            def __missing__(self, key):
                return cls.v(key)

        return eval(expr, vd())

    @tested_cached_property
    def vars(self) -> frozenset[str]:
        """Names of all the variables involved in the expression.

        >>> Expression.c("a + 2*b + c - a == 4").vars == {"b", "c"}
        True
        """
        return frozenset(self.coeffs)

    @tested_cached_property
    def is_bool(self) -> bool:
        """`True` for a Boolean expression, `False` otherwise.

        >>> Expression.c("a + 3").is_bool
        False
        >>> Expression.c("a + 3 > 0").is_bool
        True
        """
        return self.op is not None

    @tested_cached_property
    def is_int(self) -> bool:
        """`True` for an integer expression, `False` otherwise.

        >>> Expression.c("a + 3").is_int
        True
        >>> Expression.c("a + 3 > 0").is_int
        False
        """
        return self.op is None

    @tested_cached_property
    def neg(self) -> Self:
        """Negate a Boolean expression.

        Raises:
            ValueError: When not a Boolean expression.

        >>> print(Expression.c("a > 0").neg)
        a <= 0
        >>> print(Expression.c("a >= 0").neg)
        a < 0
        >>> print(Expression.c("a < 0").neg)
        a >= 0
        >>> print(Expression.c("a <= 0").neg)
        a > 0
        >>> print(Expression.c("a == 0").neg)
        a != 0
        >>> print(Expression.c("a != 0").neg)
        a == 0
        >>> Expression.c("a + 1").neg
        Traceback (most recent call last):
        ...
        ValueError: not a Boolean expression
        """
        if self.op is None:
            raise ValueError("not a Boolean expression")
        return self.__class__(self.const, self.coeffs, _NEGOP[self.op])

    @tested_cached_property
    def true(self) -> bool:
        """`True` for a Boolean expression that always evals to `True`, `False` otherwise.

                Raises:
            ValueError: When not a Boolean expression.

        >>> Expression.c("a + a - 2*a == 0").true
        True
        >>> Expression.c("1 + a - 2*a == 0").true
        False
        >>> Expression(5, {}, ">").true  # 5 > 0
        True
        >>> Expression(5, {}, "==").true  # 5 == 0
        False
        >>> Expression.c("a + 1").true
        Traceback (most recent call last):
        ...
        ValueError: not a Boolean expression
        """
        if self.op is None:
            raise ValueError("not a Boolean expression")
        return _PYOPS[self.op](self.const, 0)

    @cache
    def __int__(self) -> int:
        """Get value of a constant-valuated integer expression.

        Raises:
            ValueError: When not a constant expression, or not an integer expression.

        >>> int(Expression.c("a + 2 - a"))
        2
        >>> int(Expression.c("a + 1"))
        Traceback (most recent call last):
        ...
        ValueError: not a constant expression
        >>> int(Expression.c("a > 0"))
        Traceback (most recent call last):
        ...
        ValueError: not an integer expression
        """
        if self.op is not None:
            raise ValueError("not an integer expression")
        elif self.coeffs:
            raise ValueError("not a constant expression")
        return self.const

    @cache
    def __bool__(self) -> bool:
        """Get value of a constant-valuated expression as a Boolean.

        Note that for an integer expression `expr`, this is equivalent to `bool(int(expr))`.

        Raises:
            ValueError: When not a constant expression.

        >>> bool(Expression.c("a + 2 - a > 0"))
        True
        >>> bool(Expression.c("a + 2 - a < 0"))
        False
        >>> bool(Expression(2))
        True
        >>> bool(Expression(0))
        False
        >>> bool(Expression.c("a + 1"))
        Traceback (most recent call last):
        ...
        ValueError: not a constant expression
        """
        if self.coeffs:
            raise ValueError("not a constant expression")
        elif self.op is None:
            return bool(self.const)
        else:
            return _PYOPS[self.op](self.const, 0)

    def __add__(self, other: Self | int) -> Self:
        """Add an integer expression with another or with an int.

        Raises:
            ValueError: When one operand is a Boolean expression.

        >>> print(Expression.v("a") + Expression.c("a + b + 2"))
        2a+b+2
        >>> print(Expression.v("a") + Expression.v("a") + Expression.v("b") + 2)
        2a+b+2
        >>> Expression.c("a > 0") + 2
        Traceback (most recent call last):
        ...
        ValueError: cannot add Boolean expression
        """
        if self.op is not None:
            raise ValueError("cannot add Boolean expression")
        if isinstance(other, int):
            return self.__class__(self.const + other, self.coeffs)
        if other.op is not None:
            raise ValueError("cannot add Boolean expression")
        coeffs = defaultdict(int, self.coeffs)
        for v, c in other.coeffs.items():
            coeffs[v] += c
        return self.__class__(self.const + other.const, coeffs)

    def __radd__(self, other: int) -> Self:
        """Handle addition with `int` at the left-hand-side.

        >>> print(2 + Expression.v("a"))
        a+2
        """
        return self.__add__(other)

    def __mul__(self, other: int) -> Self:
        """Multiply an integer expression by a constant.

        >>> print(Expression.c("a + 1") * 2)
        2a+2
        >>> print(Expression.c("a + 1") * 0)
        0
        >>> Expression.c("a > 0") * 2
        Traceback (most recent call last):
        ...
        ValueError: cannot multiply Boolean expression
        """
        if self.op is not None:
            raise ValueError("cannot multiply Boolean expression")
        m = int(other)
        return self.__class__(
            self.const * m, {v: c * m for v, c in self.coeffs.items()}
        )

    def __rmul__(self, other: int) -> Self:
        """Handle multiplication with `int` at the left-hand-side.

        >>> print(2 * Expression.v("a"))
        2a
        """
        return self.__mul__(other)

    def __sub__(self, other: Self | int) -> Self:
        """Subtract from an integer expression another expression or an int.

        Raises:
            ValueError: When one operand is a Boolean expression.

        >>> print(Expression.v("a") - Expression.c("a + b + 2"))
        -b-2
        >>> print(Expression.v("a") - Expression.v("a") - Expression.v("b") - 2)
        -b-2
        >>> Expression.c("a > 0") - 2
        Traceback (most recent call last):
        ...
        ValueError: cannot add Boolean expression
        """
        return self + (other * -1)

    def __rsub__(self, other: int) -> Self:
        """Handle subtraction with `int` at the left-hand-side.

        >>> print(2 - Expression.v("a"))
        -a+2
        """
        return (self * -1) + other

    def __gt__(self, other: Self | int) -> Self:  # type: ignore
        """Compare two integer expressions, resulting in a Boolean expression.

        Raises:
            ValueError: When one operand is a Boolean expression.

        >>> print(Expression.v("a") > 0)
        a > 0
        >>> print(Expression.v("a") > Expression.c("b + 1"))
        a-b > 1
        >>> print(Expression.v("a") > Expression.c("a - 1"))
        1 > 0
        >>> print(Expression.c("a > 0") > 0)
        Traceback (most recent call last):
        ...
        ValueError: cannot compare Boolean expression
        """
        if self.op is not None:
            raise ValueError("cannot compare Boolean expression")
        if isinstance(other, int):
            return self.__class__(self.const - int(other), self.coeffs, ">")
        else:
            return self - other > 0

    def __ge__(self, other: Self | int) -> Self:  # type: ignore
        """Compare two integer expressions, resulting in a Boolean expression.

        Raises:
            ValueError: When one operand is a Boolean expression.

        >>> print(Expression.v("a") >= 0)
        a >= 0
        >>> print(Expression.v("a") >= Expression.c("b + 1"))
        a-b >= 1
        >>> print(Expression.v("a") >= Expression.c("a - 1"))
        1 >= 0
        >>> print(Expression.c("a > 0") >= 0)
        Traceback (most recent call last):
        ...
        ValueError: cannot compare Boolean expression
        """
        if self.op is not None:
            raise ValueError("cannot compare Boolean expression")
        if isinstance(other, int):
            return self.__class__(self.const - int(other), self.coeffs, ">=")
        else:
            return self - other >= 0

    def __lt__(self, other: Self | int) -> Self:  # type: ignore
        """Compare two integer expressions, resulting in a Boolean expression.

        Raises:
            ValueError: When one operand is a Boolean expression.

        >>> print(Expression.v("a") < 0)
        a < 0
        >>> print(Expression.v("a") < Expression.c("b + 1"))
        a-b < 1
        >>> print(Expression.v("a") < Expression.c("a - 1"))
        1 < 0
        >>> print(Expression.c("a > 0") < 0)
        Traceback (most recent call last):
        ...
        ValueError: cannot compare Boolean expression
        """
        if self.op is not None:
            raise ValueError("cannot compare Boolean expression")
        if isinstance(other, int):
            return self.__class__(self.const - int(other), self.coeffs, "<")
        else:
            return self - other < 0

    def __le__(self, other: Self | int) -> Self:  # type: ignore
        """Compare two integer expressions, resulting in a Boolean expression.

        Raises:
            ValueError: When one operand is a Boolean expression.

        >>> print(Expression.v("a") <= 0)
        a <= 0
        >>> print(Expression.v("a") <= Expression.c("b + 1"))
        a-b <= 1
        >>> print(Expression.v("a") <= Expression.c("a - 1"))
        1 <= 0
        >>> print(Expression.c("a > 0") <= 0)
        Traceback (most recent call last):
        ...
        ValueError: cannot compare Boolean expression
        """
        if self.op is not None:
            raise ValueError("cannot compare Boolean expression")
        if isinstance(other, int):
            return self.__class__(self.const - int(other), self.coeffs, "<=")
        else:
            return self - other <= 0

    def __eq__(self, other: Self | int) -> Self:  # type: ignore
        """Compare two integer expressions, resulting in a Boolean expression.

        Raises:
            ValueError: When one operand is a Boolean expression.

        >>> print(Expression.v("a") == 0)
        a == 0
        >>> print(Expression.v("a") == Expression.c("b + 1"))
        a-b == 1
        >>> print(Expression.v("a") == Expression.c("a - 1"))
        1 == 0
        >>> print(Expression.c("a > 0") == 0)
        Traceback (most recent call last):
        ...
        ValueError: cannot compare Boolean expression
        """
        if self.op is not None:
            raise ValueError("cannot compare Boolean expression")
        if isinstance(other, int):
            return self.__class__(self.const - int(other), self.coeffs, "==")
        else:
            return self - other == 0

    def __ne__(self, other: Self | int) -> Self:  # type: ignore
        """Compare two integer expressions, resulting in a Boolean expression.

        Raises:
            ValueError: When one operand is a Boolean expression.

        >>> print(Expression.v("a") != 0)
        a != 0
        >>> print(Expression.v("a") != Expression.c("b + 1"))
        a-b != 1
        >>> print(Expression.v("a") != Expression.c("a - 1"))
        1 != 0
        >>> print(Expression.c("a > 0") != 0)
        Traceback (most recent call last):
        ...
        ValueError: cannot compare Boolean expression
        """
        if self.op is not None:
            raise ValueError("cannot compare Boolean expression")
        if isinstance(other, int):
            return self.__class__(self.const - int(other), self.coeffs, "!=")
        else:
            return self - other != 0

    def sub(self, vars: Mapping[str, Self | int] = {}, **more_vars: Self | int) -> Self:
        """Substitute variables by integer expressions or values.

        Args:
            vars: Mapping of variables to integer expressions of `int`.
            more_vars: More variables to substitute, but passed as keywords arguments.

        Raises:
            ValueError: When one replacement is a Boolean expression.

        >>> expr = Expression.c("a + 2*b + 3*c")
        >>> print(expr.sub(a=1, b=Expression.c("b+1")))
        2b+3c+3
        >>> print(expr.sub(a=expr, b=expr, c=expr))
        6a+12b+18c
        >>> print(expr.sub(a=1, b=Expression.c("b>0")))
        Traceback (most recent call last):
        ...
        ValueError: cannot substitute variable with a Boolean expression
        """
        more_vars.update(vars)
        const = self.const
        coeffs = defaultdict(int)
        for sv, sc in self.coeffs.items():
            if (a := more_vars.get(sv)) is None:
                coeffs[sv] += sc
            elif isinstance(a, int):
                const += sc * a
            else:
                if a.op is not None:
                    raise ValueError(
                        "cannot substitute variable with a Boolean expression"
                    )
                const += sc * a.const
                for av, ac in a.coeffs.items():
                    coeffs[av] += sc * ac
        return self.__class__(const, coeffs, self.op)

    @classmethod
    def solver(
        cls,
        dom: Mapping[str, Iterable[int]],
        *exprs: Self,
        **more_dom: Iterable[int],
    ) -> z3.Solver:
        """Build a `z3.Solver` instance and add domains and expressions to it.

        Each variable `x` in `dom` is added as a constraint `z3.Or(x==d, ...)` for each `d` in `dom`.
        Each expression in `exprs` is translated into a `z3` expression and added to the solver.

        Note that no check is made to ensure that all the variables involved in `exprs` have their domains defined in `dom`, which allows to solve systems with unbounded variables.

        Args:
            dom: Mapping of variables to their domains.
            exprs: Expressions to be added to the solver.
            more_dom: More domains but given as keyword arguments.

        Raises:
            ValueError: When an integer expression is passed.
        """
        if not all(x.is_bool for x in exprs):
            raise ValueError("only Boolean expressions can be solved")
        more_dom.update(dom)
        solver = z3.Solver()
        solver.add(*(z3.Or(*(z3.Int(v) == n for n in d)) for v, d in more_dom.items()))
        solver.add(*(x.z3 for x in exprs))
        return solver

    @classmethod
    def sat(
        cls,
        dom: Mapping[str, Iterable[int]],
        *exprs: Self,
        **more_dom: Iterable[int],
    ) -> bool:
        """Check if the system formed by `exprs` is satisfiable.

        Note: to extract a model, use `s = Expression.solver(...)` then `s.check()` and `s.model()`.

        >>> a = Expression.c("a > 0")
        >>> b = Expression.c("b > 2")
        >>> c = Expression.c("c == a + b")
        >>> Expression.sat({}, a, b, c, a=range(5), b=range(5), c=range(5))
        True
        >>> Expression.sat({}, a, b, c, a=range(2), b=range(2), c=range(2))
        False
        """
        return cls.solver(dom, *exprs, **more_dom).check() == z3.sat

    @tested_cached_property
    def sympy(self) -> sp.Expr:
        """Sympy version of an expression.

        >>> expr = Expression.c("a + 2*b > 0")
        >>> print(expr)
        a+2b > 0
        >>> expr.sympy
        a + 2*b > 0
        """
        return _SPOPS[self.op](
            sp.Add(
                sp.Integer(self.const),
                *(sp.Symbol(v, integer=True) * c for v, c in self.coeffs.items()),
            ),
            0,
        )

    # NOTE: keep this property at the end to avoid masking name z3 within the class
    @tested_cached_property
    def z3(self) -> z3.ExprRef:
        """Z3 version of an expression.

        >>> expr = Expression.c("a + 2*b > 0")
        >>> print(expr)
        a+2b > 0
        >>> expr.z3
        0 + a*1 + b*2 > 0
        """
        return _Z3OPS[self.op](
            z3.Sum(
                z3.IntVal(self.const),
                *(z3.Int(v) * c for v, c in self.coeffs.items()),
            ),
            0,
        )


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
    assign: Mapping[str, Expression]
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
                self.assign | cons.assign,  # type: ignore
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

    @tested_cached_property
    def test_vars(self) -> frozenset[str]:
        return reduce(operator.or_, (g.vars for g in self.guard), frozenset())

    @tested_cached_property
    def read_vars(self) -> frozenset[str]:
        return reduce(operator.or_, (a.vars for a in self.assign.values()), frozenset())

    @tested_cached_property
    def get_vars(self) -> frozenset[str]:
        return self.test_vars | self.read_vars

    @tested_cached_property
    def set_vars(self) -> frozenset[str]:
        return frozenset(self.assign)

    @tested_cached_property
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

    @tested_cached_property
    def vars(self) -> frozenset[str]:
        return frozenset(v.name for v in self.variables)

    @tested_cached_property
    def constraints(self) -> tuple[Action, ...]:
        """The constrains in the model."""
        return tuple(a for a in self.actions if a.kind == ActionKind.cons)

    @tested_cached_property
    def rules(self) -> tuple[Action, ...]:
        """The rules in the model."""
        return tuple(a for a in self.actions if a.kind == ActionKind.rule)

    @tested_cached_property
    def ticks(self) -> tuple[Action, ...]:
        """The clock ticks in the model."""
        return tuple(a for a in self.actions if a.kind == ActionKind.tick)

    @tested_cached_property
    def expr_cls(self) -> type[Expression]:
        return self.actions[0].guard[0].__class__

    @tested_cached_property
    def v(self) -> dict[str, Variable]:
        return {v.name: v for v in self.variables}

    @tested_cached_property
    def a(self) -> dict[str, Action]:
        return {a.name: a for a in self.actions}

    @tested_cached_property
    def c(self) -> dict[str, Action]:
        return {a.name: a for a in self.actions if a.kind == ActionKind.cons}

    @tested_cached_property
    def r(self) -> dict[str, Action]:
        return {a.name: a for a in self.actions if a.kind == ActionKind.rule}

    @tested_cached_property
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
