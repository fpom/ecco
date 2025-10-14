import ast
import operator
import pathlib
import re
import sys
import subprocess
from collections import defaultdict
from collections.abc import Collection, Generator, Sequence, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property, reduce
from itertools import chain
from operator import or_
from types import SimpleNamespace
from typing import Literal, NoReturn, Self, cast, TextIO

import sympy as sp
import z3
from sympy.codegen.ast import Assignment as sp_Assign

from .mrrparse import Indenter, Lark_StandAlone, ParseError, Token, Transformer, v_args
from .pattparse import Lark_StandAlone as Lark_StandAlone_Pattern

#
# errors
#


def _error(line, column, message, errcls=ParseError) -> NoReturn:
    if column is not None:
        err = errcls(f"[{line}:{column}] {message}")
    else:
        err = errcls(f"[{line}] {message}")
    setattr(err, "line", line)
    setattr(err, "column", column)
    raise err


class BindError(ParseError):
    pass


#
# source preprocessing
#

_lno = re.compile(r'^# (\d+) "<stdin>"$')


def cpp(text: str, **var: str | int):
    out = subprocess.run(
        ["cpp", "--traditional", "-C"] + [f"-D{k}={v}" for k, v in var.items()],
        input=text,
        encoding="utf-8",
        capture_output=True,
    ).stdout
    src, num = [], -1
    for line in out.splitlines():
        if match := _lno.match(line):
            n = int(match.group(1)) - 1
            if n > num >= 0:
                src.append("\n" * (n - num))
            num = n
        elif num >= 0:
            src.append(line.split("#", 1)[0].rstrip() + "\n")
            num += 1
    return "".join(src)


#
# Python AST handlers
#


class Py2Z3(ast.NodeVisitor):
    def __init__(self, zvars: dict[str, int | z3.ArithRef]):
        self.z = zvars

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    _BoolOp = {"And": z3.And, "Or": z3.Or}

    def visit_BoolOp(self, node: ast.BoolOp):
        name = node.op.__class__.__name__
        assert name in self._BoolOp or _error(
            node.lineno, node.col_offset, f"unsupported operator '{ast.unparse(node)}'"
        )
        return self._BoolOp[name](*(self.visit(val) for val in node.values))

    _CmpOp = {
        "Lt": operator.lt,
        "LtE": operator.le,
        "Gt": operator.gt,
        "GtE": operator.ge,
        "Eq": operator.eq,
        "NotEq": operator.ne,
    }

    def visit_Compare(self, node: ast.Compare):
        cmp = []
        left = self.visit(node.left)
        for op, right in zip(node.ops, node.comparators):
            _right = self.visit(right)
            name = op.__class__.__name__
            assert name in self._CmpOp or _error(
                node.lineno,
                node.col_offset,
                f"unsupported operator '{ast.unparse(op)}'",
            )
            cmp.append(self._CmpOp[name](left, _right))
            left = _right
        return z3.And(*cmp)

    def visit_Constant(self, node: ast.Constant):
        return cast(int, node.value)

    def visit_Name(self, node: ast.Name):
        return self.z[node.id]

    _BinOp = {
        "Add": operator.add,
        "Sub": operator.sub,
        "Mult": operator.mul,
        "Div": operator.truediv,
        "FloorDiv": operator.truediv,
        "Mod": operator.mod,
    }

    def visit_BinOp(self, node: ast.BinOp):
        name = node.op.__class__.__name__
        assert name in self._BinOp or _error(
            node.lineno, node.col_offset, f"unsupported operator '{ast.unparse(node)}'"
        )
        return self._BinOp[name](self.visit(node.left), self.visit(node.right))

    _UnaryOp = {
        "Not": z3.Not,
        "UAdd": lambda x: 0 + x,
        "USub": lambda x: 0 - x,
    }

    def visit_UnaryOp(self, node: ast.UnaryOp):
        name = node.op.__class__.__name__
        assert name in self._UnaryOp or _error(
            node.lineno, node.col_offset, f"unsupported operator '{ast.unparse(node)}'"
        )
        return self._UnaryOp[name](self.visit(node.operand))  # type: ignore


class Binder(ast.NodeTransformer):
    def __init__(self, vv):
        self.v = vv

    def visit_Name(self, node: ast.Name):
        if node.id in self.v:
            return ast.Constant(self.v[node.id])
        else:
            return node


#
# model
#


class ModItem(StrEnum):
    VAR = "variable"
    CON = "constraint"
    RUL = "rule"
    CLK = "clock"


class Path(tuple[str]):
    def __truediv__(self, other: tuple[str] | str):
        if isinstance(other, str):
            return Path(self + (other,))
        else:
            return Path(self + other)

    def __matmul__(self, other: tuple[int | None, ...]) -> tuple[str | int, ...]:
        return tuple(v for v in chain(*zip(self, other)) if v is not None)


@dataclass(frozen=True)
class _Element:
    def _error(self, message, errcls=ParseError) -> NoReturn:
        line = getattr(self, "line")
        column = getattr(self, "column", None)
        _error(line, column, message, errcls)


@dataclass(frozen=True)
class VarDecl(_Element):
    line: int  # source line number
    name: str  # variable name
    domain: frozenset[int]  # variable domain (possible values)
    size: int | None  # if not None: array length
    init: (
        None | frozenset[int] | tuple[frozenset[int], ...]
    )  # omega clock | non-array initial values | array initial values
    description: str  # textual description
    isbool: bool  # Boolean or int variable
    clock: str | None = None  # name of clock that tick it
    loc: "Location" = field(init=False, repr=False, hash=False, compare=False)

    def copy(self) -> "VarDecl":
        return VarDecl(
            self.line,
            self.name,
            self.domain,
            self.size,
            self.init,
            self.description,
            self.isbool,
            self.clock,
        )

    def __post_init__(self):
        assert self.size is None or self.size > 0 or self._error("invalid size")
        super().__setattr__("domain", frozenset(self.domain))
        if self.size is None:
            assert not isinstance(self.init, tuple) or self._error(
                f"invalid initial state for non-array variable {self.init}"
            )
            super().__setattr__("init", self._init_dom(self.init))
        elif isinstance(self.init, Sequence):
            assert len(self.init) == self.size or self._error(
                "array/init size mismatch"
            )
            super().__setattr__(
                "init",
                tuple(self._init_dom(i) for i in self.init),
            )
        else:
            super().__setattr__(
                "init",
                (self._init_dom(self.init),) * self.size,
            )

    def _init_dom(self, init: None | int | Collection[int]) -> None | frozenset[int]:
        if init is None:
            if self.clock is None:
                return self.domain
            else:
                return None
        elif isinstance(init, int):
            assert (intr := frozenset({init}) & self.domain) or self._error(
                f"invalid initial state {set({init})}"
            )
            return intr
        else:
            assert (intr := frozenset(init) & self.domain) or self._error(
                f"invalid initial state {set(init)}"
            )
            return intr

    def make(self, loc: "Location"):
        super().__setattr__("loc", loc)

    def __matmul__(self, other: tuple[int | None, ...]) -> tuple[str | int, ...]:
        return self.loc.path @ other

    def __getitem__(self, idx: int):
        if self.size is None:
            raise ValueError("non-array variable cannot be indexed")
        if not 0 <= idx < self.size:
            raise ValueError("out-of-bound index")
        new = VarDecl(
            self.line,
            f"{self.name}[{idx}]",
            self.domain,
            None,
            self.init
            if self.init is None or isinstance(self.init, frozenset)
            else self.init[idx],
            self.description,
            self.isbool,
            self.clock,
        )
        super(VarDecl, new).__setattr__("loc", self.loc)
        return new

    def _init(self, value: None | frozenset[int]):
        if value is None:
            return "*"
        elif len(value) == 1:
            return f"{next(iter(value))}"
        elif frozenset(range((m := min(value)), (M := max(value)) + 1)):
            return f"{m}..{M}"
        else:
            return "|".join(str(i) for i in value)

    def __str__(self):
        idx = "" if self.size is None else f"[{self.size}]"
        clk = "" if self.clock is None else f"{self.clock}: "
        if self.init is None or isinstance(self.init, frozenset):
            init = self._init(self.init)
        elif (_init := self.init[0]) and all(i == _init for i in self.init[1:]):
            init = self._init(_init)
        else:
            init = str(tuple(set(i) for i in self.init))
        return (
            f"{{{clk}{min(self.domain)}..{max(self.domain)}}}{idx} {self.name}"
            f" = {init}: {self.description}"
        )

    def __html__(self, h):
        space = False
        if self.isbool:
            if self.size is not None:
                h.write("bool")
                space = True
        elif self.clock is not None:
            h.write(f"{{{self.clock}: {min(self.domain)}..{max(self.domain)}}}")
            space = True
        else:
            h.write(f"{{{min(self.domain)}..{max(self.domain)}}}")
            space = True
        if self.size is not None:
            with h("span", style="color:#808;"):
                h.write(f"[{self.size}]")
                space = True
        if space:
            h.write(" ")
        if self.size is None and self.isbool:
            if self.init == {0}:
                with h("span", style="color:#800;"):
                    h.write(f"{self.name}-")
            elif self.init == {1}:
                with h("span", style="color:#080;"):
                    h.write(f"{self.name}+")
            elif self.init == {0, 1}:
                with h("span", style="color:#008;"):
                    h.write(f"{self.name}*")
            else:
                self._error(f"invalid initial value in {self}")
        elif self.init is None:
            h.write(f"{self.name} = ")
            with h("span", style="color:#008;"):
                h.write("*")
        else:
            h.write(f"{self.name} = ")
            if self.size is None:
                self._html_init(h, self.init)
            elif all(
                i == cast(tuple[int | frozenset[int]], self.init)[0] for i in self.init
            ):
                self._html_init(h, cast(tuple[int | frozenset[int]], self.init)[0])
            else:
                h.write("(")
                for n, i in enumerate(self.init):
                    if n:
                        h.write(", ")
                    self._html_init(h, i)
                h.write(")")
        with h("span", style="color:#880;"):
            h.write(f": {self.description}")

    def _html_init(self, h, init):
        if self.domain == init or init is None:
            h.write("*")
        elif self.isbool and self.size is None:
            if init == {0}:
                h.write("-")
            elif init == {1}:
                h.write("+")
            else:
                self._error(f"invalid initial state {init}")
        else:
            h.write("|".join(f"{i}" for i in init))


@dataclass(frozen=True)
class Expr(_Element):
    src: str
    tree: ast.Expression = field(
        init=False,
        repr=False,
        hash=False,
        compare=False,
    )
    names: frozenset[str] = field(
        init=False,
        repr=False,
        hash=False,
        compare=False,
    )
    const: bool = field(init=False, repr=False, hash=False, compare=False)

    def __post_init__(self):
        super().__setattr__("tree", ast.parse(self.src, mode="eval"))
        super().__setattr__(
            "names",
            frozenset(
                node.id for node in ast.walk(self.tree) if isinstance(node, ast.Name)
            ),
        )
        super().__setattr__("const", not self.names)
        super().__setattr__("src", ast.unparse(self.tree))

    def bind(self, vv):
        tree = ast.parse(self.src)
        return Expr(ast.unparse(Binder(vv).visit(tree)))

    def __call__(self, **args: int | z3.ArithRef):
        if not self.const and any(isinstance(a, z3.ArithRef) for a in args):
            return self._toz3(args)
        else:
            return eval(self.src, args)

    def _toz3(self, zvars: dict[str, int | z3.ArithRef]):
        trans = Py2Z3(zvars)
        assert self.tree is not None
        return trans.visit(self.tree)

    def __str__(self):
        return self.src

    def __html__(self, h):
        h.write(re.sub(r"\b[a-z][a-z0-9]*\b", self._html_name, self.src, flags=re.I))

    def _html_name(self, match):
        return f'<span style="color:#808;">{match.group(0)}</span>'


@dataclass(frozen=True)
class Quanti(_Element):
    any: frozenset[str] = field(default_factory=frozenset)
    all: frozenset[str] = field(default_factory=frozenset)
    cond: Expr | None = None

    def __post_init__(self):
        if err := self.all & self.any:
            raise ValueError(
                f" {', '.join(repr(v) for v in err)} quantified both 'all' and 'any'"
            )
        if self.cond and (err := self.cond.names - (self.any | self.all)):
            raise ValueError(
                f" {', '.join(repr(v) for v in err)} used in 'if' but not quantified"
            )

    def bind(self, vv):
        vvnames = set(vv)
        newany = self.any - vvnames
        newall = self.all - vvnames
        if newany or newall:
            return Quanti(
                newany, newall, None if self.cond is None else self.cond.bind(vv)
            )
        else:
            return None

    def solve(
        self, *indexes: tuple[Expr, int]
    ) -> Generator[tuple[dict[str, int], dict[str, set[int]]]]:
        solver = z3.Solver()
        zvars = {}
        for var in self.all | self.any:
            _var = zvars[var] = z3.Int(var)
        if self.cond is not None:
            solver.add(self.cond(**zvars))
        for expr, size in indexes:
            _expr = expr(**zvars)
            solver.add(_expr >= 0, _expr < size)
        sol: defaultdict[tuple[int, ...], defaultdict[str, set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )
        anyvars = tuple(v for v in sorted(self.any))
        while solver.check() == z3.sat:
            mod = solver.model()
            sub = sol[
                tuple(cast(z3.IntNumRef, mod[zvars[v]]).as_long() for v in anyvars)
            ]
            for var in self.all:
                sub[var].add(cast(z3.IntNumRef, mod[zvars[var]]).as_long())
            solver.add(z3.Or(*(z != mod[z] for z in zvars.values())))
        assert sol or self._error(
            "quantification yields no solutions", errcls=BindError
        )
        for key, val in sol.items():
            assert all(v for v in val.values()) or self._error(
                "quantification yields not solutions", errcls=BindError
            )
            yield dict(zip(anyvars, key)), dict(val)

    def __str__(self):
        parts = []
        if self.all:
            parts.extend(["for all", ", ".join(self.all)])
        if self.any:
            parts.extend(["for any", ", ".join(self.any)])
        if self.cond is not None:
            parts.extend(["if", str(self.cond)])
        return " ".join(parts)

    def __html__(self, h):
        if self.all:
            with h("span", style="color:#008; font-weight:bold;"):
                h.write("for all")
            h.write(" " + ", ".join(self.all))
        if self.any:
            if self.all:
                h.write(" ")
            with h("span", style="color:#008; font-weight:bold;"):
                h.write("for any")
            h.write(" " + ", ".join(self.any))
        if self.cond is not None:
            h.write(" ")
            with h("span", style="color:#008; font-weight:bold;"):
                h.write("if")
            h.write(f" {self.cond}")


@dataclass(frozen=True)
class VarUse(_Element):
    line: int  # source line number
    column: int | None  # source line column
    name: str  # variable name
    index: int | Expr | None = None  # constant index | expr index | no index
    locrel: Literal[None, "outer", "inner"] = None  # variable in same/other location
    locname: str | None = None  # inner location name
    locidx: int | Expr | None = None  # constant index | expr index | no index
    decl: VarDecl = field(
        init=False, repr=False, hash=False, compare=False
    )  # var declaration
    names: set[str] = field(
        init=False, repr=False, hash=False, compare=False
    )  # free names in indexes

    def __post_init__(self):
        assert self.locrel in (None, "outer", "inner") or self._error(
            f"invalid location relation {self.locrel!r}",
        )
        assert (
            self.locrel is not None
            or self.locname is None
            or self._error(
                "locname should be None",
            )
        )
        assert (
            self.locrel is not None
            or self.locidx is None
            or self._error(
                "locidx should be None",
            )
        )
        assert (
            self.locrel is not None
            or self.locname is None
            or self._error(
                "outer location cannot be named",
            )
        )
        assert (
            self.locrel != "outer"
            or self.locidx is None
            or self._error(
                "outer location cannot be indexed",
            )
        )
        assert (
            self.locrel != "inner"
            or self.locname is not None
            or self._error(
                "missing value for locname",
            )
        )
        if isinstance(self.index, Expr) and self.index.const:
            super().__setattr__("index", self.index())
        if isinstance(self.locidx, Expr) and self.locidx.const:
            super().__setattr__("locidx", self.locidx())
        super().__setattr__(
            "names",
            (self.index.names if isinstance(self.index, Expr) else set())
            | (self.locidx.names if isinstance(self.locidx, Expr) else set()),
        )

    def __matmul__(self, other: tuple[int | None, ...]) -> tuple[str | int, ...]:
        path = self.decl @ other
        if isinstance(self.locidx, int) and self.locname is not None:
            return path + (self.locname, self.locidx)
        elif self.locname is not None:
            return path + (self.locname,)
        else:
            return path

    def __str__(self):
        if self.index is None:
            idx = ""
        else:
            idx = f"[{self.index}]"
        if self.locrel is None:
            at = ""
        elif self.locrel == "outer":
            at = "@@"
        elif self.locidx is None:
            at = f"@{self.locname}"
        else:
            at = f"@{self.locname}[{self.locidx}]"
        return f"{self.name}{idx}{at}"

    def __html__(self, h):
        h.write(self.name)
        if self.index is not None:
            with h("span", style="color:#808;"):
                h.write(f"[{self.index}]")
        if self.locrel is not None:
            with h("span", style="color:#088;"):
                if self.locrel == "outer":
                    h.write("@@")
                else:
                    h.write(f"@{self.locname}")
            if self.locidx is not None:
                with h("span", style="color:#808;"):
                    h.write(f"[{self.locidx}]")

    def make(self, loc: "Location"):
        if self.locrel is None:
            assert self.name in loc.var or self._error(f"no local variable {self.name}")
            super().__setattr__("decl", loc.var[self.name])
        elif self.locrel == "outer":
            assert loc.parent is not None or self._error("no outer location")
            assert self.name in loc.parent.var or self._error(
                f"no outer variable {self.name}"
            )
            super().__setattr__("decl", loc.parent.var[self.name])
        else:
            assert self.locname in loc.sub or self._error(
                f"no inner location {self.locname!r}"
            )
            assert self.name in loc.sub[self.locname].var or self._error(
                f"no variable {self.name} in {self.locname}",
            )
            assert (
                loc.sub[self.locname].size is None
                or self.locidx is not None
                or self._error(
                    f"indexing non-array location {self.locname!r}",
                )
            )
            assert (
                self.locidx is None
                or loc.sub[self.locname].size is not None
                or self._error(
                    f"invalid index for {self.locname!r}[{loc.sub[self.locname].size}]",
                )
            )
            decl = loc.sub[self.locname].var[self.name]
            assert (
                not isinstance(self.locidx, int)
                or (
                    decl.loc is not None
                    and decl.loc.size is not None
                    and 0 <= self.locidx < decl.loc.size
                )
                or self._error(f"location index '{self.locidx}' out of range")
            )
            super().__setattr__("decl", decl)
        assert (
            self.index is None
            or ((decl := self.decl) is not None and decl.size is not None)
            or self._error("cannot index non-array variable")
        )
        assert (
            not isinstance(self.index, int)
            or (
                (decl := self.decl) is not None
                and decl.size is not None
                and 0 <= self.index < decl.size
            )
            or self._error(f"index '{self.index}' out of range")
        )
        assert (
            self.locidx is None
            or ((locdecl := self.decl.loc) is not None and locdecl.size is not None)
            or self._error(
                f"indexing non-array location {self.locname!r}",
            )
        )

    @cached_property
    def indexes(self) -> frozenset[tuple[Expr, int]]:
        return (
            frozenset([(self.locidx, cast(int, self.decl.loc.size))])
            if isinstance(self.locidx, Expr)
            else frozenset()
        ) | (
            frozenset([(self.index, cast(int, self.decl.size))])
            if isinstance(self.index, Expr)
            else frozenset()
        )

    def bind(self, vv):
        new = VarUse(
            self.line,
            self.column,
            self.name,
            self.index
            if self.index is None or isinstance(self.index, int)
            else self.index.bind(vv),
            self.locrel,
            self.locname,
            self.locidx
            if self.locidx is None or isinstance(self.locidx, int)
            else self.locidx.bind(vv),
        )
        super(VarUse, new).__setattr__("decl", self.decl)
        return new


def _sp_sub(a, b):
    return sp.Add(a, -b)


@dataclass(frozen=True)
class BinOp(_Element):
    line: int  # source line number
    column: int | None  # source line column
    left: VarUse | int  # expression left-hand side
    op: str  # infix operator
    right: VarUse | Self | int | None  # expression right-hand side
    names: frozenset[str] = field(init=False, repr=False, hash=False, compare=False)

    def __post_init__(self):
        super().__setattr__(
            "names",
            (self.left.names if isinstance(self.left, VarUse) else set())
            | (self.right.names if isinstance(self.right, (VarUse, BinOp)) else set()),
        )

    _sp_op = {
        "==": sp.Eq,
        "!=": sp.Ne,
        "<": sp.Lt,
        "<=": sp.Le,
        ">": sp.Gt,
        ">=": sp.Ge,
        "=": sp_Assign,
        "+": sp.Add,
        "-": _sp_sub,
    }

    def sympy(
        self, star: str | int | sp.Expr, vname: Callable[[VarUse], str]
    ) -> sp.Expr:
        _star = sp.Symbol(star) if isinstance(star, str) else star
        args: list[int | sp.Symbol | sp.Expr] = []
        for a in [self.left, self.right]:
            if a is None:
                args.append(_star)
            elif isinstance(a, int):
                args.append(a)
            elif isinstance(a, VarUse):
                args.append(sp.Symbol(vname(a)))
            else:
                args.append(a.sympy(_star, vname))
        return self._sp_op[self.op](*args)

    def __str__(self):
        right = self.right if self.right is not None else "*"
        return f"{self.left}{self.op}{right}"

    def __html__(self, h):
        if (
            isinstance(self.left, VarUse)
            and (decl := self.left.decl) is not None
            and decl.isbool
            and self.op == "=="
            and self.right in (0, 1)
        ):
            if self.right:
                with h("span", style="color:#080;"):
                    self.left.__html__(h)
                    h.write("+")
            else:
                with h("span", style="color:#800;"):
                    self.left.__html__(h)
                    h.write("-")
        elif self.right is None:
            assert isinstance(self.left, VarUse)
            self.left.__html__(h)
            with h("span", style="color:#008;"):
                h.write("*")
        else:
            if isinstance(self.left, int):
                h.write(f"{self.left}")
            else:
                self.left.__html__(h)
            h.write(self.op)
            if isinstance(self.right, int):
                h.write(f"{self.right}")
            else:
                self.right.__html__(h)

    def make(self, loc: "Location"):
        if isinstance(self.left, VarUse):
            self.left.make(loc)
        if isinstance(self.right, (VarUse, BinOp)):
            self.right.make(loc)
        assert (
            self.right is not None
            or (
                isinstance(self.left, VarUse)
                and self.left.decl is not None
                and self.left.decl.clock is not None
            )
            or self._error(
                "cannot assign '*' to non-clocked variable",
            )
        )

    @cached_property
    def indexes(self) -> frozenset[tuple[Expr, int]]:
        return (self.left.indexes if isinstance(self.left, VarUse) else frozenset()) | (
            self.right.indexes
            if isinstance(self.right, (VarUse, BinOp))
            else frozenset()
        )

    def bind(self, vv):
        return BinOp(
            self.line,
            self.column,
            self.left if isinstance(self.left, int) else self.left.bind(vv),
            self.op,
            self.right
            if isinstance(self.right, int) or self.right is None
            else self.right.bind(vv),
        )


@dataclass(frozen=True)
class Assignment(_Element):
    line: int  # source line number
    column: int | None  # source line column
    target: VarUse  # assigned variable
    value: VarUse | BinOp | int | None  # expression assigned to variable
    names: frozenset[str] = field(init=False, repr=False, hash=False, compare=False)

    def __post_init__(self):
        super().__setattr__(
            "names",
            self.target.names
            | (self.value.names if isinstance(self.value, (VarUse, BinOp)) else set()),
        )

    def __str__(self):
        if isinstance(self.value, BinOp) and self.value.left == self.target:
            return f"{self.target}{self.value.op}={self.value.right}"
        else:
            return f"{self.target}={self.value if self.value is not None else '*'}"

    def __html__(self, h):
        if (decl := self.target.decl) is not None and decl.isbool:
            if self.value == 0:
                with h("span", style="color:#800;"):
                    self.target.__html__(h)
                    h.write("-")
            elif self.value == 1:
                with h("span", style="color:#080;"):
                    self.target.__html__(h)
                    h.write("+")
            else:
                self._error(f"invalid Boolean assignment {self}")
        else:
            self.target.__html__(h)
            if self.value is None:
                with h("span", style="color:#008;"):
                    h.write("*")
            elif isinstance(self.value, int):
                h.write(f"={self.value}")
            else:
                h.write(f"=")
                self.value.__html__(h)

    def make(self, loc: "Location"):
        self.target.make(loc)
        if self.value is None:
            assert (
                self.target.decl is not None and self.target.decl.clock is not None
            ) or self._error(
                "cannot assign '*' to non clocked variable",
            )
        elif isinstance(self.value, int):
            assert (
                self.target.decl is not None and self.value in self.target.decl.domain
            ) or self._error(
                f"out-of-domain assignment {self}",
            )
        else:
            self.value.make(loc)

    @cached_property
    def indexes(self) -> frozenset[tuple[Expr, int]]:
        return self.target.indexes | (
            self.value.indexes
            if isinstance(self.value, (VarUse, BinOp))
            else frozenset()
        )

    def bind(self, vals):
        return Assignment(
            self.line,
            self.column,
            self.target.bind(vals),
            self.value
            if isinstance(self.value, int) or self.value is None
            else self.value.bind(vals),
        )


@dataclass(frozen=True)
class Action(_Element):
    line: int  # source line number
    left: tuple[BinOp, ...]  # conditions
    right: tuple[Assignment, ...]  # assignments
    tags: tuple[str, ...] = ()  # tags
    quantifier: Quanti | None = None  # free indexes quantifiers
    name: str = field(
        init=False, repr=False, hash=False, compare=False
    )  # unique name within location
    bound: dict[str, int | set[int]] = field(
        default_factory=dict, init=False, repr=False, hash=False, compare=False
    )  # bound names
    names: frozenset[str] = field(
        init=False, repr=False, hash=False, compare=False
    )  # names used in indexes
    loc: "Location" = field(init=False, repr=False, hash=False, compare=False)
    origin: Self | None = field(
        default=None, init=False, repr=False, hash=False, compare=False
    )

    def __post_init__(self):
        super().__setattr__(
            "names",
            reduce(operator.or_, (lr.names for lr in self.left + self.right), set()),
        )
        assert (
            self.quantifier is not None
            or not (miss := self.names - {"self"})
            or self._error(f"{','.join(miss)} not quantified")
        )
        assert (
            self.quantifier is None
            or not (miss := (self.quantifier.any | self.quantifier.all) - self.names)
            or self._error(f"{','.join(miss)} quantified but not used")
        )
        assert (
            self.quantifier is None
            or not (
                miss := self.names
                - (self.quantifier.all | self.quantifier.any | {"self"})
            )
            or self._error(f"{','.join(miss)} not quantified")
        )

    @cached_property
    def indexes(self) -> frozenset[tuple[Expr, int]]:
        return frozenset(
            chain.from_iterable(lr.indexes for lr in self.left + self.right),
        )

    @cached_property
    def boundname(self):
        "name with bound free variables"
        assert self.name is not None or self._error("action should ne named")
        if self.bound:
            b = [
                f"{k}={v}"
                if isinstance(v, int)
                else f"{k}={{{','.join(str(i) for i in v)}}}"
                for k, v in self.bound.items()
            ]
            return f"{self.name}[{','.join(b)}]"
        else:
            return self.name

    def __str__(self):
        return "".join(
            [
                f"[{', '.join(self.tags)}] " if self.tags else "",
                ", ".join(str(c) for c in self.left),
                " >> ",
                ", ".join(str(a) for a in self.right),
                "" if self.quantifier is None else f" {self.quantifier}",
            ]
        )

    def __html__(self, h):
        if self.tags:
            with h("span", style="color:#088;"):
                h.write(f"[{', '.join(self.tags)}]")
            h.write(" ")
        for i, cond in enumerate(self.left):
            if i:
                h.write(", ")
            cond.__html__(h)
        with h("span", style="color:#008; font-weight:bold;"):
            h.write(" &gt;&gt; ")
        for i, assign in enumerate(self.right):
            if i:
                h.write(", ")
            assign.__html__(h)
        if self.quantifier is not None:
            h.write(" ")
            self.quantifier.__html__(h)

    def make(self, loc: "Location", name: str):
        super().__setattr__("name", name)
        super().__setattr__("loc", loc)
        for cond in self.left:
            cond.make(loc)
        for assign in self.right:
            assign.make(loc)
        assigned = set()
        for a in self.right:
            if (
                a.target.name,
                a.target.locname,
                a.target.locidx,
                a.target.index,
            ) in assigned:
                self._error(f"'{a.target}' assigned twice")
            assigned.add(
                (a.target.name, a.target.locname, a.target.locidx, a.target.index)
            )

    def bind(self, vv: dict[str, int]):
        new = Action(
            self.line,
            tuple(left.bind(vv) for left in self.left),
            tuple(right.bind(vv) for right in self.right),
            self.tags,
            None if self.quantifier is None else self.quantifier.bind(vv),
        )
        super(Action, new).__setattr__("name", self.name)
        super(Action, new).__setattr__("loc", self.loc)
        super(Action, new).__setattr__(
            "bound", self.bound | {k: v for k, v in vv.items() if k in self.names}
        )
        super(Action, new).__setattr__("origin", self)
        return new

    def expand(self, locidx: int | None):
        if locidx is None:
            vv = {}
            sb = self
        else:
            vv = {"self": locidx}
            sb = self.bind(vv)
        if self.quantifier is None:
            yield sb
        else:
            for vany, vall in self.quantifier.solve(*self.indexes):
                new = self.bind(vv | vany)
                if vall:
                    yield new._expand_all(vall)
                else:
                    yield new

    def _expand_all(self, vv: dict[str, set[int]]):
        if self.quantifier is None:
            return self
        bound = self.quantifier.all & set(vv)
        assert not (free := self.quantifier.all - bound) or self._error(
            f"{','.join(free)} not bound"
        )
        assert not self.quantifier.any or self._error(
            "cannot expand any quantified action"
        )
        sides = SimpleNamespace(left=self.left, right=self.right)
        for sd in ("left", "right"):
            for var in bound:
                old = getattr(sides, sd)
                new: list[BinOp | Assignment] = []
                for item in old:
                    if var in item.names:
                        new.extend(item.bind({var: v}) for v in vv[var])
                    else:
                        new.append(item)
                setattr(sides, sd, tuple(new))
        act = Action(
            self.line,
            sides.left,
            sides.right,
            self.tags,
            None,
        )
        super(Action, act).__setattr__("name", self.name)
        super(Action, act).__setattr__("loc", self.loc)
        super(Action, act).__setattr__(
            "bound", self.bound | cast(dict[str, int | set[int]], vv)
        )
        super(Action, act).__setattr__("origin", self.origin or self)
        return act


@dataclass(frozen=True)
class Location(_Element):
    line: int  # source line number
    variables: tuple[VarDecl, ...]  # local variables
    constraints: tuple[Action, ...]  # constraints
    rules: tuple[Action, ...]  # rules
    locations: tuple[Self, ...]  # nested locations
    name: str | None = None  # name if nested
    size: int | None = None  # size of array
    parent: Self | None = field(init=False, repr=False, hash=False, compare=False)
    clocks: dict[str, str] = field(default_factory=dict, hash=False, compare=False)
    clocked: dict[str, list[VarDecl]] = field(
        default_factory=dict, init=False, repr=False, hash=False, compare=False
    )

    @cached_property
    def path(self) -> Path:
        if self.parent is None:
            return Path()
        assert self.name is not None
        return self.parent.path / self.name

    @cached_property
    def sub(self):
        "sub-locations by name"
        return {loc.name: loc for loc in self.locations}

    @cached_property
    def var(self):
        "variables by name"
        return {v.name: v for v in self.variables}

    def __post_init__(self):
        assert self.size is None or self.size > 0 or self._error("invalid array size")

    def __str__(self):
        parts = ["@"]
        if self.name is not None:
            parts.append(f"{self.name}")
        if self.size is not None:
            parts.append(f"[{self.size}]")
        parts.append(
            f"(V={len(self.variables)},"
            f"C={len(self.constraints)},"
            f"R={len(self.rules)},"
            f"L={len(self.locations)})"
        )
        return "".join(parts)

    def __html__(self, h, indent=0):
        tab = " " * (indent * 4)
        if self.name is not None:
            with h("span", style="font-weight:bold;", BREAK=True):
                h.write(tab)
                with h("span", style="color:#008;"):
                    h.write("location")
                with h("span", style="color:#088;"):
                    h.write(f" {self.name}")
                if self.size is not None:
                    with h("span", style="color:#808;"):
                        h.write(f"[{self.size}]")
                with h("span", style="color:#008;"):
                    h.write(":")
            indent += 1
            tab += "    "
        if self.clocks:
            h.write(tab)
            with h("span", style="font-weight:bold;color:#008;", BREAK=True):
                h.write("clocks:")
            for clk, desc in self.clocks.items():
                h.write(tab)
                h.write(f"    {clk}")
                with h("span", style="color:#880;", BREAK=True):
                    h.write(f": {desc}")
        if self.variables:
            h.write(tab)
            with h("span", style="font-weight:bold;color:#008;", BREAK=True):
                h.write("variables:")
            for var in self.variables:
                h.write(tab + "    ")
                var.__html__(h)
                h.write("\n")
        for loc in self.locations:
            loc.__html__(h, indent + 1)
        for sec in ("constraints", "rules"):
            num = 1
            actions = getattr(self, sec)
            if actions:
                h.write(tab)
                with h("span", style="font-weight:bold;color:#008;", BREAK=True):
                    h.write(f"{sec}:")
                for act in actions:
                    h.write(tab + "    ")
                    act.__html__(h)
                    with h("span", style="color:#888;", BREAK=True):
                        h.write(f"  # {sec[0].upper()}{num}")
                    num += 1

    @property
    def actions(self):
        yield from self.constraints
        yield from self.rules

    def make(self, top=None, parent=None):
        super().__setattr__("parent", parent)
        if top is None:
            top = self
        for var in self.variables:
            var.make(self)
            if var.clock:
                assert var.clock in top.clocks or var._error(
                    f"clock {var.clock} is not defined"
                )
                top.clocked[var.clock].append(var)
        for loc in self.locations:
            loc.make(top, self)
        for kind in ("constraints", "rules"):
            for num, act in enumerate(getattr(self, kind), start=1):
                cast(Action, act).make(self, f"{kind[0].upper()}{num}")
        for k, v in self.clocked.items():
            assert v or self._error(f"clock {k} not used")

    def _flatten(
        self,
        *locidx: int | None,
    ) -> Generator[tuple[ModItem, tuple[int | None, ...], VarDecl | Action]]:
        for var in self.variables:
            if var.size is not None:
                for idx in range(var.size):
                    yield ModItem.VAR, locidx, var[idx]
            else:
                yield ModItem.VAR, locidx, var
        for loc in self.locations:
            if loc.size is not None:
                for idx in range(loc.size):
                    yield from loc._flatten(*locidx, idx)
            else:
                yield from loc._flatten(*locidx, None)
        for kind, group in (
            (ModItem.CON, self.constraints),
            (ModItem.RUL, self.rules),
        ):
            for act in group:
                last = locidx[-1] if locidx else None
                for a in act.expand(last):
                    yield kind, locidx, a

    def flatten(
        self,
    ) -> Generator[
        tuple[
            ModItem,
            tuple[int | None, ...] | str,
            VarDecl | Action | list[tuple[tuple[int | None, ...], VarDecl]],
        ]
    ]:
        clocked = defaultdict(list)
        for kind, locidx, item in self._flatten():
            if isinstance(item, VarDecl) and item.clock is not None:
                clocked[item.clock].append((locidx, item))
            yield kind, locidx, item
        for k, v in clocked.items():
            yield ModItem.CLK, k, v

    def _expand(self) -> "Location":
        new = Location(
            self.line,
            tuple(v.copy() for v in self.variables),
            tuple(x for a in self.constraints for x in a.expand(None)),
            tuple(x for a in self.rules for x in a.expand(None)),
            tuple(loc._expand() for loc in self.locations),
            self.name,
            self.size,
        )
        new.clocks.update(self.clocks)
        new.clocked.update((c, []) for c in self.clocks)
        return new

    def expand(self) -> "Location":
        new = self._expand()
        new.make()
        return new

    def save(self, out: str | pathlib.Path | TextIO = sys.stdout, indent: str = ""):
        if isinstance(out, (str, pathlib.Path)):
            out = open(out, "w")
        if self.clocks:
            out.write("clocks:\n")
            for clk, txt in self.clocks.items():
                out.write(f"    {clk}: {txt}\n")
        if self.variables:
            out.write(f"{indent}variables:\n")
            for var in self.variables:
                out.write(f"{indent}    {var}\n")
        if self.locations:
            for loc in self.locations:
                idx = "" if loc.size is None else f"[{loc.size}]"
                out.write(f"{indent}location {loc.name}{idx}:\n")
                loc.save(out, indent + "    ")
        for name in ("constraints", "rules"):
            attr = getattr(self, name)
            if attr:
                out.write(f"{indent}{name}:\n")
                org = None
                for act in attr:
                    if act.origin and act.origin is not org:
                        out.write(f"{indent}    # {act.origin}\n")
                        org = act.origin
                    out.write(f"{indent}    {act}\n")


#
# parser
#


def _int(tok, min=None, max=None, message="invalid int litteral"):
    try:
        val = int(tok.value)
        if min is not None:
            assert val >= min or _error(tok.line, tok.column, message)
        if max is not None:
            assert val <= max or _error(tok.line, tok.column, message)
        return val
    except ParseError:
        raise
    except Exception:
        _error(tok.line, tok.column, message)


BOOL = frozenset({0, 1})


@v_args(inline=True)
class MRRTrans(Transformer):  # pyright: ignore[reportMissingTypeArgument]
    def __init__(self, text):
        super().__init__()
        self.text = text

    def start(self, clocks, model):
        if clocks:
            model.clocks.update(clocks)
            model.clocked.update((c, []) for c in clocks)
        model.make()
        return model

    def model(self, *args):
        variables, *locations, constraints, rules = args
        line = None
        for obj in chain(variables, constraints, rules, locations):
            line = getattr(obj, "line", None)
            if line is not None:
                break
        return Location(
            line or 0, variables or (), constraints or (), rules or (), tuple(locations)
        )

    def pattern(self, *rules):
        return rules

    def sequence(self, *args):
        return args

    def dictseq(self, first, *rest):
        d = dict(first)
        for r in rest:
            d.update(r)
        return d

    def location(self, name, size, model):
        object.__setattr__(model, "name", name.value)
        object.__setattr__(model, "line", name.line)
        if size is not None:
            object.__setattr__(model, "size", _int(size, 1, None, "invalid array size"))
        return model

    def clockdecl(self, name, desc):
        return {name.value: desc.value}

    def vardecl_long(self, typ, name, init, desc):
        typ, size = typ
        if isinstance(typ, tuple):
            clock, typ = typ
        else:
            clock = None
        if isinstance(init, tuple):
            init = tuple(i & typ for i in init)
        elif init is None:
            if not clock:
                init = typ
        else:
            init = init & typ
        if size is None:
            assert not isinstance(init, tuple) or _error(
                name.line,
                None,
                "cannot initialise scalar with array",
            )
        else:
            if not isinstance(init, tuple):
                init = (init,) * size
            elif len(init) != size:
                _error(name.line, None, "type/init size mismatch")
        return VarDecl(
            name.line,
            name.value,
            typ,
            size,
            init,
            desc.value.strip(),
            typ is BOOL,
            clock,
        )

    def vardecl_short(self, name, sign, desc):
        if sign.value == "+":
            init = {1}
        elif sign.value == "-":
            init = {0}
        elif sign.value == "*":
            init = {0, 1}
        else:
            _error(name.line, None, f"unexpected initial value {sign.value!r}")
        return VarDecl(
            name.line, name.value, BOOL, None, frozenset(init), desc.value.strip(), True
        )

    def type(self, dom, size):
        if size is None:
            return dom, None
        else:
            return dom, _int(size, 1, None, "invalid array size")

    def type_bool(self):
        return BOOL

    def type_interval(self, start, end):
        return set(range(_int(start), _int(end) + 1))

    def type_clock(self, clock, start, end):
        return clock.value, self.type_interval(start, end)

    def init(self, *args):
        if any(a is None for a in args):
            return None
        else:
            return reduce(or_, args)

    def init_int(self, min, max):
        if max is None:
            return {_int(min)}
        else:
            return set(range(_int(min), _int(max) + 1))

    def init_all(self):
        return None

    def actdecl(self, tags, *args):
        if tags is not None:
            line = tags.line
            tags = tuple(ts for t in tags.value.split(",") if (ts := t.strip()))
        else:
            line = args[0].line
        left, right, quanti = [], [], None
        for arg in args:
            if isinstance(arg, BinOp):
                left.append(arg)
            elif isinstance(arg, Assignment):
                right.append(arg)
            elif isinstance(arg, Quanti):
                quanti = arg
            else:
                assert (
                    arg is None
                    or isinstance(arg, str)
                    or _error(
                        line,
                        None,
                        f"invalid condition or assignment {arg}",
                    )
                )
        return Action(line, tuple(left), tuple(right), tags or (), quantifier=quanti)

    def quantifier(self, *args):
        if isinstance(args[-1], Expr) or args[-1] is None:
            *quant, cond = args
        else:
            quant, cond = args, None
        both = {"any": set(), "all": set()}
        for q, names in quant:
            both[q].update(names)
        return Quanti(**{k: frozenset(v) for k, v in both.items()}, cond=cond)

    def quant(self, q, *names):
        ids = set()
        for n in names:
            if (i := n.value) in ids:
                _error(n.line, n.column, f"duplicated name {i!r}", ParseError)
            ids.add(i)
        return q.value, ids

    def condition_long(self, left, op, right):
        line, column = left.line, left.column
        if isinstance(left, Token):
            column -= len(left.value) - 1
            left = _int(left)
        if isinstance(right, Token):
            right = _int(right)
        return BinOp(line, column, left, op, right)

    def condition_short(self, var, op):
        if op.value == "+":
            return BinOp(var.line, var.column, var, "==", 1)
        elif op.value == "-":
            return BinOp(var.line, var.column, var, "==", 0)
        elif op.value == "*":
            return BinOp(var.line, var.column, var, "==", None)
        else:
            _error(
                op.line,
                op.column + 1 - len(op.value),
                f"invalid condition {op.value!r}",
            )

    def var(self, name, index, at):
        if at is None:
            locrel = locname = locidx = None
        elif at == "outer":
            locrel, locname, locidx = "outer", None, None
        else:
            locrel, locname, locidx = at
        return VarUse(
            name.line,
            name.column + 1 - len(name.value),
            name.value,
            index,
            locrel,
            locname,
            locidx,
        )

    def at_inner(self, name, index):
        return "inner", name.value, index

    def at_outer(self, _):
        return "outer"

    def intexpr(self, *atoms):
        return self.expr(int, *atoms)

    def boolexpr(self, *atoms):
        return self.expr(bool, *atoms)

    def expr(self, cast, *atoms):
        src = self.text[atoms[0].start_pos : atoms[-1].end_pos]
        try:
            x = Expr(src)
            if x.const:
                return cast(x())
            else:
                return x
        except Exception as err:
            _error(atoms[0].line, atoms[0].column + 1, str(err))

    def assignment_long(self, target, assign, value):
        if isinstance(value, Token):
            if value.value == "*":
                value = None
            else:
                value = _int(value)
        if assign.value[0] == "+":
            value = BinOp(target.line, target.column, target, "+", value)
        elif assign.value[0] == "-":
            value = BinOp(target.line, target.column, target, "-", value)
        return Assignment(target.line, target.column, target, value)

    def assignment_short(self, var, op):
        if op == "++":
            return Assignment(
                var.line, var.column, var, BinOp(var.line, var.column, var, "+", 1)
            )
        elif op == "--":
            return Assignment(
                var.line, var.column, var, BinOp(var.line, var.column, var, "-", 1)
            )
        elif op == "+":
            return Assignment(var.line, var.column, var, 1)
        elif op == "-":
            return Assignment(var.line, var.column, var, 0)
        elif op == "~":
            return Assignment(
                var.line, var.column, var, BinOp(var.line, var.column, 1, "-", var)
            )
        elif op == "*":
            return Assignment(var.line, var.column, var, None)
        else:
            _error(op.line, op.column, "invalid assignment {op.value!r}")


class MRRIndenter(Indenter):
    NL_type = "_NL"  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    OPEN_PAREN_types = []  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    CLOSE_PAREN_types = []  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    INDENT_type = "_INDENT"  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    DEDENT_type = "_DEDENT"  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]
    tab_len = 4  # pyright: ignore[reportAssignmentType,reportIncompatibleMethodOverride]


def parse_src(src, *defs, pattern=False):
    vardef = {}
    for d in defs:
        try:
            v, t = d.split("=", 1)
            vardef[v] = t
        except Exception:
            vardef[d] = None
    if pattern:
        lsa = Lark_StandAlone_Pattern
    else:
        lsa = Lark_StandAlone
    _src = cpp(src, **vardef)
    parser = lsa(transformer=MRRTrans(_src), postlex=MRRIndenter())
    return parser.parse(_src)


def parse(path, *defs, pattern=False):
    return parse_src(open(path).read(), *defs, pattern=pattern)
