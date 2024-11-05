import re
import subprocess

from collections import defaultdict
from dataclasses import dataclass, field
from functools import reduce
from itertools import chain, product
from operator import or_
from typing import Optional, Tuple, Set, Union, Sequence, Dict, Self

from .mrrparse import Indenter, Lark_StandAlone, ParseError, Token, Transformer, v_args
from .pattparse import Lark_StandAlone as Lark_StandAlone_Pattern


#
# errors
#


def _error(line, column, message, errcls=ParseError):
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


class BindError(Exception):
    pass


#
# source preprocessing
#


_line_dir = re.compile(r'^#\s+([0-9]+)\s+"<stdin>"\s*$')
_cpp_head = subprocess.run(
    ["cpp", "--traditional", "-C"], input="", encoding="utf-8", capture_output=True
).stdout
_comment = re.compile(r"(/\*.*?\*/)", re.S)
_cpp_head = _comment.split(_cpp_head)[1::2]


def cpp(text, **var):
    out = subprocess.run(
        ["cpp", "--traditional", "-C"] + [f"-D{k}={v}" for k, v in var.items()],
        input=text,
        encoding="utf-8",
        capture_output=True,
    ).stdout
    for hd in _cpp_head:
        out = out.replace(hd, "")
    # we don't use cpp -P but remove line directives because otherwise some
    # empty lines are missing at the beginning
    # + we remove comments
    return "".join(
        (line.split("#", 1)[0]).rstrip() + "\n"
        for line in out.splitlines()
        if not _line_dir.match(line)
    )


#
# model
#


def dictfield():
    return field(default_factory=dict)


@dataclass(unsafe_hash=True)
class _Element:
    def _error(self, message, errcls=ParseError):
        line = getattr(self, "line")
        column = getattr(self, "column", None)
        _error(line, column, message, errcls)

    def _assert(self, test, message, errcls=ParseError):
        line = getattr(self, "line")
        column = getattr(self, "column", None)
        _assert(test, line, column, message, errcls)


@dataclass(unsafe_hash=True)
class VarDecl(_Element):
    line: int  # source line number
    name: str  # variable name
    domain: Set[int]  # variable domain (possible values)
    size: Optional[int]  # if not None: array length
    init: Union[
        None,  # omega clock
        Set[int],  # non-array initial values
        Tuple[Set[int], ...],
    ]  # array initial values
    description: str  # textual description
    isbool: bool  # Boolean or int variable
    clock: Optional[str] = None  # name of clock that tick it

    def __post_init__(self):
        self._assert(self.size is None or self.size > 0, "invalid size")
        if self.size is None:
            self.init = self._init_dom(self.init)
        else:
            if isinstance(self.init, Sequence):
                self._assert(len(self.init) == self.size, "array/init size mismatch")
                self.init = tuple(self._init_dom(i) for i in self.init)
            else:
                self.init = (self._init_dom(self.init),) * self.size

    def _init_dom(self, init):
        if init is None:
            if self.clock is None:
                return self.domain
            else:
                return None
        elif isinstance(init, int):
            init = {init} & self.domain
        elif isinstance(init, set):
            init = init & self.domain
        else:
            self._error(f"invalid initial state {init!r}")
        self._assert(init, f"invalid initial state {init!r}")
        return init

    def __str__(self):
        if self.size is None:
            idx = ""
        else:
            idx = f"[{self.size}]"
        return (
            f"{{{min(self.domain)}..{max(self.domain)}}}{idx} {self.name}"
            f" = {self.init}: {self.description}"
        )

    def __html__(self, h):
        space = False
        if self.isbool:
            if self.size is not None:
                h.write("bool")
                space = True
        elif self.clock is not None:
            h.write(f"{{{self.clock}: " f"{min(self.domain)}..{max(self.domain)}}}")
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
            elif all(i == self.init[0] for i in self.init):
                self._html_init(h, self.init[0])
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


@dataclass(unsafe_hash=True)
class VarUse(_Element):
    line: int  # source line number
    column: Optional[int]  # source line column
    name: str  # variable name
    index: Optional[
        Union[
            int,  # constant index
            str,  # quantified index
            tuple[str, str, int],
        ]  # shifted quantified index
    ] = None  # no var index by default
    locrel: Optional[str] = None  # variable in other location
    locname: Optional[str] = None  # inner location name
    locidx: Optional[
        Union[
            int,  # constant index
            str,  # quantified index
            tuple[str, str, int],
        ]  # shifted quantified index
    ] = None  # no inner loc idx by default

    def __post_init__(self):
        self._assert(
            self.locrel in (None, "outer", "inner"),
            f"invalid location relation {self.locrel!r}",
        )
        self._assert(
            self.locname is None if self.locrel is None else True,
            "locname should be None",
        )
        self._assert(
            self.locidx is None if self.locrel is None else True,
            "locidx should be None",
        )
        self._assert(
            self.locname is None if self.locrel == "outer" else True,
            "outer location cannot be named",
        )
        self._assert(
            self.locidx is None if self.locrel == "outer" else True,
            "outer location cannot be indexed",
        )
        self._assert(
            self.locname is not None if self.locrel == "inner" else True,
            "missing value for locname",
        )

    def __str__(self):
        if self.index is None:
            idx = ""
        elif isinstance(self.index, tuple):
            idx = f"[{''.join(str(i) for i in self.index)}]"
        else:
            idx = f"[{self.index}]"
        if self.locrel is None:
            at = ""
        elif self.locrel == "outer":
            at = f"@@"
        elif self.locidx is None:
            at = f"@{self.locname}"
        elif isinstance(self.locidx, tuple):
            at = f"@{self.locname}[{''.join(str(i) for i in self.locidx)}]"
        else:
            at = f"@{self.locname}[{self.locidx}]"
        return f"{self.name}{idx}{at}"

    def __html__(self, h):
        h.write(self.name)
        if self.index is not None:
            with h("span", style="color:#808;"):
                if isinstance(self.index, tuple):
                    h.write(f"[{''.join(str(i) for i in self.index)}]")
                else:
                    h.write(f"[{self.index}]")
        if self.locrel is not None:
            with h("span", style="color:#088;"):
                if self.locrel == "outer":
                    h.write(f"@@")
                else:
                    h.write(f"@{self.locname}")
            if self.locidx is not None:
                with h("span", style="color:#808;"):
                    if isinstance(self.locidx, tuple):
                        h.write(f"[{''.join(str(i) for i in self.locidx)}]")
                    else:
                        h.write(f"[{self.locidx}]")

    def make(self, loc, index):
        if self.locrel is None:
            self._assert(self.name in loc.var, f"no local variable {self.name}")
            self.decl = loc.var[self.name]
        elif self.locrel == "outer":
            self._assert(loc.parent is not None, "no outer location")
            self._assert(self.name in loc.parent.var, f"no outer variable {self.name}")
            self.decl = loc.parent.var[self.name]
        else:
            self._assert(self.locname in loc.sub, f"no inner location {self.locname!r}")
            self._assert(
                self.name in loc.sub[self.locname].var,
                f"no variable {self.name} in {self.locname}",
            )
            self._assert(
                loc.sub[self.locname].size is not None
                if self.locidx is not None
                else True,
                f"indexing non-array location {self.locname!r}",
            )
            self._assert(
                0 <= self.locidx < loc.sub[self.locname].size
                if isinstance(self.locidx, int)
                else True,
                f"invalid index for {self.locname!r}" f"[{loc.sub[self.locname].size}]",
            )
            self.decl = loc.sub[self.locname].var[self.name]
        if isinstance(self.index, str) and self.index != "self":
            index.setdefault(self.index, set(self.decl.domain))
            index[self.index] &= self.decl.domain
        elif isinstance(self.index, tuple) and self.index[0] != "self":
            index.setdefault(self.index[0], set(self.decl.domain))
            index[self.index[0]] &= self.decl.domain
        if isinstance(self.locidx, str):
            index.setdefault(self.locidx, set(range(loc.sub[self.locname].size)))
            index[self.locidx] &= set(range(loc.sub[self.locname].size))
        elif isinstance(self.locidx, tuple):
            index.setdefault(self.locidx[0], set(range(loc.sub[self.locname].size)))
            index[self.locidx[0]] &= set(range(loc.sub[self.locname].size))

    def _bind_index(self, index, vals, doms):
        if index is None:
            return None
        elif isinstance(index, int):
            return index
        elif isinstance(index, str):
            if index not in vals:
                return index
            elif index == "self":
                return vals["self"]
            elif (ret := vals[index]) in doms[index]:
                return ret
        elif index[0] in vals:
            idx = vals[index[0]]
            if index[1] == "+":
                if (ret := idx + index[2]) in doms[index[0]]:
                    return ret
            elif index[1] == "-":
                if (ret := idx - index[2]) in doms[index[0]]:
                    return ret
        raise BindError("unsupported index {index!r}")

    def bind(self, vals, doms):
        var = VarUse(
            self.line,
            self.column,
            self.name,
            self._bind_index(self.index, vals, doms),
            self.locrel,
            self.locname,
            self._bind_index(self.locidx, vals, doms),
        )
        var.decl = self.decl
        return var

    def gal(self, path):
        if self.decl.isbool:
            vtype = "bool"
        elif self.decl.clock is not None:
            vtype = "clock"
        else:
            vtype = "int"
        if self.index is None:
            name = self.name
        else:
            name = f"{self.name}.{self.index}"
        if self.locrel is None:
            where = ".".join(path) + ("." if path else "")
        elif self.locrel == "outer":
            where = ".".join(path[:-1]) + ("." if path[:-1] else "")
        else:
            where = ".".join(path) + ("." if path else "") + f"{self.locname}."
        if self.locidx is not None:
            where += f"{self.locidx}."
        return f"{vtype}_{where}{name}"

    def vars(self, path):
        if self.locrel is None:
            yield path, self
        elif self.locrel == "outer":
            yield path[:-1], self
        elif self.locidx is None:
            yield path + [self.locname], self
        else:
            yield path + [f"{self.locname}[]"], self


@dataclass(unsafe_hash=True)
class BinOp(_Element):
    line: int  # source line number
    column: Optional[int]  # source line column
    left: Union[VarUse, int]  # expression left-hand side
    op: str  # infix operator
    right: Union[VarUse, Self, int, None]  # expression right-hand side

    def __str__(self):
        right = self.right if self.right is not None else "*"
        return f"{self.left}{self.op}{right}"

    def __html__(self, h):
        if (
            isinstance(self.left, VarUse)
            and self.left.decl.isbool
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

    def make(self, loc, index):
        if isinstance(self.left, VarUse):
            self.left.make(loc, index)
        if isinstance(self.right, VarUse):
            self.right.make(loc, index)
        if self.right is None:
            self._assert(
                isinstance(self.left, VarUse) and self.left.decl.clock is not None,
                f"cannot assign '*' to non-clocked variable",
            )

    def bind(self, vals, doms):
        return BinOp(
            self.line,
            self.column,
            self.left if isinstance(self.left, int) else self.left.bind(vals, doms),
            self.op,
            self.right
            if isinstance(self.right, int) or self.right is None
            else self.right.bind(vals, doms),
        )

    def gal(self, loc):
        left = f"{self.left}" if isinstance(self.left, int) else self.left.gal(loc)
        if self.right is None:
            right = -1
        elif isinstance(self.right, int):
            right = self.right
        else:
            right = self.right.gal(loc)
        if isinstance(self.right, BinOp):
            return f"{left} {self.op} ({right})"
        else:
            return f"{left} {self.op} {right}"

    def vars(self, path):
        if isinstance(self.left, VarUse):
            yield from self.left.vars(path)
        if isinstance(self.right, (VarUse, BinOp)):
            yield from self.right.vars(path)


@dataclass(unsafe_hash=True)
class Assignment(_Element):
    line: int  # source line number
    column: Optional[int]  # source line column
    target: VarUse  # assigned variable
    value: Union[VarUse, BinOp, int, None]  # expression assigned to variable

    def __str__(self):
        return f"{self.target}={self.value if self.value is not None else '*'}"

    def __html__(self, h):
        if self.target.decl.isbool:
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

    def make(self, loc, index):
        self.target.make(loc, index)
        if self.value is None:
            self._assert(
                self.target.decl.clock is not None,
                "cannot assign '*' to non clocked variable",
            )
        elif not isinstance(self.value, int):
            self.value.make(loc, index)
        self._assert(
            not isinstance(self.value, int)
            or self.value is None
            or self.value in self.target.decl.domain,
            f"out-of-domain assignment {self}",
        )

    def bind(self, vals, doms):
        return Assignment(
            self.line,
            self.column,
            self.target.bind(vals, doms),
            self.value
            if isinstance(self.value, int) or self.value is None
            else self.value.bind(vals, doms),
        )

    def gal(self, loc):
        tgt = self.target.gal(loc)
        if self.value is None:
            yield f"{tgt} = -1;"
        elif isinstance(self.value, int):
            yield f"{tgt} = {self.value};"
        else:
            yield f"{tgt} = {self.value.gal(loc)};"
            yield f"if ({tgt} < {min(self.target.decl.domain)}) {{ abort; }}"
            yield f"if ({tgt} > {max(self.target.decl.domain)}) {{ abort; }}"

    def cond(self):
        return BinOp(self.line, self.column, self.target, "==", self.value)

    def vars(self, path):
        yield from self.target.vars(path)
        if isinstance(self.value, (VarUse, BinOp)):
            yield from self.value.vars(path)


@dataclass(unsafe_hash=True)
class Action(_Element):
    line: int  # source line number
    left: Tuple[BinOp, ...]  # conditions
    right: Tuple[Assignment, ...]  # assignments
    tags: Tuple[str, ...] = ()  # tags
    quantifier: dict = dictfield()  # free indexes quantifiers
    index: dict = dictfield()  # free indexes domains

    def __str__(self):
        q = []
        if qall := [v for v, q in self.quantifier.items() if q == "all"]:
            q.append("for all " + ", ".join(qall))
        if qany := [v for v, q in self.quantifier.items() if q == "any"]:
            q.append("for any " + ", ".join(qany))
        return "".join(
            [
                f"[{', '.join(self.tags)}] " if self.tags else "",
                ", ".join(str(c) for c in self.left),
                " >> ",
                ", ".join(str(a) for a in self.right),
                " " if q else "",
                ", ".join(q),
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
        comma = False
        for q in ("all", "any"):
            if quanti := [k for k, v in self.quantifier.items() if v == q]:
                if comma:
                    h.write(",")
                h.write(" ")
                with h("span", style="color:#008; font-weight:bold;"):
                    h.write(f"for {q}")
                h.write(" " + ", ".join(quanti))
                comma = True

    def make(self, loc):
        for cond in self.left:
            cond.make(loc, self.index)
        for assign in self.right:
            assign.make(loc, self.index)
        if (idx := set(self.index)) < (quanti := set(self.quantifier)):
            self._error(f"unknown domains for {', '.join(quanti - idx)}")
        elif idx > quanti:
            self._error(f"missing quantifier for {', '.join(idx - quanti)}")

    def expand_any(self, binding={}):
        if not any(q == "any" for q in self.quantifier.values()):
            yield self.bind(binding).expand_all(), None
        else:
            names = [v for v, q in self.quantifier.items() if q == "any"]
            for prod in product(*(self.index[v] for v in names)):
                vals = dict(zip(names, prod))
                vals.update(binding)
                try:
                    yield self.bind(vals).expand_all(), vals
                except BindError:
                    pass

    def bind(self, vals):
        return Action(
            self.line,
            tuple(left.bind(vals, self.index) for left in self.left),
            tuple(right.bind(vals, self.index) for right in self.right),
            self.tags,
            {v: q for v, q in self.quantifier.items() if v not in vals},
            {v: d for v, d in self.index.items() if v not in vals},
        )

    def expand_all(self):
        doms = {v: self.index[v] for v, q in self.quantifier.items() if q == "all"}
        names = list(doms)
        expand = {}
        for side in ("left", "right"):
            newside = defaultdict(set)
            oldside = getattr(self, side)
            for item in oldside:
                for prod in product(*(doms[n] for n in names)):
                    vals = dict(zip(names, prod))
                    try:
                        newside[item].add(item.bind(vals, doms))
                    except BindError:
                        pass
            expand[side] = reduce(or_, newside.values())
        return Action(
            self.line,
            expand["left"],
            expand["right"],
            self.tags,
            {v: q for v, q in self.quantifier.items() if v not in doms},
            {v: d for v, d in self.index.items() if v not in doms},
        )


@dataclass(unsafe_hash=True)
class Location(_Element):
    line: int  # source line number
    variables: Tuple[VarDecl, ...]  # local variables
    constraints: Tuple[Action, ...]  # constraints
    rules: Tuple[Action, ...]  # rules
    locations: Tuple["Location", ...]  # nested locations
    name: Optional[str] = None  # name if nested
    size: Optional[int] = None  # size of array
    clocks: Dict[str, str] = dictfield()  # clock names -> description

    def __post_init__(self):
        self._assert(self.size is None or self.size > 0, "invalid array size")
        self.variables = tuple(self.variables)
        self.constraints = tuple(self.constraints)
        self.rules = tuple(self.rules)
        self.locations = tuple(self.locations)
        self.sub = {loc.name: loc for loc in self.locations}
        self.var = {v.name: v for v in self.variables}

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

    def __iter__(self):
        yield from self._iter()

    def _iter(self, path=[]):
        yield path, self
        for loc in self.locations:
            if loc.size:
                yield from loc._iter(path + [f"{loc.name}[]"])
            else:
                yield from loc._iter(path + [loc.name])

    def make(self, top=None, parent=None, path=[]):
        self.parent = parent
        if top is None:
            top = self
        for var in self.variables:
            if var.clock:
                var._assert(
                    var.clock in top.clocks, f"clock {var.clock} is not defined"
                )
                top.clocked[var.clock].append(var)
        for loc in self.locations:
            if loc.size is None:
                sub = path + [loc.name]
            else:
                sub = path + [f"{loc.name}[]"]
            loc.make(top, self, sub)
        for kind in ("constraints", "rules"):
            for num, act in enumerate(getattr(self, kind), start=1):
                act.make(self)
                act.name = ".".join(path + [f"{kind[0].upper()}{num}"])


#
# parser
#


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


BOOL = {0, 1}


@v_args(inline=True)
class MRRTrans(Transformer):
    def start(self, clocks, model):
        if clocks:
            model.clocks.update(clocks)
            model.clocked = {c: [] for c in clocks}
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
            line, variables or (), constraints or (), rules or (), locations
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
        model.name = name.value
        model.line = name.line
        if size is not None:
            model.size = _int(size, 1, None, "invalid array size")
        return model

    def clockdecl(self, name, desc):
        return {name.value: desc.value}

    def vardecl_long(self, typ, name, init, desc):
        typ, size = typ
        if isinstance(typ, tuple):
            clock, typ = typ
        else:
            clock = None
        if isinstance(init, Tuple):
            init = [i & typ for i in init]
        elif init is None:
            if not clock:
                init = typ
        else:
            init = init & typ
        if size is None:
            _assert(
                not isinstance(init, Tuple),
                name.line,
                None,
                "cannot initialise scalar with array",
            )
        else:
            if not isinstance(init, Tuple):
                init = [init] * size
            elif init.shape != typ.size:
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
            name.line, name.value, BOOL, None, init, desc.value.strip(), True
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
        left, right, quanti = [], [], {}
        for arg in args:
            if isinstance(arg, BinOp):
                left.append(arg)
            elif isinstance(arg, Assignment):
                right.append(arg)
            elif isinstance(arg, dict):
                quanti.update(arg)
            else:
                _assert(
                    arg is None or isinstance(arg, str),
                    line,
                    None,
                    f"invalid condition or assignment {arg}",
                )
        return Action(line, left, right, tags or (), quantifier=quanti)

    def quantifier(self, op, *args):
        quanti = {}
        for arg in args:
            if isinstance(arg, dict):
                quanti.update(arg)
            else:
                quanti[arg.value] = op.value
        return quanti

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

    def at_outer(self, at):
        return "outer"

    def index(self, idx, op=None, inc=None):
        if op is not None and inc is not None:
            return idx.value, op.value, int(inc.value)
        else:
            try:
                idx = idx.value
                idx = int(idx)
            except ValueError:
                pass
            return idx

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
    NL_type = "_NL"
    OPEN_PAREN_types = []
    CLOSE_PAREN_types = []
    INDENT_type = "_INDENT"
    DEDENT_type = "_DEDENT"
    tab_len = 4


def parse_src(src, *defs, pattern=False):
    vardef = {}
    for d in defs:
        try:
            v, t = d.split("=", 1)
            vardef[v] = t
        except Exception:
            vardef[d] = None
    if pattern:
        LS = Lark_StandAlone_Pattern
    else:
        LS = Lark_StandAlone
    parser = LS(transformer=MRRTrans(), postlex=MRRIndenter())
    return parser.parse(cpp(src, **vardef))


def parse(path, *defs, pattern=False):
    return parse_src(open(path).read(), *defs, pattern=pattern)
