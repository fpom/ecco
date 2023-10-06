import re, subprocess

from functools import reduce
from itertools import chain
from operator import or_
from typing import Set, Sequence, Optional, Union

from .mrrparse import Lark_StandAlone, Transformer, v_args, Token, ParseError, Indenter

##
## errors
##

def _error (line, column, message, errcls=ParseError) :
    if column is not None :
        err = errcls(f"[{line}:{column}] {message}")
    else :
        err = errcls(f"[{line}] {message}")
    err.line, err.column = line, column
    raise err

def _assert (test, line, column, message, errcls=ParseError) :
    if not test :
        _error(line, column, message, errcls)

##
## source preprocessing
##

_cpp_head = subprocess.run(["cpp", "--traditional", "-C"],
                           input="", encoding="utf-8",
                           capture_output=True).stdout

_line_dir = re.compile('^#\s+([0-9]+)\s+"<stdin>"\s*$')

def cpp (text, **var) :
    out = subprocess.run(["cpp", "--traditional", "-C"]
                         + [f"-D{k}={v}" for k, v in var.items()],
                         input=text, encoding="utf-8", capture_output=True).stdout
    out = out.replace(_cpp_head, "")
    # we don't use cpp -P but remove line directives because otherwise some
    # empty lines are missing at the beginning
    # + we remove comments
    return "".join((line.split("#", 1)[0]).rstrip() + "\n"
                   for line in out.splitlines()
                   if not _line_dir.match(line))

##
## model
##

class _Element (object) :
    def __init__ (self, line: int, column: Optional[int]=None) :
        self.line = line
        self.column = column
    def _error (self, message, errcls=ParseError) :
        _error(self.line, self.column, message, errcls)
    def _assert (self, test, message, errcls=ParseError) :
        _assert(test, self.line, self.column, message, errcls)

def _repr_pretty_ (obj, fields, p, cycle) :
    cls = obj.__class__.__name__
    if cycle :
        p.text("<%s ...>" % cls)
    else :
        with p.group(len(cls)+1, f"{cls}(", ")") :
            for i, name in enumerate(fields) :
                if i :
                    p.text(", ")
                    p.breakable()
                with p.group(len(name)+1, f"{name}=", "") :
                    value = getattr(obj, name)
                    if isinstance(value, tuple) :
                        value = list(value)
                    p.pretty(value)

class VarDecl (_Element) :
    def __init__ (self,
                  # source line number
                  line: int,
                  # variable name
                  name: str,
                  # variable domain (possible values)
                  domain: Set[int],
                  # if not None: array length
                  size: Optional[int],
                  # initial values
                  init: Union[int, Sequence[int]],
                  # textual description
                  description: str) :
        super().__init__(line)
        self._assert(size is None or size > 0, "invalid size")
        init = self._init_dom(init, domain)
        if size is None :
            self._assert(isinstance(init, set), "invalid initial state")
        elif isinstance(init, set) :
            init = [init] * size
        else :
            self._assert(len(init) == size, "array/init size mismatch")
        self.name = name
        self.domain = domain
        self.size = size
        self.init = init
        self.description = description
    def _init_dom (self, init, dom) :
        if init is None :
            init = dom
        elif isinstance(init, int) :
            init = {init} & dom
        elif isinstance(init, set) :
            init = init & dom
        elif isinstance(init, Sequence) :
            return [self._init_dom(i, dom) for i in init]
        else :
            self._error("invalid initial state")
        self._assert(init, "invalid initial state")
        return init
    def __str__ (self) :
        if self.size is None :
            idx = ""
        else :
            idx = f"[{self.size}]"
        return (f"{{{min(self.domain)}..{max(self.domain)}}}{idx} {self.name}"
                f" = {self.init} : {self.description}")
    def __repr__ (self) :
        return (f"{self.__class__.__name__}({self.line}, {self.column}, {self.name!r},"
                f" {self.domain!r}, {self.size}, {self.init!r}, {self.description!r})")

class VarUse (_Element) :
    def __init__ (self,
                  # source line/column numbers
                  line: int, column: int,
                  # variable name
                  name: str,
                  # if not None, array index
                  index: Optional[Union[int, str]]=None,
                  # if not None, variable in other location
                  locrel: Optional[str]=None,
                  # if not None, inner location name
                  locname: Optional[str]=None,
                  # if not None, inner location index
                  locidx: Optional[Union[int, str]]=None) :
        super().__init__(line, column)
        self._assert(locrel in (None, "outer", "inner"),
                     f"invalid location relation {locrel!r}",
                     ValueError)
        self._assert(locname is None if locrel is None else True,
                     "locname should be None",
                     TypeError)
        self._assert(locidx is None if locrel is None else True,
                     "locidx should be None",
                     TypeError)
        self._assert(locname is None if locrel == "outer" else True,
                     "outer location cannot be named",
                     TypeError)
        self._assert(locidx is None if locrel == "outer" else True,
                     "outer location cannot be indexed",
                     TypeError)
        self._assert(locname is not None if locrel == "inner" else True,
                     "missing value for locname",
                     TypeError)
        self.name = name
        self.index = index
        self.locrel = locrel
        self.locname = locname
        self.locidx = locidx
    def __str__ (self) :
        if self.index is None :
            idx = ""
        else :
            idx = f"[{self.index}]"
        if self.locrel is None :
            at = ""
        elif self.locrel == "outer" :
            at = f"@@"
        elif self.locidx is None :
            at = f"@{self.locname}"
        else :
            at = f"@{self.locname}[{self.locidx}]"
        return f"{self.name}{idx}{at}"
    def __repr__ (self) :
        args = [str(self.line), str(self.column), repr(self.name), repr(self.index)]
        if self.locrel is not None :
            args.append(repr(self.locrel))
            if self.locname is not None :
                args.append(repr(self.locname))
                if self.locidx is not None :
                    args.append(repr(self.locidx))
        return f"{self.__class__.__name__}({', '.join(args)})"
    def make (self, loc, index) :
        if self.locrel is None :
            self._assert(self.name in loc.var,
                         f"no local variable {self.name}")
            self.decl = loc.var[self.name]
        elif self.locrel == "outer" :
            self._assert(loc.parent is not None, "no outer location")
            self._assert(self.name in loc.parent.var,
                         f"no outer variable {self.name}")
            self.decl = loc.parent.var[self.name]
        else :
            self._assert(self.locname in loc.sub,
                         f"no inner location {self.locname!r}")
            self._assert(self.name in loc.sub[self.locname].var,
                         f"no outer variable {self.name}")
            self._assert(loc.sub[self.locname].size is not None
                         if self.locidx is not None else True,
                         f"indexing non-array location {self.locname!r}")
            self._assert(0 <= self.locidx < loc.sub[self.locname].size
                         if isinstance(self.locidx, int) else True,
                         f"invalid index for {self.locname!r}"
                         f"[{loc.sub[self.locname].size}]")
            self.decl = loc.sub[self.locname].var[self.name]
        if isinstance(self.index, str) :
            index.setdefault(self.index, set(self.decl.domain))
            index[self.index] &= self.decl.domain
        if isinstance(self.locidx, str) :
            index.setdefault(self.locidx, set(range(loc.sub[self.locname].size)))
            index[self.locidx] &= set(range(loc.sub[self.locname].size))

class BinOp (_Element) :
    def __init__ (self,
                  # source line/column numbers
                  line: int, column: int,
                  # expression left-hand side
                  left: Union[VarUse, int],
                  # infix operator
                  op: str,
                  # expression right-hand side
                  right: Union[VarUse, int]) :
        super().__init__(line, column)
        self.left = left
        self.op = op
        self.right = right
    def __str__ (self) :
        return f"{self.left}{self.op}{self.right}"
    def __repr__ (self) :
        return (f"{self.__class__.__name__}({self.line}, {self.column},"
                f" {self.left!r}, {self.op!r}, {self.right!r})")
    def make (self, loc, index) :
        if isinstance(self.left, VarUse) :
            self.left.make(loc, index)
        if isinstance(self.right, VarUse) :
            self.right.make(loc, index)

class Assignment (_Element) :
    def __init__ (self,
                  # source line/column numbers
                  line: int, column: int,
                  # assigned variable
                  target: VarUse,
                  # expression assigned to variable
                  value: Union[VarUse, BinOp, int]) :
        super().__init__(line, column)
        self.target = target
        self.value = value
    def __str__ (self) :
        return f"{self.target}={self.value}"
    def __repr__ (self) :
        return f"{self.__class__.__name__}({self.line}, {self.target!r}, {self.value!r})"
    def _repr_pretty_ (self, p, cycle) :
        _repr_pretty_(self, ["target", "value", "line", "column"], p, cycle)
    def make (self, loc, index) :
        self.target.make(loc, index)
        if not isinstance(self.value, int) :
            self.value.make(loc, index)

class Action (_Element) :
    def __init__ (self,
                  # source line number
                  line: int,
                  # conditions
                  left: Sequence[BinOp],
                  # assignments
                  right: Sequence[Assignment],
                  # tags
                  tags: Sequence[str]=[]) :
        super().__init__(line)
        self.left = tuple(left)
        self.right = tuple(right)
        self.tags = tuple(tags)
        self.index = {}
    def __str__ (self) :
        return "".join([f"[{', '.join(self.tags)}] " if self.tags else "",
                        ", ".join(str(c) for c in self.left),
                        " >> ",
                        ", ".join(str(a) for a in self.right)])
    def __repr__ (self) :
        return (f"{self.__class__.__name__}({self.line}"
                f", [{', '.join(repr(c) for c in self.left)}]"
                f", [{', '.join(repr(a) for a in self.right)}]"
                f", {self.tags!r}")
    def _repr_pretty_ (self, p, cycle) :
        _repr_pretty_(self, ["left", "right", "tags", "line"], p, cycle)
    def make (self, loc) :
        for cond in self.left :
            cond.make(loc, self.index)
        for assign in self.right :
            assign.make(loc, self.index)
        for idx in self.index :
            pass # check that idx is not a variable

class Location (_Element) :
    def __init__ (self,
                  # source line number
                  line: int,
                  # local variables
                  variables: Sequence[VarDecl],
                  # constraints
                  constraints: Sequence[Action],
                  # rules
                  rules: Sequence[Action],
                  # nested locations
                  locations: Sequence["Location"],
                  # name if nested
                  name: Optional[str]=None,
                  # size of array
                  size: Optional[int]=None) :
        super().__init__(line)
        if size == 0 :
            raise ValueError("invalid array size")
        self.variables = tuple(variables)
        self.constraints = tuple(constraints)
        self.rules = tuple(rules)
        self.locations = tuple(locations)
        self.name = name
        self.size = size
        self.sub = {loc.name : loc for loc in locations}
        self.var = {v.name : v for v in variables}
    def __str__ (self) :
        parts = ["@"]
        if self.name is not None :
            parts.append(f"{self.name}")
        if self.size is not None :
            parts.append(f"[{self.size}]")
        parts.append(f"(V={len(self.variables)},"
                     f"C={len(self.constraints)},"
                     f"R={len(self.rules)},"
                     f"L={len(self.locations)})")
        return "".join(parts)
    def __repr__ (self) :
        args = [str(self.line),
                repr(self.variables),
                repr(self.constraints),
                repr(self.rules),
                repr(self.locations)]
        if self.name is not None :
            args.append(repr(self.name))
        if self.size is not None :
            args.append(repr(self.size))
        return f"{self.__class__.__name__}({', '.join(args)})"
    def _repr_pretty_ (self, p, cycle) :
        _repr_pretty_(self, ["variables", "constraints", "rules", "locations",
                             "name", "size", "line"], p, cycle)
    @property
    def actions (self) :
        yield from self.constraints
        yield from self.rules
    def make (self, parent=None) :
        self.parent = parent
        for loc in self.locations :
            loc.make(self)
        for act in self.actions :
            act.make(self)

##
## parser
##

def _int (tok, min=None, max=None, message="invalid int litteral") :
    try :
        val = int(tok.value)
        if min is not None :
            assert val >= min
        if max is not None :
            assert val <= max
        return val
    except :
        _error(tok.line, tok.column, message)

@v_args(inline=True)
class MRRTrans (Transformer) :
    def start (self, model) :
        model.make()
        return model
    def model (self, variables, constraints, rules, *locations) :
        line = None
        for obj in chain(variables, constraints, rules, locations) :
            line = getattr(obj, "line", None)
            if line is not None :
                break
        return Location(line,
                        variables or (),
                        constraints or (),
                        rules or (),
                        locations)
    def sequence (self, *args) :
        return args
    def location (self, name, size, model) :
        model.name = name.value
        model.line = name.line
        if size is not None :
            model.size = _int(size, 1, None, "invalid array size")
        return model
    def vardecl_long (self, typ, name, init, desc) :
        typ, size = typ
        if isinstance(init, Sequence) :
            init = [i & typ for i in init]
        else :
            init & typ
        if size is None :
            _assert(not isinstance(init, Sequence), name.line, None,
                    "cannot initialise scalar with array")
        else :
            if not isinstance(init, Sequence) :
                init = [init] * size
            elif init.shape != typ.size :
                _error(name.line, None, "type/init size mismatch")
        return VarDecl(name.line, name.value, typ, size, init, desc.value.strip())
    def vardecl_short (self, name, sign, desc) :
        if sign.value == "+" :
            init = {1}
        elif sign.value == "-" :
            init = {0}
        elif sign.value == "*" :
            init = {0, 1}
        else :
            _error(name.line, None, f"unexpected initial value {sign.value!r}")
        return VarDecl(name.line, name.value, {0, 1}, None, init, desc.value.strip())
    def type (self, dom, size) :
        if size is None :
            return dom, None
        else :
            return dom, _int(size, 1, None, "invalid array size")
    def type_bool (self) :
        return {0, 1}
    def type_interval (self, start, end) :
        return set(range(_int(start), _int(end)+1))
    def init (self, *args) :
        if any(a is None for a in args) :
            return None
        else :
            return reduce(or_, args)
    def init_int (self, min, max) :
        if max is None :
            return {_int(min)}
        else :
            return set(range(_int(min), _int(max)+1))
    def init_all (self) :
        return None
    def actdecl (self, tags, *args) :
        if tags is not None :
            tags = tuple(t.strip() for t in tags.value[1:-1].split(",") if t.strip())
            line = tags.line
        else :
            line = args[0].line
        left, right = [], []
        for arg in args :
            if isinstance(arg, BinOp) :
                left.append(arg)
            else :
                right.append(arg)
        return Action(line, left, right, tags or ())
    def condition_long (self, left, op, right) :
        line, column = left.line, left.column
        if isinstance(left, Token) :
            column -= len(left.value) - 1
            left = _int(left)
        if isinstance(right, Token) :
            right = _int(right)
        return BinOp(line, column, left, op, right)
    def condition_short (self, var, op) :
        if op.value == "+" :
            return BinOp(var.line, var.column, var, "==", 1)
        elif op.value == "-" :
            return BinOp(var.line, var.column, var, "==", 0)
        else :
            _error(op.line, op.column + 1 - len(op.value),
                   f"invalid condition {op.value!r}")
    def var (self, name, index, at) :
        if at is None :
            locrel = locname = locidx = None
        elif at == "outer" :
            locrel, locname, locidx = "outer", None, None
        else :
            locrel, locname, locidx = at
        return VarUse(name.line, name.column + 1 - len(name.value), name.value, index,
                      locrel, locname, locidx)
    def at_inner (self, name, index) :
        return "inner", name.value, index
    def at_outer (self, at) :
        return "outer"
    def index (self, idx) :
        try :
            idx = idx.value
            idx = int(idx)
        except :
            pass
        return idx
    def assignment_long (self, target, assign, value) :
        if isinstance(value, Token) :
            value = _int(value)
        if assign.value[0] == "+" :
            value = BinOp(target.line, target.column, target, "+", value)
        elif assign.value[0] == "-" :
            value = BinOp(target.line, target.column, target, "-", value)
        return Assignment(target.line, target.column, target, value)
    def assignment_short (self, var, op) :
        if op == "++" :
            return Assignment(var.line, var.column,
                              var, BinOp(var.line, var.column, var, "+", 1))
        elif op == "--" :
            return Assignment(var.line, var.column,
                              var, BinOp(var.line, var.column, var, "-", 1))
        elif op == "+" :
            return Assignment(var.line, var.column, var, 1)
        elif op == "-" :
            return Assignment(var.line, var.column, var, 0)
        elif op == "~" :
            return Assignment(var.line, var.column,
                              var, BinOp(var.line, var.column, 1, "-", var))
        else :
            _error(op.line, op.column, "invalid assignment {op.value!r}")

class MRRIndenter (Indenter) :
    NL_type = "_NL"
    OPEN_PAREN_types = []
    CLOSE_PAREN_types = []
    INDENT_type = "_INDENT"
    DEDENT_type = "_DEDENT"
    tab_len = 4

def parse (text) :
    parser = Lark_StandAlone(transformer=MRRTrans(), postlex=MRRIndenter())
    return parser.parse(cpp(text))
