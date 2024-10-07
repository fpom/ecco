import re
import sympy
import networkx as nx

from .. import CompileError, ExprVisitor, Record

# next statement doesn't work because sympy.printing.pretty is masking name pretty
# >>> import sympy.printing.pretty.pretty as pretty
# workaround:
import importlib
pretty = importlib.import_module("sympy.printing.pretty.pretty")

from itertools import product, chain

_NEXTLOC = sympy.Symbol(".NEXTLOC")
_THISLOC = sympy.Symbol(".THISLOC")
_FORALOC = sympy.Symbol(".FORALOC")

class Star (object) :
    _count = 0
    _snums = re.compile(r"^\*[0-9]+\*$")
    def __call__ (self) :
        s = sympy.Symbol("*%s*" % self._count)
        self.__class__._count += 1
        return s
    @classmethod
    def star (cls, obj) :
        try :
            return cls._snums.match(str(obj))
        except :
            return False

newstar = Star()

class RenewStars (ExprVisitor) :
    def __init__ (self, expr) :
        self.renew = {}
        self(expr)
        self.new = expr.subs(self.renew, simultaneous=True)
    def Symbol (self, expr) :
        if Star.star(expr) :
            self.renew[expr] = newstar()
        self.GENERIC(expr)
    def GENERIC (self, expr) :
        for a in expr.args :
            self(a)

class GetThisConst (ExprVisitor) :
    def __init__ (self, *expressions) :
        self.found = {}
        for expr in expressions :
            self(expr)
    def Indexed (self, expr) :
        if (len(expr.indices) == 1
            and expr.indices[0].is_constant()) :
            self.found[expr] = expr.indices[0]
    def GENERIC (self, expr) :
        for a in expr.args :
            self(a)

def subs (expr, args) :
    e = expr.subs(args, simultaneous=True)
    return e.subs(GetThisConst(e).found)

class GetLocQueries (ExprVisitor) :
    def __init__ (self, *expressions, match=_NEXTLOC) :
        self.match = match
        self.found = []
        for expr in expressions :
            self(expr)
    def Indexed (self, expr) :
        if expr.base.label == self.match :
            self.found.append(expr)
    def GENERIC (self, expr) :
        for a in expr.args :
            self(a)

class _Record (Record) :
    def __str__ (self) :
        _str = getattr(self, "_str", [])
        return "<%s%s%s>" % (self.__class__.__name__,
                             " " if _str else "",
                             ":".join(str(getattr(self, name)) for name in _str))

class Instance (_Record) :
    _fields = ["name", "location", "arguments", "labels"]
    _str = ["name"]

class Variable (_Record) :
    _fields = ["name", "lower", "upper", "init"]
    _options = ["decl"]
    _str = ["name"]
    _multiline = False
    def subs (self, args) :
        # this declaration hides a previous declaration or parameter
        args.pop(self.name, None)
        return self.__class__(name=self.name,
                              lower=self.lower,
                              upper=self.upper,
                              init=subs(self.init, args),
                              decl=self)

class PrintExpr (pretty.PrettyPrinter) :
    def _print_Indexed (self, expr) :
        if len(expr.indices) == 1 :
            if expr.base.label == _THISLOC :
                return pretty.prettyForm(str(self._print(expr.indices[0])))
            else :
                return pretty.prettyForm("%s@%s" % (self._print(expr.indices[0]),
                                                    self._print(expr.base)))
        elif expr.base.label in (_NEXTLOC, _FORALOC) :
            var, *path = expr.indices
            parts = []
            for p in path :
                fwd, edge, node = p.indices
                if fwd :
                    parts.append("?")
                else :
                    parts.append("!")
                if edge is not sympy.S.true :
                    parts.append("{%s}" % self._print(p.indices[0]))
                if Star.star(p.base.label) :
                    parts.append("*")
                else :
                    parts.append(str(p.base))
                if node is not sympy.S.true :
                    parts.append("{%s}" % self._print(p.indices[1]))
            if expr.base.label == _FORALOC :
                return pretty.prettyForm("[%s@%s]" % (var, "".join(parts)))
            else :
                return pretty.prettyForm("%s@%s" % (var, "".join(parts)))
        else :
            return pretty.prettyForm("<Unsupported expression: %r>" % expr)
    def _print_Equality (self, expr) :
        return pretty.prettyForm(" == ".join(self.doprint(a) for a in expr.args))
    def _print_Tuple (self, expr) :
        return pretty.prettyForm("%s := %s" % (self.doprint(expr[0]),
                                               self.doprint(expr[1])))
    def _print_Or (self, expr) :
        return pretty.prettyForm(" | ".join(self.doprint(a) for a in expr.args))
    def _print_And (self, expr) :
        return pretty.prettyForm(" & ".join(self.doprint(a) for a in expr.args))
    def _print_Not (self, expr) :
        txt = str(pretty.PrettyPrinter._print_Not(self, expr))
        return pretty.prettyForm("~" + txt[1:])
    def _print_Symbol (self, expr) :
        if expr.name.startswith("CONST:") :
            name = expr.name[6:]
        else :
            name = expr.name
        return pretty.prettyForm(name)

txt = PrintExpr().doprint

class Rule (_Record) :
    _fields = ["condition", "assignment"]
    _options = ["decl", "specification", "location"]
    def subs (self, args) :
        return self.__class__(condition=[c for c in (subs(cc, args)
                                                     for cc in self.condition)
                                         if c != sympy.S.true],
                              assignment=[subs(a, args) for a in self.assignment],
                              decl=self,
                              specification=self.specification,
                              location=self.location)
    def expand (self) :
        return self.__class__(condition=list({e for c in self.condition
                                              for e in self._expand(c)}
                                             - {sympy.S.true}),
                              assignment=list({e for a in self.assignment
                                               for e in self._expand(a)}
                                              - {sympy.S.true}),
                              decl=self,
                              specification=self.specification,
                              location=self.location)
    def _expand (self, expr) :
        queries = GetLocQueries(expr, match=_FORALOC).found
        if not queries :
            return [expr]
        matches = []
        replace = []
        for q in queries :
            s, l, m = self._qmatch(q)
            if not all(Star.star(k) for d in m for k in d) :
                raise CompileError("Matching node name is forbidden within %r" % txt(q))
            matches.append(m)
            replace.append(sympy.Indexed(s, l))
        concrete = expr.subs(dict(zip(queries, replace)))
        expanded = []
        for match in product(*matches) :
            d = dict(chain.from_iterable(m.items() for m in match))
            expanded.append(concrete.subs(d))
        return expanded
    def _qmatch (self, query) :
        svar = query.indices[-1].base.label
        lvar = query.indices[0]
        match = []
        path = self.specification.graph.path
        for p in path(self.location.name,
                      [(n.indices[0], n.indices[1], n.base.label, n.indices[2])
                       for n in query.indices[1:]]) :
            inst = p.get(svar.name, None)
            if inst and lvar.name in self.specification.instances[inst].varmap :
                match.append(p)
        return svar, lvar, match
    def split (self) :
        queries = GetLocQueries(*self.condition, *self.assignment).found
        replace = []
        matches = []
        for q in queries :
            svar, lvar, m = self._qmatch(q)
            replace.append(sympy.Indexed(svar, lvar))
            matches.append(m)
        concrete = self.subs(dict(zip(queries, replace)))
        for prod in product(*matches) :
            yield concrete.subs(dict(chain.from_iterable(p.items() for p in prod)))
    _txt = "->"
    def txt (self) :
        return " ".join([", ".join(txt(c) for c in self.condition),
                         self._txt,
                         ", ".join(txt(a) for a in self.assignment)])

class Constraint (Rule) :
    _txt = "=>"

class Location (_Record) :
    _fields = ["name", "parameters", "variables", "constraints", "rules", "labels"]
    _options = ["decl", "instance", "specification"]
    _str = ["name"]
    def __init__ (self, **args) :
        super().__init__(**args)
        self.varmap = {v.name : v for v in self.variables}
        for r in chain(self.constraints, self.rules) :
            r.specification = self.specification
            r.location = self
    def __call__ (self, spec, instance) :
        args = dict(zip(self.parameters, instance.arguments))
        args[_THISLOC] = sympy.Symbol(instance.name)
        return self.__class__(name=instance.name,
                              parameters=[],
                              variables=[v.subs(args) for v in self.variables],
                              constraints=[c.subs(args) for c in self.constraints],
                              rules=[r.subs(args) for r in self.rules],
                              labels=subs(sympy.FiniteSet(self.name)
                                          | self.labels
                                          | instance.labels,
                                          args),
                              decl=self,
                              instance=instance,
                              specification=spec)
    def txt (self) :
        return "%s(%s)" % (self.decl.name,
                           ", ".join(str(a) for a in self.instance.arguments))
    def expand (self) :
        return self.__class__(name=self.name,
                              parameters=[],
                              variables=self.variables,
                              constraints=[c.expand() for c in self.constraints],
                              rules=[r.expand() for r in self.rules],
                              labels=self.labels,
                              decl=self.decl,
                              instance=self.instance,
                              specification=self.specification)

class Spec (_Record) :
    _fields = ["locations", "instances", "graph"]
    def __init__ (self, **args) :
        super().__init__(**args)
        self.locations = {loc.name : loc for loc in self.locations}
        self.instances = {inst.name : self.locations[inst.location](self, inst)
                          for inst in self.instances}
        for name, inst in self.instances.items() :
            self.graph.add_node(inst.name, labels=inst.labels)
        for name, inst in self.instances.items() :
            self.instances[name] = inst.expand()

def match (expr, labels) :
    env = {"CONST:%s" % l if isinstance(l, (int, sympy.Integer)) else str(l) : True
           for l in labels}
    env.update((str(v), False) for v in expr.free_symbols if str(v) not in env)
    return expr.subs(env) is sympy.S.true

class Graph (nx.DiGraph) :
    def __init__ (self, init) :
        nx.DiGraph.__init__(self)
        if isinstance(init, str) :
            self.add_node(init)
        else :
            for g in init :
                self.add_nodes_from(g.nodes.data())
                self.add_edges_from(g.edges.data())
    def path (self, start, spec) :
        (forward, edge, var, cond), *rest = spec
        if var is not None :
            var = str(var)
        if forward :
            onestep = self.successors
        else :
            onestep = self.predecessors
        for succ in onestep(start) :
            if forward :
                elabels = self.edges[start, succ]["labels"]
            else :
                elabels = self.edges[succ, start]["labels"]
            if (match(edge, elabels) and match(cond, self.nodes[succ]["labels"])) :
                if rest :
                    for tail in self.path(succ, rest) :
                        if var is not None and var not in tail :
                            yield dict(tail, **{var: succ})
                        elif var is None :
                            yield tail
                elif var is None :
                    yield {}
                else :
                    yield {var : succ}
