import sympy

from .. import CompileError, ExprVisitor
from .st import txt

##
## check whether a DR spec can be translated to RR
##

class DReIsRR (ExprVisitor) :
    def __init__ (self, spec, loc, expr) :
        self.spec = spec
        self.loc = loc
        self.variable = False
        self.value = False
        self(expr)
        if not (self.variable and self.value) :
            raise CompileError("unsupported condition %r" % txt(expr))
    def Indexed (self, expr) :
        if self.variable :
            raise CompileError("unsupported comparison %r" % txt(expr))
        loc = expr.base.label.name
        if loc not in self.spec.instances :
            raise CompileError("undefined location %r in %r" % (loc, txt(expr)))
        elif len(expr.indices) != 1 or not isinstance(expr.indices[0], sympy.Symbol) :
            raise CompileError("undefined variable in %r" % txt(expr))
        var = expr.indices[0].name
        if var not in self.spec.instances[loc].varmap :
            raise CompileError("undefined variable '%s@%s' in location %r"
                               % (var, loc, self.loc.txt()))
        self.variable = True
    def Symbol (self, expr) :
        if self.variable :
            raise CompileError("unsupported comparison %r" % txt(expr))
        elif expr.name not in self.loc.varmap :
            raise CompileError("undefined variable %r in location %r"
                               % (expr.name, self.loc.txt()))
        self.variable = True
    def One (self, expr) :
        if self.value :
            raise CompileError("unsupported condition %r" % txt(expr))
        self.value = True
    def Zero (self, expr) :
        if self.value :
            raise CompileError("unsupported condition %r" % txt(expr))
        self.value = True
    def Equality (self, expr) :
        if len(expr.args) != 2 :
            raise CompileError("unsupported condition %r" % txt(expr))
        for a in expr.args :
            self(a)
    def GENERIC (self, expr) :
        raise CompileError("unsupported expression %r" % txt(expr))

class DRaIsRR (ExprVisitor) :
    def __init__ (self, spec, expr) :
        self.spec = spec
        self(expr)
    def Tuple (self, expr) :
        if len(expr) != 2 :
            raise CompileError("unsupported assignement %r" % txt(expr))
        elif expr[1] not in [0, 1] :
            raise CompileError("non-Boolean assignement %r" % txt(expr))
    def GENERIC (self, expr) :
        raise CompileError("unsupported expression %r" % txt(expr))

class CheckRR (object) :
    def __init__ (self, spec) :
        for loc in spec.instances.values() :
            for var in loc.variables :
                if var.lower != 0 or var.upper != 1 :
                    raise CompileError("unsupported range %s..%s for variable %s@%s"
                                       % (var.lower, var.upper,
                                          var.name, var.location.name))
        for loc in spec.instances.values() :
            for kind in ["constraints", "rules"] :
                for rule in getattr(loc, kind) :
                    for sub in rule.split() :
                        for cond in sub.condition :
                            DReIsRR(spec, loc, cond)
                        for assign in sub.assignment :
                            DRaIsRR(spec, assign)

##
## translate sympy expressions to RR
##

class DRe2RR (ExprVisitor) :
    def __init__ (self, loc) :
        self.loc = loc
    def Equality (self, expr) :
        return "".join(self(arg) for arg in expr.args)
    def Indexed (self, expr) :
        return "%s_AT_%s" % (expr.args[1], expr.args[0])
    def Symbol (self, expr) :
        return "%s_AT_%s" % (expr.name, self.loc.name)
    def Integer (self, expr) :
        return "-+"[expr]
    def Zero (self, expr) :
        return "-"
    def One (self, expr) :
        return "+"

class DRa2RR (ExprVisitor) :
    def Tuple (self, expr) :
        self.target = expr[0]
        return self(expr[0]) + self(expr[1])
    def Indexed (self, expr) :
        return "%s_AT_%s" % (expr.args[1], expr.args[0])
    def Add (self, expr) :
        return "".join(self(arg) for arg in expr.args if arg != self.target)
    def Integer (self, expr) :
        return "-+"[expr]
    def Zero (self, expr) :
        return "-"
    def One (self, expr) :
        return "+"
    def NegativeOne (self, expr) :
        return "-"

##
## translate DR spec to RR
##

class DR2RR (object) :
    @classmethod
    def rr (cls, spec, out) :
        cls(spec, out).do_spec()
    def __init__ (self, spec, out) :
        CheckRR(spec)
        self.spec, self.out = spec, out
    def do_spec (self) :
        # variables declarations
        for loc in self.spec.instances.values() :
            self.out.write("# declarations for '%s:=%s'\n" % (loc.name, loc.txt()))
            self.out.write("%s_AS_%s:\n" % (loc.decl.name, loc.name))
            for var in loc.variables :
                self.out.write("    %s_AT_%s%s: implements '%s := %s'\n"
                               % (var.name, loc.name, "+" if var.init else "-",
                                  var.name, var.init))
            self.out.write("\n")
        # constraints and rules
        for i, kind in enumerate(["constraints", "rules"]) :
            if i :
                self.out.write("\n")
            if getattr(loc, kind) :
                self.out.write("%s:\n" % kind)
            else :
                self.out.write("# no %s\n" % kind)
            for loc in self.spec.instances.values() :
                self.out.write("    ## %s for '%s := %s'\n"
                               % (kind, loc.name, loc.txt()))
                for rule in getattr(loc, kind) :
                    self.do_rule(rule, loc)
    def do_rule (self, rule, location) :
        self.out.write("    # %s\n" % rule.txt())
        for sub in rule.split() :
            e2r = DRe2RR(sub.location)
            self.out.write("    [%s] " % location.name)
            for i, cond in enumerate(sub.condition) :
                if i :
                    self.out.write(", ")
                self.out.write(e2r(cond))
            self.out.write(" >> ")
            a2r = DRa2RR()
            for i, assign in enumerate(sub.assignment) :
                if i :
                    self.out.write(", ")
                self.out.write(a2r(assign))
            self.out.write("\n")
