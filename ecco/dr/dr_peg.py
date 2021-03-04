import io, re
import sympy
import fastidious

from . import st

##
## insert IDENT/DEDENT tokens
##

_indent = re.compile(r"^\s*")

def indedent (text, indent="↦", dedent="↤") :
    "insert INDENT/DEDENT tokens into text"
    stack = [0]
    if isinstance(text, str) :
        src = io.StringIO(text)
    else :
        src = text
    out = io.StringIO()
    for i, line in enumerate(src) :
        if line.rstrip() :
            width = _indent.match(line).end()
            if width > stack[-1] :
                out.write(indent)
                stack.append(width)
            elif width < stack[-1] :
                while width < stack[-1] :
                    out.write(dedent)
                    stack.pop()
                if width != stack[-1] :
                    raise IndentationError("line %s" % (i+1))
        out.write(line)
    while stack[-1] > 0 :
        out.write(dedent)
        stack.pop()
    return out.getvalue()

##
## parser
##

def symlabels (labels=None) :
    if labels :
        return sympy.FiniteSet(*(sympy.Symbol(l) if isinstance(l, str) else l
                                 for l in labels))
    else :
        return sympy.EmptySet

class DeerParser (fastidious.Parser) :
    __default__ = "INPUT"

    def __INIT__ (self) :
        self.locations = {}
        self.instances = {}

    __grammar__ = r"""
    INPUT  <- NL? :spec EOF {@spec}
    NL     <- (_ (~"#[^\n]*")? "\n" _)+  {_drop}
    AT     <- _ "@" _ {p_flatten}
    LP     <- _ "(" _ {p_flatten}
    RP     <- _ ")" _ {p_flatten}
    LSB    <- _ "[" _ {p_flatten}
    RSB    <- _ "]" _ {p_flatten}
    LCB    <- _ "{" _ {p_flatten}
    RCB    <- _ "}" _ {p_flatten}
    TILDA  <- _ "~" _ {p_flatten}
    STAR   <- _ "*" _ {p_flatten}
    COMMA  <- _ "," _ {p_flatten}
    COLON  <- _ ":" _ {_drop}
    INDENT <- _ "↦" NL? _ {_drop}
    DEDENT <- _ "↤" NL? _ {_drop}
    EOF    <- !.
    _      <- [\t ]* {_drop}
    NUMBER <- _ [+-]? _ [0-9]+ _
    NAME   <- _ [a-z]i ([a-z0-9_]i)* _ {p_flatten}
    """
    def _drop (self, match) :
        return ""
    def _list (self, match) :
        """callback to transform matched comma-separated lists
        <- item (COMMA item)*
        => [item, ...]
        """
        lst = []
        for m in match :
            if isinstance(m, list) :
                lst.extend(i[1] for i in m)
            else :
                lst.append(m)
        return lst
    def on_NUMBER (self, match) :
        return sympy.Integer(self.p_flatten(match))

    __grammar__ += """
    spec <- locations:(locdef+) instances:(instance+) :graph"""
    def on_spec (self, match, locations, instances, graph) :
        return st.Spec(locations=locations,
                       instances=instances,
                       graph=graph)

    __grammar__ += """
    locdef    <- AT name:NAME labels:(labels?) LP params:(params?) RP COLON NL
                INDENT location:location DEDENT
    location  <- variables:(vardef+) rules:(rule+)
    params    <- NAME (COMMA NAME)* {_list}
    labels    <- LCB lbl:(arguments?) RCB {@lbl}
    arguments <- arg (COMMA arg)* {_list}
    arg       <- NUMBER / NAME
    """
    def on_locdef (self, match, name, labels, params, location) :
        if name in self.locations :
            self.p_parse_error("location %r is already defined" % name, self.last_pos)
        loc = self.locations[name] = st.Location(name=name,
                                                 parameters=params,
                                                 labels=symlabels(labels),
                                                 **location)
        return loc
    def on_location (self, match, variables, rules) :
        return {"variables" : variables,
                "constraints" : [r for r in rules if isinstance(r, st.Constraint)],
                "rules" : [r for r in rules if not isinstance(r, st.Constraint)]}

    __grammar__ += """
    vardef     <- name:NAME bounds:(boundaries?) _ ":=" _ init:expr NL
    boundaries <- COLON lower:NUMBER _ ".." _ upper:NUMBER
    """
    def on_vardef (self, match, name, bounds, init) :
        if bounds :
            lower, upper = bounds
        else :
            lower, upper = 0, 1
        return st.Variable(name=name,
                           lower=lower,
                           upper=upper,
                           init=init)
    def on_boundaries (self, match, lower, upper) :
        return min(lower, upper), max(lower, upper)

    __grammar__ += """
    rule        <- :conditions _ priority:("=>" / "->") _ (:assignments) NL
    conditions  <- cond (COMMA cond)* {_list}
    cond        <- left:expr _ op:("==" / "<=" / ">=" / "<" / ">" / "!=") _ right:expr
    assignments <- assign (COMMA assign)* {_list}
    assign      <- target:variable _ op:("+=" / "-=" / ":=") _ expr:expr
    """
    def on_rule (self, match, conditions, priority, assignments) :
        if priority == "=>" :
            return st.Constraint(condition=conditions,
                                 assignment=assignments)
        else :
            return st.Rule(condition=conditions,
                           assignment=assignments)
    _cmpop = {
        "==" : sympy.Eq,
        "<=" : sympy.Le,
        ">=" : sympy.Ge,
        "<" : sympy.Lt,
        ">" : sympy.Gt,
        "!=" : lambda a, b: sympy.Not(sympy.Eq(a, b)),
    }
    def on_cond (self, match, left, op, right) :
        return self._cmpop[op](left, right)
    def on_assign (self, match, target, op, expr) :
        if op == "+=" :
            expr = sympy.Add(target, expr)
        elif op == "-=" :
            expr = sympy.Add(target, -expr)
        return sympy.Tuple(target, expr)

    __grammar__ += """
    variable <- name:NAME location:(atloc?) / LSB name:NAME AT forall:(neighbor+) RSB
    atloc    <- AT (name:NAME / path:(neighbor+))
    expr     <- first:term rest:([+-] term)*
    term     <- first:fact rest:([*/%] fact)* {on_expr}
    fact     <- (LP fact:expr RP) / fact:atom {@fact}
    atom     <- value:(NUMBER / variable) {@value}
    neighbor <- _ direction:[?!] _ edge:(lblmatch?) name:(NAME / STAR) loc:(lblmatch?)
    lblmatch <- LCB expr:lblexpr RCB {@expr}
    lblexpr  <- first:lblterm rest:(_ "|" _ lblterm)* {on_expr}
    lblterm  <- first:lblfact rest:(_ "&" _ lblterm)* {on_expr}
    lblfact  <- (TILDA neg:lblfact) / (LP sub:lblexpr RP) / atom:arg
    """
    def on_variable (self, match, name, location=None, forall=None) :
        if forall is not None :
            return sympy.Indexed(st._FORALOC, name, *forall)
        elif not location :
            return sympy.Indexed(st._THISLOC, name)
        elif isinstance(location, list) :
            return sympy.Indexed(st._NEXTLOC, name, *location)
        else :
            return sympy.Indexed(location, name)
    def on_atloc (self, match, name, path=None) :
        if path is None :
            return name
        else :
            return path
    def on_neighbor (self, match, direction, edge, name, loc) :
        if name == "*" :
            return sympy.Indexed(st.newstar(), direction == "?",
                                 edge or sympy.S.true, loc or sympy.S.true)
        else :
            return sympy.Indexed(name,  direction == "?",
                                 edge or sympy.S.true, loc or sympy.S.true)
    def on_lblfact (self, match, sub=None, neg=None, atom=None) :
        if neg is not None :
            return sympy.Not(neg)
        elif isinstance(atom, (int, sympy.Integer)) :
            return sympy.Symbol("CONST:%s" % atom)
        elif isinstance(atom, str) :
            return sympy.Symbol(atom)
        else :
            return sub
    _binop = {
        "+" : sympy.Add,
        "-" : lambda a, b: sympy.Add(a, -b),
        "*" : sympy.Mul,
        "/" : lambda a, b: sympy.Mul(a, b**-1),
        "%" : sympy.Mod,
        "|" : sympy.Or,
        "&" : sympy.And,
    }
    def on_expr (self, match, first, rest) :
        expr = first
        for op, tr in rest :
            expr = self._binop[op](expr, tr)
        return expr

    __grammar__ += """
    instance  <- name:NAME labels:(labels?) _ ":=" _ location:NAME
                 LP args:(arguments?) RP NL
    """
    def on_instance (self, match, name, labels, location, args) :
        if name in self.instances :
            self.p_parse_error("instance %r is already defined" % name, self.last_pos)
        elif location not in self.locations :
            self.p_parse_error("location %r is not defined" % location, self.last_pos)
        elif len(args) < len(self.locations[location].parameters) :
            self.p_parse_error("not enough arguments for instance %r of %r"
                               % (name, location), self.last_pos)
        elif len(args) > len(self.locations[location].parameters) :
            self.p_parse_error("too many arguments for instance %r of %r"
                               % (name, location), self.last_pos)
        inst = self.instances[name] = st.Instance(name=name,
                                                  location=location,
                                                  arguments=args,
                                                  labels=symlabels(labels))
        return inst

    __grammar__ += r"""
    graph <- first:subg rest:(edge? subg)*
    subg  <- __ node:NAME __ / __ "(" sub:graph ")" __
    edge  <- left:ARC labels:(arguments?) right:ARC
    ARC   <- __ arc:[<>] __ {@arc}
    __    <- ~"([ \t\n\r↦↤]*(#[^\n]*)?)*" {_drop}
    """
    def on_graph (self, match, first, rest) :
        graphs = [first] + [g for _, g in rest]
        g = st.Graph(graphs)
        for edge, (source, target) in zip([e for e, g in rest],
                                          zip(graphs, graphs[1:])) :
            if not edge :
                continue
            way, lbl = edge
            for src in source :
                for tgt in target :
                    if ">" in way :
                        old = g.edges.get((src, tgt), {}).get("labels", symlabels())
                        g.add_edge(src, tgt, labels=old | lbl)
                    if "<" in way :
                        old = g.edges.get((tgt, src), {}).get("labels", symlabels())
                        g.add_edge(tgt, src, labels=old | lbl)
        return g
    def on_subg (self, match, node=None, sub=None) :
        if sub is not None :
            return sub
        elif node not in self.instances :
            self.p_parse_error("instance %r is not defined" % node, self.last_pos)
        else :
            return st.Graph(node)
    def on_edge (self, match, left, labels, right) :
        return left+right, symlabels(labels)
