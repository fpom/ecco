import collections, ast, token, itertools, re, pathlib

from unidecode import unidecode

from .. import CompileError, Record
from .rrparse import rrParser

# to be imported from the outside
from tatsu.exceptions import FailedParse

def flatten (value) :
    if isinstance(value, (str, bytes)) :
        yield value
    else :
        try :
            for item in value :
                for child in flatten(item) :
                    yield child
        except TypeError :
            yield value

##
## data classes
##

sign2sign = {True : True,
             "+" : True,
             False : False,
             "-" : False,
             None : None,
             "*" : None}

sign2char = {True : "+",
             False : "-",
             None : "*"}

sign2neg = {True : False,
            False : True,
            None : None}

class State (Record) :
    _fields = ["name", "sign"]
    def __init__ (self, name, sign) :
        super().__init__(name, sign)
        self.sign = sign2sign[self.sign]
    def __str__ (self) :
        return "".join([self.name, sign2char[self.sign]])
    def neg (self) :
        return self.__class__(self.name, sign2neg[self.sign])
    def __hash__ (self) :
        return hash(str(self))
    def __eq__ (self, other) :
        try :
            return self.name == other.name and self.sign == other.sign
        except :
            return False
    def __ne__ (self, other) :
        return not self.__eq__(other)
    def __lt__ (self, other) :
        return str(self) < str(other)
    def __le__ (self, other) :
        return str(self) <= str(other)
    def __gt__ (self, other) :
        return not self.__le__(other)
    def __ge__ (self, other) :
        return not self.__lt__(other)
    def __invert__ (self) :
        return self.__class__(self.name, sign2neg[self.sign])
    def __pos__ (self) :
        return self.__class__(self.name, True)
    def __neg__ (self) :
        return self.__class__(self.name, False)

class Rule (Record) :
    _fields = ["left", "right"]
    _options = ["parent", "num", "label"]
    def vars (self) :
        return set(s.name for s in self.left + self.right)
    def normalise (self) :
        missing = [s for s in self.left if s not in self.right and ~s not in self.right]
        if missing :
            return self.__class__(self.left, self.right + missing,
                                  num=self.num, parent=self, label=self.label)
        else :
            return self
    def elementarise (self) :
        rule = self.normalise()
        toadd = [[]]
        diff = [s for s in rule.right if s not in rule.left and ~s not in rule.left]
        if not diff :
            yield rule
        else :
            for state in diff :
                toadd = ([t + [state] for t in toadd]
                         + [t + [state.neg()] for t in toadd])
            for n, t in enumerate(toadd, start=1) :
                yield rule.__class__(rule.left + list(sorted(t)), rule.right,
                                     parent=self, label=self.label, num=n)
    def text (self) :
        return "%s >> %s" % (", ".join(s.name + ("+" if s.sign else "-")
                                       for s in self.left),
                             ", ".join(s.name + ("+" if s.sign else "-")
                                       for s in self.right))
    def name (self) :
        return self.__class__.__name__[0] + str(self.num or "")
    def full_name (self) :
        if not self.parent :
            return self.name()
        else :
            return ":".join([self.parent.name(), str(self.num or "")])
    def __str__ (self) :
        if self.label :
            return "[%s] %s  # %s" % (self.label, self.text(), self.name())
        else :
            return "%s  # %s" % (self.text(), self.name())
    def __hash__ (self) :
        return hash((self.__class__.__name__, self.left, self.right))
    def __eq__ (self, other) :
        try :
            return (self.__class__ == other.__class__
                    and self.left == other.left
                    and self.right == other.right)
        except :
            return False
    def __ne__ (self, other) :
        return not self.__eq__(other)
    def __lt__ (self, other) :
        return (str(self.left), str(self.right)) < (str(other.left), str(other.right))
    def __le__ (self, other) :
        return (str(self.left), str(self.right)) <= (str(other.left), str(other.right))
    def __gt__ (self, other) :
        return not self.__le__(other)
    def __ge__ (self, other) :
        return not self.lt__(other)
    def inline (self, other) :
        left, right = set(self.left), set(self.right)
        reach = right | {s for s in left if ~s not in right}
        cond, assign = set(other.left), set(other.right)
        if cond & right and not any(~c in reach for c in cond) :
            missing = cond - reach
            yield self.__class__(self.left + [c for c in missing if c not in reach],
                                 [s for s in self.right if ~s not in assign]
                                 + [s for s in other.right if s not in reach],
                                 parent=self,
                                 label=self.label)
            for m in missing :
                yield self.__class__(self.left + [~m],
                                     self.right,
                                     parent=self,
                                     label=self.label)
        else :
            yield self

class Constraint (Rule) :
    pass

class Sort (Record) :
    _fields = ["state", "kind", "description"]
    def __lt__ (self, other) :
        if self.state.sign is None and other.state.sign is None :
            return self.state.name < other.state.name
        elif self.state.sign is None :
            return self.state.name < other.state.name or True
        elif other.state.sign is None :
            return self.state.name < other.state.name or False
        else :
            return (self.state.name, self.state.sign) < (other.state.name,
                                                         other.state.sign)
    def __eq__ (self, other) :
        return (self.state.name, self.state.sign) == (other.state.name, other.state.sign)

class Spec (Record) :
    _fields = ["meta", "constraints", "rules"]
    def __init__ (self, meta, constraints, rules) :
        super().__init__(meta, constraints, rules)
        self.meta = tuple(meta)
        self.constraints = tuple(self._renum(constraints))
        self.rules = tuple(self._renum(rules))
        self.labels = {r.name() : r.label for r in self if r.label}
        self.validate()
    def _renum (self, rules) :
        n = 1 + max((r.num or 0 for r in rules), default=0)
        for r in rules :
            if not r.num :
                r.num = n
                n += 1
            yield r
    def inline (self, constraints) :
        constraints = set(constraints)
        keep = []
        drop = []
        for c in self.constraints :
            if c.name() in constraints or c.num in constraints :
                drop.append(c)
            else :
                keep.append(c)
        rules = []
        for r in self.rules :
            todo = [r]
            for c in drop :
                done = []
                for s in todo :
                    done.extend(s.inline(c))
                todo = done
            rules.extend(todo)
        return self.__class__(self.meta, keep, rules)
    def validate (self) :
        decl = set(d.state.name for d in self.meta)
        for v in decl :
            if re.match("^[RC][0-9]+$", v) :
                raise CompileError("variable name %r is a rule/constraint name" % v)
        used = decl.copy()
        for r in itertools.chain(self.constraints, self.rules) :
            for s in itertools.chain(r.left, r.right) :
                used.discard(s.name)
                if s.name not in decl :
                    raise CompileError("'%s' not declared (used in '%s')" % (s.name, r))
        if used :
            print("! declared but not used: %s" % ", ".join(repr(n) for n in used))
    def save (self, target=None, overwrite=False) :
        if hasattr(target, "write") :
            self._save(target)
        else :
            path = pathlib.Path(target or self.path)
            if path.exists() and not overwrite :
                raise ValueError(f"file {target!r} exists")
            with open(path, "w") as out :
                self._save(out)
    def _save (self, out) :
        out.write("components:\n")
        for m in self.meta :
            out.write("    %s: %s\n" % (m.state, m.description))
        if self.constraints :
            out.write("\nconstraints:\n")
            for c in self.constraints :
                out.write("    %s\n" % c.text())
        if self.rules :
            out.write("\nrules:\n")
            for r in self.rules :
                out.write("    %s\n" % r.text())
    def __iter__ (self) :
        return itertools.chain(self.constraints, self.rules)
    def __str__ (self) :
        return "\n".join([
            "\n".join("# %s (%s): %s"
                      % (sort.state, sort.kind, sort.description)
                      for sort in self.meta),
            "\n".join(str(rule) for rule in self.constraints) or "## no constraints",
            "\n".join(str(rule) for rule in self.rules)
        ])
    def init (self) :
        return set(State(s.state.name, s.state.sign) for s in self.meta)

##
## parser
##

_tok_skip = {"COMMENT" , "ENDMARKER", "BACKQUOTE"}

for num, name in token.tok_name.items() :
    if name in _tok_skip :
        _tok_skip.remove(name)
        _tok_skip.add(num)

class Parser (object) :
    def __init__ (self, path, source=None) :
        self._path = path
        self._parser = rrParser()
        src = (source
               or open(path, encoding="utf-8", errors="replace").read()) + "\n"
        self._source = unidecode(src)
    def parse (self, rule="start") :
        self._meta = []
        self._constraints = []
        self._rules = []
        self._parser.parse(self._source, rule, semantics=self)
        count = collections.Counter(s.state.name for s in self._meta)
        duplicated = [var for var, num in count.most_common() if num > 1]
        if duplicated :
            raise CompileError("duplicated entity: %s" % ", ".join(duplicated))
        return Spec(self._meta, self._constraints, self._rules)
    def varstate (self, st) :
        """
        variable:word state:/[+-]/
        """
        return State(st.variable, st.state)
    def varinit (self, st) :
        """
        variable:word state:/[*+-]/
        """
        return State(st.variable, sign2sign[st.state])
    def vdecl (self, st) :
        """
        name:word ":" {nl}+
        decl:{varstate ":" description:/.*?$/ {nl}+}+
        """
        self._meta.extend(Sort(state, st.name, description.split("#", 1)[0].strip())
                          for state, _, description, _ in st.decl)
    def rule (self, st) :
        """
        { "[" label:/[^\\]]*/ "]" }?
        ","%{ left:varstate }+ ">>" ","%{ right:varstate }+ {nl}+
        """
        if st.label :
            label = st.label.strip() or None
        else :
            label = None
        if isinstance(st.left, list) :
            left = st.left
        else :
            left = [st.left]
        if isinstance(st.right, list) :
            right = st.right
        else :
            right = [st.right]
        return left, right, label
    def cdecl (self, st) :
        """
        "constraints" ":" {nl}+
        rules:{rule}+
        """
        start = len(self._constraints) + 1
        for i, (l, r, lbl) in enumerate(st.rules) :
            self._constraints.append(Constraint(l, r, num=start+i, label=lbl))
    def rdecl (self, st) :
        """
        "rules" ":" {nl}+
        rules:{rule}+
        """
        start = len(self._rules) + 1
        for i, (l, r, lbl) in enumerate(st.rules) :
            self._rules.append(Rule(l, r, num=start+i, label=lbl))
    def string (self, st) :
        return ast.literal_eval("".join(flatten(st)))
