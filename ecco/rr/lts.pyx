# distutils: language = c++
# distutils: include_dirs = ../pyddd ../libDDD ../libITS

import inspect
import sympy

from ddd cimport ddd, sdd, shom
from its cimport model

from ddd import ddd_save, ddd_load

cdef extern from "dddwrap.h" :
    ctypedef short val_t

cdef inline ddd s2d (sdd s) :
    for t in s.edges() :
        return t.value

cdef inline sdd d2s (ddd d) :
    return sdd.mkz(d)

cdef class LTS (object) :
    """a Labelled Transition System

    Class LTS is basically a wrapper around an `its.model` instance.
    Its attributes are:
     - `path`: GAL file used to build the model
     - `gal`: instance of `its.model` built from the GAL file
     - `init`: a `ddd.sdd` containing the initial states
     - `states`: a `ddd.sdd` containing the set of all states
     - `dead`: a `ddd.sdd` containing the set of all deadlocks
     - `hull`: a `ddd.sdd` containing the set of states forming the SCC hull
     - `succ`: a `ddd.shom` representing the successor function
     - `succ_o`: a `ddd.shom` representing the gfp of `succ`
     - `succ_s`: a `ddd.shom` representing the lfp of `succ`
     - `pred`: a `ddd.shom` representing the predessor function
     - `pred_o`: a `ddd.shom` representing the gfp of `pred`
     - `pred_s`: a `ddd.shom` representing the lfp of `pred`
     - `props`: a `dict` mapping textual properties to sets of states in the LTS
     - `tsucc`: a `dict` mapping rules and constraints names to `ddd.shom`
        successor functions
     - `tpred`: a `dict` mapping rules and constraints names to `ddd.shom`
       predecessor functions
     - `vars`: a truple of `str` representing the variables of the model
    """
    cdef readonly str path
    cdef readonly model gal
    cdef readonly sdd init, states, dead, hull
    cdef readonly shom succ, pred, succ_o, pred_o, succ_s, pred_s
    cdef readonly dict props, alias, tsucc, tpred
    cdef readonly tuple vars
    cdef dict _var2sdd
    cdef readonly bint compact
    cdef readonly shom constraints
    cdef readonly sdd transient
    cpdef void save_file (LTS self, str path) :
        """save LTS to file `path`

        Parameters:
         - `path` (`str`): path where to save the LTS
        """
        cdef dict dump = self.dump()
        cdef list ddds = dump.pop("DDD")
        ddd_save(path, *ddds, **dump)
    cpdef dict dump (LTS self) :
        cdef list ddds = [s2d(self.init), s2d(self.states),
                          s2d(self.dead), s2d(self.hull),
                          s2d(self.transient)]
        cdef list props = []
        cdef str p
        cdef sdd s
        for p, s in self.props.items() :
            props.append(p)
            ddds.append(s2d(s))
        return {"path" : self.path,
                "props" : props,
                "alias" : self.alias,
                "vars" : self.vars,
                "compact" : self.compact,
                "DDD" : ddds}
    @classmethod
    def load (cls, dict dump) :
        cdef list props
        cdef LTS lts
        cdef str p
        cdef ddd d, init, states, dead, hull, transient
        init, states, dead, hull, transient, *props = dump["DDD"]
        lts = LTS.__new__(LTS, dump["path"])
        lts.gal = model(lts.path, fmt="GAL")
        lts.vars = dump["vars"]
        lts.compact = dump["compact"]
        lts.alias = dump["alias"]
        lts.init = d2s(init)
        lts.states = d2s(states)
        lts.dead = d2s(states)
        lts.hull = d2s(hull)
        lts.transient = d2s(transient)
        for p, d in zip(dump["props"], props) :
            lts.props[p] = d2s(d)
        lts._build_succ()
        if lts.compact :
            lts._build_compact()
        lts._build_pred()
        return lts
    @classmethod
    def load_file (cls, str path) :
        """load a previously saved LTS

        Parameters:
         - `path` (`str`): path from which the LTS is loaded
        Returns: an instance of `LTS`
        """
        cdef dict dump
        cdef list ddds
        dump, ddds = ddd_load(path)
        dump["DDD"] = ddds
        return cls.load(dump)
    def __cinit__ (self, str path, object init="", bint compact=True) :
        self.path = path
        self.compact = compact
        self.props = {}
        self.alias = {}
        self.tsucc = {}
        self.tpred = {}
        self._var2sdd = {}
    cpdef LTS copy (LTS self) :
        cdef LTS lts = LTS.__new__(LTS, self.path)
        lts.gal = self.gal
        lts.init = self.init
        lts.states = self.states
        lts.dead = self.dead
        lts.hull = self.hull
        lts.succ = self.succ
        lts.pred = self.pred
        lts.succ_o = self.succ_o
        lts.pred_o = self.pred_o
        lts.succ_s = self.succ_s
        lts.pred_s = self.pred_s
        lts.props.update(self.props) # not shared among instances
        lts.alias.update(self.alias) # not shared among instances
        lts.tsucc = self.tsucc
        lts.tpred = self.tpred
        lts.vars = self.vars
        lts._var2sdd = self._var2sdd
        lts.compact = self.compact
        lts.constraints = self.constraints
        lts.transient = self.transient
        return lts
    def __eq__ (self, other) :
        return (self.path == other.path
                and self.init == other.init)
    def __hash__ (self) :
        return hash(("ecco.lts.lTS", self.path, self.init))
    def __init__ (self, str path, object init="", bint compact=True) :
        """creates an LTS instance

        Parameters:
         - `path` (`str`): path of a GAL file from which the LTS has to be created
         - `init` (`str=""` or `list[str]`): initial set of states
        """
        self.gal = model(path, fmt="GAL")
        self.vars = s2d(self.gal.initial()).vars()
        self._build_succ()
        if isinstance(init, str) :
            self._build_initial_states([init])
        else :
            self._build_initial_states(list(init))
        if init == "*" :
            self.states = self.init
        else :
            self._build_reachable_states()
        if self.compact :
            self._build_compact()
        self._build_pred()
        self._build_dead_states()
        self._build_hull_states()
    cdef void _build_succ (LTS self) :
        # build the successor relations
        cdef dict d = self.gal.transitions()
        cdef list constraints = []
        cdef str t
        cdef shom h, c
        for t, h in d.items() :
            if t.startswith("C") :
                constraints.append(h)
                if not self.compact :
                    self.tsucc[t] = h
            else :
                self.tsucc[t] = h
        self.constraints = shom.union(*constraints)
        if self.compact :
            c = self.constraints.lfp()
            self.succ = c * shom.union(*self.tsucc.values())
            for t, h in self.tsucc.items() :
                self.tsucc[t] = c * h
        else :
            self.succ = shom.union(*self.tsucc.values())
        self.succ_o = self.succ.gfp()
        self.succ_s = self.succ.lfp()
    cdef void _build_initial_states (LTS self, list init) except * :
        cdef str s
        cdef shom t
        self.init = sdd.empty()
        for s in init :
            self.init |= self._build_initial_states_one(s)
        if self.compact :
            t = self.constraints.lfp()
            self.init = t(self.init)
    cdef sdd _build_initial_states_one (LTS self, str init) :
        cdef ddd d
        cdef str v, s
        cdef dict i
        cdef int a, b
        if init == "" :
            return self.gal.initial()
        elif init == "*" :
            d = ddd.one()
            for v in reversed(self.vars) :
                d = ddd.from_range(v, 0, 1, d)
            return d2s(d)
        elif init == "+" :
            d = ddd.one()
            for v in reversed(self.vars) :
                d = ddd.from_range(v, 1, 1, d)
            return d2s(d)
        elif init == "-" :
            d = ddd.one()
            for v in reversed(self.vars) :
                d = ddd.from_range(v, 0, 0, d)
            return d2s(d)
        else :
            # note that GAL has a unique initial state
            # but init is set appropriately from ComponentGraph.from_model
            i = {v : (a, a) for v, a in next(s2d(self.gal.initial()).items()).items()}
            for s in init.split(",") :
                if s == "" :
                    pass
                elif s == "*" :
                    for v in i :
                        i[v] = (0,1)
                elif s == "+" :
                    for v in i :
                        i[v] = (1,1)
                elif s == "-" :
                    for v in i :
                        i[v] = (0,0)
                else :
                    v = s[:-1]
                    if v not in i :
                        raise ValueError(f"unknown variable {v!r}")
                    if s.endswith("*") :
                        i[v] = (0, 1)
                    elif s.endswith("+") :
                        i[v] = (1, 1)
                    elif s.endswith("-") :
                        i[v] = (0, 0)
                    else :
                        raise ValueError(f"invalid initial state {v!r}")
            d = ddd.one()
            for v in reversed(self.vars) :
                a, b = i[v]
                d = ddd.from_range(v, a, b, d)
            return d2s(d)
    cdef void _build_reachable_states (LTS self) :
        # build the set of reachable states
        cdef sdd reach
        cdef shom succ, pred_u
        reach = self.init
        self.states = sdd.empty()
        succ = self.succ | shom.ident()
        while True :
            reach = succ(reach)
            if reach == self.states :
                break
            self.states = reach
        pred_u = self.constraints.invert(self.states)
        self.transient = pred_u(self.states)
    cdef void _build_compact (LTS self) :
        # remove transient states from sdd and shom
        cdef shom h
        cdef str t
        self.states -= self.transient
        self.init -= self.transient
        # WARNING: `succ - transient` won't work, only `succ & states` does
        self.succ &= self.states
        for t, h in self.tsucc.items() :
            self.tsucc[t] = h & self.states
    cdef void _build_pred (LTS self) :
        # build the predecessor relations
        cdef str t
        cdef shom h
        self.pred = self.succ.invert(self.states)
        self.pred_o = self.pred.gfp()
        self.pred_s = self.pred.lfp()
        for t, h in self.tsucc.items() :
            self.tpred[t] = h.invert(self.states)
    cdef void _build_dead_states (LTS self) :
        # build the set of dead states
        self.dead = self.states - self.pred(self.states)
    cdef void _build_hull_states (LTS self) :
        # build the set of SCC hull states
        # intersect with self.states to remove potential transient states
        self.hull = (self.pred_o(self.states) & self.succ_o(self.states)) & self.states
    cpdef dict graph_props (LTS self, sdd states) :
        cdef dict cp = {}
        for name, method in sorted(inspect.getmembers(self, callable)) :
            if (name.startswith("is_")
                or name.startswith("isin_")
                or name.startswith("has_")) :
                cp[name] = method(states)
        return cp
    cpdef bint is_dead (LTS self, sdd states) :
        """check whether `states` is the set of all deadlocks

        Parameters:
         - `states` (`ddd.sdd`): set of states to be checked
        Returns: `True` if `states` is the set of all deadlocks, `False` otherwise
        """
        return states == self.dead
    cpdef bint isin_dead (LTS self, sdd states) :
        """check whether `states` contains only but not all deadlocks

        Parameters:
         - `states` (`ddd.sdd`): set of states to be checked
        Returns: `True` if `states` contains onlydeadlocks, `False` otherwise
        """
        return states < self.dead
    cpdef bint has_dead (LTS self, sdd states) :
        """check whether `states` contains deadlock

        Parameters:
         - `states` (`ddd.sdd`): set of states to be checked
        Returns: `True` if `states` contains deadlock, `False` otherwise
        """
        return bool(states & self.dead)
    cpdef bint is_init (LTS self, sdd states) :
        """check whether `states` is the set of all initial states

        Parameters:
         - `states` (`ddd.sdd`): set of states to be checked
        Returns: `True` if `states` the set of all initial states, `False` otherwise
        """
        return states == self.init
    cpdef bint isin_init (LTS self, sdd states) :
        """check whether `states` contains only but not all initial states

        Parameters:
         - `states` (`ddd.sdd`): set of states to be checked
        Returns: `True` if `states` contains only initial states, `False` otherwise
        """
        return states < self.init
    cpdef bint has_init (LTS self, sdd states) :
        """check whether `states` contains initial state

        Parameters:
         - `states` (`ddd.sdd`): set of states to be checked
        Returns: `True` if `states` contains initial state, `False` otherwise
        """
        return bool(states & self.init)
    cpdef bint is_hull (LTS self, sdd states) :
        """check whether `states` is a SCC hull (not necessarily the largest one)

        Parameters:
         - `states` (`ddd.sdd`): set of states to be checked
        Returns: `True` if `states` is a SCC hull, `False` otherwise
        """
        cdef sdd s = states & self.states
        return (self.succ_o(s) & self.pred_o(s)) == s
    cpdef bint is_scc (LTS self, sdd states) :
        """check whether `states` is a SCC

        Parameters:
         - `states` (`ddd.sdd`): set of states to be checked
        Returns: `True` if `states` is a SCC, `False` otherwise
        """
        cdef sdd i = states & self.states
        cdef sdd s
        if len(i) <= 1 :
            return False
        s = i.pick()
        return (self.succ_s(s) & self.pred_s(s)) == i
    cpdef sdd add_prop (LTS self, str prop, sdd states, bint union=False, str alias="") :
        """adds a property to the LTS

        Parameters:
         - `prop` (`str`): the textual property
         - `states` (`ddd.sdd`): the corresponding set of states in the LTS
         - `union` (`bool=False`): if the property exists and `union=True`,
           then `states` is added to it, otherwise, `states` is replaced
        Returns: `states` restricted to the states actually in the LTS
        """
        cdef sdd s = states & self.states
        if alias :
            self.alias[prop] = alias
        if union and (prop in self.props) :
            self.props[prop] |= s
            return self.props[prop]
        else :
            self.props[prop] = s
            return s
    cpdef sdd var2sdd (LTS self, str name) :
        cdef str v
        cdef ddd d
        if name not in self._var2sdd :
            d = ddd.one()
            for v in reversed(self.vars) :
                if v == name :
                    d = ddd.from_range(name, 1, 1, d)
                else :
                    d = ddd.from_range(v, 0, 1, d)
            self._var2sdd[name] = d2s(d)
        return self._var2sdd[name]
    cpdef object form (LTS self, sdd states, variables=None, normalise=None) :
        """describe a set of states by a Boolean formula

        Arguments:
         - `states` (`sdd`): a set of states
         - `variables=...`: a collection of variables names to restrict to
         - `normalise=...`: how the formula should be normalised, which must be either:
           - `"cnf"`: conjunctive normal form
           - `"dnf"`: disjunctive normal form
           - `None`: chose the smallest form
        Return: a sympy Boolean formulas
        """
        cdef dict seen = {}
        cdef ddd d = s2d(states)
        cdef set keep = set(variables or self.vars)
        return sympy.simplify_logic(self._form(d, seen, keep), form=normalise)
    cdef object _form (LTS self, ddd head, dict seen, set keep) :
        cdef object ret, sub
        cdef str var
        cdef val_t val
        cdef ddd child
        cdef list terms = []
        if head.stop() :
            seen[head] = ret = sympy.true
            return ret
        for var, num, val, child in head.edges() :
            if child in seen :
                sub = seen[child]
            else :
                sub = seen[child] = self._form(child, seen, keep)
            if var not in keep :
                terms.append(sub)
            elif val :
                terms.append(sympy.Symbol(var) & sub)
            else :
                terms.append(sympy.Not(sympy.Symbol(var)) & sub)
        if not terms :
            terms.append(sympy.true)
        seen[head] = ret = sympy.Or(*terms)
        return ret

cpdef enum setrel :
    HASNO = 0
    HAS = 1
    CONTAINS = 2
    ISIN = 3
    EQUALS = 4

ctypedef unsigned long long CompoID

cdef dict _CompoCache = {}

cdef class Component (object) :
    """a set of states with properties

    Components are meant to be used in graphs forming a partition of
    the states of an LTS. They are obtained from splitting into
    sub-components an inital component that contains all the states.

    Components may have so called split properties that related to states
    of the component to a set of states corresponding to a property.
    This properties are encoded as `setrel` values:
     - `EQUALS`: two sets of states are exactly the same
     - `ISIN`: the component's states are all into the property's states
     - `CONTAINS`: the component's states contains all of the property's states
     - `HAS`: the component and property have sets in common but none is included
       into the other
     - `HASNO`: the component and property have no sets in common

    Attributes:
     - `num` (`int`): an identifier that is the same for all the existing components
       that have the same set of states and the same underlying LTS
     - `states` (`ddd.sdd`): the set of states in the component
     - `graph_props`: a `dict` that map textual properties (`str`) to `bool`,
       limited to topological properties in the LTS
     - `split_props`: a `dict` that map textual properties (`str`) to `setrel`,
       limited to properties that have been used to split components as explained above
     - `lts`: the `LTS` from which the component is originated
    """
    cdef readonly CompoID num
    cdef readonly sdd states
    cdef readonly dict graph_props
    cdef readonly dict split_props
    cdef readonly LTS lts
    cdef readonly tuple on, off
    def __cinit__ (Component self, LTS lts, sdd states, dict gp={}, dict sp={}) :
        self.lts = lts
        self.states = states
        self.graph_props = dict(gp)
        self.split_props = dict(sp)
    def __init__ (Component self, LTS lts, sdd states, dict gp={}, dict sp={}) :
        """create an instance of a `Component`

        Parameters:
         - `lts`: the `LTS` from which the component is originated
         - `states`: the `ddd.sdd` containing the states of the component
         - `gp`: a `dict` of graph properties
         - `sp`: a `dict` of split properties
        Returns: newly created `Component`
        """
        global _CompoCache
        cdef set on, off
        if (states, lts) in _CompoCache :
            self.num = _CompoCache[states, lts]
        else :
            _CompoCache[states, lts] = self.num = len(_CompoCache)
        on, off = self.on_off()
        self.on = tuple(on)
        self.off = tuple(off)
    cpdef dict dump (Component self) :
        return {"DDD" : [s2d(self.states)],
                "num" : self.num,
                "graph_props" : self.graph_props,
                "split_props" : self.split_props,
                "on" : self.on,
                "off" : self.off}
    @classmethod
    def load (cls, dump, lts) :
        cdef Component compo = Component.__new__(Component,
                                                 lts,
                                                 d2s(dump["DDD"][0]),
                                                 dump["graph_props"],
                                                 dump["split_props"])
        compo._load(dump)
        return compo
    cdef void _load (Component self, dict dump) :
        self.num = dump["num"]
        self.on = dump["on"]
        self.off = dump["off"]
    cpdef Component copy (Component self, lts=None) :
        cdef Component compo
        if lts is None :
            compo = Component.__new__(Component, self.lts, self.states,
                                      self.graph_props, self.split_props)
        else :
            compo = Component.__new__(Component, lts, self.states,
                                      self.graph_props, self.split_props)
        compo._copy(self)
        return compo
    cdef void _copy (Component self, Component other) :
        self.num = other.num
        self.on = other.on
        self.off = other.off
    def __len__ (Component self) :
        "number of states in the component"
        return len(self.states)
    def __hash__ (Component self) :
        return hash((self.states, self.lts))
    def __eq__ (Component self, Component other) :
        return self.states == other.states and self.lts == other.lts
    @property
    def props (Component self) :
        """an iterator over the graph and split properties

        Yields: pairs of `str, bool` or `str, setrel`
        """
        yield from self.graph_props.items()
        yield from self.split_props.items()
    cpdef tuple props_row (Component self, bint alias=True) :
        cdef str p
        cdef setrel r
        cdef list props = [set() for r in setrel]
        for p, r in self.split_props.items() :
            if alias :
                props[r].add(self.lts.alias.get(p, p))
            else :
                props[r].add(p)
        return tuple(map(frozenset, props))
    cdef void _update_prop (Component self, dict pdict, str prop, sdd states) :
        # update pdict[prop] with appropriate setrel to denote the relation of states
        # wrt the prop evaluated as a set of states on the underlying LTS
        cdef sdd pstates = self.lts.props[prop]
        if not pstates :
            pdict[prop] = setrel.HASNO
        elif states == pstates :
            pdict[prop] = setrel.EQUALS
        elif states < pstates :
            pdict[prop] = setrel.ISIN
        elif states > pstates :
            pdict[prop] = setrel.CONTAINS
        elif states & pstates :
            pdict[prop] = setrel.HAS
        else :
            pdict[prop] = setrel.HASNO
    cpdef void tag (Component self, str name) :
        """add a dummy property to component

        Tag `name` is a dummy property whose states are those of the `Component`.
        LTS and component's properties are updated accordingly

        Parameters:
         - `name` (`str`): the tag name
        """
        self.lts.add_prop(name, self.states, union=True)
        self.split_props[name] = setrel.EQUALS
    cpdef setrel check (Component self, str prop, sdd states, str alias="") :
        """check a split property on component

        LTS and component's properties are updated accordingly

        Parameters:
         - `prop` (`str`): the textual property
         - `states` (`ddd.sdd`): the corresponding set of states
        Returns: a `setrel` indicating how the component relates to `states`
        """
        cdef sdd pstates = self.lts.add_prop(prop, states, alias=alias)
        self._update_prop(self.split_props, prop, self.states)
        return self.split_props[prop]
    cpdef tuple split (Component self, str prop, sdd states, str alias="") :
        """split component wrt a property

        LTS and component's properties are updated accordingly

        Parameters:
         - `prop` (`str`): the textual property
         - `states` (`ddd.sdd`): the corresponding set of states
        Returns: a pairs `intr, diff` corresponding respectively to the
        intersection and difference between the component's and the property's
        states. If one is empty then `None` is returned,
        otherwise a `Component` instance is returned
        """
        cdef sdd pstates = self.lts.add_prop(prop, states, alias=alias)
        cdef sdd intr = self.states & pstates
        cdef sdd diff = self.states - pstates
        self._update_prop(self.split_props, prop, self.states)
        if intr and diff :
            return self._make_split(intr), self._make_split(diff)
        elif intr :
            return self, None
        else :
            return None, self
    cdef Component _make_split (Component self, sdd part) :
        # updates the component's properties wrt a part of its states and
        # returns a new Component for these states
        cdef dict graph_props = self._copy_graph_props(part)
        cdef dict split_props = {}
        cdef str p
        cdef sdd s
        for p in self.split_props :
            self._update_prop(split_props, p, part)
        return Component(self.lts, part, graph_props, split_props)
    cdef dict _copy_graph_props (Component self, sdd states) :
        # return an updated graph_props for a subset of states by querying the LTS
        cdef str p
        cdef bint b
        return {p : getattr(self.lts, p)(states)
                for p, b in self.graph_props.items()}
    def merge (Component self, Component first, *rest) :
        """merge a component with others

        Parameters:
         - `first`, ...: a series of at least one `Component` to be merged
        """
        cdef Component oth
        if self.lts != first.lts or not all(self.lts == oth.lts for oth in rest) :
            raise ValueError("components do not belong to the same LTS")
        return self._merge([first] + [<Component>oth for oth in rest])
    def __or__ (self, other) :
        "equivalent to `self.merge(other)`"
        return self.merge(other)
    cdef Component _merge (Component self, list others) :
        # perform the actual merge
        cdef Component oth
        cdef sdd states = self.states
        cdef dict graph_props = self.lts.graph_props(states)
        cdef dict split_props = {}
        cdef set props = set(self.split_props)
        for oth in others :
            states |= oth.states
            props.update(oth.split_props)
        for p in props :
            self._update_prop(split_props, p, states)
        return Component(self.lts, states, gp=graph_props, sp=split_props)
    def succ (Component self, *others) :
        """computes the transitions from a component to others

        Parameters:
         - `others`, ...: a series of candidate successor components
        Yields: pairs `trans, comp` where transition `trans` allows to reach `compo`
        """
        cdef str t
        cdef shom h
        cdef Component c
        cdef list s = []
        for t, h in self.lts.tsucc.items() :
            for c in others :
                if self.states != c.states and h(self.states) & c.states :
                    yield t, c
        return s
    def explicit (Component self) :
        """splits a component into one-state sub-components

        Yields: a series of `Component`
        """
        cdef ddd state
        if len(self) == 1 :
            yield self
        else :
            for state in s2d(self.states).explicit() :
                yield self._explicit(d2s(state))
    cdef Component _explicit (Component self, sdd state) :
        # performs the properties update of a singleton component and
        # returns the corresponding `Component`
        cdef dict gp = {}
        cdef dict sp = {}
        cdef str p
        cdef setrel r
        cdef sdd pstates
        for p in self.graph_props :
            gp[p] = getattr(self.lts, p)(state)
        for p, r in self.split_props.items() :
            self._update_prop(sp, p, state)
        return Component(self.lts, state, gp=gp, sp=sp)
    cpdef tuple topo_split (Component self,
                            bint split_init=True,
                            bint split_entries=True,
                            bint split_exits=True,
                            bint split_hull=True,
                            bint split_dead=True) :
        """split a component into its topological parts

        Each part will be extracted as a sub-component in the following order:
         - `init`: initial states
         - `entries`: states from which the component may be entered
         - `exits`: states from which the component may be exited
         - `hull`: states forming a SCC hull within the components
         - `dead`: deadlocks and their basin
         - `rest`: other states that do not fall into one of the previous cases

        The order of extraction is importer since, for example, an
        initial state will be in the `init` part and not the `entries`
        part if it could belong to both. Which part is actually
        extracted is controlled with the method's parameters.

        Parameters:
         - `split_init` (`bool=True`): whether to extract initial states
         - `split_entries` (`bool=True`): whether to extract entry states
         - `split_exits` (`bool=True`): whether to extract exit states
         - `split_hull` (`bool=True`): whether to extract SCC hull states
         - `split_dead` (`bool=True`): whether to extract deadlocks
        Returns: a tuple `init, entries, exits, hull, dead, rest` corresponding
        to the sub-components described above, according to the call parameters.
        Each is returned as `None` if empty of a `Component` instance if non-empty
        """
        cdef sdd rest = self.states
        cdef sdd init, dead, entries, exits, hull
        if split_init :
            init = rest & self.lts.init
            rest -= init
        else :
            init = sdd.empty()
        if split_entries :
            entries = self.lts.succ(self.lts.pred(rest) - rest) & rest
            rest -= entries
        else :
            entries = sdd.empty()
        if split_exits :
            exits = self.lts.pred(self.lts.succ(rest) - rest) & rest
            rest -= exits
        else :
            exits = sdd.empty()
        if split_hull :
            hull = self.lts.succ_o(rest) & self.lts.pred_o(rest)
            rest -= hull
        else :
            hull = sdd.empty()
        if split_dead and rest :
            dead = (self.lts.pred & rest).lfp()(rest & self.lts.dead)
            rest -= dead
        else :
            dead = sdd.empty()
        return (self._make_split(init) if init else None,
                self._make_split(entries) if entries else None,
                self._make_split(exits) if exits else None,
                self._make_split(hull) if hull else None,
                self._make_split(dead) if dead else None,
                self._make_split(rest) if rest else None)
    cpdef dict count (Component self) :
        "return the number of on occurrences of each variable"
        cdef dict seen = {}
        cdef ddd d = s2d(self.states)
        return self._count(d, seen)
    cpdef dict _count (Component self, ddd head, dict seen) :
        cdef dict ret, sub
        cdef str var, v
        cdef unsigned long long num, n
        cdef val_t val
        cdef ddd child
        seen[head] = ret = {}
        if head.stop() :
            return ret
        for var, num, val, child in head.edges() :
            if child in seen :
                sub = seen[child]
            else :
                sub = seen[child] = self._count(child, seen)
            if val :
                ret[var] = len(child)
            for v, n in sub.items() :
                ret[v] = ret.get(v, 0) + sub[v]
        return ret
    cpdef tuple on_off (Component self) :
        "return a pair of sets `on, off` for the always-on and always-off variables"
        cdef dict cnt = self.count()
        cdef set on = set()
        cdef set off = set()
        cdef unsigned long long s, size
        size = len(self.states)
        for var in self.lts.vars :
            s = cnt.get(var, 0)
            if s == 0 :
                off.add(var)
            elif s == size :
                on.add(var)
        return on, off
    cpdef object form (Component self, variables=None, normalise=None) :
        """describe the component states states by a Boolean formula

        Arguments:
         - `variables=...`: a collection of variables names to restrict to
         - `normalise=...`: how the formula should be normalised, which must be either:
           - `"cnf"`: conjunctive normal form
           - `"dnf"`: disjunctive normal form
           - `None`: chose the smallest form
        Return: a sympy Boolean formulas
        """
        return self.lts.form(self.states, variables, normalise)
