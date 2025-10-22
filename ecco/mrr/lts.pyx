# distutils: language = c++
# distutils: include_dirs = ../pyddd ../libDDD ../libITS

"""`LTS` and `Component` are the building blocks for component graphs.

An `LTS` (labelled transition system) defines states through variables,
and transitions allowing to reach new states from existing ones. Here, `LTS`
are symbolic and act on sets of states encoded as `ddd.sdd`, while transitions
are encoded as `ddd.shom`. In this module, variables are all integer-valuated,
with minimal provision to use them as Boolean-valuated (through the `+/-`
notation from RR).

A `Component` is a subset of the reachable states of an `LTS`. `Component`s are
aimed at forming partitions of the reachable states but this aspect is not
ensured at the level of this model, it is the job of component graphs.

An `LTS` and its `Component`s are associated with properties. At the level of
and `LTS`, a property is defined by:
 - a textual description `str`, which may be an arbitrary name or a formula in a
   temporal logic
 - a set of states `ddd.sdd`, which represents the set of states that validate
   the property

This module makes no assumption on the semantics of the properties, they are just
a text and a subset of the reachable states of an `LTS`. Then, each `Component`
may be related to each property through the relations enumerated in type `setrel`.
For instance, a component `comp` may be strictly included in a property `text/states`
(encoded by `setrel.ISIN`) if all the states of `comp` are in `states` but `states`
has other states that are not in `comp`.

The API allows to introduce new properties at the level of `Component`s through
their methods `check` and `split` essentially. When a property `test/states` is
used, for instance, to split a `Component`, it is recorded at the level of the
`LTS` and then used for the split. The split component and resulting
sub-components record their relationship with the property, but this remain a
property defined at the level of the `LTS`.
"""

cimport cython

from ddd cimport ddd, sdd, shom
from its cimport model

from ddd import ddd_save, ddd_load

import sympy
import hashlib


cdef extern from "dddwrap.h":
    ctypedef short val_t


cdef inline ddd s2d (sdd s):
    # SDD -> DDD
    for t in s.edges() :
        return t.value


cdef inline sdd d2s (ddd d):
    # DDD -> SDD
    return sdd.mkz(d)


##
## auxiliary stuff
##


cpdef enum setrel:
    """Possible relations between a set `A` and a set `B`.

    Values:
     - `HASNO`: `A` has no values from `B` (possibly `B` is empty)
     - `HAS`: `A` has some values from `B` but not all
     - `CONTAINS`: `A` has all the values from `B` plus others
     - `ISIN`: `A` is included in `B` that has other values
     - `EQUALS`: `A` is exactly the same set as `B`
    """
    HASNO = 0
    HAS = 1
    CONTAINS = 2
    ISIN = 3
    EQUALS = 4


cpdef enum topo:
    """Possible topological sub-components of a component `C`.

    Values:
     - `INIT`: initial states
     - `ENTRY`: states reachable from the outside of `C`
     - `EXIT`: states that allow to reach the outside of `C`
     - `HULL`: states forming a SCC convex hull within `C`
     - `DEAD`: deadlock states
     - `SCC`: strongly connected components
    """
    INIT = 0
    ENTRY = 1
    EXIT = 2
    HULL = 3
    DEAD = 4
    SCC = 5


cdef set topoprops = set(t.name for t in topo) | {"hull", "scc"}


##
## LTS
##


cdef class LTS:
    """Store a labelled transition systelm and its properties
    
    Attributes:
     - `str path`: GAL file from which the LTS has been loaded
     - `its.model gal`: model loaded from the GAL file
     - `ddd.sdd init`: set of initial states
     - `ddd.sdd states`: set of reachable states
     - `ddd.sdd dead`: set of deadlock states
     - `ddd.sdd hull`: set of states in the SCC convex hull
     - `ddd.shom succ`: successor relation
     - `ddd.shom pred`: predecessor relation
     - `ddd.shom succ_o`: gfp of `succ`
     - `ddd.shom pred_o`: gfp of `pred`
     - `ddd.shom succ_s`: lfp of `succ`
     - `ddd.shom pred_s`: lfp of `pred`
     - `dict[str, shom] tsucc`: map transition names to the successor relations
     - `dict[str, shom] tpred`: map transition names to the predecessor relations
     - `tuple[str, ...] vars`: variable names ordered as in the DDDs
     - `dict[str, int] vmin`: minimum value allowed for each variable
     - `dict[str, int] vmax`: maximum value allowed for each variable
     - `dict[str, ddd.sdd] props`: map properties to the corresponding sets of states
     - `dict[str, str]` alias: map props to aliases
     - `dict[str, str]` saila: map aliases to props

    An LTS records properties with names (arbitrary strings that may be formulas
    of temporal logic) corresponding to sets of states. Every LTS is created with
    some default properties:
     - `"INIT"` maps to the set of initial states
     - `"DEAD"` maps to the set of deadlock states
     - `"HULL"` maps to the states forming the SCCs convex hull
    """
    # GAL path
    cdef readonly str path
    # GAL model loaded
    cdef readonly model gal
    # initial states / all states / deadlocks / SCC hull
    cdef readonly sdd init, states, dead, hull
    # succ / pred / gfp(succ) / gfp(pred) / lfp(succ) / lft(pred)
    cdef readonly shom succ, pred, succ_o, pred_o, succ_s, pred_s
    # trans name -> succ / pred
    cdef readonly dict tsucc, tpred
    # variable names
    cdef readonly tuple vars
    # variables name -> min / max
    cdef readonly dict vmin, vmax
    # props -> sdd
    cdef readonly dict props
    # props -> aliases -> props
    cdef readonly dict alias, saila
    # allow to add attributes
    cdef dict __dict__

    ##
    ## init
    ##

    def __cinit__(self, str path, dict init=None, dict vmin={}, dict vmax={}, str cache=""):
        self.path = path
        self.tsucc = {}
        self.tpred = {}
        self.vmin = {}
        self.vmax = {}
        self.props = {}
        self.alias = {}
        self.saila = {}

    def __init__(self, str path, dict init=None, dict vmin={}, dict vmax={}, str cache=""):
        """Create a new LTS from GAL file `path`.

        Arguments:
         - `str path`: path to the GAL file to be loaded
         - `dict init`: map variable names to initial values,
           if empty (default), they are taken from the GAL 
         - `dict[str, int] vmin`: minimum value for each variable (default: `0`)
         - `dict[str, int] vmax`: maximum value for each variable (default: `1`)
         - `str cache`: path to saved states or `""` if none exists

        The set of reachable states is constructed (of no cache is provided),
        so instantiating an LTS may take a long time and consume a lot of memory.
        """
        cdef str v
        cdef set expect, got
        # load model and its vars
        self.gal = model(path, fmt="GAL")
        self.vars = s2d(self.gal.initial()).vars()
        # build vmin/vmax
        expect = set(self.vars)
        got = set(vmin)
        if not got.issubset(expect):
            raise ValueError(f"unexpected variables in vmin: {got - expect}")
        got = set(vmax)
        if not got.issubset(expect):
            raise ValueError(f"unexpected variables in vmax: {got - expect}")
        for v in self.vars:
            self.vmin[v] = vmin.get(v, 0)
            self.vmax[v] = vmax.get(v, 1)
            if self.vmin[v] > self.vmax[v]:
                raise ValueError(f"{v} has an empty domain (vmin > vmax)")
        # build transition relations and sets of states
        self._build_succ()
        if cache:
            self.load(cache)
        self._build_init(init)
        if not cache:
            self._build_states()
        self._build_pred()
        if not cache:
            self._build_dead()
            self._build_hull()

    cdef void _build_succ(LTS self):
        # build the successor relations
        cdef dict d = self.gal.transitions()
        cdef str t
        cdef shom h
        for t, h in d.items():
            self.tsucc[t] = h
        self.succ = shom.union(*self.tsucc.values())
        self.succ_o = self.succ.gfp()
        self.succ_s = self.succ.lfp()

    cdef void _build_init(LTS self, dict init):
        # build the set of initial states + property INIT
        if init is None:
            self.init = self.props["INIT"] = self.gal.initial()
        else:
            self.init = self.props["INIT"] = self(init)

    cdef void _build_states(LTS self):
        # build the set of reachable states
        cdef sdd reach
        cdef shom succ
        reach = self.init
        self.states = sdd.empty()
        succ = self.succ | shom.ident()
        while True:
            reach = succ(reach)
            if reach == self.states:
                break
            self.states = reach

    cdef void _build_pred(LTS self):
        # build the predecessor relations
        cdef str t
        cdef shom h
        self.pred = self.succ.invert(self.states)
        self.pred_o = self.pred.gfp()
        self.pred_s = self.pred.lfp()
        for t, h in self.tsucc.items() :
            self.tpred[t] = h.invert(self.states)

    cdef void _build_dead(LTS self):
        # build the set of deadlock states + DEAD property
        self.dead = self.props["DEAD"] = (self.states
                                          - self.pred(self.states))

    cdef void _build_hull(LTS self):
        # build the set of SCCs hull states + HULL property
        self.hull = self.props["HULL"] = (self.pred_o(self.states)
                                          & self.succ_o(self.states))

    ##
    ## dump/load states
    ##

    def save(LTS self, str path, **headers):
        headers["__path__"] = self.path
        headers["__hash__"] = hashlib.file_digest(open(self.path, "rb"),
                                                  "sha256").hexdigest()
        cdef ddd init = s2d(self.init) if self.init else ddd.empty()
        cdef ddd states = s2d(self.states) if self.states else ddd.empty()
        cdef ddd dead = s2d(self.dead) if self.dead else ddd.empty()
        cdef ddd hull = s2d(self.hull) if self.hull else ddd.empty()
        ddd_save(path, init, states, dead, hull, **headers)

    def load(LTS self, str path):
        cdef dict headers
        cdef list ddds
        headers, ddds = ddd_load(path)
        if len(ddds) != 4:
            raise ValueError(f"invalid dump (expected 4 DDDs, found {len(ddds)})")
        if self.path != headers.pop("__path__", None):
            raise ValueError("not from same GAL file")
        cdef str h = hashlib.file_digest(open(self.path, "rb"), "sha256").hexdigest()
        if h != headers.pop("__hash__", None):
            raise ValueError("not from same GAL model")
        self.init = d2s(ddds[0])
        self.states = d2s(ddds[1])
        self.dead = d2s(ddds[2])
        self.hull = d2s(ddds[3])
        return headers

    ##
    ## build sets of states
    ##

    cdef void _check_var(LTS self, str var) except *:
        """Check whether `var` is a declared variable.
        Arguments:
         - `str var`: variable name to be checked

        Raise:
         - `ValueError` is `var` is not a variable of the LTS
        """
        if var not in self.vmin or var not in self.vmax:
            raise ValueError(f"unknown variable {var}")

    cdef void _check_val(LTS self, str var, val_t val) except *:
        """Check whether `var` is a declared variable and `val` is a valid value for it.
        Arguments:
         - `str var`: variable name to be checked
         - `int val`: a value for `var`

        Raise:
         - `ValueError` is `var` is not a variable of the LTS or `val` is not a
           value allowed for `var`
        """
        self._check_var(var)
        if val is not None:
            if not (self.vmin[var] <= val <= self.vmax[var]):
                raise ValueError(f"invalid value {val} for variable {var}")

    cpdef sdd var_eq(LTS self, str var, val_t val) except *:
        """Return the set of states such that `var == val`.
        Arguments:
         - `str var`: a variable name
         - `int val`: a value to compare to

        Return: `ddd.sdd`
        """
        cdef ddd d = ddd.one()
        cdef str v
        self._check_var(var)
        if val < self.vmin[var] or val > self.vmax[var]:
            return sdd.empty()
        for v in reversed(self.vars):
            if v == var:
                d = ddd.from_range(v, val, val, d)
            else:
                d = ddd.from_range(v, self.vmin[var], self.vmax[var], d)
        return d2s(d) & self.states

    cpdef sdd var_le(LTS self, str var, val_t val) except *:
        """Return the set of states such that `var <= val`.
        Arguments:
         - `str var`: a variable name
         - `int val`: a value to compare to

        Return: `ddd.sdd`
        """
        cdef ddd d = ddd.one()
        cdef str v
        self._check_var(var)
        if val < self.vmin[var]:
            return sdd.empty()
        elif val > self.vmax[var]:
            val = self.vmax[var]
        for v in reversed(self.vars):
            if v == var:
                d = ddd.from_range(v, self.vmin[v], val, d)
            else:
                d = ddd.from_range(v, self.vmin[v], self.vmax[v], d)
        return d2s(d) & self.states

    cpdef sdd var_lt(LTS self, str var, val_t val) except *:
        """Return the set of states such that `var < val`.
        Arguments:
         - `str var`: a variable name
         - `int val`: a value to compare to

        Return: `ddd.sdd`
        """
        return self.var_le(var, val-1)

    cpdef sdd var_ge(LTS self, str var, val_t val) except *:
        """Return the set of states such that `var >= val`.
        Arguments:
         - `str var`: a variable name
         - `int val`: a value to compare to

        Return: `ddd.sdd`
        """
        cdef ddd d = ddd.one()
        cdef str v
        self._check_var(var)
        if val > self.vmax[var]:
            return sdd.empty()
        elif val < self.vmin[var]:
            val = self.vmin[var]
        for v in reversed(self.vars):
            if v == var:
                d = ddd.from_range(v, val, self.vmax[v], d)
            else:
                d = ddd.from_range(v, self.vmin[v], self.vmax[v], d)
        return d2s(d) & self.states

    cpdef sdd var_gt(LTS self, str var, val_t val) except *:
        """Return the set of states such that `var > val`.
        Arguments:
         - `str var`: a variable name
         - `int val`: a value to compare to

        Return: `ddd.sdd`
        """
        return self.var_ge(var, val+1)

    cpdef sdd var_ne(LTS self, str var, val_t val):
        """Return the set of states such that `var != val`.
        Arguments:
         - `str var`: a variable name
         - `int val`: a value to compare to

        Return: `ddd.sdd`
        """
        return self.var_lt(var, val) | self.var_gt(var, val)

    def __call__(LTS self, dict values):
        """Build a set of states from a partial valuation `values`.

        The variables valuated in `values` will range on the given values,
        intersected with their actual domain. Variables that are not given in
        `values` will range on their domain.

        Arguments:
         - `dict values`: map variable names to values

        Return: a `ddd.sdd` holding the specified states
        """
        cdef ddd d, t
        cdef int i
        cdef str var
        cdef set vals, dom
        cdef val_t vmin, vmax
        cdef bint full
        d = ddd.one()
        for var in reversed(self.vars):
            vmin = self.vmin[var]
            vmax = self.vmax[var]
            dom = set(range(vmin, vmax+1))
            if var in values:
                vals = set(values[var]) & dom
                if not vals:
                    raise ValueError(f"empty values for {var}")
                full = (vals == dom)
            else:
                vals = dom
                full = True
            if full:
                d = ddd.from_range(var, vmin, vmax, d)
            elif len(vals) == 1:
                vmin = min(vals)
                d = ddd.from_range(var, vmin, vmin, d)
            else:
                for i, vmin in enumerate(vals):
                    if i == 0:
                        t = ddd.from_range(var, vmin, vmin, d)
                    else:
                        t |= ddd.from_range(var, vmin, vmin, d)
                d = t
        return d2s(d)

    ##
    ## general API
    ##

    def __eq__(LTS self, object other):
        if not isinstance(other, LTS):
            return False
        return (self.path == other.path
                and self.init == other.init
                and self.vars == other.vars)

    def __hash__(LTS self):
        return hash(("ecco.lts.lTS", self.path, self.init, self.vars))

    def __len__(LTS self):
        return len(self.states)

    ##
    ## manage properties
    ##

    cdef sdd add_prop(LTS self, str prop, sdd states, str alias=None):
        """Record a new property called `prop` and corresponding to `states`.

        This is an internal method not intended to be called by end-users, but only
        from methods in `Component`.

        Arguments:
         - `str prop`: the property as a string, eg, a temporal logic formula
         - `str states`: the corresponding states

        Return: `states` restricted to the states of the LTS

        Raise:
         - `ValueError` if the property already exists and does not correspond
           to the same states
        """
        cdef sdd s = states & self.states
        if prop in self.props:
            if self.props[prop] != s:
                raise ValueError("property exists with distinct states")
        else:
            self.props[prop] = s
        if alias:
            self.alias[prop] = alias
            self.saila[alias] = prop
        return s

    ##
    ## formulas
    ##


    cpdef object form(LTS self, sdd states, variables=None, normalise=None):
        """Describe a set of states by a Boolean formula.

        Arguments:
         - `sdd states`: a set of states
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

    cdef object _form(LTS self, ddd head, dict seen, set keep):
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

##
## components
##


cdef setrel _update_prop(dict pdict, str prop, sdd states, sdd lts_states):
    """Update the relations between a component and a property.

    Arguments:
     - `dict[str, setrel] pdict`: the dict to be updated
     - `str prop`: the text property
     - `ddd.sdd states`: the states to be checked against the property
     - `ddd.sdd lts_states`: the states corresponding to the property

    Return: the `setrel` stored in `pdict`
    """
    cdef setrel r
    if not lts_states:
        r = pdict[prop] = setrel.HASNO
    elif states == lts_states:
        r = pdict[prop] = setrel.EQUALS
    elif states < lts_states:
        r = pdict[prop] = setrel.ISIN
    elif states > lts_states:
        r = pdict[prop] = setrel.CONTAINS
    elif states & lts_states:
        r = pdict[prop] = setrel.HAS
    else:
        r = pdict[prop] = setrel.HASNO
    return r


ctypedef unsigned long long CompoID
# already assigned component numbers
cdef dict _CompoCache = {}


@cython.freelist(128)
cdef class Component:
    """A component is a subset of the reachable states of an LTS.

    Attributes:
     - `int num`: a unique identifier, if two components get the same identifier
       then they are from the same LTS and hold the same states
     - `ddd.sdd states`: the states of the component
     - `LTS lts`: the LTS the component belongs to
     - `dict[str, int] consts`: maps variable names to a value if the variable
       has the same value in all the states of the component
     - `dict[tuple[str, int], int] counts`: map pairs `var, val` to the number of
       states where the variable `val` appears with value `val`
     - `dict[str, setrel] props`: maps text properties to the relation between
       the component and the states corresponding to the property in the LTS. For
       instance `comp.props[p] == ISIN` reads as 'comp ISIN p', ie we have 
       `comp.states < comp.lts.props[p]`.

    A `Component comp` has several automatic properties:
     - `"INIT"`: corresponding to `comp.lts.init`
     - `"DEAD"`: corresponding to `comp.lts.dead`
     - `"HULL"`: corresponding to `comp.lts.hull`
     - `"hull"`: corresponding to the convex hull of the SCCs within `comp`
     - `"scc"`: corresponding to `comp` if it is itself a SCC

    The two latter properties are local to the component and always included in it
    (and possibly empty). The other ones are global properties related to the LTS.
    """
    # unique id
    cdef readonly CompoID num
    # states in the component
    cdef readonly sdd states
    # the LTS it belongs to
    cdef readonly LTS lts
    # variables that do not change
    cdef readonly dict consts
    # number of occurrences of each variable for each value
    cdef readonly dict counts
    # prop -> setret
    cdef readonly dict props

    ##
    ## init
    ##

    def __cinit__(Component self, LTS lts, sdd states):
        self.lts = lts
        self.states = states
        self.consts = {}
        self.counts = {}
        self.props = {}

    def __init__(Component self, LTS lts, sdd states):
        """creates a new component.

        Arguments:
         - `LTS lts`: the LTS it belongs to
         - `ddd.sdd states`: its states
        """
        global _CompoCache
        cdef dict seen = {}
        cdef ddd d = s2d(self.states)
        if (states, lts) in _CompoCache :
            self.num = _CompoCache[states, lts]
        else :
            _CompoCache[states, lts] = self.num = len(_CompoCache)
        self.counts.update(self._count(d, seen))
        self._build_consts()
        self._build_props()

    cdef dict _count(Component self, ddd head, dict seen):
        """Count the occurrences of each valuation `var, val` in `head`.

        Arguments:
         - `ddd.ddd head`: DDD to be counted in
         - `dict[ddd.ddd, dict[str, int]]`: valuations already counted from other branches

        Return: `dict[str, int]`
        """
        cdef dict ret, sub
        cdef str var, v
        cdef unsigned long long num, n
        cdef val_t val, i
        cdef ddd child
        seen[head] = ret = {}
        if head.stop() :
            return ret
        for var, num, val, child in head.edges() :
            if child in seen :
                sub = seen[child]
            else :
                sub = seen[child] = self._count(child, seen)
            ret[var, val] = len(child)
            for (v, i), n in sub.items() :
                ret[v, i] = ret.get((v, i), 0) + sub[v, i]
        return ret

    cdef void _build_consts(Component self):
        """Extract in `self.consts` the constants found in `self.count`.
        """
        cdef unsigned long long size = len(self.states)
        cdef unsigned long long cnt
        cdef str var
        cdef val_t val
        for (var, val), cnt in self.counts.items():
            if cnt == size :
                self.consts[var] = val

    cdef void _build_props(Component self):
        """Build local/global properties.
        """
        cdef sdd s, h
        # check relation with global graph properties
        _update_prop(self.props, "INIT", self.states, self.lts.init)
        _update_prop(self.props, "DEAD", self.states, self.lts.dead)
        _update_prop(self.props, "HULL", self.states, self.lts.hull)
        # check local SCC hull and SCC
        succ = self.lts.succ_o & self.states
        pred = self.lts.pred_o & self.states
        h = succ(self.states) & pred(self.states)
        if h:
            # if h is not empty, pick a SCC insidde
            s = h.pick()
            s = self.lts.succ_s(s) & self.lts.pred_s(s)
            if len(s) <= 1:
                s = sdd.empty()
        else:
            # if h is empty => no SCC
            s = sdd.empty()
        _update_prop(self.props, "hull", self.states, h)
        _update_prop(self.props, "scc", self.states, s)

    ##
    ## general API
    ##

    def __len__(Component self):
        return len(self.states)

    def __hash__(Component self):
        return hash((self.states, self.lts))

    def __eq__(Component self, object other):
        if not isinstance(other, Component):
            return False
        return self.states == other.states and self.lts == other.lts

    ##
    ## queries
    ##

    def succ(Component self, *others):
        """Compute all the transitions going to components in `others`

        Yield: pairs `t, c` is `str t` is a transition that allows to reach
        `Component c` for each `c` in `others`.
        """
        cdef str t
        cdef shom h
        cdef Component c
        for t, h in self.lts.tsucc.items() :
            for c in others :
                if self.states != c.states and h(self.states) & c.states:
                    yield t, c

    cpdef object form(Component self, variables=None, normalise=None):
        """Describe the component states states by a Boolean formula.

        Arguments:
         - `variables=...`: a collection of variables names to restrict to
         - `normalise=...`: how the formula should be normalised, which must be either:
           - `"cnf"`: conjunctive normal form
           - `"dnf"`: disjunctive normal form
           - `None`: chose the smallest form

        Return: a sympy Boolean formulas
        """
        return self.lts.form(self.states, variables, normalise)

    ##
    ## manage properties
    ##

    cpdef setrel check(Component self, str prop, sdd states, str alias=None) except *:
        """Check property `prop/states` wrt the states of the component.

        `self.props[prop]` is updated according to the relation the component
        has with the property.

        Arguments:
         - `str prop`: the text property
         - `ddd.sdd states`: the corresponding states in the LTS
         - `str alias`: if not empty, use it instead of `prop` within `self.props`

        Return: the `setrel` stored in `self.props[prop]`
        """
        cdef sdd pstates = self.lts.add_prop(prop, states, alias)
        return _update_prop(self.props, prop, self.states, pstates)

    cpdef object forget(Component self, str prop):
        """Forget property `prop`.

        `self.props[prop]` is deleted and the corresponding `setrel` is returned,
        or `None` is `prop` was unknown.

        Arguments:
         - `str prop`: the text property

        Return: the corresponding `setrel` or `None`
        """
        # in case prop is an alias
        prop = self.lts.saila.get(prop, prop)
        if prop in self.lts.alias:
            self.lts.saila.pop(self.lts.alias.pop(prop))
        return self.props.pop(prop, None)

    cpdef void forget_all(Component self):
        """Forget all non-topological properties.
        """
        cdef tuple todo = tuple(self.props)
        cdef str prop
        for prop in todo:
            self.forget(prop)

    ##
    ## splits
    ##

    cdef _update_props(Component self, dict props):
        """Update `self.props` with all the properties in `props`.

        Only the text properties are fetched from `props`, the corresponding
        states are fetched from `self.lts.props`
        """
        cdef str p
        for p in props:
            if p not in topoprops:
                _update_prop(self.props, p, self.states, self.lts.props[p])

    cdef Component _make_split(Component self, sdd part):
        """Create a sub-component of `self` with states from `part`

        The properties of the created component are initialised from those of
        self (ie, it will have the same keys in `.props`, but probably not the
        same values).

        Arguments:
         - `ddd.sdd part`: states of the sub-component to be created

        Return: new `Component`
        """
        cdef Component c = Component(self.lts, part)
        c._update_props(self.props)
        return c

    cpdef tuple split_prop(Component self, str prop, sdd states, str alias="") except *:
        """Split component wrt property `prop/states`

        Arguments:
         - `str prop`: text property
         - `ddd.sdd states`: states of the property

        Return: a pair of components `intr, diff` where `intr` is the intersection
        of the component with the property (or `None` is empty), and `diff` is
        the difference (or `None` if empty).
        """
        cdef sdd pstates = self.lts.add_prop(prop, states, alias)
        cdef sdd intr = self.states & pstates
        cdef sdd diff = self.states - pstates
        _update_prop(self.props, prop, self.states, pstates)
        if intr and diff :
            return self._make_split(intr), self._make_split(diff)
        elif intr :
            return self, None
        else :
            return None, self

    def split_topo(Component self, *parts):
        """Split component into its topological sub-components

        Topological components, as specified by enum `topo` are sets of states
        that related to the states-graph rather that to properties. They are
        extracted in the order specified in `parts`, which greatly influences
        the result. For instance, assume that the component is a SCC, then
         - calling `self.split_topo(topo.SCC, topo.ENTRY)` will yield `self`
           only because after extraction `topo.SCC` there is no more states
           to split, and there are no remaining states
        - but calling `self.split_topo(topo.ENTRY, topo.SCC)` will yield first
          the entries of `self`, then possibly sub-SCC, and finally the
          remaining states if any
        
        Arguments:
         - `topo *parts`: a `topo` for each the topological components to extract

        Yield: a series of `Component`
        """
        cdef topo t
        cdef sdd rest = self.states
        cdef sdd p, s, r
        for t in parts:
            if t == topo.INIT:
                p = rest & self.lts.init
            elif t == topo.ENTRY:
                p = self.lts.succ(self.lts.pred(rest) - rest) & rest
            elif t == topo.EXIT:
                p = self.lts.pred(self.lts.succ(rest) - rest) & rest
            elif t == topo.HULL:
                p = self.lts.succ_o(rest) & self.lts.pred_o(rest)
            elif t == topo.DEAD:
                p = rest & self.lts.dead
            elif t == topo.SCC:
                # r = convex hull of remaining states to be split
                r = self.lts.succ_o(rest) & self.lts.pred_o(rest)
                while r:
                    # pick one SCC
                    s = r.pick()
                    p = self.lts.succ_s(s) & self.lts.pred_s(s)
                    if len(p) > 1:
                        # not trivial => yield + remove from rest
                        yield self._make_split(p)
                        rest -= p
                    # update r => remove p and retake hull
                    s = r - p
                    r = self.lts.succ_o(s) & self.lts.pred_o(s)
                continue  # yields have beeen made already
            else:
                # just in case topo is extended but not this function
                raise ValueError(f"unexpected topo {t!r}")
            if p:
                yield self._make_split(p)
                rest = rest - p
        if rest:
            yield self._make_split(rest)

    def explicit(Component self):
        """Split component into its individual states.

        Yield: singleton `Component`s, one for each state of `self`
        """
        cdef ddd d
        cdef Component c
        if len(self.states) == 1:
            yield self
        else:
            for d in s2d(self.states).explicit():
                yield self._make_split(d2s(d))

    ##
    ## merges
    ##

    cpdef Component merge_with(Component self, others) except *:
        """Merge component with `others`.

        The resulting component has the union of the states of `self` and the
        components from `others`. It has also the union of the properties from
        the merged components.

        Arguments:
         - `Iterable[Component] others`: other components to merge with `self`

        Return: the merged `Component`
        """
        cdef Component c, m
        cdef sdd s = self.states
        cdef str p
        cdef dict props = self.props.copy()
        for c in others:
            if c.lts != self.lts:
                raise ValueError("components not from the same LTS")
            s |= c.states
            props.update(c.props)
        m = Component(self.lts, s)
        m._update_props(props)
        return m

    def merge(Component self, Component first, *others):
        """Like `merge_with` but with a pythonic interface
        """
        return self.merge_with([first] + list(others))

    def __or__(Component self, Component other):
        """Like `self.merge(other)`
        """
        return self.merge_with([other])
