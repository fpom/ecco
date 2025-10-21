import re

from collections import defaultdict
from functools import partial, reduce
from operator import attrgetter, or_
from typing import Self, override

import igraph as ig
import pandas as pd
import sympy

from IPython.display import display

from .. import cached_property, hset
from ..ui import log
from ..cygraphs import Graph
from .lts import LTS, Component, setrel, topo
from .ltsprop import AnyProp as ModelChecker
from .tables import TableProxy, NodesTableProxy


pd.set_option("display.max_colwidth", None)
_re_digits = re.compile(r"(\d+)")


def _human_sortkey(text):
    "key to sort strings in natural order"
    return tuple(int(m) if m.isdigit() else m.lower() for m in _re_digits.split(text))


def _gal_sortkey(text):
    "key to sort GAL var/trans names in natural order"
    head = text.split("._.")[0]
    tail = text.split("_", 1)[-1]
    return (len(head.split(".")),) + _human_sortkey(tail)


def _gal_sortkey_0(seq):
    "key to sort sequences with seq[0] in GAL var/trans names in natural order"
    return _gal_sortkey(seq[0]) + tuple(seq[1:])


class _GCS(set):
    "a subclass of set that is tied to a component graph so it has __invert__"

    components = set()

    def __invert__(self):
        return self.__class__(self.components - self)

    def __and__(self, other):
        return self.__class__(super().__and__(other))

    def __or__(self, other):
        return self.__class__(super().__or__(other))

    def __xor__(self, other):
        return self.__class__(super().__xor__(other))


class ComponentGraph:
    """A `ComponentGraph` is a decomposition of a state-space.

    Its nodes are pairwise-disjoint symbolic sets of states, called components,
    and its edges are the explicit transitions allowing to reach one component
    from another. Usually, the union of components is the set of the reachable
    states, but user may drop components and so get just a subset of the
    reachable states.

    Note that in every method accepting arbitrary arguments `**arg`, there may
    be specific keyword arguments that are then starting with `_`.

    Two enumerated types are used together with `ComponentGraph`:

     * `setrel` whose values are `HASNO`, `HAS`, `CONTAINS`, `ISIN`, and `EQUALS`
     * `topo` whose values are `INIT`, `ENTRY`, `EXIT`, `HULL`, `DEAD`, and `SCC`
    """

    def __init__(self, model, lts, *comps):
        """Create a new `ComponentGraph` on the top of a model and its LTS.

        Arguments:
         - `ecco.mrr.Model model`: the loaded MRR model
         - `ecco.mrr.lts.LTS lts`: the LTS object built from the model
         - `ecco.mrr.lts.Component *comps`: the components of the graph
        """
        self.model = model
        self.lts = lts
        self.components = tuple(sorted(comps, key=attrgetter("num")))
        self._c = {c.num: c for c in comps}  # Component.num => Component
        self._g = {}  # Component.num => Vertex index in .g.vs
        self._traps()

    def _traps(self):
        # TODO: add an option to skip this completely
        for node in self:
            vertex = self.g.vs[self._g[node.num]]
            traps = vertex["traps"] = {}
            exits = (
                self.lts.pred(self.lts.succ(node.states) - node.states) & node.states
            )
            succ_s = self.lts.succ_s & node.states
            for edge in vertex.in_edges():
                pred = self[edge.source_vertex["node"]]
                reach = succ_s(self.lts.succ(pred.states) & node.states)
                if reach >= exits:
                    edge["trap"] = False
                else:
                    traps[pred.num] = reach
                    edge["trap"] = True

    @classmethod
    def from_model(cls, model, init=None):
        """Create a `ComponentGraph` from an `ecco.mrr.Model`.

        This method takes care of creating the LTS object the is required by
        the constructor of `ComponentGraph`.

        Arguments:
         - `ecco.mrr.Model model`: the loaded MRR model
         - `dict init`: intial states, if not `None`, should map each variable
           name to the set of its initial values

        Return: a `ComponentGraph` instance
        """
        default_init, doms, actions = model.gal()
        model._gal2act = actions
        g2m, m2g, n2n, n2t = cls._translate_names(doms, actions)
        vmin = {v: min(d) for v, d in doms.items()}
        vmax = {v: max(d) for v, d in doms.items()}
        if init is None:
            init = default_init
        elif isinstance(init, str):
            init = default_init | cls._parse_state(init, m2g)
        else:
            init = default_init | init
        lts = LTS(str(model["gal"]), init, vmin, vmax)
        lts.g2m, lts.m2g, lts.n2n, lts.n2t = g2m, m2g, n2n, n2t
        return cls(model, lts, Component(lts, lts.states))

    @classmethod
    def _translate_names(cls, vnames, tnames):
        """Add variables and transitions translation tables to lts.

        This is done in the lts so that every `ComponentGraph` can share it.

        Return: a tuple of 4 dicts
         - `g2m`: GAL name -> model name
         - `m2g`: model name -> GAL name
         - `n2n`: GAL/model name -> rank for sorting
         - `n2t`: GAL/model name -> type
        """
        index = re.compile(r"\.(\d+)(\.|\Z)")
        g2m, m2g, n2n, n2t = {}, {}, {}, {}
        for g in sorted(vnames, key=_gal_sortkey):
            t, v = g.split("_", 1)
            m = index.sub(r"[\1]\2", v)
            g2m[g] = m
            m2g[m] = g
            n2n[g] = n2n[m] = len(n2n)
            n2t[g] = n2t[m] = t
        for g in sorted(tnames, key=_gal_sortkey):
            if "._." in g:
                t, v = g.split("._.")
            else:
                t, v = g, ""
            t = index.sub(r"[\1]\2", t)
            if v:
                s = v.split(".")
                v = ",".join(f"{x}={n}" for x, n in sorted(zip(s[::2], s[1::2])))
                v = f"[{v}]"
            m = f"{t}{v}"
            g2m[g] = m
            m2g[m] = g
            n2n[g] = n2n[m] = len(n2n)
        return g2m, m2g, n2n, n2t

    @classmethod
    def _parse_state(cls, spec, m2g):
        """Parse a string as a partial valuation.

        See `make_states` for documentation about the format.

        Arguments:
         - `str spec`: specification to be parsed
         - `dict m2g`: map model names rto GAL names

        Return: a `dict` mapping GAL variable names to sets of values
        """
        values = defaultdict(set)
        for assign in (part for p in spec.split(",") if (part := p.strip())):
            try:
                if assign.endswith("+"):
                    values[m2g[assign[:-1]]].add(1)
                elif assign.endswith("-"):
                    values[m2g[assign[:-1]]].add(0)
                elif assign.endswith("*"):
                    values[m2g[assign[:-1]]].update((0, 1))
                else:
                    var, val = assign.split("=")
                    values[m2g[var]].add(int(val))
            except Exception:
                raise ValueError(f"invalid valuation {assign!r}")
        return dict(values)

    def _v2s(self, var, val):
        """Compute a `str` representation of a valuated variable.

        The representation depends on the type of the variable:
         - for `bool`, it is `VAR+` or `VAR-`
         - for idle `clock`, it is `CLK*`
         - otherwise, it is `VAR=VAL`
        """
        var = self.lts.g2m.get(var, var)
        typ = self.lts.n2t[var]
        if typ == "bool":
            return f"{var}{'+' if val else '-'}"
        elif typ == "clock" and val < 0:
            return f"{var}*"
        else:
            return f"{var}={val}"

    #
    # general API
    #

    def __getitem__(self, key):
        """Return a node or an edge.

        # Arguments

         - `key (int)`: return a `Component` from its number
         - or `key (int,int)`: return an edge from the number of its
           source/destination nodes, that is its attributes as found in the
           `.edges` table

        # Return

        `Component` or `dict`.
        """
        if isinstance(key, int):
            return self._c[key]
        else:
            try:
                src, dst = key
                eid = self.g.get_eid(self._g[src], self._g[dst])
            except Exception:
                raise KeyError(key)
            return self.g.es[eid].attributes()

    def __iter__(self):
        """Yield every `Component` in the `ComponentGraph`."""
        yield from self.components

    def __len__(self):
        """Return the number of `Component`s in the `ComponentGraph`."""
        return len(self.components)

    def __eq__(self, other):
        """Check for equality of two `ComponentGraph`s.

        They are equal if the contain exactly the same set of components.
        """
        return isinstance(other, ComponentGraph) and set(self.components) == set(
            other.components
        )

    #
    # attributes
    #

    @cached_property
    def g(self):
        """The actual graph in the `ComponentGraph`.

        This is stored as an [`igraph.Graph`](https://igraph.org) instance
        whoses nodes are labelled with the component numbers and whose edges
        are labelled with the source and destination component numbers as well
        as with the transitions.
        """
        g = ig.Graph(directed=True)
        edges = defaultdict(set)
        g2m = self.lts.g2m
        for i, c in enumerate(self.components):
            self._g[c.num] = g.add_vertex(node=c.num).index
            for t, s in c.succ(*self.components[:i], *self.components[i + 1 :]):
                edges[c.num, s.num].add(g2m[t])
        _g = self._g
        n2n = self.lts.n2n
        for (src, dst), actions in edges.items():
            g.add_edge(
                _g[src], _g[dst], src=src, dst=dst, actions=hset(actions, key=n2n.get)
            )
        return g

    @cached_property
    def n(self):
        """A proxy to `.nodes` table.

        - when printed, `.n` shows a summary of the content of `.nodes`
        - so do printing `.n[col]` or `.n[[col, ...]]`
        - deleting `.n[col]` or `.n[col, ...]` deletes columns in `.nodes`
        - setting `.n[col] = fun`, where `fun` is a function that takes a row
          as yield by `.nodes.iterrows()` and return a value, its
          creates and populates a new column `col`
        """
        return NodesTableProxy(self, self.nodes)

    @cached_property
    def e(self):
        """A proxy to `.edges` table.

        Works similarly as `.n`.
        """
        return TableProxy(self.edges)

    def _make_nc_size(self, row):
        "Compute the content of column `size`."
        return len(self._c[row.name])

    def _make_nc_consts(self, row):
        "Compute the content of column `consts`."
        c = self._c[row.name]
        return hset(
            self._v2s(k, v) for k, v in sorted(c.consts.items(), key=_gal_sortkey_0)
        )

    def _make_nc_traps(self, row):
        "Compute the content of column 'traps'"
        return hset(sorted(row["traps"]))

    def _make_ncols_setrel(self, row, rel):
        "Compute the content of column `rel`, depending on the value of `rel`."
        alias = self.lts.alias
        return hset(
            alias.get(p, p) for p, r in self._c[row.name].props.items() if r == rel
        )

    def _make_ncols(self, nodes, *cols):
        "Populate the columns `cols` of `nodes`."
        makers = []
        for c in cols:
            if (mk := getattr(self, f"_make_nc_{c}", None)) is not None:
                makers.append(mk)
            elif (rel := getattr(setrel, c, None)) is not None:
                makers.append(partial(self._make_ncols_setrel, rel=rel))
            else:
                raise ValueError(f"unknown colums {c}")

        def make(row):
            return [m(row) for m in makers]

        for c in cols:
            if c not in nodes.columns:
                nodes[c] = None
        nodes[list(cols)] = nodes.apply(make, axis=1, result_type="expand")

    @cached_property
    def nodes(self):
        """The _nodes table_ holds all about the nodes of a `ComponentGraph`.

        This is stored as a [`pandas.DataFrame`](https://pandas.pydata.org)
        whose columns are:

         - `index`: component number
         - `size`: number of states in the component
         - `consts`: set of constants in the component
         - one column for each `rel` in `setrel`: for each relation, set of
           properties checked for the component that have this relation with
           its states, eg, if prop `p` is in column `ISIN` this means that the
           set of states of the component is included in the sate of states
           that satisfy `p` in the whole LTS
        """
        nodes = self.g.get_vertex_dataframe().set_index("node")
        self._make_ncols(nodes, "size", "consts", "traps", *(r.name for r in setrel))
        nodes.sort_index(inplace=True)
        return nodes

    def _make_ec_actions(self, actions):
        "Compute the columns `actions` of table `.edges`."
        g2a = self.model._gal2act
        m2g = self.lts.m2g
        return hset(
            tag
            for act in actions
            for tag in (
                [g2a[m2g[act]]] if act.startswith("tick.") else g2a[m2g[act]].tags
            )
        )

    @cached_property
    def edges(self):
        """The _edges table_ holds all about the edges of a `ComponentGraph`.

        This is stored as a [`pandas.DataFrame`](https://pandas.pydata.org)
        whose columns are:

         - `src`: source component number
         - `dst`: destination component number
         - `actions`: rules or constraints on this edge
         - `tags`: the corresponding action labels
        """
        if len(self.g.es) == 0:
            return pd.DataFrame([], ["src", "dst"], ["actions", "tags"])
        else:
            edges = self.g.get_edge_dataframe().set_index(["src", "dst"])
            del edges["source"], edges["target"]
            edges["tags"] = edges["actions"].apply(self._make_ec_actions)
            edges.sort_index(inplace=True)
            return edges

    #
    # properties management
    #

    def update(self):
        """Update the nodes table if new properties have been checked.

        This method is normally called automatically, but if some `Component`
        are updated directly, then `update` should be called.
        """
        nodes = getattr(self, "_cached_nodes", None)
        if nodes is not None:
            self._make_ncols(nodes, *(r.name for r in setrel))
            nodes.sort_index(inplace=True)

    def _get_args(
        self,
        args,
        aliased={},
        topo_props=False,
        min_props=0,
        max_props=None,
        min_comps=0,
        max_comps=None,
    ):
        """Collect the arguments for many methods.

        Most methods accept a series a component numbers followed by a series
        of properties, with possibly aliased properties passed as keyword
        arguments. This method collects these argument while ensuring some
        constraints on what is passed.

        Arguments:
         - `args`: the arguments passed to the method
         - `aliased`: the keyword properties passed to the method
         - `bool topo_props`: should we accept `topo` values as properties
         - `int min_props` and `int max_props`: min/max number of properties
           accepted
         - `int min_comps` and `int max_comps`: min/max number of component
           numbers accepted

        Return: a pair `props, comps` where
         - `props` is a list of `p, a` where `p` is a property and `a` is an
           alias for it (a `str` or `None`)
         - `comps` is a list of `Component`
        """
        props, comps = [], []
        for i, p in enumerate(args):
            if isinstance(p, str):
                props.append((p, None))
            elif isinstance(p, topo):
                if not topo_props:
                    raise TypeError(f"unexpected topological property {p}")
                props.append((p, None))
            elif isinstance(p, int):
                comps.extend(self[a] for a in args[i:])
                break
            else:
                raise TypeError(f"unexpected argument {p!r}")
        else:
            comps.extend(self.components)
        props.extend((v, k) for k, v in aliased.items())
        if (found := len(props)) < min_props:
            raise TypeError(f"expected at least {min_props} properties, got {found}")
        elif max_props is not None and found > max_props:
            raise TypeError(f"expected at most {max_props} properties, got {found}")
        if (found := len(comps)) < min_comps:
            raise TypeError(f"expected at least {min_comps} components, got {found}")
        elif max_comps is not None and found > max_comps:
            raise TypeError(f"expected at most {max_comps} components, got {found}")
        return props, comps

    def forget(self, *args):
        """Forget about properties in components.

        Every specified component (or all if non are specified) will forget
        about every specified property.

        # Arguments

         - `str, ...`: a series of at least one property to forget
         - `int, ...`: a series of components that must forget about properties
           if none is given, all the components are used
        """
        props, comps = self._get_args(args, topo_props=True, min_props=1)
        for p, _ in props:
            for c in comps:
                if isinstance(p, topo):
                    c.forget(p.name)
                else:
                    c.forget(p)
        self.update()

    def forget_all(self, *args):
        """Forget about all properties forsome components.

        # Arguments

         - `int, ...`: the components that should forget about all properties,
           if none is given, all the components are used
        """
        _, comps = self._get_args(args, max_props=0)
        for c in comps:
            c.forget_all()
        self.update()

    def tag(self, *args):
        """Tag components with arbitrary properties.

        Arbitrary properties mean that they are just `str` that are associated
        to the states of the components. Thus they are _tags_ rather that real
        properties expressed in a particular syntax. Note that if several
        components are passed, the states corresponding to the tag will be the
        union of these components' states.

        # Arguments

         - `str, ...`: at leat one tag to add
         - `int, ...`: the components that should be tagged,
           if none is given, all the components are used
        """
        props, comps = self._get_args(args, min_props=1)
        states = reduce(or_, comps)
        for p, _ in props:
            for c in comps:
                c.check(p, states.states)
        self.update()

    def check(self, *args, _warn=True, **aliased):
        """Check properties on components.

        Properties are expressed in one of the supported syntaxes (CTL,
        topological, or functional). Each property is evaluated globally on
        the LTS and then compared with the states of the considered components.
        Their relation is the recorded in the components and the nodes table.
        Properties may be aliased, that is, an alias is recorded instead of
        the property itself, this allow producing a simpler nodes table with
        short property names instead of complex expressions.

        # Arguments

         - `str, ...`: at least of property to be checked
         - `int, ...`: the components on which to check the properties,
           if none is given, all the components are used
         - `_warn (bool=True)`: shall we print a warning if a property is found
           to match no states
         - `alias=prop, ...`: aliased properties

        # Return

        A `dict` that maps each checked component to a `dict` that
        itself maps each checked property to the corresponding `setrel`.
        """
        props, comps = self._get_args(args, aliased, min_props=1)
        ret = {c.num: {} for c in comps}
        for prop, alias in props:
            checker = ModelChecker(self, prop)
            for c in comps:
                states = checker(c.states)
                ret[c.num][alias or prop] = c.check(prop, states, alias)
            if _warn and not self.lts.props[prop]:
                log.warn(f"property {alias or prop} is empty")
        self.update()
        return ret

    #
    # queries
    #

    @cached_property
    def _GCS(self):
        "A _GCS subclass tied to the components of this graph."

        class MyGCS(_GCS):
            components = {c.num for c in self.components}

        return MyGCS

    # setrel, strict -> matching setrel
    _relmatch = {
        (setrel.HASNO, True): {setrel.HASNO},
        (setrel.HASNO, False): {setrel.HASNO},
        (setrel.HAS, True): {setrel.HAS},
        (setrel.HAS, False): {setrel.HAS, setrel.CONTAINS, setrel.ISIN, setrel.EQUALS},
        (setrel.CONTAINS, True): {setrel.CONTAINS},
        (setrel.CONTAINS, False): {setrel.CONTAINS, setrel.EQUALS},
        (setrel.ISIN, True): {setrel.ISIN},
        (setrel.ISIN, False): {setrel.ISIN, setrel.EQUALS},
        (setrel.EQUALS, True): {setrel.EQUALS},
        (setrel.EQUALS, False): {setrel.EQUALS},
    }

    def __call__(self, prop, rel=setrel.HAS, strict=False):
        """Return the component numbers that match `prop` with relation `rel`.

        If the property has not been checked already, `check` is called for it.
        Matching can be strict, in which case exactly `rel` must be found. But
        if matching is not strict, other relations can be considered, for
        instance, non-strict `HAS` can match `CONTAINS` (if a component
        contains a property, then is has its states), `ISIN`, and `EQUALS`.

        The returned set can be complemented with `~`, which allows to get
        the components that do not match a property. Moreover, several such
        queries can be combined using the usual sets operations.

        # Arguments

         - `prop (str|topo)`: property to search
         - `rel (setrel)`: expected relation
         - `strict (bool=False)`: whether matching should be strict or not

        # Return

        A set of matching component numbers.
        """
        update = False
        for c in self.components:
            prop = self.lts.saila.get(prop, prop)
            if prop not in c.props:
                update = True
                self.check(prop, c.num)
        if update:
            self.update()
        return self._GCS(
            c.num
            for c in self.components
            if c.props[prop] in self._relmatch[rel, strict]
        )

    def HASNO(self, prop):
        """Shorthand for `__call__(prop, setrel.HASNO, True)`."""
        return self(prop, setrel.HASNO, True)

    def hasno(self, prop):
        """Shorthand for `__call__(prop, setrel.HASNO, False)`."""
        return self(prop, setrel.HASNO, False)

    def HAS(self, prop):
        """Shorthand for `__call__(prop, setrel.HAS, True)`."""
        return self(prop, setrel.HAS, True)

    def has(self, prop):
        """Shorthand for `__call__(prop, setrel.HAS, False)`."""
        return self(prop, setrel.HAS, False)

    def CONTAINS(self, prop):
        """Shorthand for `__call__(prop, setrel.CONTAINS, True)`."""
        return self(prop, setrel.CONTAINS, True)

    def contains(self, prop):
        """Shorthand for `__call__(prop, setrel.CONTAINS, False)`."""
        return self(prop, setrel.CONTAINS, False)

    def ISIN(self, prop):
        """Shorthand for `__call__(prop, setrel.ISIN, True)`."""
        return self(prop, setrel.ISIN, True)

    def isin(self, prop):
        """Shorthand for `__call__(prop, setrel.ISIN, False)`."""
        return self(prop, setrel.ISIN, False)

    def EQUALS(self, prop):
        """Shorthand for `__call__(prop, setrel.EQUALS, True)`."""
        return self(prop, setrel.EQUALS, True)

    def equals(self, prop):
        """Shorthand for `__call__(prop, setrel.EQUALS, False)`."""
        return self(prop, setrel.EQUALS, False)

    def select(self, *args, _all=True, _rel=setrel.ISIN, _strict=False, **aliased):
        """Return the component numbers that match all/any properties.

        Given a series of properties `props` and series of components `comps`,
        this method returns the subset of components among `comps` that match
        all (if `_all=True`) or any (if `_all=False`) of the properties in
        `props`. Additional properties may be given using `aliased`. Matching
        is performed as in `__call__` using `_rel` and `_strict`.

        # Arguments

         - `str|topo, ...`: at least one property to be matched against the
           states of components
         - `int, ...`: the components on which to check the properties,
           if none is given, all the components are used
         - `_all (bool=True)`: whether components should match all or any of the
           properties
         - `_rel (setrel=ISIN)`: the relation used in matching
         - `_strict (bool=False)`: whether to perform strict matching or not
         - `alias=prop, ...`: aliased properties to be matched with others

        # Return

        The set of matching components.
        """
        props, comps = self._get_args(args, aliased, min_props=1)
        _rel = {_rel} if isinstance(_rel, setrel) else set(_rel)
        if not _strict:
            lose = set()
            for r in _rel:
                lose.update(self._relmatch[r, False])
            _rel.update(lose)
        found = set(c.num for c in comps) if _all else set()
        update = False
        for p, a in props:
            for c in comps:
                if p not in c.props:
                    update = True
                    self.check(p, c.num, a)
        if update:
            self.update()
        for p, a in props:
            for c in comps:
                match = {c.num for c in comps if c.props[a or p] in _rel}
                if _all:
                    found.intersection_update(match)
                else:
                    found.update(match)
        return found

    def count(self, *args, transpose=False):
        """Count how many states have each variable in each value.

        # Argument

         - `int, ...`: the components to count in, or all if non is given
         - `transpose (bool=True)`: transpose the matrix before to return it

        # Return

        A [`pandas.DataFrame`](https://pandas.pydata.org) instance whose
        columns are the component numbers and whose index is the list of
        pairs `var, val`.
        """
        _, comps = self._get_args(args, max_props=0)
        lts = self.lts
        g2m, vmin, vmax = lts.g2m, lts.vmin, lts.vmax
        data = [
            [g2m[var], val, *(c.counts.get((var, val), 0) for c in comps)]
            for var in lts.vars
            for val in range(vmin[var], vmax[var] + 1)
        ]
        df = pd.DataFrame.from_records(
            data, columns=["variable", "value", *(c.num for c in comps)]
        )
        df.set_index(["variable", "value"], inplace=True)
        df.sort_index(inplace=True)
        if transpose:
            return df.transpose()
        else:
            return df

    def form(self, *args, variables=None, normalise=None, separate=False):
        """Describe components by Boolean formulas.

        # Arguments

         - `int, ...`: a series of components number, if empty,
           all the components in the graph are considered
         - `variables=...`: a collection of variables names to restrict to
         - `normalise=...`: how the formula should be normalised, which must
           be either:
            - `"cnf"`: conjunctive normal form
            - `"dnf"`: disjunctive normal form
            - `None`: chose the smallest form
         - `separate=...`: if `False` (default) returns a single formula,
           otherwise, returns one formula for each considered component

        # Return

        A [SymPy Boolean formula](https://docs.sympy.org/latest/modules/logic.html)
        or a `dict` mapping component numbers to such formulas.
        """

        def remap(expr):
            if isinstance(expr, sympy.Symbol):
                return sympy.Symbol(self.lts.g2m[expr.name])
            else:
                return expr.func(*(remap(a) for a in expr.args))

        _, compos = self._get_args(args, max_props=0)
        if variables is not None:
            variables = [self.lts.m2g[v] for v in variables]
        forms = {c.num: remap(c.form(variables, normalise)) for c in compos}
        if separate:
            return forms
        else:
            return sympy.simplify_logic(sympy.Or(*forms.values()), form=normalise)

    def walk(self, *nodes, **filters):
        return CGT(cg=self, tips=set(nodes or self._c)).all(**filters)

    #
    # splits
    #

    def untrap(self, *args, _recurse=False):
        """Remove traps from nodes.

        A trap in a node `n` is defined by an input node `i` and the set `t` of
        states that `i` allows to reach within `n`. If `t` does not include all the
        exits of `n`, then this is a trap and this method splits `n` into `n & t` and
        `n - t`. Doing so, new traps may be created on the new nodes, in which case
        `untrap` should be applyied recursively until no traps remain.

        # Arguments

         - `int, ...`: a series of components number, if empty,
           all the components in the graph are considered
         - `_recurse (bool=True)`: shall untrap be applied recursively

        # Return

        A new `ComponentGraph` (or the original one if no split occurred).
        """
        _, comps = self._get_args(args, max_props=0)
        g = self
        new = set()
        while True:
            cgc = set(g.components) - set(comps)
            for c in comps:
                traps = g.g.vs[g._g[c.num]]["traps"]
                parts = [c]
                for n, t in traps.items():
                    parts = [
                        s
                        for p in parts
                        for s in p.split_prop(f"_trap_{n}_{c.num}", t, None)
                        if s is not None
                    ]
                    for p in parts:
                        p.forget(f"_trap_{n}_{c.num}")
                cgc.update(parts)
                new.update(parts)
            g = self.__class__(self.model, self.lts, *cgc)
            if not _recurse:
                break
            comps = [c for c in cgc & new if g.g.vs[g._g[c.num]]["traps"]]
            if not comps:
                break
        return g

    def _split(self, props, comps, rem, add, warn):
        """Split `comps` yielding the size of the result at each split.

        Auxiliary method for `split()` allowing to stop splitting at a given
        size limit.
        """
        size = len(self)  # size of the resulting ComponentGraph
        comps = set(comps)
        for prop, alias in props:
            if isinstance(prop, str):
                checker = ModelChecker(self, prop)
                for c in comps:
                    intr, diff = c.split_prop(prop, checker(c.states), alias)
                    if intr is not None and diff is not None:
                        rem.add(c)
                        add.add(intr)
                        add.add(diff)
                        size += 1
                        yield size
                if warn and not self.lts.props[prop]:
                    log.warn(f"property {alias or prop!r} is empty")
            elif isinstance(prop, topo):
                for c in comps:
                    for p in c.split_topo(prop):
                        if p != c:  # split_topo yields c if not split occurs
                            rem.add(c)
                            add.add(p)
                            yield size  # count after because 1st yield will
                            size += 1  # correspond to rem.add(c)
            else:
                raise TypeError(f"unexpected property {prop!r}")
            add.difference_update(rem)
            comps = (comps - rem) | add

    def split(self, *args, _warn=True, _limit=256, **aliased):
        """Split components with respect to properties.

        For each given property, the given components are split into the part
        that validate the property and the part that doesn't. Properties may
        be formulas given as `str` in one of the available syntaxes, or `topo`
        values. The properties used for splitting are recorded in the
        components and the nodes tables, either as such or using an alias
        if `aliased` is used.

        # Arguments

         - `int, ...`: components to be split, or all if non is given
         - `str|topo, ...`: properties to be used for splitting
         - `_warn (bool=True)`: whether to issue a warning if a property
           is found to match no states
         - `_limit (int=256)`: ask a confirmation if splitting yields more that
           the given number of components, giving `0` disables confirmation
         - `alias=prop, ...`: specify aliased properties that are stored as
           `alias` instead of `prop` in the components and the nodes table

        # Return

        A new `ComponentGraph` (or the original one if no split occurred).
        """
        props, comps = self._get_args(args, aliased, min_props=1, topo_props=True)
        limit = _limit
        rem, add = set(), set()
        for size in self._split(props, comps, rem, add, _warn):
            if limit and size > limit:
                answer = input(
                    f"We have created a graph with {size} nodes,"
                    f" do you want to continue? [N/y] "
                )
                if not answer.lower().startswith("y"):
                    break
                limit += _limit
        if not add:
            if _warn:
                log.warn("no split occurred")
            return self
        return self.__class__(
            self.model, self.lts, *((set(self.components) - rem) | add)
        )

    def explicit(self, *args, _limit=256):
        """Split components into their individual states.

        Every provided component is split into its individual states.

        # Arguments

         - `int, ...`: components to be split, or all if non is given
         - `_limit (int=256)`: as for `split()`

        # Return

        A new `ComponentGraph` (or the original one if no split occurred).
        """
        _, comps = self._get_args(args, max_props=0)
        if _limit:
            size = sum((len(c) for c in comps), 0) + len(self.components) - len(comps)
            if size > _limit:
                answer = input(
                    f"We will create a graph with {size} nodes,"
                    f" do you want to continue? [N/y] "
                )
                if not answer.lower().startswith("y"):
                    return
        rem, add, s2c = set(), set(), {}
        if (
            nodes := getattr(self, "_cached_nodes", None)
        ) is not None and "component" in nodes.columns:
            s2c.update((row.name, row.component) for _, row in nodes.iterrows())
        for c in comps:
            if len(c) > 1:
                new = tuple(c.explicit())
                s2c.update((n.num, c.num) for n in new)
                rem.add(c)
                add.update(new)
        cg = self.__class__(self.model, self.lts, *(set(self.components) - rem | add))
        cg.n["component"] = lambda row: s2c.get(row.name, row.name)
        return cg

    def _classify_classname(self, classes, row):
        "Compute class names column for `classify`."
        props = set()
        for c in ("EQUALS", "ISIN"):
            if c in row.index and row[c] and (cls := row[c] & classes):
                props.update(cls)
        return hset(props)

    def classify(
        self, *args, _split=True, _limit=256, _col=None, _src=None, _dst=None, **aliased
    ):
        """Classify components against a set of aliased properties.

        Every given component is checked or split against every given propery.
        A column is the added to the resulting nodes table to summarize in
        which class every resulting component is.

        # Arguments

         - `int, ...`: components to be classified, or all if none is given
         - `_split (bool=True)`: wether the classified components should be
           split wrt the properties or we shoud used the existing components
           as such
         - `_col (str=None)`: if not `None` add a column `_col` in the
           resulting nodes table to summarize in which class every component is
         - `_src (str=None)`: add a column to the edges table with the
           classification of the source nodes
         - `_dst (str=None)`: add a column to the edges table with the
           classification of the destination nodes
         - `classname=prop, ...`: classes to be used and the corresponding
           properties

        # Return

        A new `ComponentGraph` (or the original one with properties
        updated if `_split=False`).
        """
        _, comps = self._get_args(args, max_props=0)
        if _split:
            cg = self.split(*(c.num for c in comps), _limit=_limit, **aliased)
        else:
            cg = self
            cg.check(*comps, **aliased)
        if _col:
            cg.n[_col] = partial(self._classify_classname, set(aliased))
            if _src:
                cg.e[_src] = lambda row: cg.nodes[_col][row.name[0]]
            if _dst:
                cg.e[_dst] = lambda row: cg.nodes[_col][row.name[1]]
        return cg

    #
    # other operations
    #

    def merge(self, *args):
        """Merge components into a single one.

        # Arguments

         - `int, int, ...`: at least two components to be merged

        # Return

        A new `ComponentGraph`.
        """
        _, comps = self._get_args(args, max_props=0, min_comps=2)
        rem = set(comps)
        add = {comps[0].merge_with(comps[1:])}
        return self.__class__(self.model, self.lts, *(set(self.components) - rem | add))

    def drop(self, *args, _warn=True):
        """Drop some components.

        # Arguments

         - `int, ...`: at least one component to be dropped
         - `_warn (bool=True)`: issue a warning if all the components are to be
           dropped, which is not possible

        # Return

        A new `ComponentGraph` (or `None` if all the components are to
        be dropped).
        """
        _, comps = self._get_args(args, max_props=0, min_comps=1)
        keep = set(self.components) - set(comps)
        if _warn and len(keep) == 0:
            log.warn("cannot drop all the components")
            return
        return self.__class__(self.model, self.lts, *keep)

    def search(
        self,
        *comps,
        _col="search",
        _prune=False,
        _split=True,
        _warn=True,
        _limit=256,
        **stages,
    ):
        """Search for sequences of properties.

        The graph is first classified as with `classify()` then for every pair
        of successive class `p, q`, the components that validate `p` are
        split with respect to propery `reach_q` that checks whether it allows
        to reach the components that validate `q`. Then, if a component has
        both `p` and `reach_q` it is considered as a valid step of the searched
        sequence and this is noted in a new column of the nodes tables.

        Arguments:
         - `int, ...`: components to be used, or all if none is given
         - `str _col="search"`: column to add to the nodes table
         - `bool _prune=Fase`: whether to remove or not the components that do
           not match any of the searched properties
         - `bool _split=True`: whether to split the components wrt the searched
           properties or to use the existing ones as such
         - `bool _warn=True`: wheter to issue a warning if a property is found
           to match no states or if all the components are to be pruned
         - `int _limit=256`: as for `split()`
         - `classname=prop, classname=prop, ...`: at least two named properties
           to be used for the classification. Their order is the order in which
           reachability will be checked

        Return: a new `ComponentGraph` (or `None` if all components have to be
        pruned) with one column in its node table giving for each component the
        classes it belongs to
        """
        # FIXME: to be tested
        self._get_args(comps, max_props=0)  # just to check args
        if len(stages) < 2:
            raise TypeError(f"expected at least two properties, but got {len(stages)}")
        # first: check searched properties
        if _split:
            cg = self.split(*comps, _warn=_warn, **stages)
        else:
            self.check(*comps, _warn=_warn, **stages)
            cg = self
        # prune is required
        if _prune:
            cg = cg.drop(*(set(cg._c) - reduce(or_, (cg.isin(a) for a in stages))))
            if cg is None:
                if _warn:
                    log.warn("none of the required stages were found")
                return
        # check that each property allows to reach the next one
        stnames = tuple(stages)
        for stg, nxt in zip(stnames, stnames[1:]):
            if (src := cg.isin(stg)) and cg.isin(nxt):
                tgt_form = f"EF({stages[nxt]})"
                alias = f"reach_{nxt}"
                if _split:
                    cg = cg.split(*src, _warn=_warn, _limit=_limit, **{alias: tgt_form})
                else:
                    cg.check(*src, _warn=_warn, **{alias: tgt_form})
        # which component validates property + reachability -> new column
        both = defaultdict(set)
        for c in cg.isin(stnames[-1]):
            both[c].add(stnames[-1])
        for stg, nxt in zip(stnames, stnames[1:]):
            for c in cg.isin(stg) & cg.isin(f"reach_{nxt}"):
                both[c].add(stg)
        cg.n[_col] = lambda row: hset(both[row.name])
        return cg

    #
    # draw
    #

    def _nodes_label_str(self, txt, **args):
        "add icons to state labels"
        txt = str(txt)
        if not txt:
            return ""
        icons = []
        if "INIT" in args.get("EQUALS", ()):
            icons.append("▶")
        elif any("INIT" in args.get(rel, ()) for rel in ("HAS", "CONTAINS", "ISIN")):
            icons.append("▷")
        if "scc" in args.get("EQUALS", ()):
            icons.append("⏺")
        elif "hull" in args.get("EQUALS", ()):
            icons.append("☍")
        elif any(prop in args.get("CONTAINS", ()) for prop in ("scc", "hull")):
            icons.append("○")
        if any("DEAD" in args.get(rel, ()) for rel in ("ISIN", "EQUALS")):
            icons.append("⏹")
        elif any("DEAD" in args.get(rel, ()) for rel in ("HAS", "CONTAINS")):
            icons.append("□")
        if args.get("traps", ()):
            icons.append("⬖")
        if icons:
            return f"{txt}\n{''.join(icons)}"
        else:
            return txt

    def draw(self, **opt) -> Graph | None:
        """Draw the component graph as an interactive `cygraphs.Graph`.

        It accepts any option `opt` accepted by `cygraphs.Graph` and returns
        an interactive `cygraphs.Graph` instance that can be directly displayed
        in a Jupyter notebook when it is the result of the cell (otherwise,
        use Jupyter's function `display`).
        """
        if len(self) < 2:
            log.print("cannot draw a graph with only one node", "error")
            display(self.nodes)
            return
        options = dict(
            nodes_fill_color="size",
            nodes_fill_palette=("red-green/white", "abs"),
            nodes_label_str=self._nodes_label_str,
            edges_label="actions",
            edges_draw_style="trap",
        )
        options.update(opt)
        return Graph(self.nodes, self.edges, **options)

    def _ipython_display_(self):
        display(self.draw())

    #
    # other methods
    #

    def make_states(self, spec):
        """Parse a string as a partial valuation.

        `spec` is expacted to be a comma-separated list of valuations, each
        valuation being one of:

         - `VAR=VAL`, where `VAR` is a variable name and `VAL` a value for it
         - `VAR+` that is a shorthand for `VAR=1`
         - `VAR-` that is a shorthand for `VAR=0`
         - `VAR*` that is a shorthand for `VAR=v` for all `v` in the domain of
           `VAR`

        The specified valuations are accumulated. Variables that have not
        been valuated are assumed to range on their domains.

        # Arguments

         - `str spec`: specification to be parsed

        # Return

        A `sdd` containing the specifies states.
        """
        return self.lts(self._parse_state(spec, self.lts.m2g))
