import operator
import re
import sys
from collections import defaultdict
from itertools import product
from functools import reduce
from typing import cast

import pandas as pd
from colour import Color
import sympy as sp

from .. import BaseModel, hset
from ..cygraphs import Graph
from ..ui import HTML
from .cg import ComponentGraph
from .lts import setrel, topo
from .parser import Action, BinOp, Location, ModItem, VarDecl, VarUse, parse, parse_src


class Model(BaseModel):
    """
    Options:
    - `traps: bool = False`: whether traps should be computed automatically or not
    """

    __opts__ = {"traps": False}

    @classmethod
    def parse(cls, path, *extra, **opts):
        return cls(path, parse(path, *extra), opts=opts)

    @classmethod
    def parse_src(cls, path, source, *extra, **opts):
        return cls(path, parse_src(source, *extra), opts=opts)

    @property
    def mrr(self):
        """Show MRR source code.

        The code shown is reconstructed from parsed source, so it may have some
        differences with it (and is usually clearer).
        """
        h = HTML()
        with h("pre", style="font-family:mono;line-height:140%;"):
            self.spec.__html__(h)
        return h

    def charact(self, constraints=True, variables=None, plot=True):
        """Show variables characterisation.

        This is a table that counts for each variable how many time it is
        read and how many times it is assigned.

        # Arguments

         - `constraints (bool=True)`: shall constraints be taken into account
         - `variables (None)`: list of variables to include, or a function that
           take a variable name and return a `bool`
         - `plot (bool=True)`: if `True` add bars to the table to be displayed
           in Jupyter, otherwise, just return the data

        # Return

         - a styled `pd.io.formats.style.Styler` if `plot=True`
         - a raw `pd.DataFrame` if `plot=False`
        """
        if variables is None:
            keepvar = lambda _: True
        elif isinstance(variables, (set, list, tuple, dict)):
            keepvar = set(variables).__contains__
        elif callable(variables):
            keepvar = variables
        else:
            raise TypeError(f"unexpected {variables=}")
        count = dict()
        index = {}
        for path, loc in self.spec:
            for var in loc.variables:
                vname = ".".join(path + [var.name])
                if keepvar(vname):
                    index[*path, var.name] = vname
                    count.setdefault(("left", vname), 0)
                    count.setdefault(("right", vname), 0)
            for act in loc.actions if constraints else loc.rules:
                for side in ("left", "right"):
                    for item in getattr(act, side):
                        for pth, var in item.vars(path):
                            vname = ".".join(pth + [var.name])
                            if keepvar(vname):
                                count.setdefault((side, vname), 0)
                                count[side, vname] += 1
        index = [val for _, val in sorted((len(k), v) for k, v in index.items())]
        df = pd.DataFrame(
            [
                [count.get(("left", var), 0), count.get(("right", var), 0)]
                for var in index
            ],
            index=index,
            columns=["left", "right"],
        )
        if plot:
            return df.style.bar(width=80, height=60, color="#3fb63f")
        else:
            return df

    def ecograph(self, constraints=True, **opt):
        """Show the ecosystemic graph.

        The nodes are the variables and an edge from one variable to another
        means that the former has an influence onto the latter. there are two
        possible reasons for a variable `x` to have an influence onto `y`:

         - `x` appears in the left-hand side of an actions and `y` is assigned
           in the right-hand sign, eg: `x+ >> y+`
         - `y` is assigned with an expression involving `x`, but `x` is not
           necessarily a condition of the action, eg: `... >> y=x+1`

        # Arguments

         - `constraints (bool=True)`: whether to take the constraints into
           account
         - `key=val, ...`: any option suitable for `cygraphs.Graph`

        # Return

        A `cygraph.Graph` instance that can be directly displayed in Jupyter.
        """
        nodes = []
        edges = defaultdict(set)
        for path, loc in self.spec:
            for var in loc.variables:
                # one declared variable => one node
                vname = ".".join(path + [var.name])
                vtype = (
                    "bool" if var.isbool else f"{min(var.domain)}..{max(var.domain)}"
                )
                if var.init is None:
                    vinit = -1
                elif isinstance(var.init, (set, frozenset)):
                    vinit = sum(var.init) / len(var.init)
                else:
                    vinit = sum(
                        -1 if i is None else sum(i) / len(i) for i in var.init
                    ) / len(var.init)
                if vinit < 0:
                    color = "#ffaaff"
                else:
                    color = Color("#ffaaaa")
                    green = Color("#aaffaa")
                    vmax = max(var.domain)
                    color.hue = (color.hue * vinit + green.hue * (vmax - vinit)) / vmax
                    color = color.get_hex_l()
                nodes.append(
                    {
                        "node": vname,
                        "location": ".".join(path),
                        "size": var.size or 0,
                        "color": color,
                        "type": vtype,
                        "clock": var.clock or "",
                    }
                )
            for act in loc.actions if constraints else loc.rules:
                left, right = set(), set()
                for con in act.left:
                    # explicit conditions
                    for p, v in con.vars(path):
                        left.add(".".join(p + [v.name]))
                for ass in act.right:
                    for p, v in ass.target.vars(path):
                        # assigned variable
                        right.add(".".join(p + [v.name]))
                    for p, v in getattr(ass.value, "vars", lambda _: [])(path):
                        # variable also read but possibly not as conditions
                        left.add(".".join(p + [v.name]))
                for lft, rgt in product(left, right):
                    if lft != rgt:
                        edges[lft, rgt].add(act.name)
        nodes = pd.DataFrame.from_records(nodes, index=["node"])
        edges = pd.DataFrame.from_records(
            [{"src": s, "dst": d, "actions": hset(v)} for (s, d), v in edges.items()]
        )
        opts = dict(
            layout="circle",
            nodes_fill_color="color",
            nodes_fill_palette="red-green/white",
            nodes_shape="ellipse",
            edges_label="",
            edges_curve="straight",
        )
        opts.update(opt)
        return Graph(nodes, edges, **opts)

    def ecohyper(self, constraints=True, **opt):
        """Show the ecosystemic hypergraph.

        This is an evolution of the ecosystemic graph that is more detailed and
        precise. For each action, its has a square node that is linked to all
        the variables (round nodes) it involves. This link can be:

         - unlabelled and indirected if the variable only read by the action
         - directed towards the variable if it is assigned by the actions, in
           which case the edge is labelled by the assigned expression

        Arguments:
         - `constraints (bool=True)`: whether to take the constraints into
           account
         - `key=val, ...`: any option suitable for `cygraphs.Graph`

        Return: a `cygraph.Graph` instance that can be directly displayed
        in Jupyter
        """
        nodes = []
        edges = []
        for path, loc in self.spec:
            for var in loc.variables:
                # each declared variable is a node
                vname = ".".join(path + [var.name])
                if var.init is None:
                    vinit = -1
                elif isinstance(var.init, set):
                    vinit = sum(var.init) / len(var.init)
                else:
                    vinit = sum(
                        -1 if i is None else sum(i) / len(i) for i in var.init
                    ) / len(var.init)
                if vinit < 0:
                    color = "#ffaaff"
                else:
                    color = Color("#ffaaaa")
                    green = Color("#aaffaa")
                    vmax = max(var.domain)
                    color.hue = (color.hue * (vmax - vinit) + green.hue * vinit) / vmax
                    color = color.get_hex_l()
                nodes.append(
                    {
                        "node": vname,
                        "location": ".".join(path),
                        "kind": "variable",
                        "info": str(var),
                        "shape": "ellipse",
                        "color": color,
                    }
                )
            for kind in ("constraint", "rule"):
                if not constraints and kind == "constraint":
                    continue
                for act in getattr(loc, f"{kind}s"):
                    # each action is a node
                    nodes.append(
                        {
                            "node": act.name,
                            "location": ".".join(path),
                            "kind": kind,
                            "info": str(act),
                            "shape": "rectangle",
                            "color": ("#aaaaff" if kind == "rule" else "#ffffaa"),
                        }
                    )
                    left, right, assign = set(), set(), {}
                    for con in act.left:
                        # read variables
                        for p, v in con.vars(path):
                            left.add(".".join(p + [v.name]))
                    for ass in act.right:
                        for p, v in ass.target.vars(path):
                            # assigned variable (only one actually)
                            tgt = ".".join(p + [v.name])
                            right.add(tgt)
                            assign[tgt] = str(ass.value)
                        for p, v in getattr(ass.value, "vars", lambda _: [])(path):
                            # also read variables, skip if has no method vars
                            left.add(".".join(p + [v.name]))
                    for v in left - right:
                        # readonly => empty arrowless arc
                        edges.append(
                            {
                                "src": v,
                                "dst": act.name,
                                "lbl": "",
                                "get": "-",
                                "set": "-",
                            }
                        )
                    for v in right:
                        # written => arrowed arc labelled with assigned value
                        edges.append(
                            {
                                "src": v,
                                "dst": act.name,
                                "lbl": assign[v],
                                "get": ">",
                                "set": "-",
                            }
                        )
        opts = dict(
            nodes_shape="shape",
            nodes_fill_color="color",
            edges_label="lbl",
            edges_source_tip="get",
            edges_target_tip="set",
            inspect_nodes=["kind", "location", "info"],
            inspect_edges=[],
        )
        opts.update(opt)
        nodes = pd.DataFrame.from_records(nodes, index=["node"])
        edges = pd.DataFrame.from_records(edges, index=["src", "dst"])
        return Graph(nodes, edges, **opts)

    def __call__(self, *split, _init=None, **aliased) -> ComponentGraph:
        """Build a `cg.ComponentGraph` from the model.

        # Aguments

         - `_init (None)`: initial states given as a `dict` mapping variable
           names to sets of initial values
         - `split, ...`: arguments suitable to `ComponentGraph.split()` in
           order to perform an initial split (if none is given, the graph will
           have just one component with all the reachable states)
         - `alias=prop, ...`: other arguments for `ComponentGraph.split()`

        # Return

        A `cg.ComponentGraph` instance, possibly split as specified.
        """
        cg = ComponentGraph.from_model(self, _init)
        if split or aliased:
            return cg.split(*split, **aliased)
        else:
            return cg

    def universe(self, *split, **aliased) -> ComponentGraph:
        """Build a `cg.ComponentGraph` from the model with all possible states.

        Works as `__call__` but the inital states are automatically computed as
        all the possible states.
        """
        cg = ComponentGraph.universe(self)
        if split or aliased:
            return cg.split(*split, **aliased)
        else:
            return cg

    @property
    def maxsize(self):
        """Maximal size of a model

        This is the number of poissbile valuations that the variables may have.
        Usually this is much more than the number of reachable states.
        """

        def size(loc):
            s = 1
            for var in loc.variables:
                if var.clock:
                    s *= (len(var.domain) + 1) ** (var.size or 1)
                else:
                    s *= len(var.domain) ** (var.size or 1)
            for sub in loc.locations:
                s *= size(sub) ** (sub.size or 1)
            return s

        return size(self.spec)

    def expand(self, out=sys.stdout):
        self.spec.expand().save(out)

    def gal(self):
        """Export the model as a GAL specification.

        The specification is saved in a file called `basename/basename.gal`
        where `basename.mrr` is the file from which the MRR model was loaded.
        The GAL specification is defined with a unique initial state that is
        chosen using the smallest of initial value of each variable.

        Return: a triple `init, doms`, `actions` where
         - `init` is a `dict` suitable to initialise a `cg.ComponentGraph` that
           gathers all the initial states as specified in the MRR model (and
           not just that savec in the GAL)
         - `doms` is a `dict` that maps every variable to its domain
         - `actions` is the `set` of action names
        """
        init, doms, actions = {}, {}, {}
        flat = {k: [] for k in ModItem}
        name = re.sub("[^a-z0-9]+", "", self.base.name, flags=re.I)
        with self["gal"].open("w") as out:
            out.write(f"gal {name} {{\n")
            for kind, *rest in cast(Location, self.spec).flatten():
                flat[kind].append(rest)
            for locidx, decl in flat[ModItem.VAR]:
                self._gal_var(out, locidx, decl, init, doms)
            con_guards = []
            for locidx, act in flat[ModItem.CON]:
                con_guards.append(self._gal_act(out, locidx, act, actions, sp.true))
            extraguard = sp.Not(sp.Or(*con_guards))
            for locidx, act in flat[ModItem.RUL]:
                self._gal_act(out, locidx, act, actions, extraguard)
            for clock, vars in flat[ModItem.CLK]:
                self._gal_clk(out, clock, vars, actions, extraguard)
            out.write("}\n")
        return init, doms, actions

    def _gal_var(self, out, locidx, decl: VarDecl, init, doms):
        name = self._gal_vname(locidx, decl)
        init[name] = {-1} if decl.init is None else set(cast(frozenset[int], decl.init))
        doms[name] = set(decl.domain)
        if decl.clock is not None:
            doms[name].add(-1)
        out.write(f"  int {name} = {min(init[name])};\n")

    def _gal_vname(self, locidx, var: VarDecl | VarUse):
        decl = var if isinstance(var, VarDecl) else var.decl
        if decl.clock is not None:
            prefix = "clock"
        elif decl.isbool:
            prefix = "bool"
        else:
            prefix = "int"
        name = ".".join(var.name.rstrip("]").split("["))
        if isinstance(var, VarUse) and isinstance(var.index, int):
            path = var @ locidx + (name, var.index)
        else:
            path = var @ locidx + (name,)
        return f"{prefix}_{'.'.join(str(p) for p in path)}"

    def _gal_act(self, out, locidx, act: Action, actions, extraguard):
        def vname(var: VarUse):
            return self._gal_vname(locidx, var)

        path = act.loc.path @ locidx + (act.name,)
        name = ".".join(str(p) for p in path)
        if any(k != "self" for k in act.bound):
            name += "._." + ".".join(
                f"{k}.{v}"
                if isinstance(v, int)
                else f"{k}.{'.'.join(str(i) for i in sorted(v))}"
                for k, v in sorted(act.bound.items())
                if k != self
            )
        actions[name] = act
        cond = sp.And(*(c.sympy(-1, vname) for c in act.left))
        loop = sp.Or(
            *(
                BinOp(a.line, a.column, a.target, "!=", a.value).sympy(-1, vname)
                for a in act.right
            )
        )
        guard = sp.And(cond, loop, extraguard)
        out.write(
            f"  // {'.'.join(str(p) for p in (act.loc.path @ locidx) + (act.boundname,))}\n"
        )
        out.write(f"  // {act}\n")
        out.write(f"  transition {name} [{sp.ccode(guard)}] {{\n")
        for a in act.right:
            assert a.target.decl is not None
            assign = BinOp(a.line, a.column, a.target, "=", a.value)
            target = self._gal_vname(locidx, a.target)
            m = min(a.target.decl.domain)
            M = max(a.target.decl.domain)
            out.write(f"    {sp.ccode(assign.sympy(-1, vname))}\n")
            if isinstance(a.value, int):
                if not m <= a.value <= M:
                    out.write("    abort;  // out of bound assignment\n")
            elif a.value is not None:
                out.write(f"    if ({target} < {m}) {{ abort; }}\n")
                out.write(f"    if ({target} > {M}) {{ abort; }}\n")
        out.write("  }\n")
        return guard

    def _gal_clk(self, out, clock, variables, actions, extraguard):
        vdecl = {self._gal_vname(locidx, decl): decl for locidx, decl in variables}
        actions[f"tick.{clock}"] = clock
        skip = sp.Or(*(sp.Ne(sp.Symbol(v), -1) for v in vdecl))
        cond = sp.And(
            *(sp.Lt(sp.Symbol(name), max(var.domain)) for name, var in vdecl.items())
        )
        guard = sp.And(skip, cond, extraguard)
        out.write(f"  // tick.{clock}\n")
        out.write(f"  transition tick.{clock} [{sp.ccode(guard)}] {{\n")
        for name in vdecl:
            out.write(f"      if ({name} >= 0) {{ {name} += 1; }}\n")
        out.write("  }\n")


__extra__ = {"topo": topo, "setrel": setrel}

__extra__.update({val.name: val for cls in (topo, setrel) for val in cls})
