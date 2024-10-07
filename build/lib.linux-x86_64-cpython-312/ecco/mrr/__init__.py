import re

from itertools import product
from collections import defaultdict

import pandas as pd

from colour import Color

from .parser import parse, parse_src
from .cg import ComponentGraph
from .lts import topo, setrel
from .. import BaseModel, hset
from ..cygraphs import Graph
from ..ui import HTML


class Model(BaseModel):
    @classmethod
    def parse(cls, path, *extra):
        return cls(path, parse(path, *extra))

    @classmethod
    def parse_src(cls, path, source, *extra):
        return cls(path, parse_src(source, *extra))

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
                index[*path, var.name] = vname
                if keepvar(vname):
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

         - `_init (None)`: initial state given as a `dict` mapping variable
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

    def gal(self):
        """Export the model as a GAL specification.

        The specification is saved in a file called `basename/basename.gal`
        where `basename.mrr` is the file from which the MRR model was loaded.
        The GAL specification is defined with a unique initial state that is
        chosen using the smallest of initial value of each variable.

        Return: a pair `init, doms` where
         - `init` is a `dict` suitable to initialise a `cg.ComponentGraph` that
           gathers all the initial states as specified in the MRR model (and
           not just that savec in the GAL)
         - `doms` is a `dict` that maps every variable to its domain
        """
        init, doms, clocks = {}, {}, {}
        gnames = self.spec.gal = {}
        name = re.sub("[^a-z0-9]+", "", self.base.name, flags=re.I)
        with self["gal"].open("w") as out:
            out.write(f"gal {name} {{\n")
            self._gal_vars(out, self.spec, [], init, doms, clocks, gnames)
            guard = set()
            self._gal_actions(out, self.spec, [], "constraints", guard, None, gnames)
            if guard:
                guard = " || ".join(f"({g})" for g in sorted(guard))
            else:
                guard = None
            self._gal_actions(out, self.spec, [], "rules", guard, None, gnames)
            self._gal_clocks(out, clocks, guard, gnames)
            out.write("}\n")
        return init, doms, gnames

    def _gal_vars(self, out, loc, path, init, domains, clocks, gnames):
        "Auxiliary methd that exports variables to GAL."
        head = ".".join(path) + ("." if path else "")
        for var in loc.variables:
            if var.isbool:
                vtype = "bool"
            elif var.clock is not None:
                vtype = "clock"
                clocks.setdefault(var.clock, {})
            else:
                vtype = "int"
            if var.size is None:
                vname = f"{vtype}_{head}{var.name}"
                if var.init is None:
                    out.write(f"  int {vname} = -1;\n")
                    init[vname] = {-1}
                else:
                    out.write(f"  int {vname} = {min(var.init)};\n")
                    init[vname] = var.init
                domains[vname] = var.domain | {-1}
                gnames[vname] = var
                if var.clock:
                    clocks[var.clock][vname] = var
            else:
                for idx in range(var.size):
                    vname = f"{vtype}_{head}{var.name}.{idx}"
                    if var.init[idx] is None:
                        init[vname] = {-1}
                        out.write(f"  int {vname} = -1;\n")
                    else:
                        init[vname] = var.init[idx]
                        out.write(f"  int {vname} = {min(var.init[idx])};\n")
                    domains[vname] = var.domain
                    gnames[vname] = var
                    if var.clock:
                        clocks[var.clock][vname] = var
        for sub in loc.locations:
            if sub.size is None:
                self._gal_vars(
                    out, sub, path + [f"{sub.name}"], init, domains, clocks, gnames
                )
            else:
                for idx in range(sub.size):
                    self._gal_vars(
                        out,
                        sub,
                        path + [f"{sub.name}.{idx}"],
                        init,
                        domains,
                        clocks,
                        gnames,
                    )

    def _gal_actions(self, out, loc, path, kind, guard, locidx, gnames):
        "Auxiliary method that exports actions to GAL."
        head = ".".join(path) + ("." if path else "")
        for num, act in enumerate(getattr(loc, kind), start=1):
            name = f"{head}{kind[0].upper()}{num}"
            self._gal_act(out, act, name, guard, path, locidx, gnames)
        for sub in loc.locations:
            if sub.size is None:
                self._gal_actions(
                    out, sub, path + [f"{sub.name}"], kind, guard, locidx, gnames
                )
            else:
                for idx in range(sub.size):
                    self._gal_actions(
                        out, sub, path + [f"{sub.name}.{idx}"], kind, guard, idx, gnames
                    )

    def _gal_act(self, out, act, name, guard, path, locidx, gnames):
        "Auxiliary method that exports one action to GAL"
        if locidx is None:
            binding = {}
        else:
            binding = {"self": locidx}
        out.write(f"  // {act}\n")
        for new, val in act.expand_any(binding):
            if val:
                tail = "._." + ".".join(f"{k}.{v}" for k, v in sorted(val.items()))
            else:
                tail = ""
            gnames[f"{name}{tail}"] = new
            cond = " && ".join(sorted(f"({b.gal(path)})" for b in new.left))
            loop = " && ".join(sorted(f"({a.cond().gal(path)})" for a in new.right))
            condloop = f"{cond} && !({loop})"
            if isinstance(guard, set):
                guard.add(condloop)
                out.write(f"  transition {name}{tail} [{condloop}] {{\n")
            elif guard is None:
                out.write(f"  transition {name}{tail} [{condloop}] {{\n")
            else:
                out.write(
                    f"  transition {name}{tail} [{condloop}" f" && !({guard})] {{\n"
                )
            out.write(f"    // {new}\n")
            for right in new.right:
                out.write("    " + "\n    ".join(right.gal(path)) + "\n")
            out.write("  }\n")

    def _gal_clocks(self, out, clocks, guard, gnames):
        "Auxiliary method that generates clock ticks into GAL."
        for clock, variables in clocks.items():
            gnames[f"tick.{clock}"] = (clock, variables)
            skip = " && ".join(f"({name} == -1)" for name in variables)
            cond = " && ".join(
                f"({name} < {max(var.domain)})" for name, var in variables.items()
            )
            if guard is None:
                out.write(f"  transition tick.{clock}" f" [!({skip}) && {cond}] {{\n")
            else:
                out.write(
                    f"  transition tick.{clock}"
                    f" [!({skip}) && {cond} && !({guard})] {{\n"
                )
            for name in variables:
                out.write(f"    if ({name} >= 0) {{ {name} += 1; }}\n")
            out.write("  }\n")


__extra__ = {"topo": topo, "setrel": setrel}

__extra__.update({val.name: val for cls in (topo, setrel) for val in cls})
