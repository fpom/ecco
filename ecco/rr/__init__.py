import re

import pandas as pd

from ..mrr import Model as MRRModel, __extra__ as __xtra
from ..cygraphs import Graph

from colour import Color


class Model(MRRModel):
    @classmethod
    def parse(cls, path, *defs, **opts):
        sec = re.compile(r"^\s*([a-z0-9_]+)\s*:\s*(#.*)?$", re.I)
        ind = re.compile(r"^(\s*)")
        source = []
        hasvar = False
        indecl = True
        indent = ""
        for line in (ln.rstrip() for ln in open(path)):
            if match := sec.match(line):
                if match.group(1) in ("rules", "constraints"):
                    source.append(line)
                    indecl = False
                elif indecl:
                    if hasvar:
                        source.append(f"{indent}# {line}")
                    else:
                        hasvar = True
                        source.append(f"variables:  # {line}")
                else:
                    source.append(line)
            else:
                source.append(line)
                if match := ind.match(line):
                    indent = match.group(0)
        return super().parse_src(path, "\n".join(source), *defs, **opts)

    @property
    def rr(self):
        """Show RR source code.

        The code shown is reconstructed from parsed source, so it may have some
        differences with it (and is usually clearer).
        """
        return self.mrr

    def ecohyper(self, constraints=True, **opt) -> Graph:
        """Show the ecosystemic hypergraph.

        This is an evolution of the ecosystemic graph that is more detailed and
        precise. For each action, its has a square node that is linked to all
        the variables (round nodes) it involves. This links can have several
        tips:

         - a white dot at the action end means that the action expects
           the variable to be off
         - a black dot at the action end means that the action expects
           the variable to be on
         - no tip at the action end means that the action does not read the
           variable
         - a white dot at the variable end means that the action sets the
           variable to off
         - a black dot at the variable end means that the action sets the
           variable to on
         - no tip at the action end means that the action does not assign the
           variable

        # Arguments

         - `constraints (bool=True)`: whether to take the constraints into
           account
         - `key=val, ...`: any option suitable for `cygraphs.Graph`

        # Return

        A `cygraph.Graph` instance that can be directly displayed in Jupyter.
        """
        nodes = []
        edges = []
        for path, loc in self.spec:
            for var in loc.variables:
                # each declared variable is a node
                vname = ".".join(path + [var.name])
                vinit = sum(var.init) / len(var.init)
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
                            "color": ("#aaaaff" if kind == "rule" else "#ffaaff"),
                        }
                    )
                    left, right, assign, read = set(), set(), {}, {}
                    for con in act.left:
                        # read variables
                        for p, v in con.vars(path):
                            src = ".".join(p + [v.name])
                            left.add(src)
                            read[src] = int(str(con.right))
                    for ass in act.right:
                        for p, v in ass.target.vars(path):
                            # assigned variable (only one actually)
                            tgt = ".".join(p + [v.name])
                            right.add(tgt)
                            assign[tgt] = int(str(ass.value))
                        for p, v in getattr(ass.value, "vars", lambda _: [])(path):
                            # also read variables, skip if has no method vars
                            left.add(".".join(p + [v.name]))
                    for v in left - right:
                        # readonly => empty arrowless arc
                        edges.append(
                            {
                                "src": v,
                                "dst": act.name,
                                "get": "*" if read[v] else "o",
                                "set": "-",
                            }
                        )
                    for v in right:
                        # written => arrowed arc labelled with assigned value
                        if v not in read:
                            get = "-"
                        elif read[v]:
                            get = "*"
                        else:
                            get = "o"
                        edges.append(
                            {
                                "src": v,
                                "dst": act.name,
                                "get": get,
                                "set": "*" if assign[v] else "o",
                            }
                        )
        opts = dict(
            nodes_shape="shape",
            nodes_fill_color="color",
            edges_label="",
            edges_source_tip="set",
            edges_target_tip="get",
            inspect_nodes=["kind", "location", "info"],
            inspect_edges=[],
        )
        opts.update(opt)
        nodes = pd.DataFrame.from_records(nodes, index=["node"])
        edges = pd.DataFrame.from_records(edges, index=["src", "dst"])
        return Graph(nodes, edges, **opts)


__extra__ = __xtra
