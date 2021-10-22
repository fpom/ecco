import itertools, re, tempfile, operator, functools, subprocess, warnings, pathlib
import ptnet, prince, its, ddd
import pandas as pd
import numpy as np
import bqplot as bq
import ipywidgets as ipw
import igraph as ig

from collections import defaultdict
from IPython.display import display

from .. import BaseModel, cached_property, CompileError, load as load_model
from ..graphs import Graph, Palette
from ..ui import log, getopt, HTML
from .lts import LTS, Component, setrel
from . import ltsprop
from .st import sign2char as s2c, Parser, FailedParse, State

class hset (object) :
    "similar to frozenset with sorted values"
    # work around pandas that does not display frozensets in order
    def __init__ (self, iterable=(), key=None, reverse=False, sep=",") :
        self._values = tuple(sorted(iterable, key=key, reverse=reverse))
        self._sep = sep
    def __iter__ (self) :
        yield from iter(self._values)
    def __bool__ (self) :
        return bool(self._values)
    def isdisjoint (self, other) :
        return set(self).isdisjoint(set(other))
    def issubset (self, other) :
        return set(self).issubset(set(other))
    def __le__ (self, other) :
        return self.issubset(other)
    def __lt__ (self, other) :
        return set(self) < set(other)
    def issuperset (self, other) :
        return set(self).issuperset(set(other))
    def __ge__ (self, other) :
        return self.issuperset(other)
    def __gt__ (self, other) :
        return set(self) > set(other)
    def union (self, *others) :
        return self.__class__(set(self).union(*others))
    def __or__ (self, other) :
        return self.union(other)
    def intersection (self, *others) :
        return self.__class__(set(self).intersection(*others))
    def __and__ (self, other) :
        return self.intersection(other)
    def difference (self, *others) :
        return self.__class__(set(self).difference(*others))
    def __sub__ (self, other) :
        return self.difference(other)
    def symmetric_difference (self, other) :
        return self.__class__(set(self).symmetric_difference(other))
    def __xor__ (self, other) :
        return self.symmetric_difference(other)
    def copy (self) :
        return self
    def __repr__ (self) :
        return f"{self._sep} ".join(str(v) for v in self)
    def _repr_pretty_ (self, pp, cycle) :
        pp.text(f"{self._sep} ".join(str(v) for v in self))
    def _ipython_display_ (self) :
        display(f"{self._sep} ".join(str(v) for v in self))

def parse (path, src=None) :
    try :
        return Parser(path, src).parse()
    except FailedParse as err :
        raise CompileError("\n".join(str(err).splitlines()[:3]))

class TableProxy (object) :
    _name = re.compile("^[a-z][a-z0-9_]*$", re.I)
    def __init__ (self, table) :
        self.table = table
        self.columns = list(table.columns)
    def __getitem__ (self, key) :
        if isinstance(key, str) :
            key = [key]
        elif not isinstance(key, slice) :
            key = list(key)
        stats = self.table[key].describe(include="all").fillna("")
        stats[" "] = np.full(len(stats), "")
        for col, val in {"count" : "total number of values",
                         "unique" : "number of unique values",
                         "top" : "most common value",
                         "freq" : "top's frequency",
                         "mean" : "values average",
                         "std" : "standard deviation",
                         "min" : "minimal value",
                         "max" : "maximal value",
                         "25%" : "25th percentile",
                         "50%" : "50th percentile",
                         "75%" : "75th percentile"}.items() :
            if col in stats.columns :
                stats.loc[col," "] = val
        return stats
    def __delitem__ (self, col) :
        if col in self.columns :
            display(HTML('<span style="color:#800; font-weight:bold;">warning:</span>'
                         ' you should not delete column %r,'
                         ' but if you insist I\'ll do it next time.' % col))
            self.columns.remove(col)
        elif isinstance(col, str) and col in self.table.columns :
            self.table.drop(columns=[col], inplace=True)
        elif isinstance(col, tuple) :
            for c in col :
                del self[c]
        else :
            raise KeyError("invalid column: %r" % (col,))
    def __setitem__ (self, key, val) :
        if key in self.table.columns :
            raise KeyError("columns %r exists already" % key)
        if callable(val) :
            data = self.table.apply(val, axis=1)
        elif isinstance(val, tuple) and len(val) > 1 and callable(val[0]) :
            fun, *args = val
            data = self.table.apply(fun, axis=1, args=args)
        else :
            data = pd.DataFrame({"col" : val})
        if isinstance(key, (tuple, list)) :
            for k in key :
                if not self._name.match(k) :
                    raise ValueError("invalid column name %r" % k)
            data = pd.DataFrame.from_records(data)
            for k, c in zip(key, data.columns) :
                self.table[k] = data[c]
        else :
            if not self._name.match(key) :
                raise ValueError("invalid column name %r" % key)
            self.table[key] = data
    def _ipython_display_ (self) :
        display(self[:])
    def _repr_pretty_ (self, pp, cycle) :
        return pp.text(repr(self[:]))

class Model (BaseModel) :
    ##
    ## RR
    ##
    @cached_property
    def rr (self) :
        """show RR source code
        """
        h = HTML()
        with h("pre", style="line-height:140%") :
            sections = {}
            for d in self.spec.meta :
                sections.setdefault(d.kind, []).append(d)
            for sect, decl in sections.items() :
                with h("span", style="color:#008; font-weight:bold;", BREAK=True) :
                    h.write("%s:" % sect)
                for d in decl :
                    h.write("    ")
                    if d.state.sign is None :
                        color = "#008"
                    elif d.state.sign :
                        color = "#080"
                    else :
                        color = "#800"
                    with h("span", style="color:%s;" % color) :
                        h.write("%s" % d.state)
                    h.write(": %s\n" % d.description)
            for sect in ("constraints", "rules") :
                rules = getattr(self.spec, sect)
                if rules:
                    with h("span", style="color:#008; font-weight:bold;", BREAK=True) :
                        h.write("%s:" % sect)
                    for rule in rules :
                        h.write("    ")
                        if (rule.label or "").strip() :
                            with h("span", style="color:#888; font-weight:bold;") :
                                h.write("[%s] " % rule.label.strip())
                        for i, s in enumerate(rule.left) :
                            if i :
                                h.write(", ")
                            color = "#080" if s.sign else "#800"
                            with h("span", style="color:%s;" % color) :
                                h.write(str(s))
                        with h("span", style="color:#008; font-weight:bold;") :
                            h.write(" &gt;&gt; ")
                        for i, s in enumerate(rule.right) :
                            if i :
                                h.write(", ")
                            color = "#080" if s.sign else "#800"
                            with h("span", style="color:%s;" % color) :
                                h.write(str(s))
                        with h("span", style="color:#888;") :
                            h.write("   # %s" % rule.name())
                        h.write("\n")
        return h
    def charact (self, constraints: bool=True, variables=None) -> pd.DataFrame :
        """compute variables static characterisation
        Options:
         - constraint (True): take constraints into account
         - variables (None): the list of variables to be displayed, or all if None
        Return: a DataFrame which the counts characterising each variable
        """
        if variables is None :
            index = list(sorted(s.state.name for s in self.spec.meta))
        else :
            index = list(variables)
        counts = pd.DataFrame(index=index,
                              data={"init" : np.full(len(index), 0),
                                    "left_0" : np.full(len(index), 0),
                                    "left_1" : np.full(len(index), 0),
                                    "right_0" : np.full(len(index), 0),
                                    "right_1" : np.full(len(index), 0)})
        for s in self.spec.meta :
            if s.state.name in index and s.state.sign :
                counts.loc[s.state.name,"init"] = 1
        if constraints :
            todo = [self.spec.constraints, self.spec.rules]
        else :
            todo = [self.spec.rules]
        for rule in itertools.chain(*todo) :
            for side in ["left", "right"] :
                for state in getattr(rule, side) :
                    if state.name in index :
                        sign = "_1" if state.sign else "_0"
                        counts.loc[state.name,side+sign] += 1
        return counts
    def draw_charact (self, constraints: bool=True, variables=None, **options) :
        """draw a bar chart of nodes static characterization
        Options:
         - constraints (True): take constraints into account
         - variables (None): the list of variables to be displayed, or all if None
        Figure options:
         - fig_width (960): figure width (in pixels)
         - fig_height (300): figure height (in pixels)
         - fig_padding (0.01): internal figure margins
         - fig_title (None): figure title
        Bars options:
         - bar_spacing (1): additional space at the left of bars to fit variable names
         - bar_palette ("RGW"): color palette for bars
        """
        self.opt = opt = getopt(options,
                                fig_width=960,
                                fig_height=300,
                                fig_padding=0.01,
                                fig_title=None,
                                bar_spacing=1,
                                bar_palette="RGW")
        counts = self.charact(constraints, variables)
        shift = (counts["left_0"] + counts["left_1"]).max() + opt.bar.spacing
        last = shift + (counts["right_0"] + counts["right_1"]).max()
        xs = bq.OrdinalScale(reverse=True)
        ys = bq.LinearScale(min=-opt.bar.spacing, max=float(last+1))
        bar_left = bq.Bars(x=counts.index,
                           y=[counts["left_0"], counts["left_1"]],
                           scales={"x": xs, "y": ys},
                           padding=0.1,
                           colors=Palette.mkpal(opt.bar.palette, 2),
                           orientation="horizontal")
        bar_right = bq.Bars(x=counts.index,
                            y=[counts["right_0"] + shift,
                               counts["right_1"] + shift],
                            scales={"x": xs, "y": ys},
                            padding=0.1,
                            colors=Palette.mkpal(opt.bar.palette, 2),
                            orientation="horizontal",
                            base=float(shift))
        ax_left = bq.Axis(scale=xs, orientation="vertical", grid_lines="none",
                          side="left", offset={"scale": ys, "value": 0})
        ax_right = bq.Axis(scale=xs, orientation="vertical", grid_lines="none",
                           side="left", offset={"scale": ys, "value": shift})
        ay = bq.Axis(scale=ys, orientation="horizontal",
                     tick_values=(list(range(shift + 1 - opt.bar.spacing))
                                  + list(range(shift, last+1))),
                     tick_style={"display": "none"})
        if opt.fig.title :
            fig = bq.Figure(marks=[bar_left, bar_right],
                            axes=[ax_left, ax_right, ay],
                            padding_x=opt.fig.padding, padding_y=opt.fig.padding,
                            layout=ipw.Layout(width="%spx" % opt.fig.width,
                                              height="%spx" % opt.fig.height),
                            fig_margin={"top" : 60, "bottom" : 0,
                                        "left" : 0, "right" : 0},
                            title=opt.fig.title)
        else :
            fig = bq.Figure(marks=[bar_left, bar_right],
                            axes=[ax_left, ax_right, ay],
                            padding_x=opt.fig.padding, padding_y=opt.fig.padding,
                            layout=ipw.Layout(width="%spx" % opt.fig.width,
                                              height="%spx" % opt.fig.height),
                            fig_margin={"top" : 0, "bottom" : 0,
                                        "left" : 0, "right" : 0})
        display(ipw.VBox([fig, bq.Toolbar(figure=fig)]))
    def save (self, target, overwrite: bool=False) :
        """save rr to file
        Arguments:
         - target: target file (either its path or a writable file-like object)
        Options:
         - overwrite (True): whether path should be overwritten if it exists
        """
        self.spec.save(target, overwrite)
    def inline (self, path: str, *constraints, overwrite: bool=False) :
        """inline constraints into the rules
        Arguments:
         - path: where to save the generated RR file
         - constraints, ...: a series of constraints to be inlines
           (given by names of numbers)
        Options:
         - overwrite (False): whether path can be overwritten or not if it exists
        Return: the model obtained by inlining the constraints as requested
        """
        model = self.__class__(path, self.spec.inline(constraints))
        model.save(path, overwrite=overwrite)
        return model
    ##
    ## state space methods
    ##
    def __call__ (self, *l, **k) :
        """build a `ComponentGraph` from the model

        Arguments:
         - `compact` (`True`): whether transient states should be removed and
           transitions rules adjusted accordingly
         - `init` (`""`): initial states of the LTS. If empty: take them from
           the model. Otherwise, it must be a comma-separated sequence of states
           assignements, that is interpreted as successive assignements starting
           from the initial states specified in the model. Each assignement be either:
           - `"*"`: take all the potential states (so called universe)
           - `"+"`: set all variables to "+"
           - `"-"`: set all variables to "-"
           - `VARx`, where `VAR` is a variable name and `x` is a sign in `+`, `-`,
             or `*`: set variable as specified by the sign
           For instance, sequence `"*,V+,W-` will consider all the pontential states
           restricted to those where `V` is `+` and `W` is `-`.
           A list of such strings may be used, in which case the union of the
           corresponding sets of states is considered.
         - `split` (`True`): should the graph be initially split into its initial
           states, SCC hull, and deadlocks+basins
        Returns: newly created `ComponentGraph` instance
        """
        return ComponentGraph.from_model(self, *l, **k)
    def gal_path (self, compact=False, permissive=False) :
        return str(self[("c" if compact else "")
                        + ("p" if permissive else "")
                        + "gal"])
    def gal (self, compact=False, permissive=False, showlog=True) :
        path = self.gal_path(compact, permissive)
        with log(head="<b>saving</b>",
                 tail=path,
                 done_head="<b>saved:</b>",
                 done_tail=path,
                 verbose=showlog), open(path, "w") as out :
            name = re.sub("[^a-z0-9]+", "", self.base.name, flags=re.I)
            out.write(f"gal {name} {{\n    //*** variables ***//\n")
            # variables
            for sort in self.spec.meta :
                out.write(f"    // {sort.state}: {sort.description} ({sort.kind})\n"
                          f"    int {sort.state.name} = {bool(sort.state.sign):d};\n")
                if permissive :
                    out.write(f"    int {sort.state.name}_flip = 0;\n")
                    if self.spec.constraints :
                        out.write(f"    int {sort.state.name}_FLIP = 0;\n")
            # constraints
            ctrans, ptrans = [], []
            if self.spec.constraints :
                out.write("    //*** constraints ***//\n")
            if compact :
                label = ' label "const" '
            else :
                label = " "
            for const in self.spec.constraints :
                if permissive :
                    guard = " && ".join(f"({s.name} == {s.sign:d}"
                                        f" || {s.name}_FLIP == 1"
                                        f" || {s.name}_flip == 1)"
                                        for s in const.left)
                else :
                    guard = " && ".join(f"({s.name} == {s.sign:d})"
                                        for s in const.left)
                loop = " && ".join(f"({s.name} == {s.sign:d})"
                                   for s in const.right)
                out.write(f"    // {const}\n"
                          f"    transition C{const.num} "
                          f"[{guard} && (!({loop}))]"
                          f"{label}{{\n")
                for s in const.right :
                    out.write(f"        {s.name} = {s.sign:d};\n")
                    if permissive :
                        out.write(f"        {s.name}_FLIP = 1;\n")
                        out.write(f"        {s.name}_flip = 0;\n")
                out.write("    }\n")
                ctrans.append(f"{guard} && (!({loop}))")
            if permissive and self.spec.constraints :
                out.write("    //*** end FLIPS ***//\n")
                for sort in self.spec.meta :
                    out.write(f"    transition FLIP_{sort.state.name} "
                              f"[{sort.state.name}_FLIP == 1]"
                              f"{label}{{\n"
                              f"        {sort.state.name}_FLIP = 0;\n"
                              f"    }}\n")
                    ptrans.append(f"{sort.state.name}_FLIP == 1")
            if compact :
                out.write(f"    // constraints + identity for fixpoints in rules\n"
                          f'    transition Cx [true] label "rconst" {{\n'
                          f'        self."const";'
                          f'    }}\n'
                          f'    transition ID [true] label "rconst" {{ }}\n')
            # rules
            if ctrans or ptrans :
                prio = " && (!(%s))" % " || ".join(f"({g})"
                                                   for g in ctrans + ptrans)
            else :
                prio = ""
            out.write("    //*** rules ***//\n")
            for rule in self.spec.rules :
                if permissive :
                    guard = " && ".join(f"({s.name} == {s.sign:d}"
                                        f" || {s.name}_flip == 1)"
                                        for s in rule.left)
                else :
                    guard = " && ".join(f"({s.name} == {s.sign:d})"
                                        for s in rule.left)
                loop = " && ".join(f"({s.name} == {s.sign:d})"
                                   for s in rule.right)
                out.write(f"    // {rule}\n"
                          f"    transition R{rule.num}"
                          f" [{guard} && (!({loop})){prio}] {{\n")
                for s in rule.right :
                    out.write(f"        {s.name} = {s.sign:d};\n")
                    if permissive :
                        out.write(f"        {s.name}_flip = 1;\n")
                if compact :
                    out.write("        fixpoint {\n"
                              '            self."rconst";\n'
                              "        }\n")
                out.write("    }\n")
            if permissive :
                out.write("    //*** end flips ***//\n")
                for sort in self.spec.meta :
                    out.write(f"    transition flip_{sort.state.name} "
                              f"[{sort.state.name}_flip == 1] {{\n"
                              f"        {sort.state.name}_flip = 0;\n"
                              f"    }}\n")
                    ptrans.append(f"{sort.state.name}_flip == 1")
            transient = []
            if compact :
                out.write("    //*** skip transient initial states ***//\n"
                          "    transition __INIT__ [true] {\n"
                          '        self."const";\n'
                          "    }\n")
                transient.extend(ctrans)
            if permissive :
                transient.extend(ptrans)
            if transient :
                out.write("    TRANSIENT = (%s);\n"
                          % " || ".join(f"({g})" for g in transient))
            out.write("}\n")
            return path
    def zeno (self, witness: int=10) -> list :
        """diagnosis Zeno models

        A model is called Zeno if it has cyclic executions involving
        only constraints. This is not a faulty model in itself but in
        the compact variant, it means that compact transitions may be
        the abstraction of infinite executions.

        If the model is Zeno, a witness cycle is printed (transient
        states and constraints). There may be many others, caused by
        the same or by other constraints.

        Options:
         - witness (default=10): maximum number of witness states
           to be printed to exhibit a Zeno cycle. If witness=0, only a
           yes/no diagnosis is printed.
        Return: return the list of constraints that create the witness
           cycle. If witness=0, this list consist of all the
           constraints that belong to the SCC hull of the Zeno cycles.
           If the model is not Zeno, None is returned.
        """
        cons = {c.name() : c for c in self.spec.constraints}
        gal = its.model(self.gal(showlog=False), fmt="GAL")
        with log(head="<b>checking for Zeno executions</b>",
                 tail="{step}",
                 done_head="<b>Zeno:</b>",
                 done_tail='<span style="color:#080; font-weight:bold;">no</span>',
                 steps=["computing reachable states",
                        "extracting transient states",
                        "extracting transient SCC hull",
                        "extracting witness state",
                        "extracting looping constraints"],
                 keep=True,
                 verbose=True) :
            if not cons :
                log.finish()
                return
            init = gal.reachable()
            log.update()
            chom = {n : h for n, h in gal.transitions().items() if n in cons}
            succ = ddd.shom.union(*chom.values())
            pred = ddd.shom.union(*chom.values()).invert(init)
            transient = pred(init)
            if transient & gal.initial() :
                log.warn("initial state is transient")
            else :
                log.info("initial state is not transient")
            log.update()
            succ_h = succ.gfp()
            pred_h = pred.gfp()
            hull = succ_h(transient) & pred_h(transient)
            if hull:
                log.done_tail = '<span style="color:#800; font-weight:bold;">YES</span>'
            else :
                log.finish()
                return
            log.update()
            if witness :
                succ_s = (succ & transient).lfp()
                pred_s = (pred & transient).lfp()
                while hull :
                    st = hull.pick()
                    scc = succ_s(st) & pred_s(st)
                    if len(scc) > 1 :
                        hull = scc
                        states = list(hull)[0][0]
                        selection = []
                        for i, s in zip(range(min(len(hull), witness)),
                                        states.items()) :
                            selection.append(", ".join(f"{n}+" for n, v in s.items()
                                                       if v))
                        if len(hull) > witness :
                            selection.append(f"... {len(hull) - witness} more")
                        log.print(f"<b>Zeno witness states:</b>"
                                  + "\n".join(f"<br/><code>{s}</code>"
                                              for s in selection))
                        break
                    hull = hull - scc
                    hull = succ_h(hull) & pred_h(hull)
            log.update()
            loop = []
            for n, h in chom.items() :
                if h(hull) & hull :
                    loop.append(n)
            log.print("<b>Zeno witness constraints:</b>"
                      + "\n".join(f"<br/><code>{cons[n]}</code>" for n in loop))
            log.finish()
        return [cons[n] for n in loop]
    def CTL (self, formula, compact=True, options=[], debug=False) :
        """Model-check a CTL formula
        Arguments:
         - formula: a CTL formula fiven as a string
        Options:
         - compact (True): check the compact model instead of the full
         - options ([]): command line options to be passed to "its-ctl"
         - debug (False): show "its-ctl" output
        """
        with tempfile.NamedTemporaryFile(mode="w+", suffix=f".ctl") as tmp :
            tmp.write(formula + ";")
            tmp.flush()
            argv = ["its-ctl",
                    "-i", self.gal(compact=compact),
                    "-t", "GAL",
                    "-ctl", tmp.name,
                    "--quiet"] + options
            try :
                out = subprocess.check_output(argv, encoding="utf-8")
            except subprocess.CalledProcessError as err :
                log.err(f"<tt>its-ctl</tt> exited with status code {err.returncode}")
                out = err.output
                debug = True
        if debug :
            print(out)
        return "Formula is TRUE" in out
    def LTL (self, formula, compact=True, options=[], algo="SSLAP-FST", debug=False) :
        """Model-check a LTL formula
        Arguments:
         - formula: a LTL formula fiven as a string
        Options:
         - compact (True): check the compact model instead of the full
         - options ([]): command line options to be passed to "its-ltl"
         - algo ("SSLAP-FST"): algorithm to be used
         - debug (False): show "its-ltl" output
        """
        argv = ["its-ltl",
                "-i", self.gal(compact=compact),
                "-t", "GAL",
                "-c", "-e",
                "-ltl", formula,
                f"-{algo}",
                "--place-syntax"] + options
        try :
            out = subprocess.check_output(argv, encoding="utf-8")
        except subprocess.CalledProcessError as err :
            log.err(f"<tt>its-ltl</tt> exited with status code {err.returncode}")
            out = err.output
            debug = True
        if debug :
            print(out)
        return "Formula 0 is FALSE" in out
    ##
    ## ecosystemic (hyper)graph
    ##
    def ecograph (self, constraints: bool=True, **opt) :
        """draw the ecosystemic graph
        Options:
         - constraint (True): take constraints into account
        Figure options:
         - fig_width (960): figure width (in pixels)
         - fig_height (600): figure height (in pixels)
         - fig_padding (0.1): internal figure margins
         - fig_title (None): figure title
        Graph options:
         - graph_layout ("circo"): layout engine to compute nodes positions
        Nodes options:
         - nodes_label ("node"): column that defines nodes labels
         - nodes_shape (auto): column that defines nodes shape
         - nodes_size (35): nodes width
         - nodes_color ("init"): column that defines nodes colors
         - nodes_selected ("#EE0000"): color of the nodes selected for inspection
         - nodes_palette ("RGW"): palette for the nodes colors
         - nodes_ratio (1.2): height/width ratio of non-symmetrical nodes
        """
        nodes = []
        for sort in sorted(self.spec.meta, key=lambda s : s.state.name) :
            nodes.append({"node" : sort.state.name,
                          "init" : sort.state.sign,
                          "category" : sort.kind,
                          "description" : sort.description})
        edges = []
        if constraints :
            todo = [self.spec.constraints, self.spec.rules]
        else :
            todo = [self.spec.rules]
        for rule in itertools.chain(*todo) :
            for src in rule.left :
                for dst in rule.right :
                    edges.append({"src" : src.name, "dst" : dst.name,
                                  "rule" : rule.name(), "rr" : rule.text()})
        return Graph(pd.DataFrame.from_records(nodes, index="node"),
                     pd.DataFrame.from_records(edges, index=["src", "dst"]),
                     defaults={
                         "graph_layout" : "circo",
                         "nodes_color" : "init",
                         "nodes_colorscale" : "discrete",
                         "nodes_palette" : "RGW",
                         "nodes_shape" : "circ",
                         "marks_shape" : None
                     }, **opt)
    def ecohyper (self, constraints: bool=True, **opt) :
        """draw the ecosystemic hypergraph
        Options:
         - constraint (True): take constraints into account
        Figure options:
         - fig_width (960): figure width (in pixels)
         - fig_height (600): figure height (in pixels)
         - fig_padding (0.1): internal figure margins
         - fig_title (None): figure title
        Graph options:
         - graph_layout ("fdp"): layout engine to compute nodes positions
        Nodes options:
         - nodes_size (35): nodes width
         - nodes_selected ("#EE0000"): color of the nodes selected for inspection
        """
        nodes = []
        edges = []
        if constraints :
            rules = list(itertools.chain(self.spec.constraints, self.spec.rules))
        else :
            rules = list(self.spec.rules)
        for state, desc in sorted((s.state, s.description) for s in self.spec.meta) :
            nodes.append({"node" : state.name,
                          "info" : desc,
                          "shape" : "circ",
                          "color" : 0 if state.sign else 1})
        for rule in rules :
            name = rule.name()
            nodes.append({"node" : name,
                          "info" : rule.text(),
                          "shape" : "sbox",
                          "color" : 2 if name[0] == "C" else 3})
            for var in set(s.name for s in itertools.chain(rule.left, rule.right)) :
                v0 = State(var, False)
                v1 = State(var, True)
                if v1 in rule.left :
                    if v1 in rule.right :
                        start, end = "**"
                    elif v0 in rule.right :
                        start, end = "o*"
                    else :
                        start, end = "-*"
                elif v0 in rule.left :
                    if v1 in rule.right :
                        start, end = "*o"
                    elif v0 in rule.right :
                        start, end = "oo"
                    else :
                        start, end = "-o"
                elif v1 in rule.right :
                    start, end = "*-"
                elif v0 in rule.right :
                    start, end = "o-"
                edges.append({"src" : var, "dst" : name, "get" : start, "set" : end})
        return Graph(pd.DataFrame.from_records(nodes, index="node"),
                     pd.DataFrame.from_records(edges, index=["src", "dst"]),
                     defaults={
                         "graph_layout" : "fdp",
                         "graph_directed" : False,
                         "nodes_shape" : "shape",
                         "nodes_color" : "color",
                         "nodes_colorscale" : "discrete",
                         "nodes_palette" : ["#AAFFAA", "#FFAAAA", "#FFFF88", "#FFFFFF"],
                         "edges_tips" : ["get", "set"],
                         "gui_main" : [["layout", "size"],
                                       "figure",
                                       "toolbar",
                                       "inspect"],
                     }, **opt)
    ##
    ## Petri nets
    ##
    def petri (self) :
        n = ptnet.net.Net()
        places = {}
        for state in (s.state for s in self.spec.meta) :
            places[state] = n.place_add(str(state), 1)
            places[~state] = n.place_add(str(~state))
        for rule in self.spec.rules :
            for i, r in enumerate(rule.normalise()) :
                if set(r.left) == set(r.right) :
                    continue
                t = n.trans_add("%s.%s" % (rule.name(), i))
                for v in r.vars() :
                    on = State(v, True)
                    off = State(v, False)
                    if on in r.left and on in r.right :
                        t.cont_add(places[on])
                    elif on in r.left and off in r.right :
                        t.pre_add(places[on])
                        t.post_add(places[off])
                    elif off in r.left and off in r.right :
                        t.cont_add(places[off])
                    elif off in r.left and on in r.right :
                        t.pre_add(places[off])
                        t.post_add(places[on])
        return n
    def unfold (self) :
        n = self.petri()
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as pep,\
             tempfile.NamedTemporaryFile("rb") as cuf :
            n.write(pep)
            pep.flush()
            subprocess.run(["cunf", "-s", cuf.name, pep.name])
            u = ptnet.unfolding.Unfolding()
            u.read(cuf)
        return u

class _GCS (set) :
    "a subclass of set that is tied to a component graph so it has __invert__"
    components = set()
    def __invert__ (self) :
        return self.__class__(self.components - self)
    def __and__ (self, other) :
        return self.__class__(super().__and__(other))
    def __or__ (self, other) :
        return self.__class__(super().__or__(other))
    def __xor__ (self, other) :
        return self.__class__(super().__xor__(other))

def _rulekey (r) :
    return (r[0], int(r[1:]))

class NodesTableProxy (TableProxy) :
    def __init__ (self, compograph, table, name="nodes") :
        super().__init__(table)
        self._c = compograph
        self._n = name
    def __call__ (self, col, fun=None) :
        if fun is None :
            series = self.table[col].astype(bool)
        else :
            series = self.table[col].apply(fun).astype(bool)
        return self._c._GCS(self.table.index[series])
    def __getattr__ (self, name) :
        if name in self.table.columns :
            return functools.partial(self.__call__, name)
        else :
            raise AttributeError(f"table {self._n!r} has no column {name!r}")

class ComponentGraph (object) :
    def __init__ (self, model, compact=True, init="", lts=None, **k) :
        """create a new instance

        This method is not intended to be used directly, but it will be called by
        the other methods. Use `from_model` class method if you whish to create a
        component graph from scratch.
        """
        self.model = model
        if compact and not self.model.spec.constraints :
            log.warn("model has no constraints, setting <code>compact=False</code>")
            compact = False
        if lts is None :
            self.lts = LTS(self.model.gal(), init, compact)
        else :
            self.lts = lts
        self.components = ()
        self.compact = compact
        self._c = {} # Component.num => Component
        self._g = {} # Component.num => Vertex
    @classmethod
    def from_model (cls, model, compact=True, init="", split=True) :
        """create a `ComponentGraph` from a `Model` instance

        Arguments:
         - `model`: a `Model` instance
         - `compact` (`True`): whether transient states should be removed and
           transitions rules adjusted accordingly
         - `init` (`str:""`): initial states of the LTS. If empty: take them from
           the model. Otherwise, it must be a comma-separated sequence of states
           assignements, that is interpreted as successive assignements starting
           from the initial states specified in the model. Each assignement be either:
           - `"*"`: take all the potential states (so called universe)
           - `"+"`: set all variables to "+"
           - `"-"`: set all variables to "-"
           - `VARx`, where `VAR` is a variable name and `x` is a sign in `+`, `-`,
             or `*`: set variable as specified by the sign
           For instance, sequence `"*,V+,W-` will consider all the pontential states
           restricted to those where `V` is `+` and `W` is `-`.
           A list of such strings may be used, in which case the union of the
           corresponding sets of states is considered.
         - `split` (`True`): should the graph be initially split into its initial
           states, SCC hull, and deadlocks+basins
        """
        if isinstance(init, str) :
            init = [init]
        for i, s in enumerate(init) :
            if s not in ("*", "+", "-") :
                init[i] = ",".join(f"{s.state.name}{s2c[s.state.sign]}"
                                   for s in model.spec.meta) + "," + s
        cg = cls(compact=compact, init=init, model=model)
        c_all = Component(cg.lts, cg.lts.states,
                          gp=cg.lts.graph_props(cg.lts.states))
        if split :
            cg.components = tuple(c for c in c_all.topo_split(split_entries=False,
                                                              split_exits=False)
                                  if c is not None)
        else :
            cg.components = (c_all,)
        cg._c.update((c.num, c) for c in cg.components)
        return cg
    def __getitem__ (self, num) :
        """return a component or an edge from the `ComponentGraph`

        Arguments:
         - when `num:int`: return a `Component` instance whose number is `num`
         - when `src,dst:num,num`: return a `dict` with the attributes of an edge
           between components `src` and `dst`
        Return: `Component` or `dict`
        """
        if isinstance(num, int) :
            return self._c[num]
        else :
            try :
                src, dst = num
                eid = self.g.get_eid(self._g[src], self._g[dst])
            except :
                raise KeyError(num)
            return self.g.es[eid].attributes()
    def __iter__ (self) :
        yield from self.components
    def __eq__ (self, other) :
        """equality test

        Two component graphs are equal whenever they both contain
        exactly the same components.
        """
        return set(self.components) == set(other.components)
    def _patch (self, rem, add) :
        # return a copy of the ComponentGraph with `rem` nodes removed
        # and `add` nodes added
        lts = self.lts.copy()
        cg = self.__class__(compact=self.compact, lts=lts, model=self.model)
        rnum = set(c.num for c in rem)
        keep = tuple(c.copy(lts) for c in self.components if c.num not in rnum)
        add = tuple(c.copy(lts) for c in add)
        cg.components = keep + add
        cg._c.update((c.num, c) for c in cg.components)
        return cg
    def _add_vertex (self, g, c) :
        # add `c:Component` as a vertex in `g:igraph.Graph`
        self._c[c.num] = c
        self._g[c.num] = g.add_vertex(node=c.num).index
    def _collect_edges (self, src, targets, edges) :
        # collect the edges to be added from `src` to the nodes in `targets`
        for trans, succ in src.succ(*targets) :
            edges[src.num,succ.num].add(trans)
    def _add_edges (self, g, edges) :
        # add the collected `edges` to `g:igraph.Graph`
        _g = self._g
        for (src, dst), rules in edges.items() :
            g.add_edge(_g[src], _g[dst], src=src, dst=dst,
                       rules=hset(rules, key=_rulekey))
    @cached_property
    def g (self) :
        """the actual component graph stored as an `igraph.Graph` instance
        """
        g = ig.Graph(directed=True)
        edges = defaultdict(set)
        for c in self.components :
            self._add_vertex(g, c)
            self._collect_edges(c, self.components, edges)
        self._add_edges(g, edges)
        return g
    @cached_property
    def n (self) :
        """a `Tableproxy` wrapping the `nodes` dataframe
        """
        return NodesTableProxy(self, self.nodes)
    @cached_property
    def e (self) :
        """a `Tableproxy` wrapping the `edges` dataframe
        """
        return TableProxy(self.edges)
    @cached_property
    def nodes (self) :
        """a `pandas.DataFrame` holding the information about the nodes
        """
        nodes = self.g.get_vertex_dataframe()
        def n_col (row) :
            c = self._c[row.node]
            return [len(c),
                    hset(c.on),
                    hset(c.off),
                    hset(str(p) for p, v in c.graph_props.items() if v)]
        nodes[["size", "on", "off", "topo"]] = nodes.apply(n_col, axis=1,
                                                           result_type='expand')
        nodes.set_index("node", inplace=True)
        self._update(nodes)
        return nodes
    @cached_property
    def edges (self) :
        """a `pandas.DataFrame` holding the information about the edges
        """
        if len(self.g.es) == 0 :
            # if there is no edge, add a dummy one to build a correct dataframe
            num = self.components[0].num
            self.g.add_edge(self._g[num], self._g[num],
                            src=num, dst=num, rules=hset())
        else :
            num = None
        edges = self.g.get_edge_dataframe().set_index(["src", "dst"])
        del edges["source"], edges["target"]
        if self.model.spec.labels :
            spec_labels = self.model.spec.labels
            def lbl (rules) :
                labels = set()
                for r in rules :
                    labels.update(l.strip() for l in spec_labels.get(r).split(",")
                                  if l.strip())
                return hset(labels)
            edges["labels"] = edges["rules"].apply(lbl)
        if num is not None :
            # remove the dummy edge
            return edges.drop((num, num), axis="index")
        else :
            return edges
    def __len__ (self) :
        """number of components in the `ComponentGraph`
        """
        return len(self.components)
    def _update (self, nodes=None) :
        # updates the components' split_graph properties in a nodes dataframe
        if nodes is None :
            nodes = getattr(self, "_cached_nodes", None)
        if nodes is None :
            return
        cols = [r.name for r in setrel]
        data = pd.DataFrame.from_records([[hset(c, sep=";")
                                           for c in self._c[n].props_row()]
                                          for n in nodes.index],
                                         columns=cols, index=nodes.index)
        nodes[cols] = data
    def update (self) :
        """updates table `nodes` wrt components' properties

        This method may be needed when components shared with other component
        graphs are checked or split against new formulas.
        """
        self._update()
    def _get_args (self, args, aliased={},
                   min_props=0, max_props=-1, min_compo=0, max_compo=-1) :
        # parse args as a sequence of properties followed by a sequence of components
        properties = {v : k for k, v in aliased.items()}
        components = []
        for i, p in enumerate(args) :
            if isinstance(p, str) :
                properties[p] = ""
            elif isinstance (p, int) :
                components = args[i:]
                for c in components :
                    if not c in self._c :
                        raise TypeError(f"unexpected argument {c!r}")
                break
            else :
                raise TypeError(f"unexpected argument {p!r}")
        if max_props < 0 :
            _max_props = len(args) + len(aliased)
        else :
            _max_props = max_props
        if not min_props <= len(properties) <= _max_props :
            if min_props == max_props == 0 :
                raise TypeError(f"no properties expected,"
                                f" got {len(properties)}")
            elif max_props < 0 :
                raise TypeError(f"expected at least {min_props} properties,"
                                f" got {len(properties)}")
            else :
                raise TypeError(f"expected {min_props} to {_max_props} properties,"
                                f" got {len(properties)}")
        if max_compo < 0 :
            _max_compo = len(args)
        else :
            _max_compo = max_compo
        if not min_compo <= len(components) <= _max_compo :
            if min_compo == max_compo == 0 :
                raise TypeError(f"no components expected,"
                                f" got {len(components)}")
            elif max_compo < 0 :
                raise TypeError(f"expected at least {min_compo} components,"
                                f" got {len(components)}")
            else :
                raise TypeError(f"expected {min_compo} to {_max_compo} components,"
                                f" got {len(components)}")
        if components :
            components = tuple(self._c[c] for c in components)
        else :
            components = self.components
        return properties, components
    def forget (self, *args) :
        """forget about properties

        Remove properties from the nodes table, from the components, and from the LTS.

        Arguments:
         - `prop, ...` (`str`): at least one property to be removed
        """
        props, _ = self._get_args(args, min_props=1, max_compo=0)
        alias = {a : p for p, a in self.tls.alias.items()}
        for prop in props :
            prop = alias.get(prop, prop)
            for c in self.components :
                c.split_props.pop(prop, None)
            self.lts.alias.pop(prop, None)
            self.lts.props.pop(prop, None)
        self._update()
    def forget_all (self) :
        """forget about all properties

        Remove all properties from the nodes table, from the components,
        and from the LTS.
        """
        for c in self.components :
            c.split_props.clear()
        self.lts.props.clear()
        self.lts.alias.clear()
        self._update()
    def tag (self, *args) :
        """add a dummy properties to components

        Tag `name` is a dummy property whose states are those of the component
        to which it's added.

        Parameters:
         - `name, ...` (`str`): at least one tag name
         - `number, ...` (`int`): a series of components number, if empty, all the
           components in the graph are considered
        """
        props, compo = self._get_args(args, min_props=1)
        #TODO: log
        for p in props :
            for c in compo :
                c.tag(p)
        self._update()
    def check (self, *args, **aliased) :
        """check properties on components

        Arguments:
         - `prop, ...` (`str`): at least one property to be checked against
           the states of components
         - `number, ...` (`int`): a series of components number, if empty, all the
           components in the graph are considered

        Returns: a list of tuples of `setrel` corresponding for each property `prop`
        to the relations between each component and the states validation `prop`
        in the states of whole component graph
        """
        props, compo = self._get_args(args, aliased, min_props=1)
        ret = {c.num : dict() for c in compo}
        #TODO: log
        for prop, alias in props.items() :
            checker = ltsprop.AnyProp(self.model, self.lts, prop)
            for c in compo :
                ret[c.num][prop] = setrel(c.check(prop, checker(c.states), alias)).name
            if not self.lts.props[prop] :
                log.warn(f"property {prop!r} is empty")
        self._update()
        return ret
    def split (self, *args, **aliased) :
        """split components wrt properties

        Each considered component is split into the states that validate
        each the property and those that do not. If a property is globally
        (in)valid on one component, then no split occurs.

        Arguments:
         - `prop, ...` (`str`): at least one property to be checked against the
           states of components
         - `number, ...` (`int`): a series of components number, if empty, all the
           components in the graph are considered

        Returns: a new `ComponentGraph` instance
        """
        props, compo = self._get_args(args, aliased, min_props=1)
        rem, add, compo = set(), set(), set(compo)
        for prop, alias in props.items() :
            r, a = self._split(prop, compo, alias)
            if not self.lts.props[prop] :
                desc = self.lts.alias.get(prop, prop)
                log.warn(f"property {desc!r} is empty")
            rem.update(r)
            add.update(a)
            compo = (compo - set(r)) | set(a)
        self._update()
        add.difference_update(rem)
        rem.intersection_update(self.components)
        return self._patch(rem, add)
    def _split (self, prop, components, alias="") :
        rem, add = [], []
        checker = ltsprop.AnyProp(self.model, self.lts, prop)
        for c in components :
            intr, diff = c.split(prop, checker(c.states), alias)
            #TODO: log
            if intr is not None and diff is not None :
                rem.append(c)
                add.append(intr)
                add.append(diff)
        return rem, add
    def topo_split (self, *args,
                    init=True, entries=True, exits=True, hull=True, dead=True) :
        """split components into their topological parts

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
         - `number, ...` (`int`): a series of components number, if empty, all the
           components in the graph are considered
         - `init` (`bool=True`): whether to extract initial states
         - `entries` (`bool=True`): whether to extract entry states
         - `exits` (`bool=True`): whether to extract exit states
         - `hull` (`bool=True`): whether to extract SCC hull states
         - `dead` (`bool=True`): whether to extract deadlocks

        Returns: a new `ComponentGraph` instance
        """
        rem, add = [], []
        _, compos = self._get_args(args, max_props=0)
        # TODO: log
        for c in compos :
            parts = c.topo_split(split_init=init,
                                 split_entries=entries,
                                 split_exits=exits,
                                 split_hull=hull,
                                 split_dead=dead)
            keep = [p for p in parts if p is not None]
            if len(keep) > 1 :
                rem.append(c)
                add.extend(keep)
        return self._patch(rem, add)
    _relmatch = {(setrel.HASNO, True) : {setrel.HASNO},
                 (setrel.HASNO, False) : {setrel.HASNO},
                 (setrel.HAS, True) : {setrel.HAS},
                 (setrel.HAS, False) : {setrel.HAS, setrel.CONTAINS,
                                        setrel.ISIN, setrel.EQUALS},
                 (setrel.CONTAINS, True) : {setrel.CONTAINS},
                 (setrel.CONTAINS, False) : {setrel.CONTAINS, setrel.EQUALS},
                 (setrel.ISIN, True) : {setrel.ISIN},
                 (setrel.ISIN, False) : {setrel.ISIN, setrel.EQUALS},
                 (setrel.EQUALS, True) : {setrel.EQUALS},
                 (setrel.EQUALS, False) : {setrel.EQUALS}}
    def classify (self, col, *components, src_col=None, dst_col=None, **classes) :
        """split component graph and classify the resulting nodes wrt `classes`

        `classes` provides pairs named properties as pairs `name=prop`.
        The method first splits the component graph wrt these properties.
        Then, a column `col` is added to the `nodes` table of the resulting component
        graph such that it contains, for each resulting component, the set of properties
        that hold on the component, given by their names.

        Parameters:
         - `col` (`str`): name of the column where nodes classification is stored
         - `components, ...` (`int`): components to be classified (all if none is given)
         - `src_col` (`str=None`): name of a column where edges source nodes
           classification is stored
         - `dst_col` (`str=None`): name of a column where edges destination nodes
           classification is stored
         - `name=prop, ...` (`str`): named properties fot classification
        """
        new = self.split(*classes.values(), *components)
        c2n = {v : k for k, v in classes.items()}
        def cname (row) :
            props = set()
            for c in ("EQUALS", "ISIN") :
                if c in row.index :
                    props.update(c2n[prop] for prop in row[c] if prop in c2n)
            return hset(props)
        new.n[col] = cname
        if src_col :
            def sname (row) :
                return new.nodes[col][row.name[0]]
            new.e[src_col] = sname
        if dst_col :
            def dname (row) :
                return new.nodes[col][row.name[1]]
            new.e[dst_col] = dname
        return new
    @cached_property
    def _GCS (self) :
        class MyGCS (_GCS) :
            components = {c.num for c in self.components}
        return MyGCS
    def __call__ (self, prop, rel=setrel.HAS, strict=False) :
        """compute the set of components that match `prop`

        The returned set supports `~` to compute its complement wrt
        the set of all components in the graph.

        For each value of `setrel`, two shorthand  methods are available,
        for instance:
         - `g.has(p)` is like calling `g(p, rel=setrel.HAS, strict=False)`
         - `g.HAS(p)` is like calling `g(p, rel=setrel.HAS, strict=True)`

        Arguments:
         - `prop` (`str`): the property to check on components
         - `rel` (`setrel=HAS`): the relation the components must have with `prop`
         - `strict` (`bool=False`): if `True`, `rel` must be exactly matched,
           otherwise, `rel` is considered relaxed (eg, `HAS` can be realised by `ISIN`)
        Return: the set of component numbers that match the property
        """
        update = False
        for c in self.components :
            if prop not in c.split_props :
                update = True
                self.check(prop, c.num)
        if update :
            self._update()
        return self._GCS(c.num for c in self.components
                         if c.split_props[prop] in self._relmatch[rel,strict])
    def has (self, prop) :
        return self(prop, setrel.HAS, False)
    def HAS (self, prop) :
        return self(prop, setrel.HAS, True)
    def hasno (self, prop) :
        return self(prop, setrel.HASNO, False)
    def HASNO (self, prop) :
        return self(prop, setrel.HASNO, True)
    def contains (self, prop) :
        return self(prop, setrel.CONTAINS, False)
    def CONTAINS (self, prop) :
        return self(prop, setrel.CONTAINS, True)
    def isin (self, prop) :
        return self(prop, setrel.ISIN, False)
    def ISIN (self, prop) :
        return self(prop, setrel.ISIN, True)
    def equals (self, prop) :
        return self(prop, setrel.EQUALS, False)
    def EQUALS (self, prop) :
        return self(prop, setrel.EQUALS, True)
    def select (self, *args, all=True, rel=setrel.ISIN, **aliased) :
        """select components that validate properties

        Arguments:
         - `prop, ...` (`str`): at least one property to be checked against
           the states of components
         - `number, ...` (`int`): a series of components number, if empty, all the
           components in the graph are considered
         - `all` (`bool=True`): how the selected components from each property are
           accumulated. If `True`, selected components must validate all the properties,
           if `False`, they must validate at least one property
         - `rel` (`setrel=ISIN`): how the property should be validated by the
           components, a single relation may be provided or several in an iterable,
           in which case any of these relations is accepted

        Returns: a set of components numbers among `components` that validate
        the property
        """
        props, compos = self._get_args(args, aliased, min_props=1)
        if isinstance(rel, setrel) :
            rel = {rel}
        else :
            rel = set(rel)
        if all :
            found = set(c.num for c in compos)
        else :
            found = set()
        update = False
        for p, a in props.items() :
            for c in compos :
                if p not in c.split_props :
                    update = True
                    self.check(p, c.num, a)
            match = {c.num for c in compos if c.split_props[p] in rel}
            if all :
                found.intersection_update(match)
            else :
                found.update(match)
        if update :
            self._update()
        return found
    def explicit (self, *args, limit=256) :
        """splits components into its individual states

        Arguments:
         - `number, ...`: a series of components number, if empty, all the
           components in the graph are considered
         - `limit` (`int=256`): maximum number of components in the split graph,
           if it will have more that `limit` components, a confirmation is asked
           to the user (`limit` may be set <= 0 to avoid this user-interaction)
        Returns: a new `ComponentGraph` instance
        """
        _, compos = self._get_args(args, max_props=0)
        if limit > 0 :
            size = sum((len(c) for c in compos), 0) + len(self.components) - len(compos)
            if size  > limit :
                answer = input(f"This will create a graph with {size} nodes,"
                               f" are you sure? [N/y] ")
                if not answer.lower().startswith("y") :
                    return
        rem, add = [], []
        state2compo = {}
        if "component" in self.nodes.columns :
            state2compo.update((row.name, row.component)
                               for _, row in self.nodes.iterrows())
        for c in compos :
            new = tuple(c.explicit())
            state2compo.update((n.num, c.num) for n in new)
            if len(new) > 1 :
                rem.append(c)
                add.extend(new)
        ret = self._patch(rem, add)
        ret.n["component"] = lambda row: state2compo.get(row.name, row.name)
        return ret
    def merge (self, *args) :
        """merge two or more components

        Arguments:
         - `number, ...` (`int`): at least two components to be merged
        Returns: a new `ComponentGraph` instance
        """
        rem, add = [], []
        _, compos = self._get_args(args, max_props=0, min_compo=2)
        rem.extend(compos)
        add.append(rem[0].merge(*rem[1:]))
        return self._patch(rem, add)
    def drop (self, *args) :
        """drop one or more components from the component graph

        Arguments:
         - `number, ...` (`int`): at least one component to be removed
        Returns: a new `ComponentGraph` instance, or `None` if all the components
        are to be removed
        """
        _, compos = self._get_args(args, max_props=0, min_compo=1)
        if len(compos) == len(self) :
            log.warn("cannot drop all the components")
            return
        return self._patch(compos, [])
    def form (self, *args, variables=None, normalise=None) :
        """describe components by Boolean formulas

        Arguments:
         - `number, ...` (`int`): a series of components number, if empty, all the
           components in the graph are considered
         - `variables=...`: a collection of variables names to restrict to
         - `normalise=...`: how the formula should be normalised, which must be either:
           - `"cnf"`: conjunctive normal form
           - `"dnf"`: disjunctive normal form
           - `None`: chose the smallest form
        Return: a `dict` mapping component numbers to sympy Boolean formulas
        """
        _, compos = self._get_args(args, max_props=0)
        return {c.num : c.form(variables, normalise) for c in compos}
    def count (self, *args, transpose=False) :
        """count in how many states each variable is on in components

        Arguments:
         - `number, ...` (`int`): a series of components number, if empty, all the
           components in the graph are considered
         - `transpose` (`bool=False`): default is to have the components numbers
           as the rows, and the variables as the columns, if `transpose=True`,
           this is swapped
        Return: a `pandas.DataFrame` instance
        """
        _, compos = self._get_args(args, max_props=0)
        counts = [c.count() for c in compos]
        df = pd.DataFrame.from_records([[c.get(v, 0) for v in self.lts.vars]
                                        for c in counts],
                                       index=[c.num for c in compos],
                                       columns=list(sorted(self.lts.vars)))
        if transpose :
            return df.transpose()
        else :
            return df
    def pca (self, *args, transpose=False,
             n_components=2, n_iter=3, copy=True, check_input=True,
             engine="auto", random_state=42,
             rescale_with_mean=True, rescale_with_std=True) :
        """principal component analysis of the `count` matrix

        Arguments:
         - `number, ...` (`int`): a series of components number, if empty, all the
           components in the graph are considered
         - `transpose`: passed to method `count()`

        See https://github.com/MaxHalford/prince#principal-component-analysis-pca
        for documentation about the other arguments.

        Returns: a `pandas.DataFrame` as computed by Prince.
        """
        count = self.count(*args, transpose=transpose)
        count = count[count.sum(axis="columns") > 0]
        pca = prince.PCA(n_components=n_components,
                         n_iter=n_iter,
                         copy=copy,
                         check_input=check_input,
                         engine=engine,
                         random_state=random_state,
                         rescale_with_mean=rescale_with_mean,
                         rescale_with_std=rescale_with_std)
        pca.fit(count)
        trans = pca.transform(count)
        for idx in set(count.index) - set(trans.index) :
            trans.loc[idx] = [0, 0]
        return trans
    def draw (self, **opt) :
        """draw the component graph

        Arguments:
         - `opt`: various options to control the drawing

        Figure options:
         - fig_width (960): figure width (in pixels)
         - fig_height (600): figure height (in pixels)
         - fig_padding (0.1): internal figure margins
         - fig_title (None): figure title
         - fig_background ("#F7F7F7"): figure background color
        Graph options:
         - graph_layout ("fdp"): layout engine to compute nodes positions
        Nodes options:
         - nodes_label ("node"): column that defines nodes labels
         - nodes_shape (auto): column that defines nodes shape
         - nodes_size (35): nodes width
         - nodes_color ("size"): column that defines nodes colors
         - nodes_selected ("#EE0000"): color of the nodes selected for inspection
         - nodes_palette ("GRW"): palette for the nodes colors
         - nodes_ratio (1.2): height/width ratio of non-symmetrical nodes

        Returns: a `Graph` instance that can be directly displayed in a
        Jupyter Notebook
        """
        if len(self) < 2 :
            log.print("cannot draw a graph with only one node", "error")
            display(self.nodes)
            return
        try :
            # don't add PCA layout when PCA fails
            opt.setdefault("graph_engines", {})["PCA"] = self.pca()
        except :
            log.print("could not compute PCA, this layout is thus disabled", "warning")
        return Graph(self.nodes, self.edges,
                     defaults={
                         "nodes_color" : "size",
                         "nodes_colorscale" : "linear",
                         "nodes_palette" : "GRW",
                         "nodes_shape" : (["topo"], self._nodes_shape),
                         "marks_shape" : (["topo"], self._marks_shape),
                         "marks_palette" : ["#FFFFFF", "#000000"],
                         "marks_opacity" : .5,
                         "marks_stroke" : "#888888",
                         "fig_background" : "#F7F7F7",
                     }, **opt)
    def _nodes_shape (self, row) :
        # helper function for method draw: computes nodes shapes
        if "is_scc" in row.topo :
            return "circ"
        elif "has_dead" in row.topo :
            return "sbox"
        else :
            return "rbox"
    def _marks_shape (self, row) :
        # helper function for method draw: computes nodes decorations
        if "has_init" in row.topo :
            return "triangle-down"
        elif "is_hull" in row.topo :
            return "circle"
    def search (self, *args, prune=True, **aliased) :
        # split every component wrt every property
        props, compo = self._get_args(args, aliased, min_props=2)
        rem, add, compo = set(), set(), set(compo)
        for prop, alias in props.items() :
            r, a = self._split(prop, compo, alias)
            rem.update(r)
            add.update(a)
            compo = (compo - set(r)) | set(a)
        self._update()
        add.difference_update(rem)
        rem.intersection_update(self.components)
        # build new component graph
        new = self._patch(rem, add)
        new.g # ensure that new.g is built, so that new._g is usable
        steps = [[new._g[c.num] for c in new.components
                  if c.split_props[prop] in (setrel.EQUALS, setrel.ISIN)]
                 for prop in (one, two, *more)]
        # get shortest path between steps
        parts = new._search(steps)
        path = [one]
        for p, s in zip(parts, (two, *more)) :
            path.extend([p, s])
        # remove unwanted nodes
        if prune :
            keep = set()
            for p in parts :
                keep.update(p)
            return new._patch([c for c in new.components
                               if c.num not in keep], []), path
        else :
            return new, path
        #TODO: add info in edge table
        #TODO: upgrade Graph to allow drawing this info
    def _search (self, steps) :
        # search a shortest path going through each step
        # returns a series of paths from one step to the next one
        # if no path exists, the corresponding path in the returned series is empty
        parts = []
        found = [[[None, n]] for n in steps[0]]
        for sources, targets in zip(steps, steps[1:]) :
            extend = []
            for path in self._shortest_paths(sources, targets) :
                extend.extend(p + [path] for p in found if p[-1][-1] == path[0])
            if extend :
                found = extend
            else :
                if found :
                    found.sort(key=lambda l: sum(len(p) for p in l))
                    parts.extend(found[0])
                else :
                    parts.append([None])
                found = [[[None, n]] for n in targets]
        if extend :
            extend.sort(key=lambda l: sum(len(p) for p in l))
            parts.extend(extend[0])
        del parts[0]
        parts = [p if p[0] is not None else [] for p in parts]
        return [self.g.vs[p]["node"] for p in parts]
    def _shortest_paths (self, sources, targets) :
        with warnings.catch_warnings() :
            warnings.simplefilter("ignore")
            for src in sources :
                for path in self.g.get_shortest_paths(src, targets) :
                    if len(path) > 1 :
                        yield path
    def dump (self) :
        # dump ComponentGraph info to dict, as for LTS and Component
        return {"DDD" : [],
                "model" : self.model.path}
    def save (self, path) :
        """save component graph to `path`

        Arguments:
         - `path` (`str`): file name to be saved to
        """
        d2n = {}
        hdr = []
        for obj in (self, self.lts, *self.components) :
            dump = obj.dump()
            for i, d in enumerate(dump["DDD"]) :
                dump["DDD"][i] = d2n.setdefault(d, len(d2n))
            hdr.append(dump)
        ddd.ddd_save(path, *d2n, dumps=hdr, compact=self.compact)
    @classmethod
    def load (cls, path) :
        """reload a previously saved component graph from `path`

        Arguments:
         - `path` (`str`): file name to load from
        Returns: loaded `ComponentGraph` instance
        """
        headers, ddds = ddd.ddd_load(path)
        dump_cg, dump_lts, *dump_compos = headers["dumps"]
        for dump in (dump_cg, dump_lts, *dump_compos) :
            for i, n in enumerate(dump["DDD"]) :
                dump["DDD"][i] = ddds[n]
        lts = LTS.load(dump_lts)
        compos = tuple(Component.load(dump, lts) for dump in dump_compos)
        model = load_model(dump_cg["model"])
        cg = cls(compact=headers["compact"], lts=lts, model=model)
        cg.components = compos
        cg._c.update((c.num, c) for c in compos)
        return cg

__extra__ = ["Model", "parse", "Palette", "ComponentGraph", "setrel"]
