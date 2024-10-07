import ast
import functools
import itertools
import operator
import re
import subprocess
import sys
import tempfile
import warnings

from collections import defaultdict
from typing import Optional, Union

import bqplot as bq
import ddd
import igraph as ig
import ipywidgets as ipw
import its
import numpy as np
import pandas as pd
import prince
import sympy

from IPython.display import display
from colour import Color
from pandas import isna

import tl

from .. import BaseModel, CompileError, cached_property
from .. import load as load_model
from .. import pn
from .. import istyped
from ..cygraphs import Graph
from ..graphs import Palette
from ..ui import HTML, getopt, log
from . import ltsprop
from .lts import LTS, Component, setrel
from .st import FailedParse, Parser, State
from .st import sign2char as s2c
from .match import Match, parse_pattern


class hset(object):
    "similar to frozenset with sorted values"

    # work around pandas that does not display frozensets in order
    def __init__(self, iterable=(), key=None, reverse=False, sep=","):
        self._values = tuple(sorted(iterable, key=key, reverse=reverse))
        self._sep = sep

    def __hash__(self):
        return functools.reduce(
            operator.xor, (hash(v) for v in self._values), hash("hset")
        )

    def __eq__(self, other):
        try:
            return set(self._values) == set(other._values)
        except Exception:
            return False

    def __iter__(self):
        yield from iter(self._values)

    def __bool__(self):
        return bool(self._values)

    def isdisjoint(self, other):
        return set(self).isdisjoint(set(other))

    def issubset(self, other):
        return set(self).issubset(set(other))

    def __le__(self, other):
        return self.issubset(other)

    def __lt__(self, other):
        return set(self) < set(other)

    def issuperset(self, other):
        return set(self).issuperset(set(other))

    def __ge__(self, other):
        return self.issuperset(other)

    def __gt__(self, other):
        return set(self) > set(other)

    def union(self, *others):
        return self.__class__(set(self).union(*others))

    def __or__(self, other):
        return self.union(other)

    def intersection(self, *others):
        return self.__class__(set(self).intersection(*others))

    def __and__(self, other):
        return self.intersection(other)

    def difference(self, *others):
        return self.__class__(set(self).difference(*others))

    def __sub__(self, other):
        return self.difference(other)

    def symmetric_difference(self, other):
        return self.__class__(set(self).symmetric_difference(other))

    def __xor__(self, other):
        return self.symmetric_difference(other)

    def copy(self):
        return self

    def __repr__(self):
        return f"{self._sep} ".join(str(v) for v in self)

    def _repr_pretty_(self, pp, cycle):
        pp.text(f"{self._sep} ".join(str(v) for v in self))

    def _ipython_display_(self):
        display(f"{self._sep} ".join(str(v) for v in self))


class TableProxy(object):
    _name = re.compile("^[a-z][a-z0-9_]*$", re.I)

    def __init__(self, table):
        self.table = table

    def __contains__(self, key):
        return key in self.table.columns

    def __getitem__(self, key):
        if isinstance(key, str):
            key = [key]
        elif not isinstance(key, slice):
            key = list(key)
        stats = self.table[key].describe(include="all").fillna("")
        stats[" "] = np.full(len(stats), "")
        for col, val in {
            "count": "total number of values",
            "unique": "number of unique values",
            "top": "most common value",
            "freq": "top's frequency",
            "mean": "values average",
            "std": "standard deviation",
            "min": "minimal value",
            "max": "maximal value",
            "25%": "25th percentile",
            "50%": "50th percentile",
            "75%": "75th percentile",
        }.items():
            if col in stats.columns:
                stats.loc[col, " "] = val
        return stats

    def __delitem__(self, col):
        if isinstance(col, str) and col in self.table.columns:
            self.table.drop(columns=[col], inplace=True)
        elif isinstance(col, tuple):
            for c in col:
                del self[c]
        else:
            raise KeyError("invalid column: %r" % (col,))

    def __setitem__(self, key, val):
        if key in self.table.columns:
            raise KeyError("columns %r exists already" % key)
        if callable(val):
            data = self.table.apply(val, axis=1)
        elif isinstance(val, tuple) and len(val) > 1 and callable(val[0]):
            fun, *args = val
            data = self.table.apply(fun, axis=1, args=args)
        else:
            data = pd.DataFrame({"col": val})
        if isinstance(key, (tuple, list)):
            for k in key:
                if not self._name.match(k):
                    raise ValueError("invalid column name %r" % k)
            data = pd.DataFrame.from_records(data)
            for k, c in zip(key, data.columns):
                self.table[k] = data[c]
        else:
            if not self._name.match(key):
                raise ValueError("invalid column name %r" % key)
            self.table[key] = data

    def _ipython_display_(self):
        display(self[:])

    def _repr_pretty_(self, pp, cycle):
        return pp.text(repr(self[:]))


LATEX_RR_STY = r"""
\ProvidesPackage{rr}

\usepackage{fancyvrb}
\usepackage{color}

\newcommand\RRon[1]{{\color{green!40!black}#1+}}
\newcommand\RRoff[1]{{\color{red!40!black}#1-}}
\newcommand\RRany[1]{{\color{blue!40!black}#1*}}
\newcommand\RRsec[1]{{\fontseries{b}\selectfont #1:}}
\newcommand\RRtag[1]{{\color{purple!40!black}[#1]}}
\newcommand\RRthen{{\color{yellow!40!black}{>}{>}}}
\newcommand\RRcmt[1]{{\color{white!70!black}\# #1}}

\newcommand\rr[1]{\texttt{#1}}
"""


class Model(BaseModel):
    @classmethod
    def parse(cls, path):
        try:
            model = Parser(path).parse()
        except FailedParse as err:
            raise CompileError("\n".join(str(err).splitlines()[:3]))
        return cls(path, model)

    #
    # RR
    #

    @cached_property
    def rr(self):
        """show RR source code"""
        h = HTML()
        with h("pre", style="line-height:140%"):
            sections = {}
            for d in self.spec.meta:
                sections.setdefault(d.kind, []).append(d)
            for sect, decl in sections.items():
                with h("span", style="color:#008; font-weight:bold;", BREAK=True):
                    h.write("%s:" % sect)
                for d in decl:
                    h.write("    ")
                    if d.state.sign is None:
                        color = "#008"
                    elif d.state.sign:
                        color = "#080"
                    else:
                        color = "#800"
                    with h("span", style="color:%s;" % color):
                        h.write("%s" % d.state)
                    h.write(": %s\n" % d.description)
            for sect in ("constraints", "rules"):
                rules = getattr(self.spec, sect)
                if rules:
                    with h("span", style="color:#008; font-weight:bold;", BREAK=True):
                        h.write("%s:" % sect)
                    for rule in rules:
                        h.write("    ")
                        if (rule.label or "").strip():
                            with h("span", style="color:#888; font-weight:bold;"):
                                h.write("[%s] " % rule.label.strip())
                        for i, s in enumerate(rule.left):
                            if i:
                                h.write(", ")
                            color = "#080" if s.sign else "#800"
                            with h("span", style="color:%s;" % color):
                                h.write(str(s))
                        with h("span", style="color:#008; font-weight:bold;"):
                            h.write(" &gt;&gt; ")
                        for i, s in enumerate(rule.right):
                            if i:
                                h.write(", ")
                            color = "#080" if s.sign else "#800"
                            with h("span", style="color:%s;" % color):
                                h.write(str(s))
                        with h("span", style="color:#888;"):
                            h.write("   # %s" % rule.name())
                        h.write("\n")
        return h

    def latex(self, out, sty=None, names=False, cols=False):
        if sty is not None:
            sty.write(LATEX_RR_STY)
        out.write(r"\begin{Verbatim}[commandchars=\\\{\}]" "\n")
        sections = {}
        for d in self.spec.meta:
            sections.setdefault(d.kind, []).append(d)
        for sect, decl in sections.items():
            out.write(rf"\RRsec{{{sect}}}" "\n")
            for d in decl:
                out.write("    ")
                if d.state.sign is None:
                    out.write(rf"\RRany{{{d.state.name}}}")
                elif d.state.sign:
                    out.write(rf"\RRon{{{d.state.name}}}")
                else:
                    out.write(rf"\RRoff{{{d.state.name}}}")
                out.write(f": {d.description}\n")
        if names:
            width = max(
                len(r.text()) + (len("[" + r.label.strip() + "] ") if r.label else 0)
                for rules in (self.spec.constraints, self.spec.rules)
                for r in rules
            )
        if cols:
            out.write(r"\columnbreak" "\n")
        for sect in ("constraints", "rules"):
            rules = getattr(self.spec, sect)
            if rules:
                out.write(rf"\RRsec{{{sect}}}" "\n")
                for rule in rules:
                    out.write("  ")
                    if (rule.label or "").strip():
                        out.write(rf"\RRtag{{{rule.label.strip()}}} ")
                    for i, s in enumerate(rule.left):
                        if i:
                            out.write(", ")
                        if s.sign:
                            out.write(rf"\RRon{{{s.name}}}")
                        else:
                            out.write(rf"\RRoff{{{s.name}}}")
                    out.write(r" \RRthen ")
                    for i, s in enumerate(rule.right):
                        if i:
                            out.write(", ")
                        if s.sign:
                            out.write(rf"\RRon{{{s.name}}}")
                        else:
                            out.write(rf"\RRoff{{{s.name}}}")
                    if names:
                        out.write(
                            " "
                            * (
                                2
                                + width
                                - len(rule.text())
                                - (
                                    len("[" + rule.label.strip() + "] ")
                                    if rule.label
                                    else 0
                                )
                            )
                        )
                        out.write(rf"\RRcmt{{{rule.name()}}}")
                    out.write("\n")
        out.write(r"\end{Verbatim}" "\n")

    def charact(self, constraints: bool = True, variables=None) -> pd.DataFrame:
        """compute variables static characterisation
        Options:
         - constraint (True): take constraints into account
         - variables (None): the list of variables to be displayed,
           or all if None
        Return: a DataFrame which the counts characterising each variable
        """
        if variables is None:
            index = list(sorted(s.state.name for s in self.spec.meta))
        else:
            index = list(variables)
        counts = pd.DataFrame(
            index=index,
            data={
                "init": np.full(len(index), 0),
                "left_0": np.full(len(index), 0),
                "left_1": np.full(len(index), 0),
                "right_0": np.full(len(index), 0),
                "right_1": np.full(len(index), 0),
            },
        )
        for s in self.spec.meta:
            if s.state.name in index and s.state.sign:
                counts.loc[s.state.name, "init"] = 1
        if constraints:
            todo = [self.spec.constraints, self.spec.rules]
        else:
            todo = [self.spec.rules]
        for rule in itertools.chain(*todo):
            for side in ["left", "right"]:
                for state in getattr(rule, side):
                    if state.name in index:
                        sign = "_1" if state.sign else "_0"
                        counts.loc[state.name, side + sign] += 1
        return counts

    def draw_charact(self, constraints: bool = True, variables=None, **options):
        """draw a bar chart of nodes static characterization
        Options:
         - constraints (True): take constraints into account
         - variables (None): the list of variables to be displayed,
           or all if None
        Figure options:
         - fig_width (960): figure width (in pixels)
         - fig_height (300): figure height (in pixels)
         - fig_padding (0.01): internal figure margins
         - fig_title (None): figure title
        Bars options:
         - bar_spacing (1): additional space at the left of bars to fit
           variable names
         - bar_palette ("RGW"): color palette for bars
        """
        self.opt = opt = getopt(
            options,
            fig_width=960,
            fig_height=300,
            fig_padding=0.01,
            fig_title=None,
            bar_spacing=1,
            bar_palette="RGW",
        )
        counts = self.charact(constraints, variables)
        shift = (counts["left_0"] + counts["left_1"]).max() + opt.bar.spacing
        last = shift + (counts["right_0"] + counts["right_1"]).max()
        xs = bq.OrdinalScale(reverse=True)
        ys = bq.LinearScale(min=-opt.bar.spacing, max=float(last + 1))
        bar_left = bq.Bars(
            x=counts.index,
            y=[counts["left_0"], counts["left_1"]],
            scales={"x": xs, "y": ys},
            padding=0.1,
            colors=Palette.mkpal(opt.bar.palette, 2),
            orientation="horizontal",
        )
        bar_right = bq.Bars(
            x=counts.index,
            y=[counts["right_0"] + shift, counts["right_1"] + shift],
            scales={"x": xs, "y": ys},
            padding=0.1,
            colors=Palette.mkpal(opt.bar.palette, 2),
            orientation="horizontal",
            base=float(shift),
        )
        ax_left = bq.Axis(
            scale=xs,
            orientation="vertical",
            grid_lines="none",
            side="left",
            offset={"scale": ys, "value": 0},
        )
        ax_right = bq.Axis(
            scale=xs,
            orientation="vertical",
            grid_lines="none",
            side="left",
            offset={"scale": ys, "value": shift},
        )
        ay = bq.Axis(
            scale=ys,
            orientation="horizontal",
            tick_values=(
                list(range(shift + 1 - opt.bar.spacing)) + list(range(shift, last + 1))
            ),
            tick_style={"display": "none"},
        )
        if opt.fig.title:
            fig = bq.Figure(
                marks=[bar_left, bar_right],
                axes=[ax_left, ax_right, ay],
                padding_x=opt.fig.padding,
                padding_y=opt.fig.padding,
                layout=ipw.Layout(
                    width="%spx" % opt.fig.width, height="%spx" % opt.fig.height
                ),
                fig_margin={"top": 60, "bottom": 0, "left": 0, "right": 0},
                title=opt.fig.title,
            )
        else:
            fig = bq.Figure(
                marks=[bar_left, bar_right],
                axes=[ax_left, ax_right, ay],
                padding_x=opt.fig.padding,
                padding_y=opt.fig.padding,
                layout=ipw.Layout(
                    width="%spx" % opt.fig.width, height="%spx" % opt.fig.height
                ),
                fig_margin={"top": 0, "bottom": 0, "left": 0, "right": 0},
            )
        display(ipw.VBox([fig, bq.Toolbar(figure=fig)]))

    def save(self, target, overwrite: bool = False):
        """save rr to file
        Arguments:
         - target: target file (either its path or a writable file-like object)
        Options:
         - overwrite (True): whether path should be overwritten if it exists
        """
        self.spec.save(target, overwrite)

    def inline(self, path: str, *constraints, overwrite: bool = False):
        """inline constraints into the rules
        Arguments:
         - path: where to save the generated RR file
         - constraints, ...: a series of constraints to be inlines
           (given by names of numbers)
        Options:
         - overwrite (False): whether path can be overwritten
           or not if it exists
        Return: the model obtained by inlining the constraints as requested
        """
        model = self.__class__(path, self.spec.inline(constraints))
        model.save(path, overwrite=overwrite)
        return model

    #
    # state space methods
    #
    def __call__(self, *largs, **kargs):
        """build a `ComponentGraph` from the model

        Arguments:
         - `compact` (`False`): whether transient states should be removed and
           transitions rules adjusted accordingly
         - `init` (`""`): initial states of the LTS. If empty: take them from
           the model. Otherwise, it must be a comma-separated sequence of
           states assignements, that is interpreted as successive assignements
           starting from the initial states specified in the model.
           Each assignement be either:
           - `"*"`: take all the potential states (so called universe)
           - `"+"`: set all variables to "+"
           - `"-"`: set all variables to "-"
           - `VARx`, where `VAR` is a variable name and `x` is a sign in
             `+`, `-`, or `*`: set variable as specified by the sign
           For instance, sequence `"*,V+,W-` will consider all the pontential
           states restricted to those where `V` is `+` and `W` is `-`.
           A list of such strings may be used, in which case the union of the
           corresponding sets of states is considered.
         - `split` (`True`): should the graph be initially split into its
           initial states, SCC hull, and deadlocks+basins
        Returns: newly created `ComponentGraph` instance
        """
        return ComponentGraph.from_model(self, *largs, **kargs)

    def gal_path(self, compact=False, permissive=False):
        return str(self[("c" if compact else "") + ("p" if permissive else "") + "gal"])

    def gal(self, compact=False, permissive=False, showlog=True, init={}):
        path = self.gal_path(compact, permissive)
        with log(
            head="<b>saving</b>",
            tail=path,
            done_head="<b>saved:</b>",
            done_tail=path,
            verbose=showlog,
        ), open(path, "w") as out:
            name = re.sub("[^a-z0-9]+", "", self.base.name, flags=re.I)
            out.write(f"gal {name} {{\n    //*** variables ***//\n")
            # variables
            for sort in self.spec.meta:
                sign = bool(init.get(sort.state.name, sort.state.sign))
                out.write(
                    f"    // {sort.state}: {sort.description}"
                    f" ({sort.kind})\n"
                    f"    int {sort.state.name} = {sign:d};\n"
                )
                if permissive:
                    out.write(f"    int {sort.state.name}_flip = 0;\n")
                    if self.spec.constraints:
                        out.write(f"    int {sort.state.name}_FLIP = 0;\n")
            # constraints
            ctrans, ptrans = [], []
            if self.spec.constraints:
                out.write("    //*** constraints ***//\n")
            if compact:
                label = ' label "const" '
            else:
                label = " "
            for const in self.spec.constraints:
                if permissive:
                    guard = " && ".join(
                        f"({s.name} == {s.sign:d}"
                        f" || {s.name}_FLIP == 1"
                        f" || {s.name}_flip == 1)"
                        for s in const.left
                    )
                else:
                    guard = " && ".join(f"({s.name} == {s.sign:d})" for s in const.left)
                loop = " && ".join(f"({s.name} == {s.sign:d})" for s in const.right)
                out.write(
                    f"    // {const}\n"
                    f"    transition C{const.num} "
                    f"[{guard} && (!({loop}))]"
                    f"{label}{{\n"
                )
                for s in const.right:
                    out.write(f"        {s.name} = {s.sign:d};\n")
                    if permissive:
                        out.write(f"        {s.name}_FLIP = 1;\n")
                        out.write(f"        {s.name}_flip = 0;\n")
                out.write("    }\n")
                ctrans.append(f"{guard} && (!({loop}))")
            if permissive and self.spec.constraints:
                out.write("    //*** end FLIPS ***//\n")
                for sort in self.spec.meta:
                    out.write(
                        f"    transition FLIP_{sort.state.name} "
                        f"[{sort.state.name}_FLIP == 1]"
                        f"{label}{{\n"
                        f"        {sort.state.name}_FLIP = 0;\n"
                        f"    }}\n"
                    )
                    ptrans.append(f"{sort.state.name}_FLIP == 1")
            if compact:
                out.write(
                    f"    // constraints "
                    f"+ identity for fixpoints in rules\n"
                    f'    transition Cx [true] label "rconst" {{\n'
                    f'        self."const";'
                    f"    }}\n"
                    f'    transition ID [true] label "rconst" {{ }}\n'
                )
            # rules
            if ctrans or ptrans:
                prio = " && (!(%s))" % " || ".join(f"({g})" for g in ctrans + ptrans)
            else:
                prio = ""
            out.write("    //*** rules ***//\n")
            for rule in self.spec.rules:
                if permissive:
                    guard = " && ".join(
                        f"({s.name} == {s.sign:d}" f" || {s.name}_flip == 1)"
                        for s in rule.left
                    )
                else:
                    guard = " && ".join(f"({s.name} == {s.sign:d})" for s in rule.left)
                loop = " && ".join(f"({s.name} == {s.sign:d})" for s in rule.right)
                out.write(
                    f"    // {rule}\n"
                    f"    transition R{rule.num}"
                    f" [{guard} && (!({loop})){prio}] {{\n"
                )
                for s in rule.right:
                    out.write(f"        {s.name} = {s.sign:d};\n")
                    if permissive:
                        out.write(f"        {s.name}_flip = 1;\n")
                if compact:
                    out.write(
                        "        fixpoint {\n"
                        '            self."rconst";\n'
                        "        }\n"
                    )
                out.write("    }\n")
            if permissive:
                out.write("    //*** end flips ***//\n")
                for sort in self.spec.meta:
                    out.write(
                        f"    transition flip_{sort.state.name} "
                        f"[{sort.state.name}_flip == 1] {{\n"
                        f"        {sort.state.name}_flip = 0;\n"
                        f"    }}\n"
                    )
                    ptrans.append(f"{sort.state.name}_flip == 1")
            transient = []
            if compact:
                out.write(
                    "    //*** skip transient initial states ***//\n"
                    "    transition __INIT__ [true] {\n"
                    '        self."const";\n'
                    "    }\n"
                )
                transient.extend(ctrans)
            if permissive:
                transient.extend(ptrans)
            if transient:
                out.write(
                    "    TRANSIENT = (%s);\n" % " || ".join(f"({g})" for g in transient)
                )
            out.write("}\n")
            return path

    def zeno(self, witness: int = 10) -> Optional[list]:
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
        cons = {c.name(): c for c in self.spec.constraints}
        gal = its.model(self.gal(showlog=False), fmt="GAL")
        with log(
            head="<b>checking for Zeno executions</b>",
            tail="{step}",
            done_head="<b>Zeno:</b>",
            done_tail=('<span style="color:#080; font-weight:bold;">' "no</span>"),
            steps=[
                "computing reachable states",
                "extracting transient states",
                "extracting transient SCC hull",
                "extracting witness state",
                "extracting looping constraints",
            ],
            keep=True,
            verbose=True,
        ):
            if not cons:
                log.finish()
                return
            init = gal.reachable()
            log.update()
            chom = {n: h for n, h in gal.transitions().items() if n in cons}
            succ = ddd.shom.union(*chom.values())
            pred = ddd.shom.union(*chom.values()).invert(init)
            transient = pred(init)
            if transient & gal.initial():
                log.warn("initial state is transient")
            else:
                log.info("initial state is not transient")
            log.update()
            succ_h = succ.gfp()
            pred_h = pred.gfp()
            hull = succ_h(transient) & pred_h(transient)
            if hull:
                log.done_tail = (
                    '<span style="color:#800; font-weight:bold;">' "YES</span>"
                )
            else:
                log.finish()
                return
            log.update()
            if witness:
                succ_s = (succ & transient).lfp()
                pred_s = (pred & transient).lfp()
                while hull:
                    st = hull.pick()
                    scc = succ_s(st) & pred_s(st)
                    if len(scc) > 1:
                        hull = scc
                        states = list(hull)[0][0]
                        selection = []
                        for i, s in zip(range(min(len(hull), witness)), states.items()):
                            selection.append(
                                ", ".join(f"{n}+" for n, v in s.items() if v)
                            )
                        if len(hull) > witness:
                            selection.append(f"... {len(hull) - witness} more")
                        log.print(
                            f"<b>Zeno witness states:</b>"
                            + "\n".join(f"<br/><code>{s}</code>" for s in selection)
                        )
                        break
                    hull = hull - scc
                    hull = succ_h(hull) & pred_h(hull)
            log.update()
            loop = []
            for n, h in chom.items():
                if h(hull) & hull:
                    loop.append(n)
            log.print(
                "<b>Zeno witness constraints:</b>"
                + "\n".join(f"<br/><code>{cons[n]}</code>" for n in loop)
            )
            log.finish()
        return [cons[n] for n in loop]

    def CTL(self, formula, compact=False, options=[], debug=False, init={}):
        """Model-check a CTL formula
        Arguments:
         - formula: a CTL formula fiven as a string
        Options:
         - compact (False): check the compact model instead of the full
         - options ([]): command line options to be passed to "its-ctl"
         - debug (False): show "its-ctl" output
        """
        with tempfile.NamedTemporaryFile(mode="w+", suffix=f".ctl") as tmp:
            tmp.write(tl.parse(formula).its_ctl())
            tmp.flush()
            argv = [
                "its-ctl",
                "-i",
                self.gal(compact=compact, init=init),
                "-t",
                "GAL",
                "-ctl",
                tmp.name,
                "--quiet",
            ] + options
            try:
                out = subprocess.check_output(argv, encoding="utf-8")
            except subprocess.CalledProcessError as err:
                log.err(
                    f"<tt>its-ctl</tt> exited with status code" f" {err.returncode}"
                )
                out = err.output
                debug = True
        if debug:
            sys.stderr.write(out)
        return "Formula is TRUE" in out

    def LTL(
        self, formula, compact=False, options=[], algo="SSLAP-FST", debug=False, init={}
    ):
        """Model-check a LTL formula
        Arguments:
         - formula: a LTL formula fiven as a string
        Options:
         - compact (False): check the compact model instead of the full
         - options ([]): command line options to be passed to "its-ltl"
         - algo ("SSLAP-FST"): algorithm to be used
         - debug (False): show "its-ltl" output
        """
        argv = [
            "its-ltl",
            "-i",
            self.gal(compact=compact, init=init),
            "-t",
            "GAL",
            "-c",
            "-e",
            "-ltl",
            tl.parse(formula).its_ltl(),
            f"-{algo}",
            "--place-syntax",
        ] + options
        try:
            out = subprocess.check_output(argv, encoding="utf-8")
        except subprocess.CalledProcessError as err:
            log.err(f"<tt>its-ltl</tt> exited with status code" f" {err.returncode}")
            out = err.output
            debug = True
        if debug:
            sys.stderr.write(out)
        return "Formula 0 is FALSE" in out

    #
    # ecosystemic (hyper)graph
    #
    def ecograph(self, constraints: bool = True, **opt):
        """draw the ecosystemic graph
        Options:
         - constraint (True): take constraints into account
         - plus any option accepted by `ecco.cygraphs.Graph`
        """
        nodes = []
        for sort in sorted(self.spec.meta, key=lambda s: s.state.name):
            nodes.append(
                {
                    "node": sort.state.name,
                    "init": sort.state.sign,
                    "category": sort.kind,
                    "description": sort.description,
                }
            )
        edges = []
        if constraints:
            todo = [self.spec.constraints, self.spec.rules]
        else:
            todo = [self.spec.rules]
        for rule in itertools.chain(*todo):
            for src in rule.left:
                for dst in rule.right:
                    edges.append(
                        {
                            "src": src.name,
                            "dst": dst.name,
                            "rule": rule.name(),
                            "rr": rule.text(),
                        }
                    )
        options = dict(
            layout="circle",
            nodes_fill_color="init",
            nodes_fill_palette="red-green/white",
            nodes_shape="ellipse",
            edges_label="",
            edges_curve="straight",
        )
        options.update(opt)
        return Graph(
            pd.DataFrame.from_records(nodes, index="node"),
            pd.DataFrame.from_records(edges, index=["src", "dst"]),
            **options,
        )

    def ecohyper(self, constraints: bool = True, **opt):
        """draw the ecosystemic hypergraph
        Options:
         - constraint (True): take constraints into account
         - plus any option accepted by `ecco.cygraphs.Graph`
        """
        nodes = []
        edges = []
        if constraints:
            rules = list(itertools.chain(self.spec.constraints, self.spec.rules))
        else:
            rules = list(self.spec.rules)
        for state, desc in sorted((s.state, s.description) for s in self.spec.meta):
            nodes.append(
                {
                    "node": state.name,
                    "kind": "variable",
                    "init": state.sign,
                    "info": desc,
                    "shape": "ellipse",
                    "color": 0 if state.sign else 1,
                }
            )
        for rule in rules:
            name = rule.name()
            nodes.append(
                {
                    "node": name,
                    "kind": "constraint" if name[0] == "C" else "rule",
                    "init": "",
                    "info": rule.text(),
                    "shape": "rectangle",
                    "color": 2 if name[0] == "C" else 3,
                }
            )
            for var in set(s.name for s in itertools.chain(rule.left, rule.right)):
                v0 = State(var, False)
                v1 = State(var, True)
                if v1 in rule.left:
                    if v1 in rule.right:
                        g, s = "**"
                    elif v0 in rule.right:
                        g, s = "*o"
                    else:
                        g, s = "*-"
                elif v0 in rule.left:
                    if v1 in rule.right:
                        g, s = "o*"
                    elif v0 in rule.right:
                        g, s = "oo"
                    else:
                        g, s = "o-"
                elif v1 in rule.right:
                    g, s = "-*"
                elif v0 in rule.right:
                    g, s = "-o"
                edges.append({"src": name, "dst": var, "get": g, "set": s})
        options = dict(
            nodes_shape="shape",
            nodes_fill_color="color",
            nodes_fill_palette=("hypergraph", "lin", False),
            edges_label="",
            edges_source_tip="get",
            edges_target_tip="set",
            inspect_nodes=["kind", "init", "info"],
            ui=[
                {"Graph": [["layout", "reset_view"]]},
                {
                    "Nodes": [
                        {
                            "Fill": [
                                ["nodes_fill_color", "nodes_fill_palette"],
                                ["nodes_fill_opacity"],
                            ]
                        },
                        {
                            "Draw": [
                                ["nodes_draw_color", "nodes_draw_palette"],
                                ["nodes_draw_style", "nodes_draw_width"],
                            ]
                        },
                    ]
                },
                {"Edges tips": [["edges_tip_scale"]]},
                "graph",
                {
                    "Inspector": [
                        ["select_all", "select_none", "invert_selection"],
                        ["inspect_nodes"],
                    ]
                },
                {"Export": [["export"]]},
            ],
        )
        options.update(opt)
        return Graph(
            pd.DataFrame.from_records(nodes, index="node"),
            pd.DataFrame.from_records(edges, index=["src", "dst"]),
            **options,
        )

    #
    # Petri nets
    #
    def _petri_actions(self, transform=0):
        for prio, actions in enumerate([self.spec.rules, self.spec.constraints]):
            for act in actions:
                if transform == 0:
                    yield bool(prio), act
                elif transform == 1:
                    yield bool(prio), act.normalise()
                elif transform == 2:
                    for a in act.elementarise():
                        yield bool(prio), a
                else:
                    raise ValueError(f"unexpected value for transform: {transform}")

    def petri(self, kind="xpn"):
        kind = kind.upper()
        if kind == "EPN":
            net = pn.EPN()
            for state in (s.state for s in self.spec.meta):
                net.add_place(state.name, tokens=state.sign)
            for prio, action in self._petri_actions(1):
                trans = action.name()
                net.add_trans(trans, priority=prio)
                for var in action.vars():
                    on = State(var, True)
                    off = State(var, False)
                    if on in action.left and on in action.right:
                        net.add_read(var, trans)
                    elif off in action.left and off in action.right:
                        net.add_inhib(var, trans)
                    elif on in action.left and off in action.right:
                        net.add_cons(var, trans)
                    elif off in action.left and on in action.right:
                        net.add_inhib(var, trans)
                        net.add_prod(trans, var)
                    elif on in action.right:
                        assert on not in action.left
                        assert off not in action.left
                        net.add_reset(var, trans)
                        net.add_prod(trans, var)
                    elif off in action.right:
                        assert on not in action.left
                        assert off not in action.left
                        net.add_reset(var, trans)
                    else:
                        assert on not in action.left
                        assert off not in action.left
                        assert on not in action.right
                        assert off not in action.right
        elif kind == "XPN":
            net = pn.XPN()
            for state in (s.state for s in self.spec.meta):
                net.add_place(f"{state.name}+", tokens=state.sign)
                net.add_place(f"{state.name}-", tokens=not state.sign)
            for prio, action in self._petri_actions(1):
                trans = action.name()
                net.add_trans(trans, priority=prio)
                for var in action.vars():
                    on = State(var, True)
                    off = State(var, False)
                    if on in action.left and on in action.right:
                        net.add_read(f"{var}+", trans)
                    elif off in action.left and off in action.right:
                        net.add_read(f"{var}-", trans)
                    elif on in action.left and off in action.right:
                        net.add_cons(f"{var}+", trans)
                        net.add_prod(trans, f"{var}-")
                    elif off in action.left and on in action.right:
                        net.add_cons(f"{var}-", trans)
                        net.add_prod(trans, f"{var}+")
                    elif on in action.right:
                        assert on not in action.left
                        assert off not in action.left
                        net.add_reset(f"{var}+", trans)
                        net.add_reset(f"{var}-", trans)
                        net.add_prod(trans, f"{var}+")
                    elif off in action.right:
                        assert on not in action.left
                        assert off not in action.left
                        net.add_reset(f"{var}+", trans)
                        net.add_reset(f"{var}-", trans)
                        net.add_prod(trans, f"{var}-")
                    else:
                        assert on not in action.left
                        assert off not in action.left
                        assert on not in action.right
                        assert off not in action.right
        elif kind == "SPN":
            net = pn.SPN()
            for state in (s.state for s in self.spec.meta):
                net.add_place(f"{state.name}+", tokens=state.sign)
                net.add_place(f"{state.name}-", tokens=not state.sign)
            for prio, action in self._petri_actions(2):
                trans = action.full_name()
                net.add_trans(trans, priority=prio)
                for var in action.vars():
                    on = State(var, True)
                    off = State(var, False)
                    if on in action.left and on in action.right:
                        net.add_cons(f"{var}+", trans)
                        net.add_prod(trans, f"{var}+")
                    elif off in action.left and off in action.right:
                        net.add_cons(f"{var}-", trans)
                        net.add_prod(trans, f"{var}-")
                    elif on in action.left and off in action.right:
                        net.add_cons(f"{var}+", trans)
                        net.add_prod(trans, f"{var}-")
                    elif off in action.left and on in action.right:
                        net.add_cons(f"{var}-", trans)
                        net.add_prod(trans, f"{var}+")
                    else:
                        assert on not in action.left
                        assert off not in action.left
                        assert on not in action.right
                        assert off not in action.right
            net.remove_loops()
        else:
            raise ValueError(f"unsupported Petri net class: {kind!r}")
        return net

    def pep(self, stream, epn=True):
        states = [s.state for s in self.spec.meta]
        rules = list(self.spec.rules)
        if self.spec.constraints:
            rules.extend(self.spec.constraints)
            log.warn("constraints have been turned into rules")
        if epn:
            self._pep_epn(stream, states, (r.normalise() for r in rules))
        else:
            self._pep_ppn(stream, states, (n for r in rules for n in r.elementarise()))

    def _pep_ppn(self, stream, states, rules):
        raise NotImplementedError

    def _pep_epn(self, stream, states, rules):
        stream.write("PEP\n" "PetriBox\n" "FORMAT_N2\n")
        # places
        stream.write("PL\n")
        pnum = {}
        for state in states:
            on, off = State(state.name, True), State(state.name, False)
            pnum[on] = 1 + len(pnum)
            pnum[off] = 1 + len(pnum)
            if state.sign:
                stream.write(f'"{on}"M1\n')
                stream.write(f'"{off}"\n')
            else:
                stream.write(f'"{on}"\n')
                stream.write(f'"{off}"M1\n')
        # transitions (and arcs)
        stream.write("TR\n")
        arcs = defaultdict(list)
        for tnum, rule in enumerate(rules, start=1):
            stream.write(f'"{rule.name()}"\n')
            left = set(s.name for s in rule.left)
            right = set(s.name for s in rule.right)
            for var in rule.vars():
                on, off = State(var, True), State(var, False)
                if on in rule.left and off in rule.right:
                    # A+ >> A-
                    arcs["PT"].append(f"{pnum[on]}>{tnum}")
                    arcs["TP"].append(f"{tnum}<{pnum[off]}")
                elif on in rule.left:
                    # A+ >> A+ | (no A)
                    assert (on in rule.right) or (var not in right)
                    arcs["RD"].append(f"{pnum[on]}>{tnum}")
                elif off in rule.left and on in rule.right:
                    # A- >> A+
                    arcs["PT"].append(f"{pnum[off]}>{tnum}")
                    arcs["TP"].append(f"{tnum}<{pnum[on]}")
                elif off in rule.left:
                    # A- >> A- | (no A)
                    assert (off in rule.right) or (var not in right)
                    arcs["RD"].append(f"{pnum[off]}>{tnum}")
                elif on in rule.right:
                    # (no A) >> A+
                    assert var not in left
                    arcs["RS"].append(f"{pnum[on]}>{tnum}")
                    arcs["RS"].append(f"{pnum[off]}>{tnum}")
                    arcs["TP"].append(f"{tnum}<{pnum[on]}")
                else:
                    # (no A) >> A-
                    assert (var not in left) and (off in rule.right)
                    arcs["RS"].append(f"{pnum[on]}>{tnum}")
                    arcs["RS"].append(f"{pnum[off]}>{tnum}")
                    arcs["TP"].append(f"{tnum}<{pnum[off]}")
        # write arcs
        for kind in ["RD", "TP", "PT", "RS"]:
            # NOTE: this order is important to avoid breaking ecofolder parsing
            stream.write(f"{kind}\n")
            stream.write("\n".join(arcs[kind]))
            stream.write("\n")

    def unfold(self, petri=False, compress=True):
        """Unfold the Petri net semantics of the RR model
        The RR model is converted to a Petri net, then this net is unfolded
        using `ecofolder`, and finally the resulting unfolding is displayed
        either as a Petri net or as the corresponding event structure.

        Options:
         - petri (`False`): if `False`, display the event structure, otherwise,
           display the unfolding
         - compress (`True`): use option `-c` of `ecofolder` to aggregate
           redundant pre/post conditions
        Return: an interactive `Graph` object
        """
        # TODO: highlight initial marking (conditions with no preds)
        # ecofolder options
        # TODO: -T <name> => cut branch on trans
        # TODO: -d <depth> => cut at given depth
        # -i/-confmax => cannot use
        # -r place,place,... => subsumed by search options
        # -c/-m => done
        # -att => not useful in ecco (used by Loic's tool)
        # TODO: move search options from llnet to cli => more general than -r
        pep_path = self["ll_net"]
        prn = pep_path.stem + "_pr"
        pre_path = pep_path.with_name(prn).with_suffix(".ll_net")
        mci_path = pre_path.with_suffix(".mci")
        nodes_path = mci_path.with_name(prn + "_nodes").with_suffix(".csv")
        edges_path = nodes_path.with_name(prn + "_edges").with_suffix(".csv")
        with open(pep_path, "w") as stream:
            self.pep(stream, True)
        subprocess.check_output(
            ["pr_encoding", pep_path],
            stderr=subprocess.STDOUT,
            encoding="ascii",
            errors="replace",
        )
        subprocess.check_output(
            ["ecofolder", pre_path, "-m", mci_path] + (["-c"] if compress else []),
            stderr=subprocess.STDOUT,
            encoding="ascii",
            errors="replace",
        )
        subprocess.check_output(
            ["mci2csv", mci_path] + (["-c"] if compress else []),
            stderr=subprocess.STDOUT,
            encoding="ascii",
            errors="replace",
        )

        def tokens(val):
            return ast.literal_eval(val or "None")

        def cutoff(val):
            if val == "0":
                return False
            elif val == "1":
                return True
            else:
                return None

        if compress:
            pdopts = {"converters": {"cutoff": cutoff, "tokens": tokens}}
        else:
            pdopts = {"converters": {"cutoff": cutoff}, "dtype": {"tokens": "Int64"}}
        nodes = pd.read_csv(nodes_path, **pdopts).rename(columns={"id": "node"})
        if compress:
            c2g = {g.rsplit(",", 1)[-1]: g for g in nodes["node"]}
            pdopts = {"converters": {"src": c2g.get, "dst": c2g.get}}
        else:
            pdopts = {}
        edges = pd.read_csv(edges_path, **pdopts)
        options = {
            "layout": "dagre",
            "nodes_draw_color": "#000",
            "nodes_draw_width": 0.6,
            "nodes_label": "name",
            "edges_label": "",
        }
        if petri:
            t2s = {"condition": "ellipse", "event": "rectangle"}
            options["nodes_shape"] = nodes["type"].apply(t2s.get)
        else:
            nodes = nodes[nodes["type"] == "event"].drop(columns=["type", "tokens"])
            events = set(nodes["node"])
            succ = defaultdict(set)
            for _, row in edges.iterrows():
                succ[row["src"]].add(row["dst"])
            caus = {
                (e, s)
                for (e, conds) in succ.items()
                for c in conds
                for s in succ.get(c, [])
                if e in events
            }
            root = events - {s for _, s in caus}
            conf = {
                frozenset([a, b])
                for c, evts in succ.items()
                for a in evts
                for b in evts
                if c not in events and a != b
            }
            edges = pd.DataFrame(
                [("_", r, "causality") for r in root]
                + [(a, b, "causality") for a, b in caus]
                + [(a, b, "conflict") for a, b in conf],
                columns=["src", "dst", "type"],
            )
            nodes = pd.concat(
                [nodes, pd.DataFrame([("_", "⊥", None)], columns=nodes.columns)],
                ignore_index=True,
            )
            options["nodes_shape"] = "rectangle"
            t2t = {"conflict": "-", "causality": ">"}
            options["edges_target_tip"] = edges["type"].apply(t2t.get)
            t2s = {"conflict": ":", "causality": "|"}
            options["edges_draw_style"] = edges["type"].apply(t2s.get)

        def mkcolors(row):
            tokens = row.get("tokens", None)
            if tokens is None or isna(tokens):
                if row["cutoff"] is None:
                    return "#fff"
                elif row["cutoff"]:
                    return "#aff"
                else:
                    return "#aaf"
            elif tokens == 0:
                return "#faa"
            elif tokens == 1:
                return "#afa"
            else:
                rh, rs, rl = Color("#faa").get_hsl()
                gh, gs, gl = Color("#afa").get_hsl()
                m = sum(tokens) / len(tokens)
                c = Color(
                    hue=rh * (1.0 - m) + gh * m,
                    saturation=rs * (1.0 - m) + gs * m,
                    luminance=rl * (1.0 - m) + gl * m,
                )
                return c.get_hex()

        def fullname(row):
            if row["node"] == "_":
                return row["name"]
            else:
                return f"{row['node']}:{row['name']}"

        options["nodes_fill_color"] = nodes.apply(mkcolors, axis="columns")
        nodes["fullname"] = nodes.apply(fullname, axis="columns")
        return Graph(nodes, edges, **options)

    #
    # pattern matching
    #
    def match(self, path, count=None, merged=False, strict=False):
        pattern = parse_pattern(path)
        for match in Match.match(pattern, self.spec, merged, strict):
            yield match
            if count is not None:
                count -= 1
                if count == 0:
                    break


class _GCS(set):
    "a subclass ofset that is tied to a component graph so it has __invert__"

    components = set()

    def __invert__(self):
        return self.__class__(self.components - self)

    def __and__(self, other):
        return self.__class__(super().__and__(other))

    def __or__(self, other):
        return self.__class__(super().__or__(other))

    def __xor__(self, other):
        return self.__class__(super().__xor__(other))


def _rulekey(r):
    return (r[0], int(r[1:]))


class NodesTableProxy(TableProxy):
    def __init__(self, compograph, table, name="nodes"):
        super().__init__(table)
        self._c = compograph
        self._n = name

    def __call__(self, col, fun=None):
        if fun is None:
            series = self.table[col].astype(bool)
        else:
            series = self.table[col].apply(fun).astype(bool)
        return self._c._GCS(self.table.index[series])

    def __getattr__(self, name):
        if name in self.table.columns:
            return functools.partial(self.__call__, name)
        else:
            raise AttributeError(f"table {self._n!r} has no column {name!r}")


class ComponentGraph(object):
    def __init__(
        self, model, compact=False, init: Union[list, str] = "", lts=None, **k
    ):
        """create a new instance

        This method is not intended to be used directly, but it will be
        called by the other methods. Use `from_model` class method if
        you whish to create a component graph from scratch.
        """
        self.model = model
        if compact and not self.model.spec.constraints:
            log.warn("model has no constraints," " setting <code>compact=False</code>")
            compact = False
        if lts is None:
            self.lts = LTS(self.model.gal(), init, compact)
        else:
            self.lts = lts
        self.components = ()
        self.compact = compact
        self._c = {}  # Component.num => Component
        self._g = {}  # Component.num => Vertex

    @classmethod
    def from_model(cls, model, compact=False, init="", split=True):
        """create a `ComponentGraph` from a `Model` instance

        Arguments:
         - `model`: a `Model` instance
         - `compact` (`False`): whether transient states should be removed and
           transitions rules adjusted accordingly
         - `init` (`str:""`): initial states of the LTS. If empty: take them
           from the model. Otherwise, it must be a comma-separated sequence of
           states assignements, that is interpreted as successive assignements
           starting from the initial states specified in the model.
           Each assignement be either:
           - `"*"`: take all the potential states (so called universe)
           - `"+"`: set all variables to "+"
           - `"-"`: set all variables to "-"
           - `VARx`, where `VAR` is a variable name and `x` is a sign in
             `+`, `-`, or `*`: set variable as specified by the sign
           For instance, sequence `"*,V+,W-` will consider all the potential
           states restricted to those where `V` is `+` and `W` is `-`.
           A list of such strings may be used, in which case the union of the
           corresponding sets of states is considered.
         - `split` (`True`): should the graph be initially split into its
           initial states, SCC hull, and deadlocks+basins
        """
        if isinstance(init, str):
            init = [init]
        for i, s in enumerate(init):
            if s not in ("*", "+", "-"):
                init[i] = (
                    ",".join(
                        f"{s.state.name}{s2c[s.state.sign]}" for s in model.spec.meta
                    )
                    + ","
                    + s
                )
        cg = cls(compact=compact, init=init, model=model)
        c_all = Component(cg.lts, cg.lts.states, gp=cg.lts.graph_props(cg.lts.states))
        if split:
            cg.components = tuple(
                c
                for c in c_all.topo_split(split_entries=False, split_exits=False)
                if c is not None
            )
        else:
            cg.components = (c_all,)
        cg._c.update((c.num, c) for c in cg.components)
        return cg

    def __getitem__(self, num):
        """return a component or an edge from the `ComponentGraph`

        Arguments:
         - when `num:int`: return a `Component` instance whose number is `num`
         - when `src,dst:num,num`: return a `dict` with the attributes of an
           edge between components `src` and `dst`
        Return: `Component` or `dict`
        """
        if isinstance(num, int):
            return self._c[num]
        else:
            try:
                src, dst = num
                eid = self.g.get_eid(self._g[src], self._g[dst])
            except Exception:
                raise KeyError(num)
            return self.g.es[eid].attributes()

    def __iter__(self):
        yield from self.components

    def __eq__(self, other):
        """equality test

        Two component graphs are equal whenever they both contain
        exactly the same components.
        """
        return set(self.components) == set(other.components)

    def _patch(self, rem, add):
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

    def _add_vertex(self, g, c):
        # add `c:Component` as a vertex in `g:igraph.Graph`
        self._c[c.num] = c
        self._g[c.num] = g.add_vertex(node=c.num).index

    def _collect_edges(self, src, targets, edges):
        # collect the edges to be added from `src` to the nodes in `targets`
        for trans, succ in src.succ(*targets):
            edges[src.num, succ.num].add(trans)

    def _add_edges(self, g, edges):
        # add the collected `edges` to `g:igraph.Graph`
        _g = self._g
        for (src, dst), rules in edges.items():
            g.add_edge(
                _g[src], _g[dst], src=src, dst=dst, rules=hset(rules, key=_rulekey)
            )

    @cached_property
    def g(self):
        """the actual component graph stored as an `igraph.Graph` instance"""
        g = ig.Graph(directed=True)
        edges = defaultdict(set)
        for c in self.components:
            self._add_vertex(g, c)
            self._collect_edges(c, self.components, edges)
        self._add_edges(g, edges)
        return g

    @cached_property
    def n(self):
        """a `Tableproxy` wrapping the `nodes` dataframe"""
        return NodesTableProxy(self, self.nodes)

    @cached_property
    def e(self):
        """a `Tableproxy` wrapping the `edges` dataframe"""
        return TableProxy(self.edges)

    @cached_property
    def nodes(self):
        """a `pandas.DataFrame` holding the information about the nodes"""
        nodes = self.g.get_vertex_dataframe()

        def n_col(row):
            c = self._c[row.node]
            return [
                len(c),
                hset(c.on),
                hset(c.off),
                hset(str(p) for p, v in c.graph_props.items() if v),
            ]

        nodes[["size", "on", "off", "topo"]] = nodes.apply(
            n_col, axis=1, result_type="expand"
        )
        nodes.set_index("node", inplace=True)
        self._update(nodes)
        return nodes

    @cached_property
    def edges(self):
        """a `pandas.DataFrame` holding the information about the edges"""
        if len(self.g.es) == 0:
            # if there is no edge, add a dummy one to build a correct dataframe
            num = self.components[0].num
            self.g.add_edge(self._g[num], self._g[num], src=num, dst=num, rules=hset())
        else:
            num = None
        edges = self.g.get_edge_dataframe().set_index(["src", "dst"])
        del edges["source"], edges["target"]
        if self.model.spec.labels:
            spec_labels = self.model.spec.labels

            def lbl(rules):
                labels = set()
                for r in rules:
                    labels.update(
                        lbl.strip()
                        for lbl in (spec_labels.get(r) or "").split(",")
                        if lbl.strip()
                    )
                return hset(labels)

            edges["labels"] = edges["rules"].apply(lbl)
        if num is not None:
            # remove the dummy edge
            return edges.drop((num, num), axis="index")
        else:
            return edges

    def __len__(self):
        """number of components in the `ComponentGraph`"""
        return len(self.components)

    def _update(self, nodes=None):
        # updates the components' split_graph properties in a nodes dataframe
        if nodes is None:
            nodes = getattr(self, "_cached_nodes", None)
        if nodes is None:
            return
        cols = [r.name for r in setrel]
        data = pd.DataFrame.from_records(
            [[hset(c, sep=";") for c in self._c[n].props_row()] for n in nodes.index],
            columns=cols,
            index=nodes.index,
        )
        nodes[cols] = data

    def update(self):
        """updates table `nodes` wrt components' properties

        This method may be needed when components shared with other component
        graphs are checked or split against new formulas.
        """
        self._update()

    def _get_args(
        self, args, aliased={}, min_props=0, max_props=-1, min_compo=0, max_compo=-1
    ):
        # parse args as a sequence of properties
        # followed by a sequence of components
        properties = {v: k for k, v in aliased.items()}
        components = []
        for i, p in enumerate(args):
            if isinstance(p, str):
                properties[p] = ""
            elif isinstance(p, int):
                components = args[i:]
                for c in components:
                    if c not in self._c:
                        raise TypeError(f"unexpected argument {c!r}")
                break
            else:
                raise TypeError(f"unexpected argument {p!r}")
        if max_props < 0:
            _max_props = len(args) + len(aliased)
        else:
            _max_props = max_props
        if not min_props <= len(properties) <= _max_props:
            if min_props == max_props == 0:
                raise TypeError(f"no properties expected," f" got {len(properties)}")
            elif max_props < 0:
                raise TypeError(
                    f"expected at least {min_props} properties,"
                    f" got {len(properties)}"
                )
            else:
                raise TypeError(
                    f"expected {min_props} to {_max_props}"
                    f" properties, got {len(properties)}"
                )
        if max_compo < 0:
            _max_compo = len(args)
        else:
            _max_compo = max_compo
        if not min_compo <= len(components) <= _max_compo:
            if min_compo == max_compo == 0:
                raise TypeError(f"no components expected," f" got {len(components)}")
            elif max_compo < 0:
                raise TypeError(
                    f"expected at least {min_compo} components,"
                    f" got {len(components)}"
                )
            else:
                raise TypeError(
                    f"expected {min_compo} to {_max_compo}"
                    f" components, got {len(components)}"
                )
        if components:
            components = tuple(self._c[c] for c in components)
        else:
            components = self.components
        return properties, components

    def forget(self, *args):
        """forget about properties

        Remove properties from the nodes table, from the components,
        and from the LTS.

        Arguments:
         - `prop, ...` (`str`): at least one property to be removed
        """
        props, _ = self._get_args(args, min_props=1, max_compo=0)
        alias = {a: p for p, a in self.lts.alias.items()}
        for prop in props:
            prop = alias.get(prop, prop)
            for c in self.components:
                c.split_props.pop(prop, None)
            self.lts.alias.pop(prop, None)
            self.lts.props.pop(prop, None)
        self._update()

    def forget_all(self):
        """forget about all properties

        Remove all properties from the nodes table, from the components,
        and from the LTS.
        """
        for c in self.components:
            c.split_props.clear()
        self.lts.props.clear()
        self.lts.alias.clear()
        self._update()

    def tag(self, *args):
        """add a dummy properties to components

        Tag `name` is a dummy property whose states are those of the component
        to which it's added.

        Parameters:
         - `name, ...` (`str`): at least one tag name
         - `number, ...` (`int`): a series of components number, if empty,
           all the components in the graph are considered
        """
        props, compo = self._get_args(args, min_props=1)
        # TODO: log
        for p in props:
            for c in compo:
                c.tag(p)
        self._update()

    def check(self, *args, **aliased):
        """check properties on components

        Arguments:
         - `prop, ...` (`str`): at least one property to be checked against
           the states of components
         - `number, ...` (`int`): a series of components number, if empty,
           all the components in the graph are considered

        Returns: a `dict` mapping each component number to another `dict`
        mapping each checked property to the relation between the component and
        the property.
        """
        props, compo = self._get_args(args, aliased, min_props=1)
        ret = {c.num: dict() for c in compo}
        # TODO: log
        for prop, alias in props.items():
            checker = ltsprop.AnyProp(self.model, self.lts, prop)
            for c in compo:
                ret[c.num][alias] = setrel(c.check(prop, checker(c.states), alias)).name
            if not self.lts.props[prop]:
                desc = self.lts.alias.get(prop, prop)
                log.warn(f"property {desc!r} is empty")
        self._update()
        return ret

    def split(self, *args, _warn=True, **aliased):
        """split components wrt properties

        Each considered component is split into the states that validate
        each the property and those that do not. If a property is globally
        (in)valid on one component, then no split occurs.

        Arguments:
         - `prop, ...` (`str`): at least one property to be checked against the
           states of components
         - `number, ...` (`int`): a series of components number, if empty,
           all the components in the graph are considered

        Returns: a new `ComponentGraph` instance
        """
        props, compo = self._get_args(args, aliased, min_props=1)
        rem, add, compo = set(), set(), set(compo)
        for prop, alias in props.items():
            r, a = self._split(prop, compo, alias)
            if _warn and not self.lts.props[prop]:
                desc = self.lts.alias.get(prop, prop)
                log.warn(f"property {desc!r} is empty")
            rem.update(r)
            add.update(a)
            compo = (compo - set(r)) | set(a)
        self._update()
        add.difference_update(rem)
        rem.intersection_update(self.components)
        return self._patch(rem, add)

    def _split(self, prop, components, alias=""):
        rem, add = [], []
        checker = ltsprop.AnyProp(self.model, self.lts, prop)
        for c in components:
            intr, diff = c.split(prop, checker(c.states), alias)
            # TODO: log
            if intr is not None and diff is not None:
                rem.append(c)
                add.append(intr)
                add.append(diff)
        return rem, add

    def topo_split(
        self, *args, init=True, entries=True, exits=True, hull=True, dead=True
    ):
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
         - `number, ...` (`int`): a series of components number, if empty,
           all the components in the graph are considered
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
        for c in compos:
            parts = c.topo_split(
                split_init=init,
                split_entries=entries,
                split_exits=exits,
                split_hull=hull,
                split_dead=dead,
            )
            keep = [p for p in parts if p is not None]
            if len(keep) > 1:
                rem.append(c)
                add.extend(keep)
        return self._patch(rem, add)

    def scc_split(self, *args, limit=256):
        """split some components into their SCCs and basins to/from SCCs

        Parameters:
         - `number, ...` (`int`): a series of components to be split
           (all are taken if none is given)
         - `limit` (`int`): how much components can be generated before the
           method asks whether it continues or not (give `0` to disable), when
           interrupted, the method returns the split graph generated so far,
           with the components not yet split being SCC hulls
        """
        _limit = limit
        _, compos = self._get_args(args, max_props=0)
        g, todo = self, {c.num for c in compos}
        while todo:
            done = {c.num for c in g.components} - todo
            g = g.split(*todo, H="hull()", _warn=False)
            todo = {
                c.num
                for c in g.components
                if c.num not in done
                and not c.graph_props["is_scc"]
                and c.graph_props["is_hull"]
            }
            if not todo:
                break
            done = {c.num for c in g.components} - todo
            g = g.split(*todo, E="scc(pick(entries()))", I="scc(INIT)", _warn=False)
            if limit and len(g) > _limit:
                answer = input(
                    f"We have created a graph with {len(g)} nodes,"
                    f" do you want to continue? [N/y] "
                )
                if not answer.lower().startswith("y"):
                    break
                _limit += limit
            todo = {
                c.num
                for c in g.components
                if c.num not in done and not c.graph_props["is_scc"]
            }
        g.forget("H", "E", "I")
        return g

    def split_basins(self, *args, merge=False):
        """split some components into the basins to some other components

        The basin to a component `c` is the set of states that may
        lead to `c`, ie, it is the set of predecessors of `c` in the
        LTS. Calling, eg, `split_basins(1, 2, [3, 4])` will split
        components `1` and `2` wrt the basins of `3` and `4`.

        Parameters:
         - `number, ...` (`int`): a series of at least one component
           number to be split
         - `[number, ...]` (`list[int]`): a list of at least one
           component whose basins will be considered for splits
         - `merge` (`bool=False`): whether to merge basins leading
            only to a single component with this component
        """
        _, split = self._get_args(args[:-1], min_compo=1, max_props=0)
        _, dest = self._get_args(args[-1], min_compo=1, max_props=0)
        old = set(split)
        for d in dest:
            basin = self.lts.pred_s(d.states)
            split = [
                s
                for c in split
                for s in c.split(f"basin({d.num})", basin)
                if s is not None
            ]
        new = set(split)
        if merge:
            for c in split:
                reach = [
                    d
                    for d in dest
                    if (
                        (f"basin({d.num})", setrel.ISIN) in c.props
                        or (f"basin({d.num})", setrel.EQUALS) in c.props
                    )
                ]
                if len(reach) == 1:
                    d = reach[0]
                    new.remove(c)
                    new.add(c.merge(d))
                    old.add(d)
        return self._patch(list(old - new), list(new - old))

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

    def classify(
        self, col, *components, split=True, src_col=None, dst_col=None, **classes
    ):
        """split component graph and classify the resulting nodes wrt `classes`

        `classes` provides pairs named properties as pairs `name=prop`.
        The method first splits the component graph wrt these properties.
        Then, a column `col` is added to the `nodes` table of the resulting
        component graph such that it contains, for each resulting component,
        the set of properties that hold on the component, given by their names.

        Parameters:
         - `col` (`str`): name of the column where nodes classification is
           stored
         - `components, ...` (`int`): components to be classified
           (all if none is given)
         - `split` (`bool=True`): should the components to classify be first
           split wrt the properties. If `False`, the components are directly
           analysed and the component graph tables are modified accordingly
         - `src_col` (`str=None`): name of a column where edges source nodes
           classification is stored
         - `dst_col` (`str=None`): name of a column where edges destination
           nodes classification is stored
         - `name=prop, ...` (`str`): named properties for classification
        """
        if split:
            new = self.split(*components, **classes)
        else:
            new = self
            new.check(*components, **classes)
        cls = set(classes)

        def cname(row):
            props = set()
            for c in ("EQUALS", "ISIN"):
                if c in row.index and row[c] and row[c] <= cls:
                    props.add("+".join(sorted(row[c])))
            return hset(props)

        new.n[col] = cname
        if src_col:

            def sname(row):
                return new.nodes[col][row.name[0]]

            new.e[src_col] = sname
        if dst_col:

            def dname(row):
                return new.nodes[col][row.name[1]]

            new.e[dst_col] = dname
        return new

    @cached_property
    def _GCS(self):
        class MyGCS(_GCS):
            components = {c.num for c in self.components}

        return MyGCS

    def __call__(self, prop, rel=setrel.HAS, strict=False):
        """compute the set of components that match `prop`

        The returned set supports `~` to compute its complement wrt
        the set of all components in the graph.

        For each value of `setrel`, two shorthand  methods are available,
        for instance:
         - `g.has(p)` is like calling `g(p, rel=setrel.HAS, strict=False)`
         - `g.HAS(p)` is like calling `g(p, rel=setrel.HAS, strict=True)`

        Arguments:
         - `prop` (`str`): the property to check on components
         - `rel` (`setrel=HAS`): the relation the components must have
           with `prop`
         - `strict` (`bool=False`): if `True`, `rel` must be exactly matched,
           otherwise, `rel` is considered relaxed (eg, `HAS` can be
           realised by `ISIN`)
        Return: the set of component numbers that match the property
        """
        update = False
        for c in self.components:
            if prop not in c.split_props:
                update = True
                self.check(prop, c.num)
        if update:
            self._update()
        return self._GCS(
            c.num
            for c in self.components
            if c.split_props[prop] in self._relmatch[rel, strict]
        )

    def has(self, prop):
        return self(prop, setrel.HAS, False)

    def HAS(self, prop):
        return self(prop, setrel.HAS, True)

    def hasno(self, prop):
        return self(prop, setrel.HASNO, False)

    def HASNO(self, prop):
        return self(prop, setrel.HASNO, True)

    def contains(self, prop):
        return self(prop, setrel.CONTAINS, False)

    def CONTAINS(self, prop):
        return self(prop, setrel.CONTAINS, True)

    def isin(self, prop):
        return self(prop, setrel.ISIN, False)

    def ISIN(self, prop):
        return self(prop, setrel.ISIN, True)

    def equals(self, prop):
        return self(prop, setrel.EQUALS, False)

    def EQUALS(self, prop):
        return self(prop, setrel.EQUALS, True)

    def select(self, *args, all=True, rel=setrel.ISIN, **aliased):
        """select components that validate properties

        Arguments:
         - `prop, ...` (`str`): at least one property to be checked against
           the states of components
         - `number, ...` (`int`): a series of components number, if empty,
           all the components in the graph are considered
         - `all` (`bool=True`): how the selected components from each property
           are accumulated. If `True`, selected components must validate all
           the properties, if `False`, they must validate at least one property
         - `rel` (`setrel=ISIN`): how the property should be validated by the
           components, a single relation may be provided or several in an
           iterable, in which case any of these relations is accepted

        Returns: a set of components numbers among `components` that validate
        the property
        """
        props, compos = self._get_args(args, aliased, min_props=1)
        if isinstance(rel, setrel):
            rel = {rel}
        else:
            rel = set(rel)
        if all:
            found = set(c.num for c in compos)
        else:
            found = set()
        update = False
        for p, a in props.items():
            for c in compos:
                if p not in c.split_props:
                    update = True
                    self.check(p, c.num, a)
            match = {c.num for c in compos if c.split_props[p] in rel}
            if all:
                found.intersection_update(match)
            else:
                found.update(match)
        if update:
            self._update()
        return found

    def explicit(self, *args, limit=256):
        """splits components into its individual states

        Arguments:
         - `number, ...`: a series of components number, if empty, all the
           components in the graph are considered
         - `limit` (`int=256`): maximum number of components in the split
           graph, if it will have more that `limit` components, a confirmation
           is asked to the user (`limit` may be set <= 0 to avoid this
           user-interaction)
        Returns: a new `ComponentGraph` instance
        """
        _, compos = self._get_args(args, max_props=0)
        if limit > 0:
            size = sum((len(c) for c in compos), 0) + len(self.components) - len(compos)
            if size > limit:
                answer = input(
                    f"This will create a graph with {size} nodes,"
                    f" are you sure? [N/y] "
                )
                if not answer.lower().startswith("y"):
                    return
        rem, add = [], []
        state2compo = {}
        if "component" in self.nodes.columns:
            state2compo.update(
                (row.name, row.component) for _, row in self.nodes.iterrows()
            )
        for c in compos:
            new = tuple(c.explicit())
            state2compo.update((n.num, c.num) for n in new)
            if len(new) > 1:
                rem.append(c)
                add.extend(new)
        ret = self._patch(rem, add)
        ret.n["component"] = lambda row: state2compo.get(row.name, row.name)
        return ret

    def merge(self, *args):
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

    def drop(self, *args):
        """drop one or more components from the component graph

        Arguments:
         - `number, ...` (`int`): at least one component to be removed
        Returns: a new `ComponentGraph` instance, or `None` if all the
        components are to be removed
        """
        _, compos = self._get_args(args, max_props=0, min_compo=1)
        if len(compos) == len(self):
            log.warn("cannot drop all the components")
            return
        return self._patch(compos, [])

    def form(self, *args, variables=None, normalise=None, separate=False):
        """describe components by Boolean formulas

        Arguments:
         - `number, ...` (`int`): a series of components number, if empty,
           all the components in the graph are considered
         - `variables=...`: a collection of variables names to restrict to
         - `normalise=...`: how the formula should be normalised, which must
           be either:
           - `"cnf"`: conjunctive normal form
           - `"dnf"`: disjunctive normal form
           - `None`: chose the smallest form
         - `separate=...`: if `False` (default) returns a single formula,
           otherwise, returns one formula for each considered component
        Return: a sympy Boolean formula or a `dict` mapping component numbers
        to such formulas
        """
        _, compos = self._get_args(args, max_props=0)
        forms = {c.num: c.form(variables, normalise) for c in compos}
        if separate:
            return forms
        else:
            return sympy.simplify_logic(sympy.Or(*forms.values()), form=normalise)

    def count(self, *args, transpose=False):
        """count in how many states each variable is on in components

        Arguments:
         - `number, ...` (`int`): a series of components number, if empty,
           all the components in the graph are considered
         - `transpose` (`bool=False`): default is to have the components
           numbers as the rows, and the variables as the columns,
           if `transpose=True`, this is swapped
        Return: a `pandas.DataFrame` instance
        """
        _, compos = self._get_args(args, max_props=0)
        counts = [c.count() for c in compos]
        df = pd.DataFrame.from_records(
            [[c.get(v, 0) for v in self.lts.vars] for c in counts],
            index=[c.num for c in compos],
            columns=list(sorted(self.lts.vars)),
        )
        if transpose:
            return df.transpose()
        else:
            return df

    def pca(
        self,
        *args,
        transpose=False,
        n_components=2,
        n_iter=3,
        copy=True,
        check_input=True,
        engine="sklearn",
        random_state=42,
        rescale_with_mean=True,
        rescale_with_std=True,
    ):
        """principal component analysis of the `count` matrix

        Arguments:
         - `number, ...` (`int`): a series of components number, if empty,
           all the components in the graph are considered
         - `transpose`: passed to method `count()`

        See https://github.com/MaxHalford/prince#principal-component-analysis-pca
        for documentation about the other arguments.

        Returns: a `pandas.DataFrame` as computed by Prince.
        """
        count = self.count(*args, transpose=transpose)
        count = count[count.sum(axis="columns") > 0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pca = prince.PCA(
                n_components=n_components,
                n_iter=n_iter,
                copy=copy,
                check_input=check_input,
                engine=engine,
                random_state=random_state,
                rescale_with_mean=rescale_with_mean,
                rescale_with_std=rescale_with_std,
            )
            pca.fit(count)
            trans = pca.transform(count)
        for idx in set(count.index) - set(trans.index):
            trans.loc[idx] = [0, 0]
        return trans

    def _convert_nodes_label(self, txt, **args):
        "convert labels for states"
        txt = str(txt)
        if not txt:
            return ""
        icons = []
        try:
            if "has_init" in args["topo"]:
                icons.append("▶")
            if "has_scc" in args["topo"] or "is_hull" in args["topo"]:
                icons.append("⏺")
            if "has_dead" in args["topo"] or "is_dead" in args["topo"]:
                icons.append("⏹")
        except Exception:
            pass
        if icons:
            return f"{txt}\n{''.join(icons)}"
        else:
            return txt

    def draw(self, **opt):
        """draw the component graph

        Arguments: any option accepted by `ecco.cygraphs.Graph`

        Returns: a `Graph` instance that can be directly displayed in a
        Jupyter Notebook
        """
        if len(self) < 2:
            log.print("cannot draw a graph with only one node", "error")
            display(self.nodes)
            return
        try:
            # don't add PCA layout when PCA fails
            pca = self.pca()
            pca *= 80 / pca.abs().max().max()
            pos = {str(i): tuple(r) for i, r in pca.iterrows()}
            layout_extra = {"PCA": pos}
        except Exception:
            layout_extra = {}
            log.print("could not compute PCA, this layout is thus disabled", "warning")
        options = dict(
            nodes_fill_color="size",
            nodes_fill_palette=("red-green/white", "abs"),
            layout_extra=layout_extra,
            nodes_label_str=self._convert_nodes_label,
            edges_label="rules",
        )
        options.update(opt)
        return Graph(self.nodes, self.edges, **options)

    def search(self, *args, prune=False, stutter=False, col="search", **props):
        """search a sequence of properties in component graph

        The graph is first classified against the searched properties
        (see method `classify`) and then paths respecting the given sequence
        are searched.

        Arguments:
         - `prop, prop, ...` (`str`): sequence of properties
         - `...` (`int`): components to be considered
         - `prune` (`bool`): if `True`, remove components that
           match none of the searched properties (default to `False`)
         - `stutter`: allow consecutive properties to be realised by a single
           component
         - `col` (`str`, default: `"search"`): name of the columns to be added
           to `nodes` and `edges` tables

        Returns: a new component graph where searched properties have be split
        appart, with a column `col` in its `nodes` table showing which
        properties in the sequence is validated by each node, and one column
        in the `edges` table for each path found showing which edge is part
        of the path.
        """
        # check arguments
        self._get_args(args, max_props=0)
        if len(props) < 2:
            raise TypeError(
                f"expected at least two properties," f" but {len(props)} given"
            )
        elif len(set(props.values())) != len(props):
            raise TypeError(f"properties must be pairwise distinct")
        # classify nodes and prune
        classes = {f"{n}:{a}": p for n, (a, p) in enumerate(props.items())}
        cg = self.classify("search", *args, **classes)
        scol = cg.nodes["search"]
        if prune:
            drop = list(cg.nodes.index[~scol.map(bool)])
            if drop:
                cg = cg.drop(*drop)
                if cg is None:
                    log.warn("none of the required steps were found")
                    return
                cg.nodes[col] = scol[cg.nodes.index]
        cg.g  # ensure cg._q is populated
        # construct sequences of searched properties
        seq = [[] for i in range(len(props))]

        def getnum(s):
            return int(s.split(":", 1)[0])

        for node, steps in scol.iteritems():
            for s in map(getnum, steps):
                seq[s].append(cg._g[node])
        seq = [s for s in seq if s]
        if len(seq) <= 1:
            log.warn(f"only {len(seq)} non-empty properties," f" no path can be found")
            return cg
        paths = {((None, s),) for s in seq[0]}
        for tgt in seq[1:]:
            new = set()
            for start in paths:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    short = cg.g.get_shortest_paths(start[-1][1], tgt)
                for suite in short:
                    if not suite or (len(suite) == 1 and not stutter):
                        continue
                    new.add(start + tuple(zip(suite, suite[1:])))
            if new:
                paths = new
            else:
                paths = {p + ((None, t),) for p in paths for t in tgt}
        # save found sequences in edges table
        minlen = min(map(len, paths))
        nodes = cg.g.vs["node"]
        edges = {
            tuple((nodes[s], nodes[t]) for s, t in p if s is not None and t is not None)
            for p in paths
            if len(p) == minlen
        }
        for num, edg in enumerate(edges, start=1):
            name = f"{col}#{num}"
            cg.edges[name] = [0] * len(cg.edges.index)
            cg.edges.loc[cg.edges.index.isin(set(edg)), name] = 1
        return cg

    def dump(self):
        # dump ComponentGraph info to dict, as for LTS and Component
        return {"DDD": [], "model": self.model.path}

    def save(self, path):
        """save component graph to `path`

        Arguments:
         - `path` (`str`): file name to be saved to
        """
        d2n = {}
        hdr = []
        for obj in (self, self.lts, *self.components):
            dump = obj.dump()
            for i, d in enumerate(dump["DDD"]):
                dump["DDD"][i] = d2n.setdefault(d, len(d2n))
            hdr.append(dump)
        ddd.ddd_save(path, *d2n, dumps=hdr, compact=self.compact)

    @classmethod
    def load(cls, path):
        """reload a previously saved component graph from `path`

        Arguments:
         - `path` (`str`): file name to load from
        Returns: loaded `ComponentGraph` instance
        """
        headers, ddds = ddd.ddd_load(path)
        dump_cg, dump_lts, *dump_compos = headers["dumps"]
        for dump in (dump_cg, dump_lts, *dump_compos):
            for i, n in enumerate(dump["DDD"]):
                dump["DDD"][i] = ddds[n]
        lts = LTS.load(dump_lts)
        compos = tuple(Component.load(dump, lts) for dump in dump_compos)
        model = load_model(dump_cg["model"])
        cg = cls(compact=headers["compact"], lts=lts, model=model)
        cg.components = compos
        cg._c.update((c.num, c) for c in compos)
        return cg


__extra__ = ["Model", "Palette", "ComponentGraph", "setrel"]
