import math, inspect, base64, mimetypes, tempfile, pathlib, subprocess, io

import ipycytoscape as cy
import pandas as pd
import numpy as np
import ipywidgets as ipw
import igraph as ig
import networkx as nx

from IPython.display import display
from pandas.api.types import is_numeric_dtype
from colour import Color
from math import pi

from . import scm, bqcm, tikz

def maybeColor (c) :
    try :
        return Color(c).hex
    except :
        return None

class Palette (object) :
    palettes = {"red-green" : ["#F00", "#0F0"],
                "green-red" : ["#0F0", "#F00"],
                "red-green/white" : ["#FAA", "#AFA"],
                "green-red/white" : ["#AFA", "#FAA"],
                "black-white" : ["#000", "#FFF"],
                "white-black" : ["#FFF", "#000"],
                "gray" : ["#333", "#DDD"],
                "yarg" : ["#DDD", "#333"],
                "black" : ["#000"],
                "white" : ["#FFF"],
                "hypergraph" : ["#AFA", "#FAA", "#FF8", "#AAF"]}
    def __init__ (self, name, mode="lin", sort=True) :
        self.colors = self.palettes[name]
        self.name = name
        self.mode = mode
        self.sort = sort
    def __repr__ (self) :
        return f"{self.__class__.__name__}({self.name!r}, {self.mode!r}, {self.sort})"
    @property
    def mode (self) :
        return self._mode
    @mode.setter
    def mode (self, value) :
        if value not in ("abs", "lin", "log") :
            raise ValueError(f"invalid palette mode {value!r}")
        self._mode = value
    @property
    def sort (self) :
        return self._sort
    @sort.setter
    def sort (self, value) :
        self._sort = bool(value)
    def _ipython_display_ (self, width=400) :
        w = max(1, width / len(self.colors))
        s = f"background-color: {{col}}; padding: 8px {w}px; color: {{col}};"
        c = "color" if len(self.colors) == 1 else "colors"
        display(ipw.VBox([ipw.HTML(f"<b>{self.name}</b> ({len(self.colors)} {c})"),
                          ipw.HTML("".join(f"<span style={s.format(col=col)!r}></span>"
                                           for col in self.colors))]))
    def mkcolors (self, values, circular="h", linear="sl") :
        src = pd.DataFrame.from_records((Color(c).hsl for c in self.colors),
                                        columns=["h", "s", "l"])
        if self.mode == "abs" :
            values = values.astype(float)
            pal = pd.DataFrame(index=values.index, columns=["h", "s", "l"])
            m, M = values.min(), values.max()
            if m == M :
                return [self.colors[0]] * len(values)
            pal["pos"] = (values.astype(float) - m) * (len(self.colors) - 1.0) / (M - m)
        else :
            unique = pd.Series(values.unique())
            if self.sort :
                unique.sort_values(inplace=True)
            count = len(unique)
            pal = pd.DataFrame(index=range(count), columns=["h", "s", "l"])
            if self.mode == "lin" :
                pal["pos"] = np.linspace(0.0, len(self.colors)-1.0, count, dtype=float)
            else :
                log = np.logspace(0.0, 1.0, count, base=10, dtype=float)
                pal["pos"] = (log - 1.) / 9.
        pal["left"] = np.floor(pal["pos"]).astype(int)
        pal["right"] = np.ceil(pal["pos"]).astype(int)
        P = pal["pos"] = pal["pos"] - pal["left"]
        for col in circular.lower() :
            L = pal["left"].map(src[col].get) * 2*pi
            R = pal["right"].map(src[col].get) * 2*pi
            pal[col] = np.mod(np.arctan2(P * np.sin(R) + (1-P) * np.sin(L),
                                         P * np.cos(R) + (1-P) * np.cos(L)) / (2*pi),
                              1.0)
        for col in linear.lower() :
            L = pal["left"].map(src[col].get)
            R = pal["right"].map(src[col].get)
            pal[col] = P*R + (1-P)*L
        colors = [Color(hsl=row).hex
                  for row in pal[["h","s","l"]].itertuples(index=False)]
        if self.mode == "abs" :
            return colors
        else :
            cmap = dict(zip(unique, colors))
            return [cmap[c] for c in values]

for module in (bqcm, scm) :
    for name in sorted(getattr(module, "__all__")) :
        Palette.palettes[name] = getattr(module, name)

class _Desc (object) :
    def __set_name__ (self, obj, name) :
        if not hasattr(obj, "_opts") :
            obj._opts = {}
        obj._opts[name] = self

class PaletteDesc (_Desc) :
    def __init__ (self, default, mode="lin", sort=True) :
        self.palette = Palette(default, mode, sort)
    def __set_name__ (self, obj, name) :
        super().__set_name__(obj, name)
        base, end = name.rsplit("_", 1)
        assert end == "palette", f"invalid palette attribute name: {name!r}"
        self.name = name
        self.color = f"{base}_color"
    def __get__ (self, obj, type=None) :
        return self.palette
    def __set__ (self, obj, value) :
        if isinstance(value, str) :
            value = [value]
        self.palette = Palette(*value)
        setattr(obj, self.color, None)

class _TableDesc (_Desc) :
    def __init__ (self, table, store, column, default) :
        self.table = table
        self.store = store
        self.column = column
        self.default = default
    def __set_name__ (self, obj, name) :
        super().__set_name__(obj, name)
        if not hasattr(obj, "_ncol") :
            obj._ncol = {}
        if not hasattr(obj, "_ecol") :
            obj._ecol = {}
        if name.startswith("nodes_") :
            obj._ncol[name] = self.column
        elif name.startswith("edges_") :
            obj._ecol[name] = self.column
    def __get__ (self, obj, type=None) :
        store = getattr(obj, self.store)
        if self.column not in store.columns :
            self.__set__(obj, self.default)
        return store[self.column]

class TableColorDesc (_TableDesc) :
    def __init__ (self, table, store, column, default) :
        super().__init__(table, store, column, default)
        self.last = default
    def __set_name__ (self, obj, name) :
        super().__set_name__(obj, name)
        base, end = name.rsplit("_", 1)
        assert end == "color", f"invalid color attribute name: {name!r}"
        self.name = name
        self.palette = f"{base}_palette"
    def __set__ (self, obj, val) :
        store = getattr(obj, self.store)
        table = getattr(obj, self.table)
        if val is None :
            val = self.last
        if isinstance(val, str) and maybeColor(val) is not None :
            obj._legend[self.name] = None
            colors = [Color(val).hex] * len(table)
        elif isinstance(val, str) and val in table.columns :
            obj._legend[self.name] = val
            colors = [maybeColor(c) for c in table[val]]
            if any(c is None for c in colors) :
                palette = getattr(obj, self.palette)
                if not is_numeric_dtype(table[val]) and palette.mode == "abs" :
                    palette.mode = "lin"
                colors = palette.mkcolors(table[val])
        elif (isinstance(val, (list, tuple, pd.Series, np.ndarray))
              and len(val) == len(table)) :
            obj._legend[self.name] = None
            colors = [maybeColor(c) for c in val]
            if any(c is None for c in colors) :
                pos = colors.index(None)
                raise ValueError(f"invalid color {colors[pos]!r}")
        else :
            raise ValueError(f"cannot use {val!r} as color(s)")
        self.last = val
        store[self.column] = colors

class TableNumberDesc (_TableDesc) :
    def __init__ (self, table, store, column, default, mini, maxi, cls=None) :
        if cls is None :
            cls = type(default)
        self.cls = cls
        super().__init__(table, store, column, default)
        assert mini < maxi, "mini should be strictly less than maxi"
        self.mini = cls(mini)
        self.maxi = cls(maxi)
    def _check (self, n) :
        if n < self.mini :
            raise ValueError(f"value should be at least {self.mini}")
        elif n > self.maxi :
            raise ValueError(f"value should be at most {self.maxi}")
    def __set__ (self, obj, val) :
        store = getattr(obj, self.store)
        table = getattr(obj, self.table)
        if isinstance(val, (int, float)) :
            numbers = self.cls(val)
            self._check(numbers)
        elif ((isinstance(val, str) and val in table.columns)
              or (isinstance(val, (list, tuple, pd.Series, np.ndarray))
                  and len(val) == len(table))) :
            if isinstance(val, str) and val in table.columns :
                numbers = table[val].astype(self.cls)
            else :
                numbers = pd.Series(data=val, index=table.index, dtype=self.cls)
            numbers.map(self._check)
        else :
            raise ValueError(f"cannot use {val!r} as {self.cls.__name__}(s)")
        store[self.column] = numbers

class TableTipDesc (_TableDesc) :
    tips = {"triangle", "triangle-tee", "circle-triangle",
            "triangle-cross", "triangle-backcurve", "vee", "tee",
            "square", "circle", "diamond", "chevron", "none"}
    alias = {">" : "filled-triangle",
             "<" : "filled-triangle",
             "*" : "filled-circle",
             "o" : "hollow-circle",
             "|" : "filled-tee",
             "-" : "none"}
    def __init__ (self, table, store, column, default) :
        super().__init__(table, store, column, default)
        base, end = column.rsplit("-", 1)
        assert end == "shape", f"invalid tip column name: {column!r}"
        self.fill_column = f"{base}-fill"
    def _split (self, spec) :
        spec = self.alias.get(spec, spec)
        if spec.startswith(("filled-", "hollow-")) :
            fill, shape = spec.split("-", 1)
        else :
            fill, shape = "filled", spec
        if shape not in self.tips :
            raise ValueError(f"invalid arrow tip {shape!r}")
        return fill, shape
    def __set__ (self, obj, val) :
        store = getattr(obj, self.store)
        table = getattr(obj, self.table)
        if isinstance(val, str) and val in table.columns :
            fill, shape = list(zip(*table[val].apply(self._split)))
        elif isinstance(val, str) :
            fill, shape = self._split(val)
        elif (isinstance(val, (list, tuple, pd.Series, np.ndarray))
              and len(val) == len(table)) :
            fill, shape = list(zip(map(self._split, val)))
        else :
            raise ValueError(f"cannot use {val!r} as arrow tip(s)")
        store[self.fill_column] = fill
        store[self.column] = shape

class _TableEnumDesc (_TableDesc) :
    values = {}
    alias = {}
    def __init__ (self, table, store, column, default, description) :
        super().__init__(table, store, column, default)
        self.description = description
    def _check (self, spec) :
        val = self.alias.get(spec, spec)
        if val not in self.values :
            raise ValueError(f"invalid {self.description} {spec!r}")
        return val
    def __set__ (self, obj, val) :
        store = getattr(obj, self.store)
        table = getattr(obj, self.table)
        if isinstance(val, str) and val in table.columns :
            shapes = table[val].apply(self._check)
        elif isinstance(val, str) :
            shapes = self._check(val)
        elif (isinstance(val, (list, tuple, pd.Series, np.ndarray))
              and len(val) == len(table)) :
            shapes = list(map(self._check, val))
        else :
            raise ValueError(f"cannot use {val!r} as arrow tip(s)")
        store[self.column] = shapes

class TableShapeDesc (_TableEnumDesc) :
    values = {"ellipse", "triangle", "round-triangle", "rectangle",
              "round-rectangle", "bottom-round-rectangle",
              "cut-rectangle", "barrel", "rhomboid", "diamond",
              "round-diamond", "pentagon", "round-pentagon",
              "hexagon", "round-hexagon", "concave-hexagon",
              "heptagon", "round-heptagon", "octagon", "round-octagon",
              "star", "tag", "round-tag", "vee"}
    def __init__ (self, table, store, column, default) :
        super().__init__(table, store, column, default, "shape")

class TableCurveDesc (_TableEnumDesc) :
    values = {"bezier", "straight"}
    def __init__ (self, table, store, column, default) :
        super().__init__(table, store, column, default, "curve")

class TableStyleDesc (_TableEnumDesc) :
    values = {"solid", "dotted", "dashed", "double"}
    alias = {"-" : "solid",
             "|" : "solid",
             ":" : "dotted",
             "!" : "dashed",
             "=" : "double"}
    def __init__ (self, table, store, column, default) :
        super().__init__(table, store, column, default, "style")

def _str (txt, **args) :
    return str(txt)

class TableStrDesc (_TableDesc) :
    def __init__ (self, table, store, column, default, convert=_str) :
        super().__init__(table, store, column, default)
        self.convert = convert
    def __set__ (self, obj, val) :
        store = getattr(obj, self.store)
        table = getattr(obj, self.table)
        if isinstance(val, str) and val in table.columns :
            strings = table[val].apply(self.convert)
        elif isinstance(val, str) :
            strings = []
            for idx, row in table.iterrows() :
                if not isinstance(idx, (tuple, list)) :
                    idx = [idx]
                d = dict(zip(table.index.names, idx))
                d.update(row)
                strings.append(self.convert(val.format(**d), **d))
        elif (isinstance(val, (list, tuple, pd.Series, np.ndarray))
              and len(val) == len(table)) :
            strings = list(map(self.convert, val))
        else :
            raise ValueError(f"cannot use {val!r} as string(s)")
        store[self.column] = strings

class GroupDesc (_Desc) :
    def __init__ (self, *members) :
        self.members = tuple(members)
    def __set_name__ (self, obj, name) :
        super().__set_name__(obj, name)
        if not hasattr(obj, "_ncol") :
            obj._ncol = {}
        if not hasattr(obj, "_ecol") :
            obj._ecol = {}
        if name.startswith("nodes_") :
            obj._ncol[name] = self.members
        elif name.startswith("edges_") :
            obj._ecol[name] = self.members
    def __get__ (self, obj, type=None) :
        return tuple(getattr(obj, m) for m in self.members)
    def __set__ (self, obj, val) :
        for m in self.members :
            setattr(obj, m, val)

class TableValignDesc (_TableEnumDesc) :
    values = {"top", "center", "bottom"}
    def __init__ (self, table, store, column, default) :
        super().__init__(table, store, column, default, "label align")

class TableHalignDesc (_TableEnumDesc) :
    values = {"left", "center", "right"}
    def __init__ (self, table, store, column, default) :
        super().__init__(table, store, column, default, "label align")

class TableAlignDesc (_Desc) :
    g2a = {"n" : "top",
           "s" : "bottom",
           "m" : "center",
           "c" : "center",
           "e" : "right",
           "w" : "left"}
    h = "emcw"
    v = "nmcs"
    def __init__ (self, h, v) :
        self.hopt = h
        self.vopt = v
        self.a2g = {v : k for k, v in self.g2a.items()}
    def __get__ (self, obj, type=None) :
        return getattr(obj, self.hopt), getattr(obj, self.vopt)
    def valid (self, txt) :
        if not isinstance(txt, str) :
            return False
        h = v = None
        for c in txt.lower() :
            if h is None and c in self.h :
                h = c
            elif v is None and c in self.v :
                v = c
            else :
                return False
        return h is not None or v is not None
    def split (self, txt) :
        h = v = None
        for c in txt.lower() :
            if h is None and c in self.h :
                h = self.g2a[c]
            elif v is None and c in self.v :
                v = self.g2a[c]
            else :
                raise ValueError(f"invalid alignment {txt!r}")
        return h, v
    def __set__ (self, obj, val) :
        table = getattr(obj, obj._opts[self.hopt].table)
        if ((isinstance(val, str) and val in table.columns)
              or (isinstance(val, (list, tuple, pd.Series, np.ndarray))
                  and len(val) == len(table))) :
            if isinstance(val, str) :
                align = pd.DataFrame(table[val].map(self.split).tolist(),
                                     columns=["h", "v"])
            else :
                align = pd.DataFrame(pd.Series(val).map(self.split).tolist(),
                                     columns=["h", "v"])
            setattr(obj, self.hopt, align["h"].tolist())
            setattr(obj, self.vopt, align["v"].tolist())
        elif isinstance(val, str) :
            h, v = self.split(val)
            setattr(obj, self.hopt, h or "center")
            setattr(obj, self.vopt, v or "center")
        else :
            raise ValueError(f"cannot use {val!r} as alignment(s)")

class TableWrapDesc (_TableEnumDesc) :
    values = {"wrap", "none", "ellipsis"}
    def __init__ (self, table, store, column, default) :
        super().__init__(table, store, column, default, "text wrap")

class TableAngleDesc (_TableDesc) :
    def __set__ (self, obj, val) :
        store = getattr(obj, self.store)
        table = getattr(obj, self.table)
        if isinstance(val, str) and val in ("autorotate", "auto") :
            angles = "autorotate"
        elif isinstance(val, str) and val in table.columns :
            angles = table[val].astype(float)
        elif isinstance(val, (int, float)) :
            angles = float(val)
        elif (isinstance(val, (list, tuple, pd.Series, np.ndarray))
              and len(val) == len(table)) :
            angles = list(map(float, val))
        else :
            raise ValueError(f"cannot use {val!r} as angle(s)")
        store[self.column] = angles

class LayoutDesc (_Desc) :
    values = {"random", "grid", "circle", "concentric",
              "breadthfirst", "cose", "dagre", "preset",
              "gv-dot", "gv-neato", "gv-twopi", "gv-circo", "gv-fdp"}
              # "ig-large", "ig-drl", "ig-fr", "ig-kk",
              # "nx-shell", "nx-spring", "nx-spectral", "nx-spiral"}
    alias = {}
    def __init__ (self, default) :
        self.value, self.args = self._check(default)
    def _check (self, spec) :
        if isinstance(spec, (tuple, list)) :
            spec, args = spec
        else :
            args = {}
        val = self.alias.get(spec, spec)
        if val not in self.values :
            raise ValueError(f"invalid graph layout {spec!r}")
        return val, dict(args)
    def __get__ (self, obj, type=None) :
        return self.value, self.args
    def __set__ (self, obj, val) :
        self.value, self.args = self._check(val)
        val, args = self.get_layout(self.value, obj)
        if args is None :
            args = self.args
        else :
            args.update(self.args)
        obj._cy.set_layout(name=val, **args)
    def get_layout (self, layout, graph) :
        for node in graph._cy.graph.nodes :
            node.position = {}
        if layout == "grid" :
            return "grid", {"rows" : math.ceil(math.sqrt(len(graph.nodes)))}
        elif layout.startswith(("nx-", "gv-", "ig-", "xx-")) :
            if layout.startswith("xx-") :
                pos = graph.layout_extra[layout[3:]]
                if callable(pos) :
                    pos = pos()
                pos = {str(k) : v for k, v in pos.items()}
            else :
                algo = layout[3:]
                n = nx.DiGraph()
                for nid in graph.nodes.index :
                    if layout.startswith("gv-") and nid in graph.groups :
                        n.add_node(f"cluster_{nid}",
                                   original_nodes=graph.groups[nid])
                    else :
                        n.add_node(nid)
                n.add_edges_from(graph.edges.index)
                if layout.startswith("nx-") :
                    fun = getattr(nx, f"{algo}_layout")
                    pos = fun(n, scale=80)
                elif layout.startswith("gv-") :
                    pos = nx.nx_pydot.pydot_layout(n, prog=algo)
                    pos = {k[8:] if k.startswith("cluster_") and k[8:] in graph.groups
                           else k : (x, -y) for k, (x, y) in pos.items()}
                else :
                    g = ig.Graph.from_networkx(n)
                    l = g.layout(algo)
                    l.fit_into((150, 150))
                    pos = dict(zip(graph.nodes.index, l.coords))
            for node in graph._cy.graph.nodes :
                x, y = pos[node.data["id"]]
                node.position = {"x" : x, "y" : y}
            return "preset", {"fit" : True}
        else :
            return layout, None

def state_str (txt, **args) :
    txt = str(txt)
    if not txt :
        return ""
    icons = []
    try :
        if "has_init" in args["topo"] :
            icons.append("▶")
        if "has_scc" in args["topo"] or "is_hull" in args["topo"] :
            icons.append("⏺")
        if "has_dead" in args["topo"] or "is_dead" in args["topo"] :
            icons.append("⏹")
    except :
        pass
    if icons :
        return f"{txt}\n{''.join(icons)}"
    else :
        return txt

_export_html = "".join('''<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
<a download="{filename}" href="data:{mimetype};charset=utf-8;base64;,{payload}">
Download {filename}
</a>
</body>
</html>'''.splitlines())

class Graph (object) :
    """An interactive graph that can be displayed in Jupyter

    Arguments:
     - `nodes` (`pandas.DataFrame`): nodes table whose index is called `node`
     - `edges` (`pandas.DataFrame`): edges table whose index has two columns
       called `src` and |dst`

    Graph options:
     - `layout` (`str` or `tuple[str,dict]`): a layout algorithm, or a pair
       `(algo, options)` where options are the parameters for the chosen algorithm
     - `layout_extra` (`dict`): additional layout algorithms to be added to the UI,
       mapping names to functions or static positions

    Most nodes and edges options may be given as a single value that
    is applied to every nodes/edges, or as the name of a column in
    `nodes`/`edges` containing the values to apply to each node/edge,
    or as an array of values whose length is the number of
    nodes/edges. When a column or an array is provided, it must
    contain valid values for the expected property, except for the
    color that can be computed in every case.

    Colors are obtained from a pair of options `..._color` and
    `..._palette` as follows:
     - a named color (`"red"`, etc.) is applied to evey nodes/edges
     - a HTML colors (`"#C0FFEE"`, etc.) also
     - a column/array of values is used to compute a color for each unique value
       according to the palette (see below)

    A palette is provided as either:
     - a name (`str`) that maps to a series of colors, see `Palette.palettes`
     - an optional mode (one of `"lin"` (default), `"log"`, or `"abs"`) that
       specifies how colors in the palette are interpolated
       - when `mode="lin"` the set of unique values is linearly distributed
         onto the list of colors in the palette, and unique colors are computed
         by interpolation. Consider for example a palette of 4 colors
         with 5 unique values, they can be distributed as:
         ```
         palette:   "#AFA"   "#FAA"   "#FF8"   "#AAF"
         values:    a       b       c       d       e
         ```
         The color for `a` will be `"#AFA"`, the color for `e` will be `"#AAF"`,
         the color for `c` will be 50% `"#FAA"` and 50% `"#FF8"`, the color for `b`
         will be mostly `"#FFA"` with a touch of `"#AFA"`, etc.
       - when `mode="log"` the same principle os applied but the values are
         distributed on a logarithmic scale (values are close one to each other
         at the beginning and get farther as we move to the right)
       - when `mode="abs"` the values must be numbers and are directly used as the
         position within the palette (after being scaled to the appropriate range)
     - an optional `bool` (default: `True`) that specifies if values are sorted
       before to generate colors for them

    Nodes options:
     - `nodes_shape`: shape as listed in `TableShapeDesc.values`
     - `nodes_width`: width as `int` or `float`
     - `nodes_height`: height as `int` or `float`
     - `nodes_size`: both width and eight, as `int` or `float`
     - `node_fill_color`: background color
     - `nodes_fill_palette`: palette for background color
     - `nodes_fill_opacity`: background opacity (`float` between `0.0` and `1.0`)
     - `nodes_draw_color`: color for the border of nodes
     - `nodes_draw_palette`: palette for `nodes_draw_color`
     - `nodes_draw_style`: line style for the border of nodes,
       as listed in `TableStyleDesc.values`
     - `nodes_draw_width`: line width for the border of nodes

    Node labels options:
     - `nodes_label`: text drawn onto nodes
     - `nodes_label_wrap`: how labels text is wrapped,
       as listed in `TableWrapDesc.values`
     - `nodes_label_size`: size of labels
     - `nodes_label_halign`: horizontal placement of labels wrt nodes,
       as listed in `TableHalignDesc.values`
     - `nodes_label_valign`: vertical placement of labels wrt nodes,
       as listed in `TableValignDesc.values`
     - `node_label_align`: both horizontal and vertical placement of labels wrt nodes,
       given as a one- ot two-chars `str` combining:
        - `"n"`: north
        - `"s"`: south
        - `"e"`: east
        - `"w"`: west
        - `"c"` or `"m"`: center or middle (they are synonymous)
       If only one align is given, they other default to `"m"`, for instance,
       `"n"` is equivalent to `"nm"` (or `"nc"`). If the first letter is `"c"`
       or `"m"`, it will be used for horizontal alignment, so `"cn"` will fail
       because it specifies twice horizontal alignement. To avoid this, start with
       a character in `"nsew"` that is tight to one direction.
     - `nodes_label_angle`: the rotation of labels, in degrees
     - `nodes_label_outline_width`: width of the outline around node labels
     - `nodes_label_outline_color`: color of the outline around node labels
     - `nodes_label_outline_palette`: palette for `nodes_label_outline_color`
     - `nodes_label_outline_opacity`: opacity of label outline

    Edges options:
     - `edges_curve`:
        - when `"bezier"`, two opposite edges will be bent automatically to avoid
          their overlapping, which is desired when they have labels
        - when `straight`, two opposite edges will overlap, which may be desired
          when they have no labels
     - `edges_draw_color`, `edges_draw_palette`, `edges_draw_style`,
       and `edges_draw_width`: like `nodes_draw_...` but for edges (which means that
       information is taken from table `edges` is column names are used)
     - `edges_label`, `edges_label_angle`, `edges_label_outline_width`,
       `edges_label_outline_color`, `edges_label_outline_palette`,
       `edges_label_outline_opacity`: like `nodes_label...` but for edges.
       Note that `edges_label_angle="auto"` allows to automatically rotate the label
       following the slope of each edge.

    Edges tips:
     - `edges_tip_scale`: scaling factor that applies to all edge tips, this is
       a single value (default `0.6`) and cannot be a column/array of values
     - `edges_target_tip`: shape of edge tip at target node, as listed in
       `TableTipDesc.values`. Values may be prefixed with `"filled-"` or `"hollow-"`
       to fill or not the tip drawing. `TableTipDesc.alias` provides short names
       for some edge tips.
     - `edges_target_color` and `edges_target_palette`: color and palette of edge tips
       at target node
     - `edges_source_tip`, `edges_source_color`, `edges_source_palette`: like
       `edges_source_...` but for the tips as edges source

    Additional properties gather other properties, so assigned them will actually
    assign several other properties, and reading them will return a tuple of
    other properties:
     - `nodes_color`: `nodes_draw_color` and `nodes_fill_color`
     - `nodes_palette`: `nodes_draw_palette` and `nodes_fill_palette`
     - `edges_tip_color`: `edges_target_color` and `edges_source_color`
     - `edges_tip_palette`: `edges_target_palette` and `edges_source_palette`
     - `edges_color`: `edges_draw_color` and `edges_tip_color`
     - `edges_palette`: `edges_draw_palette` and `edges_tip_palette`

    User interface (UI) option `ui` may be given as a `list` of UI elements
    organised as nested lists and dicts:
     - top-level `list` is a vbox
     - a vbox arranges its content into a `ipywidget.VBox`
     - a `list` in a vbox arranges its content into an `ipywidget.HBox`
     - a `dict` in a vbox arranges its content as an `ipywidget.Accordion`
       whose sections are the keys of the dict, each containing a vbox.
     - no other nesting is allowed
     - atomic elements are:
        - the name of an option listed above (except the additional ones at the end)
          to build a widget allowing to tune this option
        - `"graph"`: the interactive graph itself
        - `"reset_view"`: a button to recompute the layout and reset the graph
          display, which is useful when one has zoomed too much and the graph is
          not visible anymore
        - `"select_all"`, `"select_none"`, and `"invert_selection"`: buttons to
          (un)select nodes
        - `"inspect_nodes"` and `"inspect_edges"`: view of `nodes` and `edges`
          tables limited to the selected nodes and the edges connected to them.
          Options `inspect_nodes` and `inspect_edges` may be provided
          (as lists of columns) to restrict the columns displayed here

    All the options presented above, except those related to the building of the UI,
    are also attributes of a `Graph` instance. Which means they can be read or
    assigned to change a graph appearance. In this case, it is mandatory to callable
    method `update` to actually update the display. `Graph` instance also supports
    item assignement to update one option for just one node or edge,
    see `__setitem__` documentation.
    """
    layout = LayoutDesc("cose")
    layout_extra = {}
    # nodes
    nodes_shape = TableShapeDesc("nodes", "_ns", "shape", "round-rectangle")
    nodes_width = TableNumberDesc("nodes", "_ns", "width", 20., 0., 100.)
    nodes_height = TableNumberDesc("nodes", "_ns", "height", 20., 0., 100.)
    nodes_fill_color = TableColorDesc("nodes", "_ns", "background-color", "size")
    nodes_fill_palette = PaletteDesc("red-green/white", "abs")
    nodes_fill_opacity = TableNumberDesc("nodes", "_ns", "background-opacity",
                                         1., 0., 1.)
    nodes_draw_color = TableColorDesc("nodes", "_ns", "border-color", "black")
    nodes_draw_palette = PaletteDesc("black")
    nodes_draw_width = TableNumberDesc("nodes", "_ns", "border-width", 0., 0., 10.)
    nodes_draw_style = TableStyleDesc("nodes", "_ns", "border-style", "|")
    # nodes labels
    nodes_label = TableStrDesc("nodes", "_ns", "label", "{node}", state_str)
    nodes_label_wrap = TableWrapDesc("nodes", "_ns", "text-wrap", "wrap")
    nodes_label_size = TableNumberDesc("nodes", "_ns", "font-size", 8., 0., 40.)
    nodes_label_valign = TableValignDesc("nodes", "_ns", "text-valign", "center")
    nodes_label_halign = TableHalignDesc("nodes", "_ns", "text-halign", "center")
    nodes_label_angle = TableAngleDesc("nodes", "_ns", "text-rotation", 0)
    nodes_label_outline_color = TableColorDesc("nodes", "_ns",
                                               "text-outline-color", "white")
    nodes_label_outline_palette = PaletteDesc("white")
    nodes_label_outline_opacity = TableNumberDesc("nodes", "_ns",
                                                  "text-outline-opacity", 0., 0., 1.)
    nodes_label_outline_width = TableNumberDesc("nodes", "_ns", "text-outline-width",
                                                0., 0., 10.)
    # edges
    edges_curve = TableCurveDesc("edges", "_es", "curve-style", "bezier")
    edges_draw_color = TableColorDesc("edges", "_es", "line-color", "black")
    edges_draw_palette = PaletteDesc("black")
    edges_draw_width = TableNumberDesc("edges", "_es", "width", .6, 0., 10.)
    edges_draw_style = TableStyleDesc("edges", "_es", "line-style", "|")
    # edges labels
    edges_label = TableStrDesc("edges", "_es", "label", "{src}-{dst}")
    edges_label_size = TableNumberDesc("edges", "_es", "font-size", 6., 0., 40.)
    edges_label_angle = TableAngleDesc("edges", "_es", "text-rotation", "autorotate")
    edges_label_outline_color = TableColorDesc("edges", "_es",
                                               "text-outline-color", "white")
    edges_label_outline_palette = PaletteDesc("white")
    edges_label_outline_opacity = TableNumberDesc("edges", "_es",
                                                  "text-outline-opacity", .75, 0., 1.)
    edges_label_outline_width = TableNumberDesc("edges", "_es", "text-outline-width",
                                                3., 0., 10.)
    # tips
    edges_tip_scale = TableNumberDesc("edges", "_es", "arrow-scale", .6, 0., 5.)
    edges_target_tip = TableTipDesc("edges", "_es", "target-arrow-shape", ">")
    edges_target_color = TableColorDesc("edges", "_es", "target-arrow-color", "black")
    edges_target_palette = PaletteDesc("black")
    edges_source_tip = TableTipDesc("edges", "_es", "source-arrow-shape", "-")
    edges_source_color = TableColorDesc("edges", "_es", "source-arrow-color", "black")
    edges_source_palette = PaletteDesc("black")
    # meta-properties
    nodes_size = GroupDesc("nodes_width", "nodes_height")
    nodes_color = GroupDesc("nodes_draw_color", "nodes_fill_color")
    nodes_palette = GroupDesc("nodes_draw_palette", "nodes_fill_palette")
    nodes_label_align = TableAlignDesc("nodes_label_halign", "nodes_label_valign")
    edges_tip_color = GroupDesc("edges_target_color", "edges_source_color")
    edges_tip_palette = GroupDesc("edges_target_palette", "edges_source_palette")
    edges_color = GroupDesc("edges_draw_color", "edges_tip_color")
    edges_palette = GroupDesc("edges_draw_palette", "edges_tip_palette")
    #
    ui = [{"Graph" :
           [["layout", "reset_view"]]},
          {"Nodes" :
           [["nodes_shape", "nodes_size"],
            {"Fill" :
             [["nodes_fill_color", "nodes_fill_palette"],
              ["nodes_fill_opacity"]]},
            {"Draw" :
             [["nodes_draw_color", "nodes_draw_palette"],
              ["nodes_draw_style", "nodes_draw_width"]]},
            {"Labels" :
             [["nodes_label", "nodes_label_size"],
              ["nodes_label_align", "nodes_label_angle"],
              {"Outline" :
               [["nodes_label_outline_color", "nodes_label_outline_palette"],
                ["nodes_label_outline_opacity", "nodes_label_outline_width"]]}]}]},
          {"Edges" :
           [["edges_draw_color", "edges_draw_palette"],
            ["edges_draw_width", "edges_draw_style"],
            {"Labels" :
             [["edges_label", "edges_label_size"],
              ["edges_label_angle"],
              {"Outline" :
               [["edges_label_outline_color", "edges_label_outline_color"],
                ["edges_label_outline_opacity", "edges_label_outline_width"]]}]},
            {"Tips" :
             [["edges_tip_scale"],
              {"Source" :
               [["edges_source_tip"],
                ["edges_source_color", "edges_source_palette"]]},
              {"Target" :
               [["edges_target_tip"],
                ["edges_target_color", "edges_target_palette"]]}]}]},
          "graph",
          {"Inspector" :
           [["select_all", "select_none", "invert_selection"],
            ["inspect_nodes"],
            ["inspect_edges"]]},
          {"Export" :
           [["export"]]}]
    def __init__ (self, nodes, edges, **args) :
        """Build a new `Graph` instance

        Arguments:
         - `nodes`: a `pandas.DataFrame` with nodes
         - `edges`: a `pandas.DataFrame` with edges
         - `**args`: varied options to control graph appearance
        See `Graph` class documentation for details.
        """
        self.nodes = nodes.reset_index()
        self.nodes["node"] = self.nodes["node"].map(str)
        self.nodes.index = self.nodes.pop("node")
        self.edges = edges.reset_index()
        self.edges["src"] = self.edges["src"].map(str)
        self.edges["dst"] = self.edges["dst"].map(str)
        self.edges.index = [self.edges.pop("src"), self.edges.pop("dst")]
        self._ns = pd.DataFrame(index=self.nodes.index)
        self._es = pd.DataFrame(index=self.edges.index)
        self._inspect_nodes_cols = args.pop("inspect_nodes", self.nodes.columns)
        self._inspect_edges_cols = args.pop("inspect_edges", self.edges.columns)
        self.style = [{"selector" : "node.selected",
                       "style" : {"underlay-color" : "black",
                                  "underlay-padding" : 4,
                                  "underlay-opacity" : .15}},
                      {"selector" : "node.hierarchical",
                       "style" : {"label" : "",
                                  "border-width" : .0,
                                  "background-color" : "#F8F8F2"}}]
        self._legend = {}
        self.selection = set()
        self.xy = {}
        self.groups = {}
        hierarchy = args.pop("hierarchy", None)
        if hierarchy :
            self.nodes.sort_values(by=hierarchy, inplace=True, na_position="first")
        n = [{"data" : {"id" : idx}} for idx in self.nodes.index]
        if hierarchy :
            hcol = nodes[hierarchy]
            for node in reversed(n) :
                data = node["data"]
                idx = data["id"]
                if hcol[idx] :
                    p = str(hcol[idx])
                    data["parent"] = p
                    self.groups.setdefault(p, set())
                    self.groups[p].add(idx)
                if idx in self.groups :
                    node["classes"] = " ".join(["hierarchical"]
                                               + node.get("classes", "").split())
        e = [{"data" : {"id" : f"{src}-{dst}",
                        "source" : f"{src}",
                        "target" : f"{dst}"}}
             for src, dst in self.edges.index]
        self._cy = cy.CytoscapeWidget()
        self._cy.graph.add_graph_from_json({"nodes" : n, "edges" : e},
                                           directed=True)
        for opt, desc in self._opts.items() :
            if isinstance(desc, TableStrDesc) and f"{opt}_str" in args :
                desc.convert = args.pop(f"{opt}_str")
            if opt in args :
                setattr(self, opt, args.pop(opt))
            else :
                getattr(self, opt)
        for opt in list(args) :
            if hasattr(self.__class__, opt) :
                setattr(self, opt, args.pop(opt))
        self._inspect_nodes = self._inspect_edges = None
        self._widgets = {}
        self._ui = self._make_ui_vbox(self.ui)
        self._cy.on("node", "click", self._on_node_click)
        if args :
            raise TypeError(f"unexpected arguments: {', '.join(args)}")
    def _ipython_display_ (self) :
        self.update()
        display(self._ui)
    def update (self, layout=False) :
        """Update the display after some options have been changed

        Arguments:
         - `layout` (default: `False`): whether update the graph layout, which is
           only needed when option `layout` has been assigned (or when one wants to
           redraw the graph)
        """
        self._cy.set_style(
            [{"selector": f"node[id='{n}']", "style": r.to_dict()}
             for n, r in self._ns.iterrows()]
            + [{"selector": f"edge[id='{s}-{t}']", "style": r.to_dict()}
               for (s, t), r in self._es.iterrows()]
            + self.style)
        if layout :
            self._cy.relayout()
            self.xy = {}
            self._update_xy()
    def __setitem__ (self, key, val) :
        """Update an option for one specific node or edge

        For instance, to change nodes, we use `"option":node` indexing, and
        for edges, we use `"option":source:target` indexing:

        >>> g = Graph(nodes, edges, nodes_fill_color="red", edges_color="red")
        >>> g["color":1] = "#C0FFEE"    # node 1 is turned to light blue
        >>> g["color":1:2] = "#DEFEC7"  # edge from 1 to 2 is turned to light green
        >>> g.update()                  # update display

        `"options"` above is not prefixed by either `"nodes_"` or `"edges_"` as this
        is deduced from the indexing used. So in the second line, `"color"` is
        expanded to `nodes_color` option, and in the third line, `"color"` is expanded
        to `edges_color` option. (Note that both are compound options that gather
        several other ones.)
        """
        if key.step is None :
            prefix = "nodes"
            colnames = self._ncol
            idx = str(key.stop)
            table = self._ns
        else :
            prefix = "edges"
            colnames = self._ecol
            idx = (str(key.stop), str(key.step))
            table = self._es
        todo = [colnames[f"{prefix}_{key.start}"]]
        while todo :
            c = todo.pop(0)
            if isinstance(c, (list, tuple)) :
                todo.extend(colnames[x] for x in c)
            else :
                table.loc[idx,c] = val
    ##
    ## legend
    ##
    def legend (self, option="nodes_fill", values=None,
                vertical=True, span=1, pos=None) :
        """Build a `Graph` that shows the legend for a given choice of colors

        Optional arguments:
         - `option` (default: `"nodes_fill"`): the color/palette option whose
           legend must be constructed
         - `values`: the values corresponding to the colors, if the colors
           have been computed from a column, this parameter can be left to `None`
           and the values will be retrieved automatically, otherwise, they have to
           be provided explicitly
         - `vertical` (default: `True`): shall the legend be drawn vertically or
           horizontally
         - `span` (default: `1`): number of columns (or lines if `vertical=False`)
           the legend will be spanned on
         - `pos` (defaut: auto from `vertical`): where to place the labels,
           specified as one direction char from `"nsewcm"`

        Returns: an instance of `Graph` arranged as a grid, that shows
        each value next to its corresponding color.
        """
        color = f"{option}_color"
        table = getattr(self, option.split("_")[0])
        if values is None :
            col = self._legend.get(color, None)
            if col is None :
                raise ValueError("unknown color source, use parameter 'values'")
            data = table[col]
        elif isinstance(values, str) and values in table.columns :
            data = table[values]
        elif (isinstance(values, (list, tuple, pd.Series, np.ndarray))
              and len(values) == len(table)) :
            data = pd.DataFrame(list(values), index=table.index)
        else :
            raise ValueError("cannot interpret {values!r} as values")
        data = list(sorted(set(zip(data, getattr(self, color)))))
        nodes = pd.DataFrame.from_records([{"node" : n, "label" : v, "color" : c}
                                           for n, (v, c) in enumerate(data)],
                                          index=["node"])
        edges = pd.DataFrame.from_records([{"src" : n, "dst" : n+1}
                                           for n in range(len(data)-1)],
                                          index=["src", "dst"])
        gridopt = {"fit" : True, "condense" : True, "avoidOverlapPadding" : 5}
        if vertical :
            gridopt["cols"] = span
        else :
            gridopt["rows"] = span
        if pos is None :
            if vertical :
                pos, dx, dy = "e", 5., 0.
            else :
                pos, dx, dy = "s", 0., 5.
        elif pos in ("m", "c") :
            dx = dy = 0.
        elif pos == "s" :
            dx, dy = 0., 5.
        elif pos == "e" :
            dx, dy = 5., 0.
        elif pos == "n" :
            dx, dy = 0., -5.
        elif pos == "w" :
            dx, dy = -5., 0.
        else :
            raise ValueError(f"unexpected value for pos: {pos!r}")
        g = Graph(nodes, edges,
                  nodes_label="label",
                  nodes_label_align=pos,
                  nodes_fill_color="color",
                  nodes_shape="rectangle",
                  nodes_size=20.,
                  edges_label="",
                  edges_target_tip="-",
                  edges_source_tip="-",
                  edges_draw_width=0,
                  layout=("grid", gridopt),
                  ui=["graph"])
        g._ns["text-margin-x"] = dx
        g._ns["text-margin-y"] = dy
        g._cy.graph.nodes.sort(key=lambda n : int(n.data["id"]))
        g._cy.relayout()
        return g
    ##
    ## ui
    ##
    def _select_node (self, node) :
        node.classes = " ".join(node.classes.split() + ["selected"])
        self.selection.add(node.data["id"])
    def _unselect_node (self, node) :
        node.classes = " ".join(c for c in node.classes.split() if c != "selected")
        self.selection.discard(node.data["id"])
    def _toggle_select_node (self, node) :
        classes = node.classes.split()
        if "selected" in classes :
            node.classes = " ".join(c for c in classes if c != "selected")
            self.selection.discard(node.data["id"])
        else :
            node.classes = " ".join(classes + ["selected"])
            self.selection.add(node.data["id"])
    def _update_inspector (self) :
        if self._inspect_nodes is not None :
            self._inspect_nodes.clear_output()
            if self.selection :
                with self._inspect_nodes :
                    sub = self.nodes[self.nodes.index.isin(self.selection)]
                    display(sub[self._inspect_nodes_cols])
        if self._inspect_edges is not None :
            self._inspect_edges.clear_output()
            if self.selection :
                with self._inspect_edges :
                    def insel (idx) :
                        return idx[0] in self.selection or idx[1] in self.selection
                    sub = self.edges[self.edges.index.map(insel)]
                    display(sub[self._inspect_edges_cols])
    def _on_node_click (self, event) :
        nid = event["data"]["id"]
        found = [node for node in self._cy.graph.nodes if nid == node.data["id"]]
        if not found :
            return
        self._toggle_select_node(found[0])
        self._update_inspector()
    def _make_ui_vbox (self, items) :
        children = []
        for item in items :
            if isinstance(item, str) :
                children.append(self._make_ui_widget(item))
            elif isinstance(item, list) :
                children.append(self._make_ui_hbox(item))
            elif isinstance(item, dict) :
                children.append(self._make_ui_accordion(item))
            else :
                raise ValueError(f"unexpected UI spec: {item!r}")
        return ipw.VBox(children)
    def _make_ui_hbox (self, items) :
        children = []
        for item in items :
            if isinstance(item, str) :
                children.append(self._make_ui_widget(item))
            else :
                raise ValueError(f"unexpected UI spec: {item!r}")
        return ipw.HBox(children)
    def _make_ui_accordion (self, items) :
        children = []
        titles = []
        for k, v in items.items() :
            titles.append(k)
            if isinstance(v, str) :
                children.append(self._make_ui_widget(v))
            elif isinstance(v, list) :
                children.append(self._make_ui_vbox(v))
            elif isinstance(v, dict) :
                children.append(self._make_ui_accordion(v))
            else :
                raise ValueError(f"unexpected UI spec: {v!r}")
        acc = ipw.Accordion(children, selected_index=None)
        for i, t in enumerate(titles) :
            acc.set_title(i, t)
        return acc
    def _make_ui_widget (self, name) :
        opt = self._opts.get(name, None)
        make = getattr(self, f"_make_ui_widget_{name}", None)
        if make is not None :
            self._widgets[name] = w = make(name, opt)
            return w
        for cls in inspect.getmro(opt.__class__) :
            make = getattr(self, f"_make_ui_widget_{cls.__name__}", None)
            if make is not None :
                self._widgets[name] = w = make(name, opt)
                return w
        return ipw.Label(f"[{name}]")
    def _make_ui_widget_graph (self, name, opt) :
        return self._cy
    def _make_ui_widget_inspect_nodes (self, name, opt) :
        self._inspect_nodes = ipw.Output()
        return self._inspect_nodes
    def _make_ui_widget_inspect_edges (self, name, opt) :
        self._inspect_edges = ipw.Output()
        return self._inspect_edges
    def _make_ui_widget_reset_view (self, name, opt) :
        button = ipw.Button(description="reset layout/view")
        def on_click (event) :
            self.update(True)
        button.on_click(on_click)
        return button
    def _make_ui_widget_select_all (self, name, opt) :
        button = ipw.Button(description="select all nodes")
        def on_click (event) :
            for node in self._cy.graph.nodes :
                self._select_node(node)
            self._update_inspector()
        button.on_click(on_click)
        return button
    def _make_ui_widget_select_none (self, name, opt) :
        button = ipw.Button(description="unselect all nodes")
        def on_click (event) :
            for node in self._cy.graph.nodes :
                self._unselect_node(node)
            self._update_inspector()
        button.on_click(on_click)
        return button
    def _make_ui_widget_invert_selection (self, name, opt) :
        button = ipw.Button(description="invert nodes selection")
        def on_click (event) :
            for node in self._cy.graph.nodes :
                self._toggle_select_node(node)
            self._update_inspector()
        button.on_click(on_click)
        return button
    def _make_ui_widget_LayoutDesc (self, name, opt) :
        default, _ = getattr(self, name)
        options = list(sorted((opt.values | set(f"xx-{x}" for x in self.layout_extra))
                              - {"preset"}))
        libs = {"gv" : "GraphViz",
                "nx" : "NetworkX",
                "ig" : "igraph",
                "xx" : "extra"}
        for i, o in enumerate(options) :
            if o.startswith(("nx-", "gv-", "ig-", "xx-")) :
                code, algo = o.split("-", 1)
                options[i] = (f"{algo} ({libs[code]})", o)
            else :
                options[i] = (o, o)
        drop = ipw.Dropdown(options=options,
                            value=default,
                            description=name.split("_")[-1])
        def on_change (event) :
            value, args = opt.get_layout(event["new"], self)
            if args is None :
                setattr(self, name, value)
            else :
                setattr(self, name, (value, args))
            self.update(True)
        drop.observe(on_change, names="value")
        return drop
    def _make_ui_widget__TableEnumDesc (self, name, opt) :
        default = getattr(self, name)
        unique = default.unique()
        if len(unique) == 1 :
            default = unique[0]
        table = getattr(self, opt.table)
        allowed = set(opt.values) | set(opt.alias)
        columns = [col for col in table.columns if table[col].isin(allowed).all()]
        drop = ipw.Dropdown(options=["(default)"] + columns,
                            value="(default)",
                            description=name.split("_")[-1])
        def on_change (event) :
            choice = event["new"]
            if choice == "(default)" :
                setattr(self, name, default)
            else :
                setattr(self, name, choice)
            self.update()
        drop.observe(on_change, names="value")
        return drop
    def _make_ui_widget_PaletteDesc (self, name, opt) :
        palette = getattr(self, name)
        drop_name = ipw.Dropdown(options=list(sorted(Palette.palettes, key=str.lower)),
                                 value=palette.name,
                                 description="palette")
        drop_mode = ipw.Dropdown(options=[("absolute", "abs"),
                                          ("linear", "lin"),
                                          ("logarithmic", "log")],
                                 value=palette.mode)
        drop_sort = ipw.Checkbox(value=palette.sort,
                                 description="sort values",
                                 indent=False)
        def on_change (event) :
            setattr(self, name, (drop_name.value, drop_mode.value, drop_sort.value))
            self.update()
        drop_name.observe(on_change, names="value")
        drop_mode.observe(on_change, names="value")
        drop_sort.observe(on_change, names="value")
        return ipw.HBox([drop_name, drop_mode, drop_sort])
    def _make_ui_widget_TableNumberDesc (self, name, opt) :
        table = getattr(self, opt.table)
        default = opt.default
        columns = [col for col in table.columns if is_numeric_dtype(table[col])]
        drop = ipw.Dropdown(options=["(default)", "(value)"] + columns,
                            value="(default)",
                            description=name.split("_")[-1])
        if opt.cls is int :
            entry = ipw.BoundedIntText(value=opt.default,
                                       min=opt.mini,
                                       max=opt.maxi,
                                       step=max(1, (opt.maxi - opt.mini) / 100),
                                       disabled=True)
        else :
            entry = ipw.BoundedFloatText(value=opt.default,
                                         min=opt.mini,
                                         max=opt.maxi,
                                         step=(opt.maxi - opt.mini) / 100.,
                                         disabled=True)
        def drop_change (event) :
            choice = event["new"]
            if choice == "(default)" :
                entry.disabled = True
                setattr(self, name, default)
            elif choice == "(value)" :
                entry.disabled = False
                setattr(self, name, entry.value)
            else :
                entry.disabled = True
                setattr(self, name, choice)
            self.update()
        drop.observe(drop_change, names="value")
        def entry_change (event) :
            setattr(self, name, event["new"])
            self.update()
        entry.observe(entry_change, names="value")
        return ipw.HBox([drop, entry])
    def _make_ui_widget_nodes_size (self, name, opt) :
        return self._make_ui_widget_TableNumberDesc(name, self._opts[opt.members[0]])
    def _make_ui_widget_TableColorDesc (self, name, opt) :
        table = getattr(self, opt.table)
        default = opt.default
        palette = name.rsplit("_", 1)[0] + "_palette"
        drop = ipw.Dropdown(options=["(default)"] + list(sorted(table.columns,
                                                                key=str.lower)),
                            value="(default)",
                            description="color")
        def on_change (event) :
            choice = event["new"]
            if choice == "(default)" :
                setattr(self, name, default)
            else :
                setattr(self, name, choice)
            pal = self._widgets.get(palette, None)
            if pal is not None :
                pal.children[1].value = getattr(self, palette).mode
            self.update()
        drop.observe(on_change, names="value")
        return drop
    def _make_ui_widget_TableStrDesc (self, name, opt) :
        table = getattr(self, opt.table)
        default = opt.default
        drop = ipw.Dropdown(options=["(default)"] + list(sorted(table.columns,
                                                                key=str.lower)),
                            value="(default)",
                            description=name.split("_")[-1])
        def on_change (event) :
            choice = event["new"]
            if choice == "(default)" :
                setattr(self, name, default)
            else :
                setattr(self, name, choice)
            self.update()
        drop.observe(on_change, names="value")
        return drop
    def _make_ui_widget_nodes_label_align (self, name, opt) :
        table = getattr(self, self._opts[opt.hopt].table)
        h, v = getattr(self, name)
        default = h.map(opt.a2g.get) + v.map(opt.a2g.get)
        options = ["(default)"]
        for col in sorted(table.columns, key=str.lower) :
            if table[col].apply(opt.valid).all() :
                options.append(col)
        drop = ipw.Dropdown(options=options,
                            value="(default)",
                            description=name.split("_")[-1])
        def on_change (event) :
            choice = event["new"]
            if choice == "(default)" :
                setattr(self, name, default)
            else :
                setattr(self, name, choice)
            self.update()
        drop.observe(on_change, names="value")
        return drop
    def _make_ui_widget_TableAngleDesc (self, name, opt) :
        table = getattr(self, opt.table)
        default = opt.default
        options = ["(default)", "(auto)", "(value)"]
        options.extend(col for col in table.columns if is_numeric_dtype(table[col]))
        drop = ipw.Dropdown(options=options,
                            value="(default)",
                            description=name.split("_")[-1])
        entry = ipw.IntText(value=0, step=1, disabled=True)
        def drop_change (event) :
            choice = event["new"]
            if choice == "(value)" :
                entry.disabled = False
                setattr(self, name, (float(entry.value) % 360) * pi / 180)
            elif choice == "(default)" :
                setattr(self, name, default)
                entry.disabled = True
            elif choice == "(auto)" :
                setattr(self, name, "autorotate")
                entry.disabled = True
            else :
                setattr(self, name, (table[choice].astype(float) % 360) * pi / 180)
            self.update()
        drop.observe(drop_change, names="value")
        def entry_change (event) :
            setattr(self, name, (float(event["new"]) % 360) * pi / 180)
            self.update()
        entry.observe(entry_change, names="value")
        return ipw.HBox([drop, entry])
    def _make_ui_widget_TableTipDesc (self, name, opt) :
        table = getattr(self, opt.table)
        default = opt.default
        allowed = (set(opt.tips)
                   | {f"hollow-{t}" for t in opt.tips}
                   | {f"filled-{t}" for t in opt.tips}
                   | set(opt.alias))
        options = ["(default)", "(value)"]
        for col in sorted(table.columns, key=str.lower) :
            if table[col].isin(allowed).all() :
                options.append(col)
        drop = ipw.Dropdown(options=options,
                            value="(default)",
                            description=name.split("_")[-1])
        tips = ([a for a in opt.alias if a != "<"]
                + list(sorted(opt.tips))
                + [f"hollow-{t}" for t in sorted(opt.tips) if t != "none"])
        entry = ipw.Dropdown(options=tips,
                             value=default,
                             disabled=True)
        def drop_change (event) :
            choice = event["new"]
            if choice == "(default)" :
                entry.disabled = True
                setattr(self, name, default)
            elif choice == "(value)" :
                entry.disabled = False
                setattr(self, name, entry.value)
            else :
                entry.disabled = True
                setattr(self, name, table[choice])
            self.update()
        drop.observe(drop_change, names="value")
        def entry_change (event) :
            setattr(self, name, event["new"])
            self.update()
        entry.observe(entry_change, names="value")
        return ipw.HBox([drop, entry])
    def _make_ui_widget_export (self, name, opt) :
        progress = ipw.IntProgress(value=0,
                                   min=0,
                                   max=len(self.nodes),
                                   step=1,
                                   description="captured:",
                                   orientation="horizontal")
        choice = ipw.Dropdown(options=[("PDF", "pdf"),
                                       ("TikZ", "tikz"),
                                       ("LaTeX", "tex"),
                                       ("PNG", "png")],
                              value="pdf",
                              description="format",
                              disabled=False)
        button = ipw.Button(description="export",
                            disabled=True)
        output = ipw.Output()
        def update () :
            progress.value = len(self.xy)
            button.disabled = (len(self.xy) != len(self.nodes))
        self._update_xy = update
        def cb (event) :
            self.xy[event["data"]["id"]] = (event["position"]["x"],
                                            event["position"]["y"])
            update()
        self._cy.on("node", "mousemove", cb)
        def on_click (event) :
            output.clear_output()
            data = base64.b64encode(self._export(choice.value)).decode()
            path = f"graph.{choice.value}"
            mime = mimetypes.guess_type(path, False)[0] or "text/plain"
            html = _export_html.format(filename=path,
                                       payload=data,
                                       mimetype=mime)
            with output :
                display(ipw.HTML(html))
        button.on_click(on_click)
        return ipw.VBox([ipw.Label("Move mouse over nodes to capture their positions."
                                   " When progress bar is full, click 'export'."),
                         ipw.HBox([progress, choice, button]),
                         output])
    def _update_xy (self) :
        pass
    ##
    ## export
    ##
    def _export (self, fmt) :
        handler = getattr(self, f"_export_{fmt}", None)
        if len(self.xy) != len(self.nodes) :
            raise ValueError("missing node positions")
        elif handler is None :
            raise ValueError(f"unsupported export format {fmt!r}")
        nodes = self._ns.copy()
        nodes["x"] = nodes.index.map(lambda n : self.xy[n][0])
        nodes["y"] = nodes.index.map(lambda n : self.xy[n][1])
        edges = self._es.copy()
        data = handler(nodes, edges)
        if isinstance(data, str) :
            return data.encode("utf-8")
        else :
            return data
    def _export_tikz (self, nodes, edges) :
        return tikz.tikz(nodes, edges)
    def _export_tex (self, nodes, edges) :
        return tikz.latex(nodes, edges)
    def _export_pdf (self, nodes, edges) :
        return self._export_image(nodes, edges, "pdf")
    def _export_png (self, nodes, edges) :
        return self._export_image(nodes, edges, "png")
    def _export_image (self, nodes, edges, fmt) :
        with tempfile.TemporaryDirectory() as tmp :
            with (pathlib.Path(tmp) / "graph.tex").open("w") as tex :
                tex.write(self._export_tex(nodes, edges))
            try :
                subprocess.check_output(["lualatex", "-interaction=batchmode",
                                         "graph.tex"],
                                        stdin=subprocess.DEVNULL,
                                        stderr=subprocess.STDOUT,
                                        encoding="utf-8", errors="replace",
                                        cwd=tmp)
            except subprocess.CalledProcessError as err :
                err.tex = (pathlib.Path(tmp) / "graph.tex").read_text()
                err.log = (pathlib.Path(tmp) / "graph.log").read_text()
                raise err
            output = f"graph.{fmt}"
            if fmt != "pdf" :
                subprocess.check_output(["pdftoppm", "-singlefile",
                                         f"-{fmt}", "-r", "300",
                                         "graph.pdf", "graph"],
                                        stdin=subprocess.DEVNULL,
                                        stderr=subprocess.STDOUT,
                                        encoding="utf-8", errors="replace",
                                        cwd=tmp)
            with (pathlib.Path(tmp) / output).open("rb") as img :
                return img.read()
