import functools, inspect, pathlib
import networkx as nx
import pandas as pd
import numpy as np
import bqplot as bq
import ipywidgets as ipw
from IPython.display import display, clear_output
from math import pi
from colour import Color
from . import Record, cached_property, scm, bqcm
from .ui import getopt, now

##
## palettes and colors
##

class Palette (Record) :
    _fields = ["scheme"]
    _options = ["count"]
    palettes = {}
    DEFAULT_PALETTE = "RGW"
    def __init__ (self, scheme, count=None, **options) :
        colors = self.palettes[scheme]
        super().__init__(scheme, count or len(colors))
        self.colors = self.mkpal(colors, self.count)
        self.opt = opt = getopt(options,
                                fig_title=self.scheme,
                                fig_width=960,
                                fig_height=150)
        data = np.arange(0, len(self.colors))
        cs = bq.ColorScale(colors=self.colors)
        hm = bq.HeatMap(color=np.array([data, data]), scales={"color" : cs})
        self.figure = bq.Figure(marks=[hm],
                                layout=ipw.Layout(width="%spx" % opt.fig["width"],
                                                  height="%spx" % opt.fig["height"]),
                                padding_x=0.0, padding_y=0.0,
                                fig_margin={"top" : 60, "bottom" : 0,
                                            "left" : 0, "right" : 0},
                                title=opt.fig["title"])
        self.vbox = ipw.VBox([self.figure, bq.Toolbar(figure=self.figure)])
    def _ipython_display_ (self) :
        display(self.vbox)
    def save (self, path=None) :
        if path is None :
            path = "%s-%s.png" % (self.scheme, now())
        fmt = pathlib.Path(path).suffix.lstrip(".").lower()
        if fmt == "png" :
            self.figure.save_png(str(path))
        elif fmt == "svg" :
            self.figure.save_svg(str(path))
        else :
            raise ValueError("unsupported output format %r" % fmt)
    @classmethod
    def mkpal (cls, palette, count=None, circular="h", linear="sl") :
        if isinstance(palette, str) :
            palette = cls.palettes[palette]
        if count is None :
            count = len(palette)
        if len(palette) == count :
            return palette
        src = pd.DataFrame.from_records((Color(c).hsl for c in palette),
                                        columns=["h", "s", "l"])
        pal = pd.DataFrame(index=range(count), columns=["h", "s", "l"])
        pal["pos"] = np.linspace(0.0, len(palette)-1.0, count, dtype=float)
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
        return [Color(hsl=row).hex_l for row in
                pal[["h","s","l"]].itertuples(index=False)]
    @classmethod
    def hexcolor (cls, color) :
        return Color(color).hex_l

for module in (bqcm, scm) :
    for name in getattr(module, "__all__") :
        Palette.palettes[name] = getattr(module, name)

Palette.palettes["RG"] = ["#FF0000", "#00FF00"]
Palette.palettes["GR"] = ["#00FF00", "#FF0000"]
Palette.palettes["RGW"] = ["#FFAAAA", "#AAFFAA"]
Palette.palettes["GRW"] = ["#AAFFAA", "#FFAAAA"]

for pref in ("Pastel28", "Set28", "Pastel19", "Set19", "Set312") :
    if pref in Palette.palettes :
        Palette.DEFAULT_PALETTE = pref
        break

##
## layout engines
##

layout_engines = {prog : functools.partial(nx.nx_pydot.pydot_layout, prog=prog)
                  for prog in ("dot", "neato", "fdp", "twopi", "circo", "sfdp")}
for name, obj in inspect.getmembers(nx, callable) :
    if not name.endswith("_layout") :
        continue
    sig = inspect.signature(obj)
    try :
        if list(sig.bind(None).arguments) == ["G"] :
            layout_engines[name[:-7]] = obj
    except :
        pass

# discard buggy engines
for name in ["planar", "sfdp"] :
    layout_engines.pop(name, None)

DEFAULT_LAYOUT = "neato"

##
## graphs
##

def to_graph (nodes, nodecol, edges, srccol, dstcol, data=True) :
    g = nx.DiGraph()
    if data :
        g.add_nodes_from((n.pop(nodecol), n)
                         for n in nodes.to_dict(orient="index").values())
        g.add_edges_from((e.pop(srccol), e.pop(dstcol), e)
                         for e in edges.to_dict(orient="index").values())
    else :
        g.add_nodes_from(nodes[nodecol])
        g.add_edges_from(edges[[srccol, dstcol]].itertuples(index=False))
    return g

class CancelUpdate (Exception) :
    pass

def gui_update (*what) :
    def decorator (method) :
        @functools.wraps(method)
        def wrapper (self, *l, **k) :
            try :
                method(self, *l, **k)
            except CancelUpdate :
                return
            widgets = set()
            for w in what :
                widgets.update(getattr(self, "gui_update_" + w)())
            for w in widgets :
                w.send_state()
        return wrapper
    return decorator

class col2num (object) :
    def __init__ (self, data) :
        if isinstance(data, pd.DataFrame) :
            data = data.apply(tuple, axis=1)
        self.c2n = {v : i for i, v in enumerate(sorted(data.unique()))}
    def __call__ (self, row) :
        if isinstance(row, pd.Series) :
            row = tuple(row)
            if len(row) == 1 :
                return self.c2n[row[0]]
            else :
                return self.c2n[row]
        else :
            return self.c2n[row]

NODES_SHAPES = ("circ", "sbox", "rbox", "vbox", "hbox", "vegg", "hegg")

class col2shape (col2num) :
    def __call__ (self, row) :
        return NODES_SHAPES[super().__call__(row) % len(NODES_SHAPES)]

def _to_apply (value) :
    col, fun = value
    if not isinstance(col, list) :
        col = [col]
    return col, fun

def _str (value) :
    return str(value) or " "

class Graph (Record) :
    _fields = ["nodes", "nodes_columns",
               "edges", "edges_columns",
               "nodecol", "srccol", "dstcol",
               "defaults"]
    def __init__ (self, nodes, edges,
                  nodecol="node", srccol="src", dstcol="dst", defaults={},
                  **options) :
        ncols = list(nodes.columns)
        ecols = list(edges.columns)
        super().__init__(nodes.copy(), ncols,
                         edges.copy(), ecols,
                         nodecol, srccol, dstcol,
                         defaults)
        self.nodes.reset_index(inplace=True)
        self.edges.reset_index(inplace=True)
        self.opt = opt = getopt(options, self.defaults,
                                gui_main=[["layout", "color", "palette"],
                                          ["label", "shape", "size"],
                                          "figure",
                                          "toolbar",
                                          "inspect"],
                                fig_width=960,
                                fig_height=600,
                                fig_padding=0.01,
                                fig_title=None,
                                graph_layout=DEFAULT_LAYOUT,
                                graph_engines={},
                                graph_directed=True,
                                nodes_size=35,
                                nodes_label=self.nodecol,
                                nodes_shape="circ",
                                nodes_color=0,
                                nodes_colorscale="linear",
                                nodes_selected="#EE0000",
                                nodes_palette=Palette.DEFAULT_PALETTE,
                                nodes_ratio=1.2,
                                # assign nodes_pos at the end because it depends
                                # on the others
                                nodes_pos=None,
                                edges_tips="-",
                                marks_color=0,
                                marks_colorscale="linear",
                                marks_stroke="#888888",
                                marks_palette=["#0000FF"],
                                marks_size=10,
                                marks_x=None,
                                marks_y=None,
                                marks_shape=None,
                                marks_opacity=.5)
        if opt.nodes.pos is None :
            opt.nodes.pos = opt.graph.layout
        if opt.marks.x is None :
            opt.marks.x = opt.nodes.size / 3.0
        if opt.marks.y is None :
            opt.marks.y = opt.nodes.size / 3.0
        self.nodes["_xy_min"] = np.full(len(self.nodes), 0)
        self.nodes["_x_max"] = np.full(len(self.nodes), opt.fig.width)
        self.nodes["_y_max"] = np.full(len(self.nodes), opt.fig.height)
        # assign edges before nodes because of dependencies
        opt.assign(self, "edges", "nodes", "marks", "gui")
    def _ipython_display_ (self) :
        display(self.gui_main)
    def save (self, path=None) :
        if path is None :
            path = "graph-%s.png" % now()
        fmt = pathlib.Path(path).suffix.lstrip(".").lower()
        if fmt == "png" :
            self.gui_figure.save_png(str(path))
        elif fmt == "svg" :
            self.gui_figure.save_svg(str(path))
        else :
            raise ValueError("unsupported output format %r" % fmt)
    def to_graph (self, data=False) :
        return to_graph(self.nodes[self.nodes_columns], self.nodecol,
                        self.edges[self.edges_columns], self.srccol, self.dstcol,
                        data)
    ##
    ## gui
    ##
    @cached_property
    def gui_main (self) :
        pass # setter is called first, then cached value is returned
    @gui_main.setter
    def gui_main (self, value) :
        vbox = []
        for hbox in value :
            if isinstance(hbox, list) :
                vbox.append(ipw.HBox([getattr(self, "gui_" + widget)
                                      for widget in hbox]))
            elif isinstance(hbox, str) :
                vbox.append(getattr(self, "gui_" + hbox))
            else :
                vbox.append(hbox)
        self._cached_gui_main = ipw.VBox(vbox)
    @cached_property
    def gui_layout (self) :
        choice = ipw.Dropdown(description="Layout",
                              options=(list(self.opt.graph.engines)
                                       + list(layout_engines)),
                              value=self.opt.graph.layout)
        choice.observe(self.on_layout, "value")
        return choice
    @gui_update("nodes_pos", "marks_pos", "tips_pos")
    def on_layout (self, event) :
        self.nodes_pos = event.new
    @cached_property
    def gui_color (self) :
        color = self.opt.nodes.color
        if isinstance(color, str) and color in self.nodes_columns :
            choice = ipw.Dropdown(description="Color",
                                  options=self.nodes_columns,
                                  value=color)
        else :
            choice = ipw.Dropdown(description="Color",
                                  options=["(default)"] + self.nodes_columns,
                                  value="(default)")
        choice.observe(self.on_color, "value")
        return choice
    @gui_update("nodes_color", "nodes_palette")
    def on_color (self, event) :
        if event.new == "(default)" :
            self.nodes_color = self.opt.nodes.color
        elif np.issubdtype(self.nodes[event.new].dtype, np.number) :
            self.nodes_color = event.new
        else :
            self.nodes_color = event.new, col2num(self.nodes[event.new])
    @cached_property
    def gui_palette (self) :
        choice = ipw.Dropdown(description="Palette",
                              options=list(sorted(Palette.palettes,
                                                  key=str.lower)),
                              value=self.opt.nodes.palette)
        choice.observe(self.on_palette, "value")
        return choice
    @gui_update("nodes_palette")
    def on_palette (self, event) :
        self.nodes_palette = event.new
    @cached_property
    def gui_label (self) :
        choice = ipw.Dropdown(description="Label",
                              options=self.nodes_columns,
                              value=self.opt.nodes.label)
        choice.observe(self.on_label, "value")
        return choice
    @gui_update("nodes_data")
    def on_label (self, event) :
        self.nodes_label = event.new
    @cached_property
    def gui_shape (self) :
        shape = self.opt.nodes.shape
        if isinstance(shape, str) and shape in self.nodes_columns :
            choice = ipw.Dropdown(description="Shape",
                                  options=self.nodes_columns,
                                  value=shape)
        else :
            choice = ipw.Dropdown(description="Shape",
                                  options=["(default)"] + self.nodes_columns,
                                  value="(default)")
        choice.observe(self.on_shape, "value")
        return choice
    @gui_update("nodes_data")
    def on_shape (self, event) :
        if event.new == "(default)" :
            self.nodes_shape = self.opt.nodes.shape
        else :
            self.nodes_shape = event.new, col2shape(self.nodes[event.new])
    @cached_property
    def gui_size (self) :
        choice = ipw.BoundedIntText(description="Size",
                                    min=0, max=100, step=1,
                                    value=self.opt.nodes.size)
        choice.observe(self.on_size, "value")
        return choice
    @gui_update("nodes_data", "marks_pos", "tips_pos")
    def on_size (self, event) :
        self.nodes_size = int(event.new)
    @cached_property
    def gui_figure (self) :
        self._x_scale = bq.LinearScale(min=0, max=self.opt.fig.width)
        self._y_scale = bq.LinearScale(min=0, max=self.opt.fig.height)
        if self.opt.fig.title is None :
            title, top = " ", 0
        else :
            title = str(self.opt.fig.title).strip() or " "
            top = 60 if title != " " else 0
        title = str(self.opt.fig.title) or " " if self.opt.fig.title else " "
        figure = bq.Figure(marks=([self.gui_graph]
                                  + list(self.gui_marks.values())
                                  + list(self.gui_tips.values())),
                           layout=ipw.Layout(width="%spx" % self.opt.fig.width,
                                             height="%spx" % self.opt.fig.height),
                           padding_x=self.opt.fig.padding,
                           padding_y=self.opt.fig.padding,
                           fig_margin={"top" : top,
                                       "bottom" : 0,
                                       "left" : 0,
                                       "right" : 0},
                           title=title)
        return figure
    @cached_property
    def gui_graph (self) :
        ncs = bq.ColorScale(colors=self.nodes_palette)
        graph = bq.Graph(node_data=self.node_data,
                         link_data=self.link_data,
                         link_type="line",
                         directed=self.opt.graph.directed,
                         color=self.nodes_color,
                         scales={"x": self._x_scale,
                                 "y": self._y_scale,
                                 "color": ncs},
                         x=self.nodes_x,
                         y=self.nodes_y)
        return graph
    def gui_update_nodes_data (self) :
        self.gui_graph.node_data = self.node_data
        yield self.gui_graph
    def gui_update_nodes_color (self) :
        self.gui_graph.color = self.nodes_color
        yield self.gui_graph
    def gui_update_nodes_palette (self) :
        self.gui_graph.scales["color"].colors = self.nodes_palette
        yield self.gui_graph
    def gui_update_nodes_pos (self) :
        self.gui_graph.x = self.nodes_x
        self.gui_graph.y = self.nodes_y
        yield self.gui_graph
    _marks_shapes = ["circle", "cross", "diamond", "square", "triangle-down",
                     "triangle-up", "arrow", "rectangle", "ellipse"]
    @cached_property
    def gui_marks (self) :
        sc_s = bq.LinearScale(min=5, max=10)
        sc_o = bq.LinearScale(min=0, max=1)
        sc_c = bq.ColorScale(colors=self.marks_palette)
        marks = {}
        for shape in self._marks_shapes :
            nodes = self.nodes[self.nodes["_marks_shape"] == shape]
            scatt = bq.Scatter(x=nodes["_marks_x"],
                               y=nodes["_marks_y"],
                               size=nodes["_marks_size"],
                               opacity=nodes["_marks_opacity"],
                               color=nodes["_marks_color"],
                               marker=shape,
                               stroke=self.opt.marks.stroke,
                               scales={"x" : self._x_scale,
                                       "y" : self._y_scale,
                                       "size" : sc_s,
                                       "opacity" : sc_o,
                                       "color" : sc_c})

            marks[shape] = scatt
        return marks
    def gui_update_marks_pos (self) :
        marks = self.gui_marks
        for shape, scatt in marks.items() :
            nodes = self.nodes[self.nodes["_marks_shape"] == shape]
            scatt.x = nodes["_marks_x"]
            scatt.y = nodes["_marks_y"]
            yield scatt
    _tips_shapes = ["arrow", "circle"]
    @cached_property
    def gui_tips (self) :
        sc_r = bq.LinearScale(min=0, max=np.pi)
        sc_c = bq.ColorScale(colors=Palette.mkpal(["#000000", "#FFFFFF"]))
        tips = {}
        for shape in self._tips_shapes :
            for end in ("src", "dst") :
                edges = self.edges[self.edges["_shape_" + end] == shape]
                scatt = bq.Scatter(x=edges["_x_" + end],
                                   y=edges["_y_" + end],
                                   marker=shape,
                                   rotation=edges["_angle_" + end],
                                   color=edges["_color_" + end],
                                   stroke="#000000",
                                   scales={"x" : self._x_scale,
                                           "y" : self._y_scale,
                                           "color" : sc_c,
                                           "rotation" : sc_r,
                                   })
                tips[end,shape] = scatt
        return tips
    def gui_update_tips_pos (self) :
        tips = self.gui_tips
        for (end, shape), scatt in tips.items() :
            edges = self.edges[self.edges["_shape_" + end] == shape]
            scatt.x = edges["_x_" + end]
            scatt.y = edges["_y_" + end]
            scatt.rotation = edges["_angle_" + end]
            yield scatt
    @cached_property
    def gui_toolbar (self) :
        return bq.Toolbar(figure=self.gui_figure)
    @cached_property
    def gui_inspect (self) :
        self.gui_nodes_inspect = ipw.Output()
        self.gui_edges_inspect = ipw.Output()
        inspect = ipw.Tab([self.gui_nodes_inspect,
                           self.gui_edges_inspect,
                           self.gui_move_nodes])
        inspect.set_title(0, "Selected nodes")
        inspect.set_title(1, "Selected edges")
        inspect.set_title(2, "Move nodes")
        graph = self.gui_graph
        graph.selected_style = {"stroke" : self.opt.nodes.selected,
                                "stroke-width" : "4"}
        graph.on_element_click(self.on_node_click)
        graph.on_background_click(self.on_bg_click)
        self.gui_selected = set()
        return inspect
    @cached_property
    def gui_move_nodes (self) :
        self.gui_move_nodes_x = ipw.IntSlider(min=0, max=self.opt.fig.width,
                                              description="x position:",
                                              disabled=True,
                                              continuous_update=False,
                                              layout=ipw.Layout(width="95%"))
        self.gui_move_nodes_y = ipw.IntSlider(min=0, max=self.opt.fig.height,
                                              description="y position:",
                                              disabled=True,
                                              continuous_update=False,
                                              layout=ipw.Layout(width="95%"))
        self.gui_move_nodes_x.observe(self.on_move_x, "value")
        self.gui_move_nodes_y.observe(self.on_move_y, "value")
        return ipw.VBox([self.gui_move_nodes_x,
                         self.gui_move_nodes_y])
    def on_node_click (self, graph, event) :
        # update selection
        nid = event["data"]["id"]
        if nid in self.gui_selected :
            self.gui_selected.remove(nid)
        else :
            self.gui_selected.add(nid)
        self.gui_graph.selected = list(self.gui_selected)
        # update tables
        with self.gui_nodes_inspect :
            clear_output()
            if self.gui_selected :
                cols = self.nodes_columns
                nodes = self.nodes[self.nodes["_id"].isin(self.gui_selected)]
                self.gui_move_nodes_x.disabled = True
                self.gui_move_nodes_x.value = nodes["_x"].mean()
                self.gui_move_nodes_y.disabled = True
                self.gui_move_nodes_y.value = nodes["_y"].mean()
                display(nodes[cols])
                self.gui_selected_nodes = nodes
        with self.gui_edges_inspect :
            clear_output()
            if self.gui_selected :
                cols = self.edges_columns
                edges = self.edges[self.edges["_source"].isin(self.gui_selected)
                                   | self.edges["_target"].isin(self.gui_selected)]
                display(edges[cols])
        self.gui_move_nodes_x.disabled = not self.gui_selected
        self.gui_move_nodes_y.disabled = not self.gui_selected
    def on_bg_click (self, graph, event) :
        self.gui_selected.clear()
        self.gui_graph.selected = []
        with self.gui_nodes_inspect :
            clear_output()
        with self.gui_edges_inspect :
            clear_output()
        self.gui_move_nodes_x.disabled = True
        self.gui_move_nodes_y.disabled = True
    @gui_update("nodes_pos", "marks_pos", "tips_pos")
    def on_move_x (self, event) :
        if self.gui_move_nodes_x.disabled or not hasattr(self, "gui_selected_nodes") :
            raise CancelUpdate()
        nodes = self.gui_selected_nodes
        shift = event.new - event.old
        self.nodes.loc[nodes.index,"_x"] += shift
        self.nodes["_x"] = self.nodes[["_x", "_x_max"]].min(axis=1)
        self.nodes["_x"] = self.nodes[["_x", "_xy_min"]].max(axis=1)
        self._pos_updated()
    @gui_update("nodes_pos", "marks_pos", "tips_pos")
    def on_move_y (self, event) :
        if self.gui_move_nodes_y.disabled or not hasattr(self, "gui_selected_nodes") :
            raise CancelUpdate()
        nodes = self.gui_selected_nodes
        shift = event.new - event.old
        self.nodes.loc[nodes.index,"_y"] += shift
        self.nodes["_y"] = self.nodes[["_y", "_y_max"]].min(axis=1)
        self.nodes["_y"] = self.nodes[["_y", "_xy_min"]].max(axis=1)
        self._pos_updated()
    ##
    ## generic setters
    ##
    def _setter (self, table, cols, value, types=(int, float), astype=None) :
        if not isinstance(cols, list) :
            cols = [cols]
        if not isinstance(astype, list) :
            astype = [astype] * len(cols)
        if len(table) == 0 :
            for c in cols :
                if c not in table.columns :
                    table[c] = []
            return
        if isinstance(value, str) and value in table.columns :
            data = table[[value]]
        elif isinstance(value, list) and all(isinstance(c, str) and c in table.columns
                                             for c in value) :
            data = table[value]
        elif isinstance(value, pd.Series) and len(value) == len(table) :
            data = value
        elif isinstance(value, types) :
            data = pd.DataFrame({c : np.full(len(table), value) for c in cols})
        elif isinstance(value, tuple) :
            col, fun = value
            if not isinstance(col, list) :
                col = [col]
            data = self.nodes[col].apply(fun, axis=1)
        else :
            raise ValueError("invalid data for %r: %r"
                             % ([c.rsplit("_", 1)[-1] for c in cols], value))
        if not isinstance(data, pd.DataFrame) :
            data = pd.DataFrame(data)
        for t, c, a in zip(cols, data.columns, astype) :
            if a is None :
                table[t] = data[c].values
            else :
                table[t] = data[c].values.astype(a)
    ##
    ## marks
    ##
    @property
    def marks_color (self) :
        return self.nodes["_marks_color"]
    @marks_color.setter
    def marks_color (self, value) :
        self._setter(self.nodes, "_marks_color", value)
    @property
    def marks_palette (self) :
        return self._marks_palette
    @marks_palette.setter
    def marks_palette (self, value) :
        if isinstance(value, str) and value in Palette.palettes :
            self._marks_palette = Palette.palettes[value]
        elif isinstance(value, list) :
            self._marks_palette = [Palette.hexcolor(c) for c in value]
        else :
            raise ValueError("unknown palette %r" % value)
        if self.opt.marks.colorscale == "linear" :
            count = self.marks_color.max() + 1
        else :
            count = self.marks_color.nunique()
        if len(self._marks_palette) < count :
            self._marks_palette = Palette.mkpal(self._marks_palette, int(count))
    @property
    def marks_size (self) :
        return self.nodes["_marks_size"]
    @marks_size.setter
    def marks_size (self, value) :
        self._setter(self.nodes, "_marks_size", value)
    @property
    def marks_x (self) :
        return self.nodes["_marks_x"]
    @marks_x.setter
    def marks_x (self, value) :
        self.nodes["_marks_x"] = self.nodes["_x"] + value
    @property
    def marks_y (self) :
        return self.nodes["_marks_y"]
    @marks_y.setter
    def marks_y (self, value) :
        self.nodes["_marks_y"] = self.nodes["_y"] + value
    @property
    def marks_opacity (self) :
        return self.nodes["_marks_opacity"]
    @marks_opacity.setter
    def marks_opacity (self, value) :
        self._setter(self.nodes, "_marks_opacity", value)
    @property
    def marks_shape (self) :
        return self.nodes["_marks_shape"]
    @marks_shape.setter
    def marks_shape (self, value) :
        self._setter(self.nodes, "_marks_shape", value, (str, type(None)))
    ##
    ## edges
    ##
    @property
    def edges_tips (self) :
        return self.edges[["_tip_src", "tip_dst"]]
    @edges_tips.setter
    def edges_tips (self, value) :
        if len(self.edges) == 0 :
            for col in ("_tip_src", "_tip_dst") :
                if col not in self.edges.columns :
                    self.edges[col] = []
            return
        if (isinstance(value, str)
            and 1 <= len(value) <= 2
            and all(c in "<>-o*" for c in value)) :
            self.edges.loc[self.edges.index,"_tip_src"] = value[0]
            self.edges.loc[self.edges.index,"_tip_dst"] = value[-1]
        else :
            self._setter(self.edges, ["_tip_src", "_tip_dst"], value, ())
    ##
    ## nodes
    ##
    @property
    def nodes_pos (self) :
        return self.nodes[["_x", "_y"]]
    @property
    def nodes_x (self) :
        return self.nodes["_x"]
    @property
    def nodes_y (self) :
        return self.nodes["_y"]
    @nodes_pos.setter
    def nodes_pos (self, value) :
        if isinstance(value, list) :
            pos = self.nodes[value]
        elif isinstance(value, pd.DataFrame) :
            pos = value
        elif isinstance(value, tuple) :
            col, fun = value
            if not isinstance(col, list) :
                col = [col]
            pos = self.nodes[col].apply(fun, axis=1)
        elif isinstance(value, str) and value in layout_engines :
            fun = layout_engines[value](self.to_graph()).get
            pos = pd.DataFrame.from_records(self.nodes[self.nodecol].map(fun).tolist())
        elif isinstance(value, str) and value in self.opt.graph.engines :
            pos = self.opt.graph.engines[value]
        else :
            raise ValueError("invalid position: %r" % value)
        for dim, col, scale in zip(("_x", "_y"),
                                   pos.columns,
                                   (self.opt.fig.width, self.opt.fig.height)) :
            pos[col] -= pos[col].min()
            pos[col] /= pos[col].max() or 1.0
            self.nodes[dim] = pos[col] * scale
        self._pos_updated()
    def _pos_updated (self) :
        if not "_x" in self.nodes.columns or not "_y" in self.nodes.columns :
            return
        # update marks positions
        self.marks_x = self.opt.marks.x
        self.marks_y = self.opt.marks.y
        # update arrow tips
        edges = self.edges[self.edges_columns + ["_tip_" + e for e in ("src", "dst")]]
        nodes = self.nodes[[self.nodecol, "_x", "_y", "_size"]]
        edges = edges.merge(nodes, left_on=self.srccol, right_on=self.nodecol)
        edges = edges.merge(nodes, left_on=self.dstcol, right_on=self.nodecol,
                            suffixes=["_src", "_dst"])
        slope = np.arctan2(edges["_y_dst"] - edges["_y_src"],
                           edges["_x_dst"] - edges["_x_src"])
        edges["_angle_src"] = edges["_angle_src"] = np.pi/2.0 - slope
        edges["_angle_dst"] = edges["_angle_dst"] = np.pi/2.0 - slope
        for tip, shape, angle, color, factor in zip(
                ("<",     ">",     "-",  "o",      "*"),      # tip
                ("arrow", "arrow", None, "circle", "circle"), # shape
                (np.pi,   0,       None, None,     None),     # angle
                (0,       0,       0,     1,       0),        # color
                (1.5,     1.5,     1,     1.8,     1.8)) :    # factor
            for end, sign in zip(("src", "dst"),  # end
                                 (1,     -1)) :   # sign
                idx = edges[edges["_tip_" + end] == tip].index
                edges.loc[idx,"_x_" + end] = (edges.loc[idx,"_x_" + end]
                                              + (np.cos(slope[idx])
                                                 * edges.loc[idx,"_size_" + end]
                                                 / factor) * sign)
                edges.loc[idx,"_y_" + end] = (edges.loc[idx,"_y_" + end]
                                              + (np.sin(slope[idx])
                                                 * edges.loc[idx,"_size_" + end]
                                                 / factor) * sign)
                if angle is None :
                    edges.loc[idx,"_angle_" + end] = 0
                else :
                    edges.loc[idx,"_angle_" + end] += angle
                    edges.loc[idx,"_angle_" + end] %= 2 * np.pi
                edges.loc[idx,"_shape_" + end] = np.full(len(idx), shape)
                edges.loc[idx,"_color_" + end] = np.full(len(idx), color)
        self.edges = edges
    @property
    def nodes_palette (self) :
        return self._nodes_palette
    @nodes_palette.setter
    def nodes_palette (self, value) :
        if isinstance(value, str) and value in Palette.palettes :
            self._nodes_palette = Palette.palettes[value]
        elif isinstance(value, list) :
            self._nodes_palette = [Palette.hexcolor(c) for c in value]
        else :
            raise ValueError("unknown palette %r" % value)
        if self.opt.nodes.colorscale == "linear" :
            count = self.nodes_color.max() + 1
        else :
            count = self.nodes_color.nunique()
        if len(self._nodes_palette) < count :
            self._nodes_palette = Palette.mkpal(self._nodes_palette, int(count))
    @property
    def nodes_color (self) :
        return self.nodes["_color"]
    @nodes_color.setter
    def nodes_color (self, value) :
        self._setter(self.nodes, "_color", value, astype=int)
    @property
    def nodes_size (self) :
        return self.nodes["_size"]
    @nodes_size.setter
    def nodes_size (self, value) :
        self._setter(self.nodes, "_size", value)
        self._pos_updated()
        self._shape_or_size_updated()
    @property
    def nodes_label (self) :
        return self.nodes["_label"]
    @nodes_label.setter
    def nodes_label (self, value) :
        self._setter(self.nodes, "_label", value, ())
        self.nodes["_label"] = self.nodes["_label"].apply(_str)
    @property
    def nodes_shape (self) :
        return self.nodes["_shape"]
    @nodes_shape.setter
    def nodes_shape (self, value) :
        self._setter(self.nodes, "_shape", value, (str,))
        self._shape_or_size_updated()
    def _shape_or_size_updated (self) :
        if "_shape" not in self.nodes.columns or "_size" not in self.nodes.columns :
            return
        d = self.nodes[["_shape", "_size"]].apply(self._shape_attrs, axis=1)
        f = pd.DataFrame.from_records(d, columns=["_nodes_shape", "_nodes_info"])
        for col in f.columns :
            self.nodes[col] = f[col]
    def _shape_attrs (self, row) :
        ratio = self.opt.nodes.ratio
        if row["_shape"] == "circ" :
            return "circle", {"r" : row["_size"] / 2.0}
        elif row["_shape"] == "sbox" :
            return "rect", {"width" : row["_size"], "height" : row["_size"]}
        elif row["_shape"] == "rbox" :
            return "rect", {"width" : row["_size"], "height" : row["_size"],
                            "rx" : 8, "ry" : 8}
        elif row["_shape"] == "vbox" :
            return "rect", {"width" : row["_size"] / ratio,
                            "height" : row["_size"] * ratio}
        elif row["_shape"] == "hbox" :
            return "rect", {"width" : row["_size"] * ratio,
                            "height" : row["_size"] / ratio}
        elif row["_shape"] == "vegg" :
            return "ellipse", {"rx" : row["_size"] / (2.0 * ratio),
                               "ry" : row["_size"] * (ratio / 2.0)}
        elif row["_shape"] == "hegg" :
            return "ellipse", {"rx" : row["_size"] * (ratio / 2.0),
                               "ry" : row["_size"] / (2.0 * ratio)}
        else :
            raise ValueError("Invalid shape: %r" % row["_shape"])
    _node_data_map = {"_x" : "x",
                      "_y" : "y",
                      "_nodes_shape" : "shape",
                      "_label" : "label",
                      "_nodes_info" : "shape_attrs",
                      "_id" : "id"}
    @property
    def node_data (self) :
        ncol = self._node_data_map
        self.nodes["_id"] = self.nodes.index
        return [row.rename(ncol).to_dict()
                for _, row in self.nodes[list(ncol)].iterrows()]
    _link_data_map = {"_source" : "source",
                      "_target" : "target"}
    @property
    def link_data (self) :
        ecol = self._link_data_map
        node2idx = dict((v, k) for k, v in self.nodes[self.nodecol].iteritems())
        self.edges["_source"] = self.edges[self.srccol].apply(node2idx.get)
        self.edges["_target"] = self.edges[self.dstcol].apply(node2idx.get)
        return [row.rename(ecol).to_dict()
                for _, row in self.edges[list(ecol)].iterrows()]

#     def _tikz_shape (self, shape) :
#         if shape == "circ" :
#             return "circle,minimum size={size_}"
#         elif shape == "sbox" :
#             return "rectangle,minimum size={size_}"
#         elif shape == "rbox" :
#             return "rectangle,minimum size={size_},rounded corners"
#         elif shape == "vbox" :
#             return "rectangle,minimum width={size_w},minimum height={size_h}"
#         elif shape == "hbox" :
#             return "rectangle,minimum width={size_h},minimum height={size_w}"
#         elif shape == "vegg" :
#             return "ellipse,minimum width={size_w},minimum height={size_h}"
#         elif shape == "hegg" :
#             return "ellipse,minimum width={size_w},minimum height={size_h}"
#         else :
#             return ""
#     def tikz (self, indent=2, grid=.5, headers=False, label=None, nodescale=.01) :
#         if label is None :
#             label = self.nodecol
#         if isinstance(indent, int) :
#             indent = indent * " "
#         tikz = io.StringIO()
#         nodes = self.nodes.join(self._ni.rename(columns={c : c + "_" for c in
#                                                          self._ni.columns}), how="outer")
#         nodes["shape_"] = nodes["shape_"].apply(self._tikz_shape)
#         nodes["size_"] = (nodes["size_"] * nodescale).round(2)
#         nodes["size_w"] = (nodes["size_"] * self.opt.nodes.ratio).round(2)
#         nodes["size_h"] = (nodes["size_"] / self.opt.nodes.ratio).round(2)
#         if grid :
#             for col in ("x_", "y_") :
#                 nodes[col] = (nodes[col] / grid).round() * grid
#         nodeline = ("{indent}\\node[%%s,label=center:{{{%s}}}]"
#                     " at ({x_},{y_}) (N{%s}) {{}};\n" % (label, self.nodecol))
#         for _, row in nodes.iterrows() :
#             tikz.write((nodeline % row.shape_).format(indent=indent,
#                                                       **(row.to_dict())))
#         edgeline = ("{indent}\\draw (N{%s}) edge[->] (N{%s});\n"
#                     % (self.srccol, self.dstcol))
#         for _, row in self.edges.iterrows() :
#             tikz.write(edgeline.format(indent=indent,
#                                        **(row.to_dict())))
#         return tikz.getvalue()
