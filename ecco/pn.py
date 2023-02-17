import re, struct, subprocess, pathlib, tempfile, collections, warnings, operator

import igraph as ig
import pandas as pd

from .cygraphs import Graph

class NotNet (object) :
    _attrs = {"tokens", "priority"}
    _nodes = {"place", "trans", "cond", "event"}
    _edges = {"cons", "prod", "read", "reset", "inhib", "succ", "conflict"}
    _allowed_edges = {"place" : {"trans" : {"cons", "reset", "read", "inhib"}},
                      "trans" : {"place" : {"prod"}},
                      "cond"  : {"event" : {"cons", "reset", "read", "inhib"}},
                      "event" : {"cond"  : {"prod"},
                                 "event" : {"succ", "conflict"}}}
    def __init__ (self) :
        self.g = ig.Graph(directed=True)
    def copy (self) :
        cls = self.__class__
        new = cls.__new__(cls)
        new.g = self.g.copy()
        return new
    def __getitem__ (self, key) :
        if isinstance(key, tuple) :
            src, tgt = key
            return self.g.es.find(_source=self[src].index, _target=self[tgt].index)
        else :
            return self.g.vs.find(key)
    def has (self, src, tgt=None, kind=None) :
        try :
            if tgt is None and kind is None :
                self[src]
            elif tgt is None :
                return self[src]["kind"] == kind
            elif kind is None :
                self[src,tgt]
            else :
                return self[src,tgt]["kind"] == kind
            return True
        except :
            return False
    def __getattr__ (self, name) :
        if name.startswith("_") :
            if name[1:] in self._nodes :
                return list(self.g.vs.select(kind=name[1:]))
            elif name[1:] in self._attrs :
                return list(self.g.vs.select(**{f"{name[1:]}_eq" : True}))
            elif name[1:] in self._edges :
                return list(self.g.es.select(kind=name[1:]))
        elif name in self._nodes or name in self._attrs :
            return {node["name"] for node in getattr(self, f"_{name}")}
        elif name in self._edges :
            return {(self[edge.source]["name"], self[edge.target]["name"])
                    for edge in getattr(self, f"_{name}")}
        raise AttributeError(f"{self.__class__.__name__!r} object"
                             f" has no attribute {name!r}")
    def pre (self, node, kind=None) :
        found = collections.defaultdict(set)
        for edge in self.g.es.select(_target=self[node].index) :
            found[edge["kind"]].add(self[edge.source]["name"])
        if kind is None :
            return dict(found)
        else :
            return found[kind]
    def post (self, node, kind=None) :
        found = collections.defaultdict(set)
        for edge in self.g.es.select(_source=self[node].index) :
            found[edge["kind"]].add(self[edge.target]["name"])
        if kind is None :
            return dict(found)
        else :
            return found[kind]
    def add_node (self, kind, name, **attrs) :
        if kind not in self._nodes :
            raise ValueError(f"unknown node type: {kind}")
        elif self.has(name) :
            raise ValueError(f"node {name!r} exists")
        attributes = {a : None for a in self._attrs}
        attributes.update(attrs)
        self.g.add_vertex(name, kind=kind, **attributes)
    def remove_node (self, node) :
        self.g.delete_vertices(self[node].index)
    def add_arc (self, kind, source, target) :
        if kind not in self._edges :
            raise ValueError(f"unknown node type: {kind}")
        src = self[source]
        tgt = self[target]
        if self.g.es.select(_source=src.index, _target=tgt.index) :
            raise ValueError("arc from {source!r} to {target!r} exists")
        if tgt["kind"] not in self._allowed_edges[src["kind"]] :
            raise ValueError(f"no arcs allowed from {src['kind']} to {tgt['kind']}")
        elif kind not in self._allowed_edges[src["kind"]][tgt["kind"]] :
            raise ValueError(f"{kind} arcs allowed from {src['kind']} to {tgt['kind']}")
        self.g.add_edge(src.index, tgt.index, kind=kind)
    def to_tables (self) :
        nodes = pd.DataFrame({attr : self.g.vs[attr]
                              for attr in self.g.vertex_attributes()
                              if attr != "name"},
                             index=pd.Series(self.g.vs["name"], name="node"))
        edges = pd.DataFrame({attr : self.g.es[attr]
                              for attr in self.g.edge_attributes()},
                             index=pd.MultiIndex.from_tuples(
                                 [(self[e.source]["name"], self[e.target]["name"])
                                  for e in self.g.es],
                                 names=("src", "dst")))
        return nodes, edges
    _draw_shape = {"place" : "ellipse",
                   "cond" : "ellipse",
                   "trans" : "rectangle",
                   "event" : "rectangle"}
    _draw_tip = {"cons" : "vee",
                 "prod" : "vee",
                 "read" : "-",
                 "reset" : "diamond",
                 "inhib" : "o",
                 "succ" : ">",
                 "conflict" : "-"}
    _draw_fill = {True : "#ffa",
                  False : "#fff",
                  None : "#fff"}
    _draw_width = {True : 1.2,
                   False : .6,
                   None : .6}
    def _draw (self) :
        nodes, edges = self.to_tables()
        nodes["shape"] = nodes["kind"].map(self._draw_shape.get)
        if "priority" in self._attrs :
            nodes["fill"] = nodes["priority"].map(self._draw_fill.get)
            nodes["width"] = nodes["priority"].map(self._draw_width.get)
        else :
            nodes["fill"] = self._draw_fill[False]
            nodes["width"] = self._draw_width[False]
        edges["style"] = "solid"
        edges["style"][edges["kind"] == "conflict"] = "dashed"
        edges["dst-tip"] = edges["kind"].map(self._draw_tip.get)
        edges["src-tip"] = "-"
        # two opposite arcs => one bidirectional arc
        todrop = set()
        for (src, dst), row in edges.iterrows() :
            if (src, dst) not in todrop and (dst, src) in edges.index :
                edges.loc[(src, dst),"src-tip"] = edges.loc[(dst, src),"dst-tip"]
                todrop.add((dst, src))
        edges.drop(index=todrop, inplace=True)
        # /done
        def nodes_label (txt, **args) :
            if args.get("tokens", None) is True :
                return f"{txt}\n⏺"
            else :
                return str(txt)
        options = {"nodes_shape" : "shape",
                   "nodes_size" : 30,
                   "nodes_label_str" : nodes_label,
                   "nodes_fill_color" : "fill",
                   "nodes_draw_width" : "width",
                   "edges_draw_style" : "style",
                   "edges_draw_color" : "#000",
                   "edges_source_tip" : "src-tip",
                   "edges_target_tip" : "dst-tip",
                   "edges_tip_scale" : .8,
                   "edges_curve" : "straight",
                   "edges_label" : "",
                   "inspect_nodes" : ["kind", "tokens"],
                   "ui" : [{"Graph" :
                            [["layout", "reset_view"]]},
                           {"Nodes" :
                            [["nodes_shape", "nodes_size"],
                             {"Fill" :
                              [["nodes_fill_color", "nodes_fill_palette"],
                               ["nodes_fill_opacity"]]}]},
                           "graph",
                           {"Inspector" :
                            [["select_all", "select_none", "invert_selection"],
                             ["inspect_nodes"]]},
                           {"Export" :
                            [["export"]]}]}
        return nodes, edges, options
    def draw (self, **opts) :
        nodes, edges, options = self._draw()
        options.update(opts)
        return Graph(nodes, edges, **options)
    ##
    ## i/o
    ##
    def to_tina (self, stream, name="NET") :
        for kind, family in (
                # Petri nets extensions
                ("reset", "arcs"),
                # unfoldings
                ("cond", "nodes"), ("event", "nodes"),
                # event structures
                ("succ", "arcs"), ("conflict", "arcs")) :
            if getattr(self, kind, None) :
                raise ValueError(f"tina does not support {kind} {family}")
        stream.write(f"net {{{name}}}\n")
        for place in sorted(self._place, key=operator.itemgetter("name")) :
            if place["tokens"] :
                stream.write(f"pl {{{place['name']}}} (1)\n")
            else :
                stream.write(f"pl {{{place['name']}}}\n")
        for trans in sorted(self._trans, key=operator.itemgetter("name")) :
            pre = []
            for kind, op in (("cons", ""), ("read", "?1"), ("inhib", "-1")) :
                for arc in self.g.es.select(_target=trans.index, kind=kind) :
                    pre.append(f"{{{self[arc.source]['name']}}}{op}")
            post = []
            for arc in self.g.es.select(_source=trans.index, kind="prod") :
                post.append(f"{{{self[arc.target]['name']}}}")
            stream.write(f"tr {{{trans['name']}}} {' '.join(sorted(pre))}"
                         f" -> {' '.join(sorted(post))}\n")
        low_trans = self.g.vs.select(priority=False)
        for high in self.g.vs.select(priority=True) :
            for low in low_trans :
                stream.write(f"pr {{{low['name']}}} < {{{high['name']}}}\n")
    def to_pep (self, stream) :
        # headers
        stream.write("PEP\n"
                     "PetriBox\n"
                     "FORMAT_N2\n")
        num = {}
        # places
        tokens = self.tokens
        stream.write("PL\n")
        for i, place in enumerate(sorted(self.place), start=1) :
            num[place] = i
            if place in tokens :
                stream.write(f'"{place}"M1\n')
            else :
                stream.write(f'"{place}"\n')
        # transitions
        stream.write("TR\n")
        for i, trans in enumerate(sorted(self.trans), start=1) :
            num[trans] = i
            stream.write(f'"{trans}"\n')
        # arcs
        for kind, tag, sep in (("read", "RD", ">"),
                               ("prod", "TP", "<"),
                               ("cons", "PT", ">"),
                               ("reset", "RS", ">"),
                               ("inhib", "IN", ">")) :
            if kind not in self._edges :
                continue
            stream.write(f"{tag}\n")
            for src, tgt in sorted(getattr(self, kind)) :
                stream.write(f"{num[src]}{sep}{num[tgt]}\n")
    _pep_re = {
        # headers
        "HD"  : None,
        # default for blocks
        "DBL" : None,
        # default for places
        "DPL" : None,
        # default for transitions
        "DTR" : None,
        # default for arcs
        "DPT" : None,
        # blocks
        "BL" : None,
        # places
        "PL"  : re.compile(r'^([0-9]*)\"([^\"]+)\".*?([M0-9]*).*$'),
        # transitions
        "TR"  : re.compile(r'^([0-9]*)\"([^\"]+)\".*$'),
        # phantom transitions
        "PTR" : None,
        # arcs from transitions to places
        "TP"  : re.compile(r"^([0-9]+)<([0-9]+)$"),
        # arcs from places to transitions
        "PT"  : re.compile(r"^([0-9]+)>([0-9]+)$"),
        # arcs from phantom transitions to places
        "PTP" : None,
        # arcs from places to phantom transitions
        "PPT" : None,
        # texts
        "TX"  : None,
        # extension: read arcs
        "RD" : re.compile(r"^([0-9]+)>([0-9]+)$"),
        # extension: reset arcs
        "RS" : re.compile(r"^([0-9]+)>([0-9]+)$"),
        # extension: inhibitor arcs
        "IN" : re.compile(r"^([0-9]+)>([0-9]+)$")
    }
    _pep_arcs = {"PT" : "cons",
                 "TP" : "prod",
                 "RD" : "read",
                 "RS" : "reset",
                 "IN" : "inhib"}
    @classmethod
    def from_pep (cls, stream) :
        pname, tname = {}, {}
        net = cls()
        sec = "HD"
        def error (lno, msg) :
            raise ValueError(f"[{stream.name}:{lno}] {msg}")
        def match (lno, line) :
            match = cls._pep_re[sec].match(line)
            if not match :
                error(lno, f"invalid {sec} line: {line!r}")
            return match.groups()
        def readlines () :
            for lno, line in enumerate(stream, start=1) :
                txt = line.split("%", 1)[0].strip()
                if txt :
                    yield lno, txt
        for lno, line in readlines() :
            if line in cls._pep_re :
                sec = line
            elif cls._pep_re[sec] is None :
                continue
            elif sec == "PL" :
                ident, name, mark = match(lno, line)
                if ident :
                    error(lno, f"explicit identifiers not supported: {line!r}")
                elif mark and mark != "M1" :
                    error(lno, f"unsupported marking: {mark!r}")
                pname[len(pname)+1] = name
                net.add_place(name, tokens=bool(mark))
            elif sec == "TR" :
                ident, name = match(lno, line)
                if ident :
                    error(lno, f"explicit identifiers not supported: {line!r}")
                tname[len(tname)+1] = name
                net.add_trans(name)
            elif sec in cls._pep_arcs :
                if sec == "TP" :
                    sn, tn = tname, pname
                else :
                    sn, tn = pname, tname
                src, tgt = (int(x) for x in match(lno, line))
                net.add_arc(cls._pep_arcs[sec], sn[src], tn[tgt])
            else :
                raise error(lno, f"unexpected line: {line!r}")
        return net

class SPN (NotNet) :
    "Safe Petri nets"
    _nodes = {"place", "trans"}
    _edges = {"cons", "prod"}
    def add_place (self, name, **attrs) :
        self.add_node("place", name, **attrs)
    def add_trans (self, name, **attrs) :
        self.add_node("trans", name, **attrs)
    def add_cons (self, source, target) :
        self.add_arc("cons", source, target)
    def add_prod (self, source, target) :
        self.add_arc("prod", source, target)
    def remove_loops (self, copy=False, errors="warn") :
        if "reset" in self._edges and self._reset :
            if errors == "ignore" :
                pass
            elif errors == "warn" :
                warnings.warn("cannot detect loops (net has reset arcs)")
            elif errors == "raise" :
                raise ValueError("cannot detect loops (net has reset arcs)")
            else :
                raise ValueError(f"unexpected value for errors: {errors!r}")
        if copy :
            new = self.copy()
            new.remove_loops(False, "ignore")
            return new
        for trans in list(self.trans) :
            if self.pre(trans, "cons") == self.post(trans, "prod") :
                self.remove_node(trans)

class XPN (SPN) :
    "SPN extended with read- and reset-arcs"
    _edges = {"cons", "prod", "read", "reset"}
    def add_read (self, source, target) :
        self.add_arc("read", source, target)
    def add_reset (self, source, target) :
        self.add_arc("reset", source, target)
    def unfold (self, base=None) :
        with tempfile.TemporaryDirectory() as tmp :
            if base is None :
                base = pathlib.Path(tmp)
            else :
                base = pathlib.Path(base)
            mci = base.with_suffix(".mci")
            pre = (mci.parent / (mci.stem + "_pr")).with_suffix(".ll_net")
            pep = mci.with_suffix(".ll_net")
            self.to_pep(pep.open("w"))
            try :
                subprocess.check_output(["pr_encoding", str(pep)],
                                        stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as err :
                raise RuntimeError(err.stdout)
            try :
                subprocess.check_output(["ecofolder", str(pre), "-m", str(mci)],
                                        stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as err :
                raise RuntimeError(err.stdout)
            return UNF.from_mci(mci.open("rb"))

class EPN (XPN) :
    "XPN extended with inhibitor-arcs"
    _edges = {"cons", "prod", "read", "reset", "inhib"}
    def add_inhib (self, source, target) :
        self.add_arc("inhib", source, target)

class NotUnf (NotNet) :
    def add_event (self, name, **attrs) :
        self.add_node("event", name, **attrs)
    def _draw (self) :
        nodes, edges, options = super()._draw()
        def cutoff (node) :
            return self[node]["cutoff"]
        nodes["cutoff"] = nodes.index.map(cutoff)
        def shape (node) :
            node = self[node]
            if node["kind"] == "cond" :
                return "ellipse"
            elif node["cutoff"] :
                return "round-rectangle"
            else :
                return "rectangle"
        nodes["shape"] = nodes.index.map(shape)
        def nodes_label (txt, **args) :
            if args.get("tokens", False) is True :
                return f"{txt}\n{args['petri']}\n⏺"
            else :
                return f"{txt}\n{args['petri']}"
        options.update({"nodes_label_str" : nodes_label,
                        "inspect_nodes" : ["kind", "petri"] + list(self._attrs),
                        "layout" : "gv-dot"})
        return nodes, edges, options

class UNF (NotUnf) :
    "Unfoldings of Petri nets"
    _attrs = {"tokens", "cutoff", "dummy"}
    _nodes = {"cond", "event"}
    _edges = {"cons", "prod"}
    def add_cons (self, source, target) :
        self.add_arc("cons", source, target)
    def add_prod (self, source, target) :
        self.add_arc("prod", source, target)
    def add_cond (self, name, **attrs) :
        self.add_node("cond", name, **attrs)
    def es (self) :
        events = self.g.vs.select(kind="event")
        attrs = ES._attrs | {"name", "kind", "petri"}
        struct = ES()
        struct.g.add_vertices(len(events),
                              {a : events.get_attribute_values(a) for a in attrs})
        for ev in events :
            idx = ev.index
            succ = {(ev, s)
                    for c in self.g.successors(ev)
                    for s in self.g.successors(c)}
            struct.g.add_edges(succ, {"kind" : ["succ"] * len(succ)})
            conf = {(ev, x)
                    for c in self.g.predecessors(ev)
                    for x in self.g.successors(c)
                    if x > idx}
            struct.g.add_edges(conf, {"kind" : ["conflict"] * len(conf)})
        return struct
    ##
    ## i/o
    ##
    @classmethod
    def from_mci (cls, stream) :
        INTSIZE = struct.calcsize("i")
        def readint () :
            return struct.unpack("i", stream.read(INTSIZE))[0]
        numco = readint()
        numev = readint()
        unf = cls()
        ev2tr, co2pl = {}, {}
        for ev in range(1, numev+1) :
            evname = f"e{ev}"
            ev2tr[evname] = readint() - 1
            unf.add_event(evname, cutoff=False, dummy=False)
        for co in range(1, numco+1) :
            coname = f"c{co}"
            co2pl[coname] = readint() - 1
            toks = readint()
            unf.add_cond(coname, tokens=bool(toks))
            ev = readint()
            if ev :
                unf.add_prod(f"e{ev}", coname)
            while True :
                ev = readint()
                if not ev :
                    break
                unf.add_cons(coname, f"e{ev}")
        while True :
            ev = readint()
            if not ev :
                break
            node = unf[f"e{ev}"]
            node["cutoff"] = True
            if readint() :
                node["dummy"] = True
        while readint() :
            pass
        numpl = readint()
        readint() # numtr
        readint() # max len of names
        names = [n.decode() for n in stream.read().split(b'\0') if n]
        for co, pl in co2pl.items() :
            unf[co]["petri"] = names[pl]
        for ev, tr in ev2tr.items() :
            unf[ev]["petri"] = names[numpl + tr]
        return unf

class ES (NotUnf) :
    "Event structures"
    _attrs = {"cutoff", "dummy"}
    _nodes = {"event"}
    _edges = {"succ", "conflict"}
    def add_succ (self, source, target) :
        self.add_arc("succ", source, target)
    def add_conflict (self, source, target) :
        self.add_arc("conflict", source, target)
    def _draw (self) :
        nodes, edges, options = super()._draw()
        options["inspect_nodes"] = ["kind", "petri"] + list(self._attrs)
        return nodes, edges, options
