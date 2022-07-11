from pathlib import Path
from subprocess import check_call

import struct, io
import igraph as ig

class _g2dot (object) :
    def __init__ (self, g) :
        self.g = g
    def dot (self, path,
             engine="dot",
             style_dummy="dashed",
             style_cutoff="bold",
             condition_shape="circle",
             event_shape="box",
             bullet_tokens=True,
             group_conflicts=True) :
        path = Path(path)
        dot_path = path.with_suffix(".dot")
        with dot_path.open("w") as out :
            out.write("digraph {\n")
            for node in self.g.vs :
                index = node.index
                attr = node.attributes()
                attr["name"] = repr(attr["name"])[1:-1]
                out.write(f"  {index} [\n")
                if attr.get("kind") == "condition" :
                    out.write(f"    shape={condition_shape}\n")
                    if index in self.g["tokens"] :
                        if bullet_tokens :
                            attr["tokens"] = "●️"
                        else :
                            attr["tokens"] = "1"
                    else :
                        attr["tokens"] = ""
                    out.write('    label="{tokens}\\n{name}\\n{id}"\n'.format(**attr))
                else :
                    out.write('    label="{name}\\n{id}"\n'.format(**attr))
                    out.write(f"    shape={event_shape}\n")
                    if index in self.g["dummy"] :
                        out.write(f"    style={style_dummy or 'solid'}\n")
                    elif index in self.g["cutoff"] :
                        out.write(f"    style={style_cutoff or 'bold'}\n")
                out.write("  ];\n")
            edges = set()
            for edge in self.g.es :
                attr = edge.attributes()
                kind = attr.get("kind")
                src, tgt = edge.source, edge.target
                if kind == "conflict" :
                    if (src, tgt) in edges :
                        continue
                    edges.add((tgt, src))
                    if group_conflict :
                        out.write(f"  {src} -> {tgt} ["
                                  "style=dashed arrowhead=none];\n"
                                  f"  subgraph {{ rank=same; {src}; {tgt}; }}\n")
                else :
                    out.write(f"  {src} -> {tgt};\n")
            out.write("}\n")
        fmt = path.suffix.lower().lstrip(".")
        if not fmt == "dot" :
            check_call([engine, f"-T{fmt}", "-o", str(path), str(dot_path)])
            dot_path.unlink()

class Unfolding (_g2dot) :
    def events (self) :
        start = self.g["e1"]
        stop = start + self.g["events"]
        yield from ((n, self.g.vs[n].attributes()) for n in range(start, stop))
    def conditions (self) :
        start = self.g["c1"]
        stop = start + self.g["conditions"]
        yield from ((n, self.g.vs[n].attributes()) for n in range(start, stop))
    @classmethod
    def from_mci (cls, mci) :
        if isinstance(mci, str) :
            stream = open(mci, "rb")
        elif isinstance(mci, Path) :
            stream = mci.open("rb")
        elif isinstance(mci, io.IOBase) :
            stream = mci
        else :
            raise ValueError(f"don't know how to read {mci!r}")
        INTSIZE = struct.calcsize("i")
        def readint () :
            return struct.unpack("i", stream.read(INTSIZE))[0]
        ev2tr, co2pl = {}, {}
        numco = readint()
        numev = readint()
        g = ig.Graph(numco + numev,
                     directed=True,
                     graph_attrs={"cutoff" : set(),
                                  "dummy" : set(),
                                  "tokens" : set(),
                                  "events" : numev,
                                  "e1" : 0,
                                  "conditions" : numco,
                                  "c1" : numev})
        for ev in range(numev) :
            ev2tr[ev] = readint() - 1
            g.vs[ev]["kind"] = "event"
            g.vs[ev]["id"] = f"e{ev+1}"
        for co in range(numco) :
            idx = co + numev
            co2pl[co] = readint() - 1
            g.vs[idx]["kind"] = "condition"
            g.vs[idx]["id"] = f"c{co+1}"
            toks = readint()
            if toks :
                g["tokens"].add(idx)
            ev = readint()
            if ev :
                g.add_edge(ev-1, idx)
            while True :
                ev = readint()
                if not ev :
                    break
                g.add_edge(idx, ev-1)
        while True :
            ev = readint()
            if not ev :
                break
            g["cutoff"].add(ev-1)
            g["dummy"].add(readint()-1)
        while readint() :
            pass
        numpl = readint()
        numtr = readint()
        readint() # max len of names
        names = [n.decode() for n in stream.read().split(b'\0') if n]
        for ev in range(numev) :
            g.vs[ev]["name"] = names[ev2tr[ev] + numpl]
        for co in range(numco) :
            g.vs[co + numev]["name"] = names[co2pl[co]]
        return cls(g)
    def ev_struct (self) :
        return EventStructure.from_unfolding(self)

class EventStructure (_g2dot) :
    @classmethod
    def from_unfolding (cls, unf) :
        events = unf.g["events"]
        g = ig.Graph(events,
                     directed=True,
                     graph_attrs={"cutoff" : set(unf.g["cutoff"]),
                                  "dummy" : set(unf.g["dummy"])},
                     vertex_attrs={"id" : unf.g.vs["id"],
                                   "name" : unf.g.vs["name"]})
        for ev in range(events) :
            succ = {(ev, s)
                    for c in unf.g.successors(ev)
                    for s in unf.g.successors(c)}
            g.add_edges(succ, {"kind" : ["succ"] * len(succ)})
            conf = {(ev, x)
                    for c in unf.g.predecessors(ev)
                    for x in unf.g.successors(c) if x != ev}
            g.add_edges(conf, {"kind" : ["conflict"] * len(conf)})
        return cls(g)
