import ecco.ui as ui
import bz2
import numpy as np
from pathlib import Path

from ecco.tables import mkpath

ctypedef unsigned long long NODEID
ctypedef unsigned long long COUNT

##
## import DDD stuff
##

from ddd cimport ddd, sdd, shom, makeddd, makesdd
from ddd import ddd_save, ddd_load
from its cimport model
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "dddwrap.h" :
    ctypedef short val_t

cdef extern from "ddd/DDD.h" :
    cdef cppclass DDD :
        DDD ()
        DDD (const DDD &)
        DDD (int var, val_t val, const DDD &d)
        DDD (int var, val_t val)
        @staticmethod
        void varName (int var, const string &name)
        @staticmethod
        const string getvarName (int var)
        int variable() const

cdef extern from "ddd/SDD.h" :
    cdef cppclass SDD :
        pass

cdef extern from "dddwrap.h" :
    cdef bint ddd_is_STOP(DDD &d)
    cdef DDD ddd_new_ONE()
    ctypedef pair[val_t,DDD] *ddd_iterator
    cdef ddd_iterator ddd_iterator_begin (DDD d)
    cdef void ddd_iterator_next (ddd_iterator i)
    cdef bint ddd_iterator_end (ddd_iterator i, DDD d)
    cdef DDD ddd_iterator_ddd (ddd_iterator i)
    cdef val_t ddd_iterator_value (ddd_iterator i)
    cdef SDD sdd_new_SDDd (int var, const DDD &val, const SDD &s)
    cdef const SDD sdd_ONE
    cdef bint sdd_is_STOP(SDD &s)
    ctypedef DDD* DDD_star
    ctypedef vector[pair[DDD_star,SDD]] sdd_iterator
    cdef sdd_iterator sdd_iterator_begin (SDD s)
    cdef DDD *sdd_iterator_DDD_value (sdd_iterator i)
    cdef void sdd_iterator_next (sdd_iterator i)
    cdef bint sdd_iterator_end (sdd_iterator i, SDD s)

##
## SDD/DDD utilities
##

cdef inline sdd ddd2sdd (ddd d) :
    # make a sdd out of a ddd
    return sdd.mkz(d)

cdef inline ddd sdd2ddd (sdd s) :
    # make a ddd out of a sdd
    cdef DDD* d
    cdef sdd_iterator it = sdd_iterator_begin(s.s)
    if not sdd_is_STOP(s.s) and not sdd_iterator_end(it, s.s) :
        d = sdd_iterator_DDD_value(it)
        return makeddd(d[0])
    return ddd.empty()

cdef str DDD_min (str path, DDD head) :
    # get the minimal value of a DDD in lexicographic order
    cdef DDD child
    cdef val_t value
    cdef ddd_iterator it
    if ddd_is_STOP(head) :
        return path
    else :
        it = ddd_iterator_begin(head)
        while not ddd_iterator_end(it, head) :
            value = ddd_iterator_value(it)
            child = ddd_iterator_ddd(it)
            if value == 0 :
                return DDD_min(path + "0", child)
            ddd_iterator_next(it)
        return DDD_min(path + "1", child)

cdef inline str sdd_min (sdd s) :
    # get the minimal value of a sdd in lexicographic order
    cdef ddd d = sdd2ddd(s)
    return DDD_min("", d.d)

##
## Components
##

from cpython cimport array
import array
cimport cython

cdef unsigned int SIZE = len(state_variables)
cdef array.array ZERO = array.array("Q", [0] * SIZE)

cdef class count (object) :
    """simpler/faster/smaller version of `collection.Counter` to count variables
    occurrences in a `Component`
    """
    cdef array.array a
    def __init__ (count self) :
        "initialise a `count` with zeros"
        self.a = array.clone(ZERO, SIZE, zero=True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __add__ (count self, count other) -> count :
        "add two `count`"
        cdef COUNT i
        cdef count c = count.__new__(count)
        c.a = array.clone(ZERO, SIZE, zero=False)
        for i in range(SIZE) :
            c.a[i] = self.a[i] + other.a[i]
        return c
    def __setitem__ (count self, str var, COUNT val) :
        "assign the counter for a specific variable"
        self.a[vnum[var]] = val
    def __getitem__ (count self, str var) :
        "get the counter for a specific variable"
        return self.a[vnum[var]]
    def __iter__ (count self) :
        "iterate over pairs variable, counter"
        cdef str var
        cdef COUNT num
        for var, num in vnum.items() :
            yield var, self.a[num]

cdef enum kind :
    # kind of components
    init = 1
    dead = 2
    scc = 4
    hull = 8

cdef class Component (object) :
    """a subset of the reachable states
    """
    cdef readonly sdd s
    cdef readonly NODEID n
    cdef unsigned char p
    def __init__ (self, sdd s, NODEID n, unsigned char p=0) :
        self.s = s
        self.n = n
        self.p = p
    def __len__ (Component self) :
        "number of states in the component"
        return len(self.s)
    def __hash__ (Component self) :
        return self.s.__hash__()
    def __eq__ (Component self, Component other) :
        return self.s == other.s
    def __ne__ (Component self, Component other) :
        return self.s != other.s
    cdef inline bint has (Component self, kind k) :
        return bool(self.p & k)
    @property
    def init (self) :
        return self.has(kind.init)
    @property
    def dead (self) :
        return self.has(kind.dead)
    @property
    def scc (self) :
        return self.has(kind.scc)
    @property
    def hull (self) :
        return self.has(kind.hull)
    cpdef count count (Component self) :
        "return the number of on occurrences of each variable"
        cdef dict seen = {}
        cdef ddd d = sdd2ddd(self.s)
        return self._count(d, seen)
    cpdef count _count (Component self, ddd head, dict seen) :
        cdef count ret, sub
        cdef str var
        cdef int num
        cdef val_t val
        cdef ddd child
        seen[head] = ret = count()
        if head.stop() :
            return ret
        for var, num, val, child in head.edges() :
            if child in seen :
                sub = seen[child]
            else :
                sub = seen[child] = self._count(child, seen)
            if val :
                ret[var] = len(child)
            ret = ret + sub
        return ret
    cpdef tuple on_off (Component self) :
        "return a pair of states `(on, off)` for the always-on and always-off variables"
        cdef count cnt = self.count()
        cdef state on = state.none()
        cdef state off = state.none()
        cdef COUNT s, size
        size = len(self.s)
        for var, num in vnum.items() :
            s = cnt.a[num]
            if s == 0 :
                off |= var
            elif s == size :
                on |= var
        return on, off

##
## store the changes made on a graph
##

cdef class diff (object) :
    cdef readonly set rem, add
    cdef readonly list components
    def __init__ (self, compo=[]) :
        self.rem = set()
        self.add = set()
        self.components = list(compo)
    def __bool__ (self) :
        return len(self.components) > 1
    def __len__ (self) :
        return len(self.components)
    def __iadd__ (self, other) :
        self.rem.update(other.rem)
        self.add.update(other.add)
        return self
    def __iter__ (self) :
        return iter(self.components)
    def __getitem__ (self, index) :
        return self.components[index]
    cpdef void extend (diff self, diff other) :
        self += other
        self.components.extend(other.components)
    cpdef Component pop (diff self) :
        cdef Component c = self.components.pop(-1)
        cdef set drop, s
        cdef NODEID src, dst
        cdef tuple t
        for s in [self.rem, self.add] :
            drop = set()
            for src, dst in s :
                if src == c.n or dst == c.n :
                    drop.add((src, dst))
            for t in drop :
                s.discard(t)
        return c

##
## Component Graph
##

ctypedef void (*itercb)(Graph, state, sdd)
ctypedef pair[int,val_t] DDD_edge

cdef class Graph (object) :
    # gal / sdd / shom
    cdef readonly str path
    cdef readonly model m
    cdef readonly sdd initial, reachable, transient, deadlocks
    cdef readonly shom succ, pred, succ_o, pred_o, succ_s, pred_s
    cdef dict tsucc
    # user-level information
    cdef readonly bint compact, universe
    cdef readonly tuple variables
    # internal use
    cdef Logger log
    cdef NODEID last
    cdef dict _var2sdd_cached
    # graph
    cdef dict nodes
    cdef dict edges_preds, edges_succs
    ##
    ## create / save / reload / access content
    ##
    def __init__ (self, str path, bint compact, bint universe, bint verbose=False) :
        self.path = path
        self.m = model(path, fmt="GAL")
        if universe :
            self.initial = self.reachable = self._universe()
        else :
            self.initial = self.reachable = self.m.initial()
        self.transient = sdd.empty()
        self.deadlocks = sdd.empty()
        self.tsucc = {}
        #
        self.compact = compact
        self.universe = universe
        self.variables = sdd2ddd(self.initial).vars()
        #
        self.log = Logger()
        self.log.verbose = verbose
        self.last = 0
        self._var2sdd_cached = {}
        #
        self.nodes = {}
        self.edges_preds = {}
        self.edges_succs = {}
    cdef sdd _universe (self) :
        cdef ddd d = ddd.one()
        cdef ddd i = next(iter(self.m.initial()))[0]
        cdef str v
        for v in reversed(i.vars()) :
            d = ddd.from_range(v, 0, 1, d)
        return sdd.mkz(d)
    cpdef save (Graph self, str path) :
        cdef ddd i = sdd2ddd(self.initial)
        cdef ddd r = sdd2ddd(self.reachable)
        cdef ddd t = sdd2ddd(self.transient)
        cdef ddd d = sdd2ddd(self.deadlocks)
        cdef Component c
        cdef list s = []
        cdef list p = []
        cdef list n = []
        for c in self.nodes.values() :
            s.append(sdd2ddd(c.s))
            p.append(c.p)
            n.append(c.n)
        ddd_save(path, i, r, t, d, *s,
                 properties=p,
                 numbers=n,
                 gal_path=self.path,
                 compact=self.compact,
                 universe=self.universe,
                 last=self.last,
                 edges_preds=self.edges_preds,
                 edges_succs=self.edges_succs,
                 verbose=self.log.verbose,
                 transitions=bool(self.tsucc))
    @classmethod
    def load (cls, str path) :
        cdef Graph g
        cdef dict headers, e
        cdef ddd d
        cdef unsigned char p
        cdef NODEID n
        cdef list ddds
        headers, ddds = ddd_load(path)
        g = cls(headers["gal_path"], headers["compact"], headers["universe"],
                headers["verbose"])
        g.initial = ddd2sdd(ddds[0])
        g.reachable = ddd2sdd(ddds[1])
        g.transient = ddd2sdd(ddds[2])
        g.deadlocks = ddd2sdd(ddds[3])
        g.last = headers["last"]
        g.nodes = {n : Component(ddd2sdd(d), n, p)
                   for d, n, p in zip(ddds[4:],
                                      headers["numbers"],
                                      headers["properties"])}
        g.edges_preds = headers["edges_preds"]
        g.edges_succs = headers["edges_succs"]
        if headers["transitions"] :
            if g.compact :
                g.mk_compact_trans(False)
            else :
                g.mk_full_trans()
        return g
    def __getitem__ (self, item) :
        """get elements of the graph:
         - if item is an int => a component by its number
         - if item is a pair of ints => an edge between two components (or `None`)
         - if item is a rule/constraint name => the corresponding `shom`
         - if item is "*" => all potential states as a `sdd`
         - if item is "#" => all deadlocks as a `sdd`
         - if item is a variable name => the subset of "*" that valuates variable to 1
        """
        cdef NODEID x, y
        if isinstance(item, int) :
            x = item
            return self.nodes[x]
        elif isinstance(item, tuple) and len(item) == 2 :
            x, y = item
            return self.edges_succs.get(x, {}).get(y, None)
        elif isinstance(item, str) :
            if item in self.tsucc :
                return self.tsucc[item]
            elif item == "*" or item in self.variables :
                return self._var2sdd(item)
            elif item == "#" :
                return self.deadlocks
            else :
                raise ValueError("unknown variable/rule name '%s'" % item)
        else :
            raise TypeError("expected 'int', 'int,int', or 'str' as variable/rule name"
                            " but got '%s'" % item.__class__.__name__)
    cdef sdd _var2sdd (Graph self, str var) :
        cdef str v
        cdef ddd d
        if var not in self._var2sdd_cached :
            d = ddd.one()
            for v in reversed(self.variables) :
                if v == var :
                    d = ddd.from_range(var, 1, 1, d)
                else :
                    d = ddd.from_range(v, 0, 1, d)
            self._var2sdd_cached[var] = ddd2sdd(d)
        return self._var2sdd_cached[var]
    @property
    def nodes_count (self) :
        return len(self.nodes)
    @property
    def edges_count (self) :
        cdef COUNT count = 0
        cdef dict edges
        for edges in self.edges_succs.values() :
            count += len(edges)
        return count
    cpdef dict getsucc (Graph self, NODEID num) :
        """get the successors of a node
        Return: as a `dict` mapping node numbers to sets of rules/constraints names
        """
        return self.edges_succs.get(num, {})
    cpdef dict getpred (Graph self, NODEID num) :
        """get the successors of a node
        Return: as a `dict` mapping node numbers to sets of rules/constraints names
        """
        cdef dict edges = {}
        cdef set preds = self.edges_preds.get(num, set())
        cdef NODEID src
        for src in preds :
            edges[src] = self.edges_succs[src][num]
        return edges
    def __iter__ (self) :
        "iterate over the nodes (instances of `Component`)"
        cdef Component c
        for c in self.nodes.values() :
            yield c
    def edges (self) :
        "iterate over the edges as triples (source, destination, rule names)"
        cdef NODEID src, dst
        cdef dict edges
        cdef set rules
        for src, edges in self.edges_succs.items() :
            for dst, rules in edges.items() :
                yield src, dst, rules
    ##
    ## manage graph nodes/edges
    ##
    cdef void update_prop (Graph self, Component c) :
        # update the properties of a component
        cdef sdd s
        c.p = 0
        if c.s & self.initial :
            c.p |= kind.init
        if c.s & self.deadlocks :
            c.p |= kind.dead
        elif len(c.s) > 1 :
            s = c.s.pick()
            if (self.succ_s(s) & self.pred_s(s)) == c.s :
                c.p |= kind.scc
            elif (self.succ_o(c.s) & self.pred_o(c.s)) == c.s :
                c.p |= kind.hull
    cdef void _update_edges (Graph self, Component src, Component dst, diff patch) :
        cdef str t
        cdef shom h
        cdef dict succs, d
        cdef set rules, preds
        cdef bint add = False
        self.edges_succs.get(src.n, {}).pop(dst.n, None)
        self.edges_preds.get(dst.n, set()).discard(src.n)
        patch.rem.add((src.n, dst.n))
        for t, h in self.tsucc.items() :
            if h(src.s) & dst.s :
                succs = self.edges_succs.setdefault(src.n, {})
                rules = succs.setdefault(dst.n, set())
                patch.add.add((src.n, dst.n))
                rules.add(t)
                add = True
        if add :
            preds = self.edges_preds.setdefault(dst.n, set())
            preds.add(src.n)
        if dst.n in self.edges_preds and not self.edges_preds[dst.n] :
            del self.edges_preds[dst.n]
        if src.n in self.edges_succs and not self.edges_succs[src.n] :
            del self.edges_succs[src.n]
    cdef diff update_edges (Graph self, Component node, set preds, set succs) :
        cdef Component src, dst
        cdef diff patch = diff()
        for n in preds :
            if n != node.n :
                src = self.nodes[n]
                self._update_edges(src, node, patch)
        for n in succs :
            if n != node.n :
                dst = self.nodes[n]
                self._update_edges(node, dst, patch)
        return patch
    cdef diff add_compo (Graph self, sdd s, set preds=set(), set succs=set()) :
        cdef Component c = Component(s, self.last)
        cdef diff patch
        self.update_prop(c)
        self.nodes[self.last] = c
        self.last += 1
        patch = self.update_edges(c, preds, succs)
        patch.components.append(c)
        return patch
    cdef diff replace_compo (Graph self, sdd s, NODEID num,
                             set preds=set(), set succs=set()) :
        cdef Component c = Component(s, num)
        cdef diff patch
        self.update_prop(c)
        self.nodes[num] = c
        patch = self.update_edges(c, preds, succs)
        patch.components.append(c)
        return patch
    cpdef diff del_compo (Graph self, NODEID num) :
        cdef diff patch = diff([self.nodes.pop(num)])
        cdef NODEID src, dst
        for src in self.edges_preds.get(num, {}) :
            patch.rem.add((src, num))
        for dst in self.edges_succs.get(num, {}) :
            patch.rem.add((num, dst))
        for src, dst in patch.rem :
            self.edges_succs.get(src, {}).pop(dst, None)
            if src in self.edges_succs and not self.edges_succs[src] :
                del self.edges_succs[src]
            self.edges_preds.get(dst, {}).discard(src)
            if dst in self.edges_preds and not self.edges_preds[dst] :
                del self.edges_preds[dst]
        return patch
    ##
    ## build / split / merge components
    ##
    cpdef build (Graph self, bint split=True) :
        cdef diff patch
        self.mk_reachable()
        if self.compact :
            self.mk_compact_trans()
        else :
            self.mk_full_trans()
        patch = self.add_compo(self.reachable)
        if split :
            self.split_hull(patch.components[0], True, True)
    cdef void mk_reachable (Graph self) :
        cdef sdd reach
        cdef shom succ
        with self.log(head="<b>computing reachability:</b>",
                      done_head="<b>computed:</b> {done} states",
                      tail="{done} states (TIME: {time} | MEM: {memory:.1f}%)",
                      done_tail="(TIME: {time})",
                      keep=True) :
            succ = self.m.succ() | shom.ident()
            while True :
                reach = succ(self.reachable)
                if reach == self.reachable :
                    break
                self.reachable = reach
                self.log.update_to(len(reach))
    cdef void mk_full_trans (Graph self) :
        cdef str t
        cdef shom h
        cdef dict d
        with self.log(head="builing transition relations",
                      done_head="done building transition relations",
                      tail="{step} (TIME: {time} | ETA: {eta} | MEM: {memory:.1f}%)",
                      done_tail="(TIME: {time})",
                      steps=["succ", "pred", "fixpoints"]) :
            # succ
            self.succ = self.m.succ()
            d = self.m.transitions()
            for t, h in d.items() :
                self.tsucc[t] = h
            self.log.update()
            # pred
            self.pred = self.succ.invert(self.reachable)
            self.log.update()
            # fixpoint
            self.succ_o = self.succ.gfp()
            self.pred_o = self.pred.gfp()
            self.succ_s = self.succ.lfp()
            self.pred_s = self.pred.lfp()
            self.log.update()
    cdef void mk_compact_trans (Graph self, bint full=True) :
        cdef dict d
        cdef str t
        cdef shom h, succ_u, pred_u
        cdef list u
        with self.log(head="builing transition relations",
                      done_head="<b>removed:</b>",
                      tail="{step} (TIME: {time} | ETA: {eta} | MEM: {memory:.1f}%)",
                      done_tail="(TIME: {time})",
                      steps=["urgent", "transient", "succ", "pred", "fixpoints", "init"],
                      keep=True) :
            # urgent
            d = self.m.transitions()
            u = []
            for t, h in d.items() :
                if t[0] in ("c", "C") :
                    u.append(h)
            succ_u = shom.union(*u).lfp()
            pred_u = shom.union(*u).invert(self.reachable)
            self.log.update()
            # transient
            if full :
                self.transient = pred_u(self.reachable)
            self.reachable -= self.transient
            self.log.update()
            # succ
            self.succ = (succ_u * self.m.succ()) - self.transient
            self.tsucc = {}
            for t, h in d.items() :
                self.tsucc[t] = (succ_u * h) - self.transient
            self.log.update()
            # pred
            self.pred = self.succ.invert(self.reachable) - self.transient
            self.log.update()
            # fixpoint
            self.succ_o = self.succ.gfp() - self.transient
            self.pred_o = self.pred.gfp() - self.transient
            self.succ_s = self.succ.lfp() - self.transient
            self.pred_s = self.pred.lfp() - self.transient
            self.log.update()
            # init
            if full :
                self.initial = succ_u(self.initial) - self.transient
            self.log.done_tail = ("%s transient states, %s left (TIME: {time})"
                                  % (len(self.transient), len(self.reachable)))
            self.log.update()
    cpdef diff split_states (Graph self, NODEID num, sdd states) :
        cdef Component compo = self.nodes[num]
        cdef sdd inside = compo.s & states
        cdef sdd outside = compo.s - states
        cdef set succs, preds
        cdef diff patch
        if inside and outside :
            preds = {num} | set(self.edges_preds.get(num, []))
            succs = {num} | set(self.edges_succs.get(num, []))
            patch = self.replace_compo(inside, num, preds, succs)
            patch.extend(self.add_compo(outside, preds, succs))
        else :
            patch = diff([compo])
        return patch
    cpdef diff split_hull (Graph self, Component c,
                           bint deadlocks=False, bint verbose=False) :
        cdef diff patch = diff()
        cdef sdd hull, head, tail, rest, done
        cdef Component last
        with self.log(head="extracting",
                      done_head="<b>found:</b>",
                      tail="{step} (TIME: {time} | ETA: {eta} | MEM: {memory:.1f}%)",
                      done_tail="(TIME: {time})",
                      steps=["deadlocks", "SCC hull", "initial states",
                             "final states", "initial-to-final states",
                             "dead ends"],
                      verbose=verbose,
                      keep=True) :
            if deadlocks :
                self.deadlocks = self.reachable - self.pred(self.reachable)
            self.log.update()
            done = hull = self.pred_o(c.s) & self.succ_o(c.s)
            if hull == c.s or not hull :
                patch.components.append(c)
                self.log.finish()
                return patch
            else :
                patch.extend(self.split_states(c.n, hull))
                self.log.update()
            head = (self.pred_s & c.s)(hull) - hull
            if head :
                done |= head
                last = patch.pop()
                patch.extend(self.split_states(last.n, head))
            self.log.update()
            tail = (self.succ_s & c.s)(hull) - hull
            if tail :
                done |= tail
                last = patch.pop()
                patch.extend(self.split_states(last.n, tail))
                self.log.update()
                skip = (self.pred_s & c.s)(tail) - done
                if skip :
                    last = patch.pop()
                    patch.extend(self.split_states(last.n, skip))
                done |= skip
                self.log.update()
            else :
                self.log.update(2)
            rest = c.s - done
            if rest :
                last = patch.pop()
                patch.extend(self.split_states(last.n, rest))
            self.log.update()
            return patch
    cpdef diff merge (Graph self, object components) :
        cdef tuple compo = tuple(components)
        cdef set preds, succs
        cdef Component c, keep
        cdef diff patch = diff()
        cdef diff p
        keep = compo[0]
        preds = set(self.edges_preds.get(keep.n, {}))
        succs = set(self.edges_succs.get(keep.n, {}))
        for c in compo[1:] :
            keep.s |= c.s
            preds.update(self.edges_preds.get(c.n, {}))
            succs.update(self.edges_succs.get(c.n, {}))
            patch += self.del_compo(c.n)
        for c in compo[1:] :
            preds.discard(c.n)
            succs.discard(c.n)
        patch.extend(self.replace_compo(keep.s, keep.n, preds, succs))
        patch.components.extend(compo[1:])
        return patch
    ##
    ## dumps to CSV
    ##
    cpdef void to_csv (Graph self, str nodes, str edges) :
        cdef NODEID num, s, d
        cdef Component src, dst
        cdef COUNT size
        cdef count cnt
        cdef str var
        cdef state on, off
        cdef set rules
        cdef Component c
        cdef list succs, preds
        with self.log(head="saving",
                      done_head="saved",
                      tail= nodes,
                      total=self.nodes_count + self.edges_count + 1) :
            path = mkpath(nodes)
            with open(path.typ, "w") as out :
                out.write(repr({"node": "int64",
                                "size": "int64",
                                "succ": "object",
                                "pred": "object",
                                "on": "state",
                                "off": "state",
                                "init": "bool",
                                "dead": "bool",
                                "scc": "bool",
                                "hull": "bool"}) + "\n")
            with bz2.open(path.csv, "wt", encoding="utf-8") as csv :
                csv.write("node,size,succ,pred,on,off,init,dead,scc,hull\n")
                for src in self.nodes.values() :
                    size = len(src.s)
                    on, off = src.on_off()
                    succs = []
                    for s in sorted(self.edges_succs.get(src.n, {})) :
                        succs.append(str(s))
                    preds = []
                    for s in sorted(self.edges_preds.get(src.n, {})) :
                        preds.append(str(s))
                    csv.write("{node},{size},{succ},{pred},{on},{off},{init},{dead},"
                              "{scc},""{hull}\n".format(node=src.n,
                                                        size=size,
                                                        succ="|".join(succs),
                                                        pred="|".join(preds),
                                                        on=str(on),
                                                        off=str(off),
                                                        init=src.init,
                                                        dead=src.dead,
                                                        scc=src.scc,
                                                        hull=src.hull))
                    self.log.update()
            self.log.tail = edges
            path = mkpath(edges)
            with open(path.typ, "w") as out :
                out.write(repr({"src": "int64",
                                "dst": "int64",
                                "rule": "object"}) + "\n")
            self.log.update()
            with bz2.open(path.csv, "wt", encoding="utf-8") as csv :
                csv.write("src,dst,rule\n")
                for s in self.edges_succs :
                    for d, rules in self.edges_succs[s].items() :
                        csv.write("{src},{dst},{rule}"
                                  "\n".format(src=s,
                                              dst=d,
                                              rule="|".join(sorted(rules))))
    # used internally for dump_x iterator callback
    cdef object nodes_csv, edges_csv
    cdef dict states_x
    cdef Component compo_x
    cpdef void dump_x (Graph self, list components, str nodes, str edges) :
        cdef Component c
        cdef COUNT total = 0
        for c in components :
            total += len(c.s)
        nodes_path = mkpath(nodes)
        edges_path = mkpath(edges)
        with open(nodes_path.typ, "w") as out :
            out.write(repr({"node": "int64",
                            "succ": "object",
                            "on": "state",
                            "off": "state",
                            "init": "bool",
                            "dead": "bool",
                            "scc": "bool",
                            "hull": "bool",
                            "transient": "bool",
                            "component" : "int64"}) + "\n")
        with open(edges_path.typ, "w") as out :
            out.write(repr({"src": "int64",
                            "dst": "int64",
                            "rule": "object"}) + "\n")
        with bz2.open(nodes_path.csv, "wt", encoding="utf-8") as nodes_csv, \
             bz2.open(edges_path.csv, "wt", encoding="utf-8") as edges_csv, \
             self.log(head="dumping explicit graph",
                      tail="(TIME: {time} | ETA: {eta} | MEM: {memory:.1f}%)",
                      done_head="<b>saved</b>",
                      done_tail=("<code>%s</code> and <code>%s</code>"
                                 % (nodes_path.csv, edges_path.csv)),
                      total=total,
                      keep=True) :
            nodes_csv.write("node,succ,on,off,init,dead,scc,hull,transient,component\n")
            edges_csv.write("src,dst,rule\n")
            self.nodes_csv = nodes_csv
            self.edges_csv = edges_csv
            self.states_x = {}
            for c in components :
                self.compo_x = c
                self._iter(c.s, <itercb>self.add_state)
    cdef add_state (Graph self, state src, sdd asd) :
        cdef NODEID src_num = self.states_x.setdefault(src, len(self.states_x))
        cdef NODEID dst_num
        cdef set succ = set()
        cdef state dst
        cdef str rule
        cdef set rules
        cdef dict edges = {}
        cdef bint transient
        if self.compact :
            transient = False
        else :
            transient = src.transient()
        # write edges rows and collect succ
        for rule, dst in src.succ(self.compact) :
            dst_num = self.states_x.setdefault(dst, len(self.states_x))
            rules = edges.setdefault((src_num, dst_num), set())
            rules.add(rule)
        for (src_num, dst_num), rules in edges.items() :
            self.edges_csv.write("{src},{dst},{rule}\n"
                                 "".format(src=src_num,
                                           dst=dst_num,
                                           rule="|".join(sorted(rules))))
            succ.add(dst_num)
        # write node row
        self.nodes_csv.write("{node},{succ},{on},{off},{init},{dead},{scc},{hull}"
                             ",{trans},{compo}\n".format(
                                 node=src_num,
                                 succ="|".join(map(str, sorted(succ))),
                                 on=src,
                                 off=~src,
                                 init=bool(asd & self.initial),
                                 dead=bool(asd & self.deadlocks),
                                 scc=bool(self.compo_x.p & kind.scc),
                                 hull=bool(self.compo_x.p & kind.hull),
                                 trans=transient,
                                 compo=self.compo_x.n))
        self.log.update()
    cpdef sdd state2sdd (Graph self, state s) :
        cdef ddd d
        cdef str v
        d = ddd.one()
        for v in reversed(self.variables) :
            if s[v] :
                d = ddd.from_range(v, 1, 1, d)
            else :
                d = ddd.from_range(v, 0, 0, d)
        return ddd2sdd(d)
    ##
    ## iterator over states
    ##
    cdef void _iter (Graph self, sdd head, itercb act) :
        cdef ddd hd = sdd2ddd(head)
        cdef vector[DDD_edge] path
        path.resize(len(self.variables))
        self._do_iter(hd.d, state(), path, 0, act)
    cdef void _do_iter (Graph self, DDD head, state on,
                        vector[DDD_edge] path, unsigned int pos,
                        itercb act) :
        cdef DDD child, DDD_path
        cdef DDD_edge edge
        cdef val_t value
        cdef int varid
        cdef ddd_iterator it
        cdef str name
        cdef ddd ddd_path
        cdef sdd sdd_path
        if ddd_is_STOP(head) :
            DDD_path = ddd_new_ONE()
            for edge in reversed(path) :
                DDD_path = DDD(edge.first, edge.second, DDD_path)
            ddd_path = makeddd(DDD_path)
            sdd_path = ddd2sdd(ddd_path)
            act(self, on, sdd_path)
        else :
            it = ddd_iterator_begin(head)
            while not ddd_iterator_end(it, head) :
                value = ddd_iterator_value(it)
                child = ddd_iterator_ddd(it)
                varid = head.variable()
                path[pos] = DDD_edge(varid, value)
                if value :
                    name = head.getvarName(varid).decode()
                    self._do_iter(child, on|name, path, pos+1, act)
                else :
                    self._do_iter(child, on, path, pos+1, act)
                ddd_iterator_next(it)
    cdef object states_cb
    cdef _states_cb (Graph self, state s, sdd, d) :
        self.states_cb(s)
    cpdef void states (Graph self, object cb) :
        "iterate over all the states, calling cb(s) for each state s"
        cdef Component c
        self.states_cb = cb
        for c in self.nodes.values() :
            self._iter(c.s, <itercb>self._states_cb)
