from functools import cache
from itertools import product
from collections import Counter


class OpSig(dict):
    _ops = {"<": "<>", "<=": "<>", ">": "<>", ">=": "<>", "=": "=="}

    def __setitem__(self, key, val):
        return super().__setitem__(self._ops.get(key, key), val)

    def __missing__(self, key):
        return self.get(self._ops.get(key, key), 0)

    def __le__(self, other):
        return all(v <= other.get(k, 0) for k, v in self.items())

    @classmethod
    def from_action(cls, act):
        sig = cls()
        for g in act.guard:
            sig[g.op] += 1
        sig[":="] = len(act.assign)
        sig["??"] = len(act.read_vars)
        return sig


class OpSigSet(OpSig):
    def __missing__(self, key):
        k = self._ops.get(key, key)
        v = self[k] = set()
        return v


class CountSig(list):
    @classmethod
    def from_action(cls, act, counts):
        return cls([counts[v] for v in act.vars])

    def __le__(self, other):
        oth = iter(sorted(other))
        for s in sorted(self):
            try:
                while s > next(oth):
                    pass
            except StopIteration:
                return False
        return True


class VarSig(dict):
    @classmethod
    def from_actions(cls, actions):
        sig = cls()
        for act in actions:
            for g in act.guard:
                for v in g.vars:
                    sig[v][g.op].update(g.vars - {v})
            for v, x in act.assign.items():
                sig[v][":="].update(x.vars)
        return sig

    def __missing__(self, key):
        v = self[key] = OpSigSet()
        return v

    def __le__(self, other):
        e = OpSigSet()
        return all(v <= other.get(k, e) for k, v in self)


class Match:
    def __init__(self, actions, variables):
        self.actions = dict(actions)
        self.variables = dict(variables)

    @classmethod
    def match(cls, pat, mod):
        _pat = pat.constraints + pat.rules
        _mod = mod.constraints + mod.rules
        # match on operators and variables counts
        pat_sigs = [OpSig.from_action(a) for a in _pat]
        mod_sigs = [OpSig.from_action(a) for a in _mod]
        groups = []
        for ps in pat_sigs:
            groups.append(cur := [])
            for m, ms in zip(_mod, mod_sigs):
                if ps <= ms:
                    cur.append(m)
        for combo in product(*groups):
            if len(combo) == len(set(combo)) and cls._match_counts(_pat, combo):
                for v in cls._match_vars(_pat, combo):
                    yield cls(zip(_pat, combo), v)

    @classmethod
    def _match_counts(cls, pat, mod):
        pc = cls._count_sigs(pat)
        mc = cls._count_sigs(mod)
        return all(
            CountSig.from_action(p, pc) <= CountSig.from_action(m, mc)
            for p, m in zip(pat, mod)
        )

    @classmethod
    @cache
    def _count_sigs(cls, actions):
        t, r, s = Counter(), Counter(), Counter()
        for a in actions:
            t.update(a.test_vars)
            r.update(a.read_vars)
            s.update(a.set_vars)
        return {v: (t[v], r[v], s[v]) for v in t | r | s}

    @classmethod
    def _match_vars(cls, pat, mod):
        pat_sigs = VarSig.from_actions(pat)
        mod_sigs = VarSig.from_actions(mod)

    # @classmethod
    # def _match_variables(cls, act_match, pat_counts, mod_counts, vdecl, expr_cls):
    #     # match variables within an action match
    #     # 1.compute matches based on count only
    #     prematch = {}
    #     for p, pc in pat_counts.items():
    #         # for each variable p in the pattern, with count pc
    #         prematch[p] = set()
    #         for m, mc in mod_counts.items():
    #             # for each variable m in the model, with count mc
    #             if all(x >= y for x, y in zip(mc, pc)):
    #                 # if m appears never less that p, it may be a match
    #                 prematch[p].add(m)
    #     # 2. examine each potential matche
    #     for combo in product(*prematch.values()):
    #         # skip when the same model variable is used to match several pattern variables
    #         if len(set(combo)) != len(combo):
    #             continue
    #         # map pattern variables to model variables
    #         vmap = dict(zip(prematch.keys(), combo))
    #         # skip when pattern has more variables that model
    #         if not all(
    #             {vmap[v] for v in p.test_vars} <= m.test_vars
    #             and {vmap[v] for v in p.read_vars} <= m.read_vars
    #             and {vmap[v] for v in p.set_vars} <= m.set_vars
    #             for p, m in act_match.items()
    #         ):
    #             continue
    #         # check that match is satisfiable
    #         for p, m in act_match.items():
    #             pass
    #         if expr_cls.sat(dom, iff=iff):
    #             yield vmap
