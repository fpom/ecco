# from functools import cache
from itertools import product
from collections import Counter

from rich.text import Text

from .record import Printable, Printer


class OpSig(dict):
    _keys: dict[str, str] = {"<": "<>", "<=": "<>", ">": "<>", ">=": "<>", "=": "=="}
    _zero: type = int

    def __setitem__(self, key, val):
        return super().__setitem__(self._keys.get(key, key), val)

    def __missing__(self, key):
        k = self._keys.get(key, key)
        v = self[k] = self._zero()
        return v

    def __le__(self, other):
        return all(v <= other.get(k, self._zero()) for k, v in self.items())

    @classmethod
    def from_action(cls, act):
        sig = cls()
        for g in act.guard:
            sig[g.op] += 1
        sig[":="] = len(act.assign)
        sig["??"] = len(act.read_vars)
        return sig


class OpSigSet(OpSig):
    _zero = set

    def __le__(self, other):
        return all(len(v) <= len(other.get(k, self._zero())) for k, v in self.items())

    def match(self, other, vmap):
        return all(
            {vmap[x] for x in v} <= other.get(k, self._zero()) for k, v in self.items()
        )


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

    def match(self, other, vmap):
        e = OpSigSet()
        return all(v.match(other.get(vmap[k], e), vmap) for k, v in self.items())


class ConsError(Exception):
    pass


class ConsCons(dict):
    def add(self, var, pat, mod):
        if not (pat and mod):
            return
        for p in pat:
            if (var, p) in self:
                if i := self[var, p] & mod:
                    self[var, p] = i
                else:
                    raise ConsError
            else:
                self[var, p] = mod


class Match(Printable):
    def __init__(self, actions, variables):
        self.variables = dict(variables)
        self._styles = {
            ("var", m): Text.assemble(
                Text(m, "blue bold"), Text(f":{p}", "dim bold blue")
            )
            for p, m in self.variables.items()
        }
        self.actions = {}
        modl = None
        for pa, ma in dict(actions).items():
            if modl is None:
                vars = tuple(
                    ma._mod.v[m].copy(name=p) for p, m in self.variables.items()
                )
                modl = pa._mod.copy(variables=vars)
            npa = pa.copy()
            npa.__dict__["_mod"] = modl
            self.actions[npa] = ma

    def __txt__(self, styles={}):
        prn = Printer(styles)
        _styles = self._styles | styles
        with prn.match:
            return (prn / "\n")(
                Text("matched:", "red bold"),
                *(prn["  ", p.__txt__(styles)] for p in self.actions.keys()),
                prn[
                    Text("with", "red bold"),
                    " ",
                    (prn / ", ")(
                        prn[prn(p, "var"), prn(":", "op"), prn(m, "var")]
                        for p, m in self.variables.items()
                    ),
                    " ",
                    Text("as:", "red bold"),
                ],
                *(prn["  ", m.__txt__(_styles)] for m in self.actions.values()),
            )

    @classmethod
    def match(cls, pat, mod):
        _pat = pat.constraints + pat.rules
        _mod = mod.constraints + mod.rules
        pat_sigs = {a: OpSig.from_action(a) for a in _pat}
        mod_sigs = {a: OpSig.from_action(a) for a in _mod}
        for match in cls._group_sig(pat_sigs, mod_sigs, cls._match_counts):
            for vm in cls._match_vars(match, mod.v):
                yield cls(match, vm)

    @classmethod
    def _group_sig(cls, pat, mod, *checks):
        groups = [[ma for ma, ms in mod.items() if ps <= ms] for ps in pat.values()]
        p = tuple(pat.keys())
        for m in product(*groups):
            if len(m) == len(set(m)):
                if all(
                    chk[0](p, m, *chk[1:]) if isinstance(chk, tuple) else chk(p, m)
                    for chk in checks
                ):
                    yield dict(zip(pat.keys(), m))

    @classmethod
    def _match_counts(cls, pat, mod):
        pc = cls._count_sigs(pat)
        mc = cls._count_sigs(mod)
        return all(
            CountSig.from_action(p, pc) <= CountSig.from_action(m, mc)
            for p, m in zip(pat, mod)
        )

    @classmethod
    def _count_sigs(cls, actions):
        t, r, s = Counter(), Counter(), Counter()
        for a in actions:
            t.update(a.test_vars)
            r.update(a.read_vars)
            s.update(a.set_vars)
        return {v: (t[v], r[v], s[v]) for v in t | r | s}

    @classmethod
    def _match_vars(cls, match, vdecls):
        pat_sigs = VarSig.from_actions(match.keys())
        mod_sigs = VarSig.from_actions(match.values())
        for match in cls._group_sig(
            pat_sigs,
            mod_sigs,
            (cls._match_vsig, pat_sigs, mod_sigs),
            (cls._match_consts, match, vdecls),
        ):
            yield match

    @classmethod
    def _match_vsig(cls, pat, mod, pat_sigs, mod_sigs):
        return pat_sigs.match(mod_sigs, dict(zip(pat, mod)))

    @classmethod
    def _match_consts(cls, pat, mod, act_match, vdecls):
        var_match = dict(zip(pat, mod))
        cons = ConsCons()
        for pact, mact in act_match.items():
            for pe in (g for g in pact.guard if len(g.vars) == 1):
                pv = next(iter(pe.vars))
                mv = var_match[pv]
                dom = vdecls[mv].domain
                found = False
                for me in (g for g in mact.guard if g.vars == {mv}):
                    found = True
                    ps = {d for d in dom if pe.sub({pv: d}).as_bool}
                    ms = {d for d in dom if me.sub({mv: d}).as_bool}
                    try:
                        cons.add(pv, ps, ms)
                        cons.add(pv, dom - ps, dom - ms)
                    except ConsError:
                        return False
                if not found:
                    return False
            for pv, pe in ((k, v) for k, v in pact.assign.items() if not v.vars):
                mv = var_match[pv]
                dom = vdecls[mv].domain
                if mv not in mact.assign:
                    return False
                me = mact.assign[mv]
                if me.vars:
                    return False
                ps = {pe.as_int}
                ms = {me.as_int}
                try:
                    cons.add(pv, ps, ms)
                    cons.add(pv, dom - ps, dom - ms)
                except ConsError:
                    return False
        return True
