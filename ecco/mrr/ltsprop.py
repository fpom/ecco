import ast
import operator
import re

import tl
import pymc as mc
import ddd

from functools import reduce

from . import parser
from .. import cached_property


class Prop:
    syntax: str | None = None
    _cache: dict = {}
    _cache_as = ("cg.lts", "prop", "syntax")

    def __init__(self, cg, prop):
        self.cg = cg
        self.prop = prop

    @property
    def cache(self):
        key = []
        for path in self._cache_as:
            obj = self
            for attr in path.split("."):
                obj = getattr(obj, attr)
            key.append(obj)
        return self._cache.setdefault(tuple(key), {})

    def __call__(self, states):
        cache = self.cache
        if states not in cache:
            self.states = states
            cache[states] = self._get_states(states)
        return cache[states]

    def _get_states(self, states):
        raise NotImplementedError("abstract method")


class CTLProp(Prop):
    syntax = "CTL"

    def __init__(self, cg, prop, phi=None):
        if phi is None:
            phi = tl.parse(prop).ctl()
        super().__init__(cg, prop)
        self.phi = phi

    def _get_states(self, states):
        cache = self.cache
        if None in cache:
            checker = cache[None]
        else:
            checker = mc.CTL_model_checker(
                self.cg.lts.states, self.cg.lts.pred, self._atom
            )
            cache[None] = checker
        return checker.check(self.phi)

    _eval_op = {
        "==": operator.eq,
        "=": operator.eq,
        "!=": operator.ne,
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
    }
    _swap_op = {"<": ">", "<=": ">=", ">=": "<="}
    _atom_re = re.compile(r"([^=!<>]+)\s*(%s)\s*([^=!<>]+)" % "|".join(_eval_op))

    def _atom(self, text):
        text = text.strip()
        if match := self._atom_re.match(text):
            left, op, right = match.groups()
        else:
            left, op, right = text, ">=", "1"
        if left.isdigit() and right.isdigit():
            left, right = int(left), int(right)
            if self._eval_op[op](left, right):
                return self.cg.lts.states
            else:
                return ddd.sdd.empty()
        elif left.isdigit():
            left, right = right, int(left)
            op = self._swap_op.get(op, op)
        elif right.isdigit():
            right = int(right)
        else:
            raise ValueError(f"invalid atom {text!r}")
        var = self.cg.lts.m2g.get(left, left)
        if op in ("=", "=="):
            return self.cg.lts.var_eq(var, right)
        elif op == "!=":
            return self.cg.lts.var_ne(var, right)
        elif op == "<":
            return self.cg.lts.var_lt(var, right)
        elif op == ">":
            return self.cg.lts.var_gt(var, right)
        elif op == "<=":
            return self.cg.lts.var_le(var, right)
        elif op == ">=":
            return self.cg.lts.var_ge(var, right)
        raise ValueError(f"invalid atom {text!r}")


class ARCTLProp(CTLProp):
    syntax = "ARCTL"

    def __init__(self, cg, prop, phi=None):
        if phi is None:
            phi = tl.parse(prop).arctl()
        super().__init__(cg, prop, phi)

    def _get_states(self, states):
        cache = self.cache
        if None in cache:
            checker = cache[None]
        else:
            actions = {
                self.cg.lts.tpred[n]: a.tags
                for n, a in self.cg.model.spec.gal.items()
                if isinstance(a, parser.Action)
            }
            checker = mc.FARCTL_model_checker(
                self.cg.lts.states, actions, tau_label="_None", atom=self._atom
            )
            cache[None] = checker
        return checker.check(self.phi)


NOARG = object()


class LTSProxy:
    def __init__(self, cg):
        self.cg = cg
        self.lts = cg.lts

    @cached_property
    def succ(self):
        g2m = self.lts.g2m
        return {g2m[n]: h for n, h in self.lts.tsucc.items()}

    @cached_property
    def pred(self):
        g2m = self.lts.g2m
        return {g2m[n]: h for n, h in self.lts.tpred.items()}

    def __getitem__(self, spec):
        tname = self.lts.m2g.get(spec, None)
        tpred = self.lts.tpred.get(tname, None)
        if tpred is not None:
            return tpred(self.lts.states)
        else:
            return self.cg.make_states(spec) & self.lts.states


class StatePropEnv(dict):
    def __init__(self, cg, states):
        super().__init__()
        self.cg = cg
        self.states = states
        self.ltsproxy = LTSProxy(cg)

    _cnum = re.compile(r"^C(\d+)$", re.I)

    def __getitem__(self, name):
        if name == "DEAD":
            return self.cg.lts.dead
        elif name == "INIT":
            return self.cg.lts.init
        elif name == "HULL":
            return self.cg.lts.hull
        elif name == "ALL":
            return self.cg.lts.states
        elif name == "LTS":
            return self.ltsproxy
        elif match := self._cnum.match(name):
            return self.cg[int(match.group(1))].states
        elif (do := getattr(self, f"_do_{name}", None)) is not None:
            return do
        else:
            raise KeyError(name)

    def get(self, name, default=NOARG):
        try:
            return self[name]
        except KeyError:
            if default is not NOARG:
                return default
            raise

    def _do_hull(self, states=None):
        if states is None:
            states = self.states
        succ_o = (self.cg.lts.succ & states).gfp()
        pred_o = (self.cg.lts.pred & states).gfp()
        return succ_o(states) & pred_o(states)

    def _do_past(self, states=None):
        if states is None:
            states = self.states
        return self.cg.lts.pred_s(states)

    def _do_future(self, states=None):
        if states is None:
            states = self.states
        return self.cg.lts.succ_s(states)

    def _do_basin(self, states=None):
        if states is None:
            states = self.states
        basin = self._do_past(states)
        basin -= self._do_hull(basin)
        return basin - self.cg.lts.pred_s(self.cg.lts.states - basin)

    def _do_pick(self, states=None):
        if states is None:
            states = self.states
        for x in states.explicit():
            return x

    def _do_succ(self, states=None):
        if states is None:
            states = self.states
        return self.cg.lts.succ(states)

    def _do_succ_s(self, states=None):
        if states is None:
            states = self.states
        return self.cg.lts.succ_s(states)

    def _do_succ_o(self, states=None):
        if states is None:
            states = self.states
        return self.cg.lts.succ_o(states)

    def _do_pred(self, states=None):
        if states is None:
            states = self.states
        return self.cg.lts.pred(states)

    def _do_pred_s(self, states=None):
        if states is None:
            states = self.states
        return self.cg.lts.pred_s(states)

    def _do_pred_o(self, states=None):
        if states is None:
            states = self.states
        return self.cg.lts.pred_o(states)

    def _do_entries(self, states=None):
        if states is None:
            states = self.states
        return self.cg.lts.succ(self.cg.lts.pred(states) - states) & states

    def _do_exits(self, states=None):
        if states is None:
            states = self.states
        return self.cg.lts.pred(self.cg.lts.succ(states) - states) & states

    def _do_oneway(self, trans, states=None):
        if states is None:
            states = self.cg.lts.states
        pre = self.cg.lts.tpred[self.cg.lts.m2g[trans]](states)
        post = self.cg.lts.succ_s(self.cg.lts.tsucc[trans](states))
        if post & pre:
            raise ValueError(f"{trans} is not one-way")
        return post

    def _do_without(self, trans, states=None):
        if states is None:
            states = self.cg.lts.states
        succ = reduce(
            operator.or_, (h for t, h in self.cg.lts.tsucc.items() if t != trans)
        )
        return succ(states)


class StateProp(Prop):
    syntax = "STATE"

    def __init__(self, cg, prop):
        ast.parse(prop, mode="eval")  # just to check syntax
        super().__init__(cg, prop)

    def _get_states(self, states):
        ret = eval(self.prop, {}, StatePropEnv(self.cg, states))
        if not isinstance(ret, ddd.sdd):
            raise TypeError("not a state expression")
        return ret


class AnyProp(Prop):
    _order = (CTLProp, ARCTLProp, StateProp)

    def __init__(self, cg, prop, **opts):
        super().__init__(cg, prop)
        self._opts = opts
        self._checkers = list(self._order)
        self._check = self._dummy_check
        self.syntax = "GUESS"

    def _dummy_check(self, _):
        raise NotImplementedError

    def __call__(self, states):
        errors = []
        while True:
            try:
                return self._check(states)
            except Exception as err:
                if self.syntax != "GUESS":
                    errors.append((self.syntax, f"{err.__class__.__name__}: {err}"))
                while self._checkers:
                    try:
                        cls = self._checkers.pop(0)
                        self.syntax = cls.syntax
                        self._check = cls(self.cg, self.prop, **self._opts)
                        break
                    except Exception as err:
                        errors.append((self.syntax, f"{err.__class__.__name__}: {err}"))
                else:
                    raise ValueError(
                        f"Could not check formula {self.prop!r}\n"
                        + "\n".join(f"[{s}] {e}" for s, e in errors)
                    )
