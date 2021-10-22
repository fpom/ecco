import ast, functools, operator
from collections import defaultdict

import tl, pymc as mc

class Prop (object) :
    syntax = None
    _cache = {}
    _cache_as = ("lts", "prop", "syntax")
    def __init__ (self, model, lts, prop) :
        self.model = model
        self.lts = lts
        self.prop = prop
    @property
    def cache (self) :
        key = tuple(getattr(self, name) for name in self._cache_as)
        return self._cache.setdefault(key, {})
    def __call__ (self, states) :
        cache = self.cache
        if states not in cache :
            self.states = states
            cache[states] = self._get_states(states)
        return cache[states]
    def _get_states (self, states) :
        raise NotImplementedError("abstract method")

class StateProp (Prop) :
    syntax = "STATE"
    def __init__ (self, model, lts, prop) :
        ast.parse(prop, mode="eval")
        super().__init__(model, lts, prop)
    def __getitem__ (self, name) :
        if name == "DEAD" :
            return self.lts.dead
        elif name == "INIT" :
            return self.lts.init
        elif name == "HULL" :
            return self.lts.hull
        elif name == "TRANSIENT" :
            return self.lts.transient
        elif name == "ALL" :
            return self.lts.states
        elif name in self.lts.vars :
            return self.lts.var2sdd(name)
        elif name in self.lts.tsucc :
            return self.lts.tpred[name](self.lts.states)
        elif name.startswith("succ_") and name[5:] in self.lts.tsucc :
            return self.lts.tsucc[name[5:]]
        elif name.startswith("pred_") and name[5:] in self.lts.tsucc :
            return self.lts.tpred[name[5:]]
        elif getattr(self, f"_do_{name}", None) is not None :
            return getattr(self, f"_do_{name}")
        else :
            raise KeyError(name)
    def get (self, name, *default) :
        try :
            return self[name]
        except KeyError :
            if len(default) == 1 :
                return default[0]
            elif default :
                raise TypeError(f"get() takes 2 positional arguments but"
                                f" {len(default)+1} were given")
            raise
    def _do_hull (self, states=None) :
        if states is None :
            states = self.states
        succ_o = (self.lts.succ & states).gfp()
        pred_o = (self.lts.pred & states).gfp()
        return succ_o(states) & pred_o(states)
    def _do_comp (self, states=None) :
        if states is None :
            states = self.states
        return self.lts.states - states
    def _do_succ (self, states=None) :
        if states is None :
            states = self.states
        return self.lts.succ(states)
    def _do_succ_s (self, states=None) :
        if states is None :
            states = self.states
        return self.lts.succ_s(states)
    def _do_succ_o (self, states=None) :
        if states is None :
            states = self.states
        return self.lts.succ_o(states)
    def _do_pred (self, states=None) :
        if states is None :
            states = self.states
        return self.lts.pred(states)
    def _do_pred_s (self, states=None) :
        if states is None :
            states = self.states
        return self.lts.pred_s(states)
    def _do_pred_o (self, states=None) :
        if states is None :
            states = self.states
        return self.lts.pred_o(states)
    def _do_entries (self, states=None) :
        if states is None :
            states = self.states
        return self.lts.succ(self.lts.pred(states) - states) & states
    def _do_exits (self, states=None) :
        if states is None :
            states = self.states
        return self.lts.pred(self.lts.succ(states) - states) & states
    def _get_states (self, states) :
        self.states = states
        return eval(self.prop, {}, self)

class CTLProp (Prop) :
    syntax = "CTL"
    def __init__ (self, model, lts, prop, phi=None) :
        if phi is None :
            phi = tl.parse(prop).ctl()
        super().__init__(model, lts, prop)
        self.phi = phi
        self.fair = phi.fair
    def _get_states (self, states) :
        cache = self.cache
        if None in cache :
            checker = cache[None]
        else :
            if self.fair :
                checker = mc.FairCTL_model_checker(self.lts.states,
                                                   self.lts.pred,
                                                   self.fair)
            else :
                checker = mc.CTL_model_checker(self.lts.states,
                                               self.lts.pred)
            cache[None] = checker
        return checker.check(self.phi)

class ARCTLProp (Prop) :
    syntax = "ARCTL"
    def __init__ (self, model, lts, prop, phi=None) :
        if phi is None :
            phi = tl.parse(prop).arctl()
        super().__init__(model, lts, prop)
        self.phi = phi
        self.fair = phi.fair
    def _get_states (self, states) :
        cache = self.cache
        if None in cache :
            checker = cache[None]
        else :
            actions = defaultdict(list)
            _labels = self.model.spec.labels
            for rule in self.model.spec.rules :
                rname = rule.name()
                label = _labels.get(rname, None) or "_None"
                actions[rname].append(self.lts.tpred[rname])
                for lbl in (l.strip() for l in label.split(",")
                            if l.strip()) :
                    actions[lbl].append(self.lts.tpred[rname])
            lbl2shom = {}
            for label, shoms in actions.items() :
                if len(shoms) == 1 :
                    lbl2shom[label] = shoms[0]
                else :
                    lbl2shom[label] = functools.reduce(operator.or_, shoms)
            if self.fair :
                checker = mc.FairARCTL_model_checker(self.lts.states,
                                                     lbl2shom,
                                                     self.lts.pred,
                                                     self.fair,
                                                     true_label="_None")
            else :
                checker = mc.ARCTL_model_checker(self.lts.states,
                                                 lbl2shom,
                                                 self.lts.pred,
                                                 tau_label="_None")
            cache[None] = checker
        return checker.check(self.phi)

class AnyProp (Prop) :
    _order = (CTLProp, ARCTLProp, StateProp)
    def __init__ (self, model, lts, prop, **opts) :
        super().__init__(model, lts, prop)
        self._opts = opts
        self._checkers = list(self._order)
        self._check = None
        self.syntax = "GUESS"
    def __call__ (self, states) :
        errors = []
        while True :
            try :
                return self._check(states)
            except Exception as err :
                if self._check is not None :
                    errors.append((self.syntax, f"{err.__class__.__name__}: {err}"))
                while self._checkers :
                    try :
                        cls = self._checkers.pop(0)
                        self.syntax = cls.syntax
                        self._check = cls(self.model, self.lts, self.prop, **self._opts)
                        break
                    except Exception as err :
                        errors.append((self.syntax, f"{err.__class__.__name__}: {err}"))
                else :
                    raise ValueError(f"Could not evaluate formula {self.prop!r}\n"
                                     + "\n".join(f"[{s}] {e}" for s, e in errors))
