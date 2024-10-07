import re

from itertools import combinations, product, chain
from typing import Iterator, Self
from collections import namedtuple, defaultdict

from unidecode import unidecode
from rich.text import Text  # pyright: ignore

from .st import Parser, Rule

try:
    from IPython.display import display  # type: ignore
except ImportError:
    display = print

_const = re.compile(r"^\s*constraints\s*:\s*(#.*)?$")
_rules = re.compile(r"^\s*rules\s*:\s*(#.*)?$")
_spec = namedtuple("_spec", ["constraints", "rules"])


def parse_pattern(path):
    rules, const = [], []
    target = rules
    with open(path, encoding="utf-8", errors="replace") as f:
        src = unidecode(f.read())
    top_parser = Parser(path, src)
    parse = top_parser._parser.parse
    for line in src.splitlines():
        if _rules.match(line):
            target = rules
        elif _const.match(line):
            target = const
        elif ln := line.split("#", 1)[0].strip():
            left, right, _ = parse(ln + "\n", "rule", semantics=top_parser)  # type: ignore
            target.append(Rule(left, right, num=len(target) + 1))
    return _spec(const, rules)


class StateMatch(dict):
    def consistent(self):
        s2t, t2s = {}, {}
        for s, t in self.items():
            s2t.setdefault(s.name, t.name)
            t2s.setdefault(t.name, s.name)
            if s2t[s.name] != t.name:
                return False
            if t2s[t.name] != s.name:
                return False
        return len(self) == len(set(self.values()))

    def __add__(self, other: Self) -> Self | None:
        if any(self[k] != other[k] for k in self if k in other):
            # self and other disagree on some mappings
            return None
        if any(~self[k] != other[~k] for k in self if ~k in other):
            # self and other are inconsistent on mapping on state and its opposite
            return None
        add = self.__class__(self | other)
        if not add.consistent():
            return None
        return add

    def __str__(self):
        seen = set()
        items = []
        for s, t in self.items():
            if s in seen:
                continue
            elif ~s in self:
                _s = "±" if s.sign else "∓"
                _t = "±" if t.sign else "∓"
                items.append(f"{s.name}{_s} => {t.name}{_t}")
            else:
                items.append(f"{s} => {t}")
            seen.update((s, ~s))
        return " / ".join(items)


class Match:
    def __init__(self, actions, states):
        self.actions = dict(actions)
        self.states = StateMatch(states)

    def __bool__(self):
        return bool(self.actions)

    def __len__(self):
        return len(self.actions)

    def __add__(self, other: Self) -> Self | None:
        if (s := self.states + other.states) is None:
            return None
        a = {
            k: self.actions.get(k, set()) | other.actions.get(k, set())
            for k in set(self.actions) | set(other.actions)
        }
        return self.__class__(a, s)

    @classmethod
    def match(cls, src, tgt, merged=False, strict=True) -> Iterator[Self]:
        if merged:
            # fuse constraints and rules
            src = _spec((), src.constraints + src.rules)
            tgt = _spec((), tgt.constraints + tgt.rules)
        if not src.constraints or not tgt.constraints:
            # no constraints to match => match rules only
            for r in cls.match_actions(src.rules, tgt.rules, strict):
                yield r
        elif not src.rules or not tgt.rules:
            # not rules to match => match constraints only
            for c in cls.match_actions(src.constraints, tgt.constraints, strict):
                yield c
        else:
            # match both => cross product constraints/rules matches
            for c in cls.match_actions(src.constraints, tgt.constraints, strict):
                for r in cls.match_actions(src.rules, tgt.rules, strict):
                    if (m := c + r) is not None:
                        # yield only compatible matches
                        yield m

    @classmethod
    def match_actions(cls, src, tgt, strict) -> Iterator[Self]:
        size = min(len(src), len(tgt))
        for s, t in product(combinations(src, size), combinations(tgt, size)):
            # match every subset of same size
            for sm in cls.match_states(s, t, strict):
                if (m := cls.make(s, t, sm, strict)) is not None:
                    yield m

    @classmethod
    def make(cls, src, tgt, sm, strict):
        img = set(sm.values())
        am = defaultdict(set)
        for s in src:
            # match each source action => get its image
            s_left, s_right = set(sm[a] for a in s.left), set(sm[a] for a in s.right)
            for t in tgt:
                # to each target action => get its image without unmapped items
                t_left, t_right = set(t.left) & img, set(t.right) & img
                if (strict and s_left == t_left and s_right == t_right) or (
                    not strict and t_left & s_left and t_right & s_right
                ):
                    am[s].add(t)
        if not am:
            return None
        return cls(am, sm)

    @classmethod
    def match_states(cls, src, tgt, strict):
        # all the src states we need to match
        needed = {s for act in src for s in chain(act.left, act.right)}
        # tgt state -> (left, right) counts
        tgt_counts = cls._counts(tgt)
        # src state -> (left, right) counts
        min_counts = cls._counts(src)
        # src state -> max possible (left, right) counts after elementarisation
        max_counts = min_counts if strict else cls._max_counts(src, min_counts)
        # invert tgt_counts
        groups = {}
        for state, counts in tgt_counts.items():
            groups.setdefault(counts, set())
            groups[counts].add(state)
        # match each group from tgt
        matches = defaultdict(set)
        for (left, right), tgt_grp in groups.items():
            # collect src states with compatible counts
            for state, (left_min, right_min) in min_counts.items():
                left_max, right_max = max_counts[state]
                if left_min <= left <= left_max and right_min <= right <= right_max:
                    matches[state].update(tgt_grp)
        if set(matches) != needed:
            return
        # combine partial matches
        keys = tuple(matches)
        for match in product(*(matches[k] for k in keys)):
            if (m := StateMatch(zip(keys, match))).consistent():
                yield m

    @classmethod
    def _max_counts(cls, actions, keys):
        # count the max number of each state on each side, including those implicit
        # use elementarisation to make implicit states explicit
        counts = {s: [0, 0] for s in keys}
        for act in actions:
            cnt = {}
            for elm in act.elementarise():
                for state, (left, right) in cls._counts([elm]).items():
                    lft, rgt = cnt.get(state, (0, 0))
                    cnt[state] = (max(lft, left), max(rgt, right))
            for state, (left, right) in cnt.items():
                counts[state][0] += left
                counts[state][1] += right
        return counts

    @classmethod
    def _counts(cls, actions):
        cnt = {}
        for act in actions:
            for i, side in enumerate([act.left, act.right]):
                for state in side:
                    cnt.setdefault(state, [0, 0])
                    cnt[state][i] += 1
        return {k: tuple(v) for k, v in cnt.items()}

    def _markup(self):
        lines = []
        lines.append(f"[b]{self.states}[/]")
        for s, tgt in sorted(self.actions.items(), key=lambda k: k[0].num):
            lines.append(f"  [blue]{s}[/]")
            for i, t in enumerate(tgt):
                h = " |" if i else "=>"
                lines.append(f"  [dim]{h}[/] [green]{t}[/]")
        txt = Text.from_markup("\n".join(lines))
        txt.highlight_regex(r"\s*\#.*(\n|$)", "dim")
        return txt

    def __str__(self):
        return str(self._markup())

    def _ipython_display_(self):
        display(self._markup())
