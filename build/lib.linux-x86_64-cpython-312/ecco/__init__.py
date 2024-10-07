"""
ECological COmputation
"""

import functools
import importlib
import operator
import pathlib

from inspect import Parameter, Signature

from IPython.display import display
from typeguard import check_type, TypeCheckError


def istyped(obj, annot):
    """check whether an object validates type annotation or not

    Similar to isinstance except that the second argument is a type annotation.
    """
    try:
        check_type(obj, annot)
        return True
    except TypeCheckError:
        return False


class CompileError(Exception):
    pass


class ExprVisitor:
    """Sympy expressions visitor.
    """
    def __call__(self, expr):
        handler = getattr(self, expr.func.__name__,
                          getattr(self, "GENERIC", None))
        if handler is None:
            raise CompileError("unsupported operation %s" % expr.func.__name__)
        return handler(expr)


def cached_property(method):
    name = "_cached_%s" % method.__name__

    @functools.wraps(method)
    def wrapper(self, *largs, **kargs):
        if not hasattr(self, name):
            setattr(self, name, method(self, *largs, **kargs))
        return getattr(self, name)

    return property(wrapper)


class Record:
    """Base class to store information.
    """
    _fields = []
    _options = []
    _defaults = {}
    _multiline = True

    def __init__(self, *largs, **kargs):
        sig = Signature([Parameter(name, Parameter.POSITIONAL_OR_KEYWORD)
                         for name in self._fields]
                        + [Parameter(name, Parameter.POSITIONAL_OR_KEYWORD,
                                     default=self._defaults.get(name, None))
                           for name in self._options])
        self._args = sig.bind(*largs, **kargs)
        self._args.apply_defaults()
        for key, val in self._args.arguments.items():
            setattr(self, key, val)

    def _del_cached_property(self, name):
        try:
            delattr(self, "_cached_%s" % name)
        except Exception:
            pass

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__,
                           ", ".join("%s=%r" % item for item in self.items()))

    def __eq__(self, other):
        return (self.__class__ is other.__class__
                and all(value == getattr(other, name)
                        for name, value in self.items()))

    def __hash__(self):
        return functools.reduce(operator.xor,
                                (hash(item for item in self.items())),
                                hash(self.__class__.__name__))

    def items(self):
        for field in self._fields:
            yield field, getattr(self, field)
        params = self._args.signature.parameters
        for field in self._options:
            member = getattr(self, field, None)
            if member != params[field].default:
                yield field, member

    def update(self, items):
        valid = set(self._fields) | set(self._options)
        for key, val in dict(items).items():
            if key not in valid:
                raise KeyError(key)
            setattr(self, key, val)

    def _repr_pretty_(self, p, cycle):
        cls = self.__class__.__name__
        if cycle:
            p.text("<%s ...>" % cls)
        else:
            with p.group(len(cls)+1, "%s(" % cls, ")"):
                for i, (name, value) in enumerate(self.items()):
                    if i:
                        p.text(", ")
                        if self._multiline:
                            p.breakable()
                    with p.group(len(name)+1, "%s=" % name, ""):
                        p.pretty(value)


class BaseModel (object):
    """Store a model.

    # Attributes

     - `path`: file path from which the model was loaded
     - `spec`: the model itself
     - `base`: base directory in which auxiliary files are stored
    """
    def __init__(self, path, spec, base=None):
        self.path = pathlib.Path(path)
        self.spec = spec
        if base is None:
            base = self.path.with_suffix("")
        self.base = base
        if not self.base.exists():
            self.base.mkdir(parents=True)

    def __getitem__(self, args):
        if not args:
            return self.base
        if not isinstance(args, (list, tuple)):
            args = [args]
        ext = args[-1]
        if not ext.startswith("."):
            ext = "." + ext
        parts = [[]]
        for a in args[:-1]:
            if a == "/" and parts[-1]:
                parts.append([])
            elif isinstance(a, (list, tuple)):
                parts[-1].extend(a)
            else:
                parts[-1].append(a)
        if len(args) > 1:
            return self.base.joinpath(*("-".join(p) for p in parts)) \
                .with_suffix(ext)
        else:
            return self.base.joinpath(self.base.name).with_suffix(ext)


def load(path, *extra):
    fmt = path.rsplit(".", 1)[-1].lower()
    try:
        module = importlib.import_module("." + fmt, "ecco")
    except ModuleNotFoundError:
        raise ValueError("unknown model format %r" % fmt)
    cls = getattr(module, "Model", BaseModel)
    return module, cls.parse(path, *extra)


class hset(object):
    "similar to frozenset with sorted values"
    # work around pandas that does not display frozensets in order
    def __init__(self, iterable=(), key=None, reverse=False, sep=", "):
        self._values = tuple(sorted(iterable, key=key, reverse=reverse))
        self._sep = sep

    def __hash__(self):
        return functools.reduce(operator.xor,
                                (hash(v) for v in self._values),
                                hash("hset"))

    def __eq__(self, other):
        try:
            return set(self._values) == set(other._values)
        except Exception:
            return False

    def __iter__(self):
        yield from iter(self._values)

    def __len__(self):
        return len(self._values)

    def __bool__(self):
        return bool(self._values)

    def isdisjoint(self, other):
        return set(self).isdisjoint(set(other))

    def issubset(self, other):
        return set(self).issubset(set(other))

    def __le__(self, other):
        return self.issubset(other)

    def __lt__(self, other):
        return set(self) < set(other)

    def issuperset(self, other):
        return set(self).issuperset(set(other))

    def __ge__(self, other):
        return self.issuperset(other)

    def __gt__(self, other):
        return set(self) > set(other)

    def union(self, *others):
        return self.__class__(set(self).union(*others))

    def __or__(self, other):
        return self.union(other)

    def intersection(self, *others):
        return self.__class__(set(self).intersection(*others))

    def __and__(self, other):
        return self.intersection(other)

    def difference(self, *others):
        return self.__class__(set(self).difference(*others))

    def __sub__(self, other):
        return self.difference(other)

    def symmetric_difference(self, other):
        return self.__class__(set(self).symmetric_difference(other))

    def __xor__(self, other):
        return self.symmetric_difference(other)

    def copy(self):
        return self

    def __repr__(self):
        return self._sep.join(str(v) for v in self)

    def __str__(self):
        return self._sep.join(str(v) for v in self)

    def _repr_pretty_(self, pp, cycle):
        pp.text(self._sep.join(str(v) for v in self))

    def _ipython_display_(self):
        display(self._sep.join(str(v) for v in self))
