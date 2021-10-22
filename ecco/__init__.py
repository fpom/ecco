"""
ECological COmputation
"""

import functools, pathlib, importlib

from inspect import Signature, Parameter

class CompileError (Exception) :
    pass

class ExprVisitor (object) :
    """Sympy expressions visitor.
    """
    def __call__ (self, expr) :
        handler = getattr(self, expr.func.__name__, getattr(self, "GENERIC", None))
        if handler is None :
            raise CompileError("unsupported operation %s" % expr.func.__name__)
        return handler(expr)

def cached_property (method) :
    name = "_cached_%s" % method.__name__
    @functools.wraps(method)
    def wrapper (self, *l, **k) :
        if not hasattr(self, name) :
            setattr(self, name, method(self, *l, **k))
        return getattr(self, name)
    return property(wrapper)

class Record (object) :
    """Base class to store information.
    """
    _fields = []
    _options = []
    _defaults = {}
    _multiline = True
    def __init__ (self, *l, **k) :
        sig = Signature([Parameter(name, Parameter.POSITIONAL_OR_KEYWORD)
                         for name in self._fields]
                        + [Parameter(name, Parameter.POSITIONAL_OR_KEYWORD,
                                     default=self._defaults.get(name, None))
                           for name in self._options])
        self._args = sig.bind(*l, **k)
        self._args.apply_defaults()
        for key, val in self._args.arguments.items() :
            setattr(self, key, val)
    def _del_cached_property (self, name) :
        try :
            delattr(self, "_cached_%s" % name)
        except :
            pass
    def __repr__ (self) :
        return "%s(%s)" % (self.__class__.__name__,
                           ", ".join("%s=%r" % item for item in self.items()))
    def __eq__ (self, other) :
        return (self.__class__ is other.__class__
                and all(value == getattr(other, name)
                        for name, value in self.items()))
    def __hash__ (self) :
        return functools.reduce(operator.xor,
                                (hash(item for item in self.items())),
                                hash(self.__class__.__name__))
    def items (self) :
        for field in self._fields :
            yield field, getattr(self, field)
        params = self._args.signature.parameters
        for field in self._options :
            member = getattr(self, field, None)
            if member != params[field].default :
                yield field, member
    def update (self, items) :
        valid = set(self._fields) | set(self._options)
        for key, val in dict(items).items() :
            if key not in valid :
                raise KeyError(key)
            setattr(self, key, val)
    def _repr_pretty_ (self, p, cycle) :
        cls = self.__class__.__name__
        if cycle :
            p.text("<%s ...>" % cls)
        else :
            with p.group(len(cls)+1, "%s(" % cls, ")") :
                for i, (name, value) in enumerate(self.items()) :
                    if i :
                        p.text(", ")
                        if self._multiline :
                            p.breakable()
                    with p.group(len(name)+1, "%s=" % name, "") :
                        p.pretty(value)

class BaseModel (object) :
    """Store a model.
    Attributes:
     - path: file path from which the model was loaded
     - spec: the model itself
     - base: base directory in which auxiliary files are stored
    """
    def __init__ (self, path, spec, base=None) :
        self.path = pathlib.Path(path)
        self.spec = spec
        if base is None :
            base = self.path.with_suffix("")
        self.base = base
        if not self.base.exists() :
            self.base.mkdir(parents=True)
    def __getitem__ (self, args) :
        if not args :
            return self.base
        if not isinstance(args, (list, tuple)) :
            args = [args]
        ext = args[-1]
        if not ext.startswith(".") :
            ext = "." + ext
        parts = [[]]
        for a in args[:-1] :
            if a == "/" and parts[-1] :
                parts.append([])
            elif isinstance(a, (list, tuple)) :
                parts[-1].extend(a)
            else :
                parts[-1].append(a)
        if len(args) > 1 :
            return self.base.joinpath(*("-".join(p) for p in parts)).with_suffix(ext)
        else :
            return self.base.joinpath(self.base.name).with_suffix(ext)

def load (path) :
    fmt = path.rsplit(".", 1)[-1].lower()
    try :
        module = importlib.import_module("." + fmt, "ecco")
    except ModuleNotFoundError :
        raise ValueError("unknown model format %r" % fmt)
    return module, getattr(module, "Model", BaseModel)(path, module.parse(path))
