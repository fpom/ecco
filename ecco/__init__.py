"""
ECological COmputation
"""

import functools, operator, pathlib, inspect
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

def _methkey (item) :
    return item[1].__code__.co_firstlineno

def _propkey (item) :
    try :
        return item[1].fget.__code__.co_firstlineno
    except AttributeError :
        return 0

def cached_property (method) :
    name = "_cached_%s" % method.__name__
    @functools.wraps(method)
    def wrapper (self, *l, **k) :
        if not hasattr(self, name) :
            setattr(self, name, method(self, *l, **k))
        return getattr(self, name)
    return property(wrapper)

def help (cls) :
    doc = [None]
    for name, meth in sorted(inspect.getmembers(cls, inspect.isdatadescriptor),
                             key=_propkey) :
        docstr = inspect.getdoc(meth)
        if docstr and not name.startswith("_") :
            doc.append(" - %s: %s" % (name, docstr.splitlines()[0]))
    if doc[-1] is not None :
        doc[0] = "Properties:"
        doc.append(None)
    for name, meth in sorted(inspect.getmembers(cls, inspect.isfunction),
                             key=_methkey) :
        docstr = inspect.getdoc(meth)
        if docstr and not name.startswith("_") :
            doc.append(" - %s: %s" % (name, docstr.splitlines()[0]))
            spec = inspect.getfullargspec(meth)
            ret = repr(spec.annotations.get("return", None))
            if ret.startswith("<class ") :
                ret = ret[8:-2]
            elif ret.startswith("typing.Union[") :
                ret = ret[13:-1].replace(", ", " | ")
            if ret :
                meth.__doc__ += "Return: %s" % ret
    if doc[-1] is not None :
        doc[doc.index(None)] = "Methods:"
    else :
        del doc[-1]
    if doc :
        if cls.__doc__ :
            cls.__doc__ += "\n" + "\n".join(doc)
        else :
            cls.__doc__ = "\n".join(doc)
    return cls

@help
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

@help
class Model (Record) :
    """Store a model.
    Attributes:
     - path: file path from which the model was loaded
     - spec: the model itself
    """
    _fields = ["path", "spec"]
    _options = ["base"]
    def __init__ (self, *l, **k) :
        super().__init__(*l, **k)
        if self.base is None :
            self.base = pathlib.Path(self.path).with_suffix("")
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
