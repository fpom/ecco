import pathlib, collections, datetime, io
from IPython.display import display

##
## utilities
##

def now () :
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

##
## console logs
##

from ._ui import Logger
log = Logger()
log.verbose = True

##
## options
##

class _getopt (object) :
    def __getitem__ (self, name) :
        return self._d[name]
    def __getattr__ (self, name) :
        return self._d[name]
    def __init__ (self, options, *allowed, defaults={}) :
        self._d = collections.defaultdict(dict)
        for name, value in defaults.items() :
            cat, sub = name.split("_", 1)
            if allowed and cat in allowed :
                self[cat][sub] = value
        for name, value in options.items() :
            cat, sub = name.split("_", 1)
            if allowed and cat not in allowed :
                raise ValueError("option %r not supported here" % name)
            self[cat][sub] = value
    def __call__ (self, cat, **defaults) :
        opt = self[cat]
        for name, value in defaults.items() :
            if name not in opt :
                opt[name] = value
        return {cat + "_" + name : value for name, value in opt.items()}

class getopt (dict) :
    def __init__ (self, options, defaults={}, **static) :
        dict.__init__(self)
        valid = set(defaults) | set(static)
        if not all(n in valid for n in options) :
            raise ValueError("invalid option(s) %s"
                             % ", ".join(repr(n) for n in options if n not in valid))
        opt = {}
        for src in (static, defaults, options) :
            opt.update(src)
        children = {}
        for key, val in opt.items() :
            if "_" not in key :
                self[key] = val
            else :
                top, sub = key.split("_", 1)
                if top not in children :
                    children[top] = {sub : val}
                else :
                    children[top][sub] = val
        for top, content in children.items() :
            self[top] = getopt(content, content)
    def __getattr__ (self, name) :
        return self[name]
    def __setattr__ (self, name, value) :
        self[name] = value
    def _flat (self) :
        for key, val in self.items() :
            if isinstance(val, getopt) :
                for k, v in val._flat() :
                    yield key + "_" + k, v
            else :
                yield key, val
    def flat (self) :
        return dict(self._flat())
    def assign (self, obj, *keys) :
        for key in keys :
            for sub, val in self[key]._flat() :
                setattr(obj, key + "_" + sub, val)

def popopt (options, *top, defaults={}, **static) :
    sub_options = {k : v for k, v in options.items()
                   if any(k.startswith(t + "_") for t in top)}
    sub_defaults = {k : v for k, v in defaults.items()
                    if any(k.startswith(t + "_") for t in top)}
    for k in sub_options :
        options.pop(k)
    for k in sub_defaults :
        defaults.pop(k)
    return getopt(sub_options, sub_defaults, **static)

##
## check updates
##

def updated (sources, targets) :
    if not all(pathlib.Path(p).exists() for p in targets) :
        return True
    latest = max(pathlib.Path(p).stat().st_mtime for p in sources)
    if any(pathlib.Path(p).stat().st_mtime <= latest for p in targets) :
        return True
    return False

##
## HTML output
##

import ipywidgets as ipw

class HTML (object) :
    def __init__ (self, text="") :
        self.t = io.StringIO(text)
        self.t.seek(0, 2)
        self.tags = []
        self.w = ipw.HTML()
    def write (self, text) :
        self.t.write(text)
    def flush (self) :
        self.w.value = self.t.getvalue()
    def _ipython_display_ (self) :
        self.flush()
        display(self.w)
    def __call__ (self, tag, **attr) :
        self.tags.append((tag, attr))
        return self
    def __enter__ (self) :
        tag, attr = self.tags[-1]
        self.write("<%s %s>" % (tag, " ".join("%s=%r" % (a.strip("_"), v)
                                              for a, v in attr.items()
                                              if a.lower() == a)))
    def __exit__ (self, exc_type, exc_value, traceback) :
        tag, attr = self.tags.pop(-1)
        self.write("</%s>" % tag)
        if attr.get("BREAK", False) :
            self.write("\n")
