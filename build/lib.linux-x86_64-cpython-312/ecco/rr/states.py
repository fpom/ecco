import math, datetime, struct, sys, io, os, importlib, shutil, subprocess
from bitarray import bitarray
from sympy import And, Or, Symbol, pycode
from itertools import chain
from pathlib import Path

##
## code generation
##

WIDTH = struct.calcsize("I")

class Writer (object) :
    def __init__ (self) :
        self.output = io.StringIO()
        self._indent = 0
    def __call__ (self, line="") :
        self.output.write((" " * self._indent * 4) + line.strip() + "\n")
        return self
    def __enter__ (self) :
        self._indent += 1
    def __exit__ (self, exc_type, exc_value, traceback) :
        self._indent -= 1
    def getvalue (self) :
        return self.output.getvalue()

def bin8 (val) :
    return "0b" + bin(val)[2:].rjust(8, "0")

class CyGen (object) :
    def __init__ (self, spec, out, profile=False) :
        self.spec = spec
        self.out = out
        self.profile = profile
        self.variables = tuple(sorted(s.state.name for s in spec.meta))
        self.vmap = {n : (i // 8, i % 8)
                     for i, n in enumerate(self.variables)}
        self.word = math.ceil(len(self.vmap) / 8)
        self.width = self.word * 8
        self.const = {n : bitarray(("1" + i*"0").rjust(self.width, "0"))
                      for i, n in enumerate(self.variables)}
    def is_set (self, name) :
        w, b = self.vmap[name]
        return "W%s & %s" % (w, bin8(2**b))
    def set_0 (self, name) :
        w, b = self.vmap[name]
        return "W%s &= %s" % (w, bin8(255 - 2**b))
    def set_1 (self, name) :
        w, b = self.vmap[name]
        return "W%s |= %s" % (w, bin8(2**b))
    def gen_state (self) :
        self.out("cdef dict vnum = %r"
                 % {n : i for i, n in enumerate(self.variables)})
        self.out()
        with self.out("cdef class state :") :
            for i in range(self.word) :
                self.out("cdef unsigned char W%s" % i)
            with self.out("def __init__ (self, on=[]) :") :
                self.out("cdef str v")
                self.out("cdef unsigned int n")
                self.out("cdef list init")
                with self.out("if not on :") :
                    self.out("init = []")
                with self.out("elif isinstance(on, str) :") :
                    self.out("init = on.split('|')")
                with self.out("else :") :
                    self.out("init = list(on)")
                with self.out("for v in init :") :
                    self.out("n = vnum[v]")
                    for i, n in enumerate(self.variables) :
                        if i == 0 :
                            self.out("if n == %s : self.%s" % (i, self.set_1(n)))
                        else :
                            self.out("elif n == %s : self.%s" % (i, self.set_1(n)))
            with self.out("def __getitem__ (self, key) :") :
                self.out("cdef unsigned int n = vnum[key]")
                for i, n in enumerate(self.variables) :
                    if i == 0 :
                        self.out("if n == %s : return bool(self.%s)"
                                 % (i, self.is_set(n)))
                    else :
                        self.out("elif n == %s : return bool(self.%s)"
                                 % (i, self.is_set(n)))
            with self.out("def __setitem__ (self, key, val) :") :
                self.out("cdef unsigned int n = vnum[key]")
                for i, n in enumerate(self.variables) :
                    if i == 0 :
                        self.out("if n == %s :" % i)
                    else :
                        self.out("elif n == %s :" % i)
                    with self.out :
                        self.out("if val : self.%s" % self.set_1(n))
                        self.out("else : self.%s" % self.set_0(n))
            with self.out("def __iter__ (self) :") :
                for n in self.variables :
                    self.out("if self.%s : yield %r" % (self.is_set(n), n))
            with self.out("def __str__ (state self) :") :
                self.out("cdef list l = []")
                for n in self.variables :
                    self.out("if self.%s : l.append(%r)" % (self.is_set(n), n))
                self.out("return '|'.join(l)")
            with self.out("def __repr__ (state self) :") :
                self.out("cdef list l = []")
                for n in self.variables :
                    self.out("if self.%s : l.append(%r)" % (self.is_set(n), repr(n)))
                self.out("return 'state([%s])' % ', '.join(l)")
            with self.out("def __eq__ (state self, state other) :") :
                self.out("try: return "
                         + " and ".join("(self.W%s == other.W%s)" % (i, i)
                                        for i in range(self.word)))
                self.out("except : return False")
            with self.out("def __ne__ (state self, state other) :") :
                self.out("try : return "
                         + " or ".join("(self.W%s != other.W%s)" % (i, i)
                                      for i in range(self.word)))
                self.out("except : return False")
            with self.out("def __hash__ (state self) :") :
                self.out("return "
                         + " + ".join("(self.W%s << %s)"
                                      % (i, (i % WIDTH)*8 + i // WIDTH)
                                      for i in range(self.word)))
            with self.out("def __invert__ (self) : ") :
                self.out("cdef state s = state.__new__(state)")
                for i in range(self.word-1) :
                    self.out("s.W%s = ~self.W%s" % (i, i))
                last = (len(self.variables) % 8) or 0
                mask = "0" * last + "1" * (8 - last)
                self.out("s.W%s = 0b%s & ~self.W%s"
                         % (self.word - 1, mask, self.word - 1))
                self.out("return s")
            with self.out("def __or__ (state self, object other) :") :
                self.out("cdef state s = state.__new__(state)")
                self.out("cdef unsigned int n")
                with self.out("if isinstance(other, str) :") :
                    self.out("n = vnum[other]")
                    for i in range(self.word) :
                        self.out("s.W%s = self.W%s" % (i, i))
                    for i, n in enumerate(self.variables) :
                        if i == 0 :
                            self.out("if n == %s : s.%s" % (i, self.set_1(n)))
                        else :
                            self.out("elif n == %s : s.%s" % (i, self.set_1(n)))
                with self.out("elif isinstance(other, state) :") :
                    for i in range(self.word) :
                        self.out("s.W%s = self.W%s | <state>other.W%s" % (i, i, i))
                with self.out("else :") :
                    self.out("raise TypeError(\"expected 'str' or 'state'"
                             " but had '%s'\" % other.__class__.__name__)")
                self.out("return s")
            with self.out("def __and__ (state self, object other) :") :
                self.out("cdef state s")
                self.out("cdef unsigned int n")
                with self.out("if isinstance(other, str) :") :
                    self.out("s = state.__new__(state)")
                    self.out("n = vnum[other]")
                    for i, n in enumerate(self.variables) :
                        if i == 0 :
                            with self.out("if n == %s :" % i) :
                                self.out("if self.%s : s.%s"
                                         % (self.is_set(n), self.set_1(n)))
                        else :
                            with self.out("elif n == %s :" % i) :
                                self.out("if self.%s : s.%s"
                                         % (self.is_set(n), self.set_1(n)))
                with self.out("elif isinstance(other, state) :") :
                    self.out("s = state.__new__(state)")
                    for i in range(self.word) :
                        self.out("s.W%s = self.W%s & <state>other.W%s" % (i, i, i))
                with self.out("else :") :
                    self.out("raise TypeError(\"expected 'str' or 'state'"
                             " but had '%s'\" % other.__class__.__name__)")
                self.out("return s")
            with self.out("def __contains__ (self, str var) :") :
                self.out("cdef unsigned int n = vnum[var]")
                for i, n in enumerate(self.variables) :
                    if i == 0 :
                        self.out("if n == %s : return self.%s" % (i, self.is_set(n)))
                    else :
                        self.out("elif n == %s : return self.%s" % (i, self.is_set(n)))
            for rule in chain(self.spec.constraints, self.spec.rules) :
                self.gen_succ(rule)
            with self.out("cpdef bint transient (state self) :") :
                if self.spec.constraints :
                    cond = Or(*(self.gen_cond(c) for c in self.spec.constraints))
                    self.out("return %s" % pycode(cond))
                else :
                    self.out("return False")
            with self.out("cdef void _succ (state self, set acc) :") :
                if self.spec.constraints :
                    self.out("cdef bint transient = 0")
                self.out("cdef state s")
                for rule in self.spec.constraints :
                    self.out("if self.%s(acc) : transient = 1" % rule.name())
                if self.spec.constraints :
                    self.out("if transient : return")
                for rule in self.spec.rules :
                    self.out("self.%s(acc)" % rule.name())
            with self.out("cpdef set succ (state self, bint compact=False) :") :
                self.out("cdef set succ, todo, done, skip")
                self.out("cdef state s, q")
                self.out("cdef str r, c")
                self.out("todo = set()")
                self.out("self._succ(todo)")
                with self.out("if compact :") :
                    self.out("done = set()")
                    self.out("skip = set()")
                    with self.out("while todo :") :
                        self.out("r, s = todo.pop()")
                        with self.out("if s.transient() :") :
                            self.out("skip.add((r, s))")
                            self.out("succ = set() ")
                            self.out("s._succ(succ)")
                            self.out("succ.difference_update(skip)")
                            self.out("succ.difference_update(done)")
                            with self.out("for c, q in succ :") :
                                self.out("todo.add((r, q))")
                        with self.out("else :") :
                            self.out("done.add((r, s))")
                    self.out("return done")
                with self.out("else :") :
                    self.out("return todo")
            self.out("@classmethod")
            with self.out("def init (cls) :") :
                self.out("cdef state s = cls.__new__(cls)")
                init = bitarray("0" * self.width)
                for s in sorted(self.spec.meta) :
                    if s.state.sign :
                        pos, val = self.vmap[s.state.name]
                        self.out("# %s = W%s & %s" % (s.state.name, pos, bin8(2**val)))
                        init |= self.const[s.state.name]
                for i, v in enumerate(reversed(init.tobytes())) :
                    self.out("s.W%s = %s" % (i, bin8(v)))
                self.out("return s")
            self.out("@classmethod")
            with self.out("def none (cls) :") :
                self.out("return state.__new__(state)")
            self.out("@classmethod")
            with self.out("def all (cls) :") :
                self.out("cdef state s = cls.__new__(cls)")
                for i in range(self.word - 1) :
                    self.out("s.W%s = 0b11111111" % i)
                last = (len(self.variables) % 8) or 8
                mask = "0" * (8 - last) + "1" * last
                self.out("s.W%s = 0b%s" % (self.word - 1, mask))
                self.out("return s")
            self.out("@classmethod")
            with self.out("def vars (cls) :") :
                self.out("return %r" % (self.variables,))
            self.out()
    def gen_cond (self, rule, guard=None, assign=None) :
        if guard is None :
            guard = {False : bitarray("0" * self.width),
                     True : bitarray("0" * self.width)}
        if assign is None :
            assign = {False : bitarray("0" * self.width),
                      True : bitarray("0" * self.width)}
        for s in rule.left :
            guard[s.sign] |= self.const[s.name]
        for s in rule.right :
            assign[s.sign] |= self.const[s.name]
        return And(*(Symbol("(self.W%s | %s) == 0xFF" % (i, bin8(v)))
                     for i, v in enumerate(reversed((~guard[True]).tobytes()))
                     if v != 0xFF),
                   *(Symbol("(self.W%s & %s) == 0" % (i, bin8(v)))
                     for i, v in enumerate(reversed(guard[False].tobytes()))
                     if v != 0),
                   Or(*(Symbol("(self.W%s | %s) != 0xFF" % (i, bin8(v)))
                        for i, v in enumerate(reversed((~assign[True]).tobytes()))
                        if v != 0xFF),
                      *(Symbol("(self.W%s & %s) != 0" % (i, bin8(v)))
                        for i, v in enumerate(reversed(assign[False].tobytes()))
                        if v != 0)))
    def gen_succ (self, rule) :
        guard = {False : bitarray("0" * self.width),
                 True : bitarray("0" * self.width)}
        assign = {False : bitarray("0" * self.width),
                  True : bitarray("0" * self.width)}
        name = rule.name()
        cond = self.gen_cond(rule, guard, assign)
        with self.out("cdef bint %s (state self, set succ) :" % name) :
            self.out("# %s" % rule.text())
            for vname in sorted(rule.vars()) :
                pos, val = self.vmap[vname]
                self.out("# ... %s = W%s & %s" % (vname, pos, bin8(2**val)))
            self.out("cdef state s")
            self.out("if %s : return False" % pycode(~cond))
            self.out("s = state.__new__(state)")
            oz = list(zip(assign[True].tobytes(), (~assign[False]).tobytes()))
            for i, (one, zero) in enumerate(reversed(oz)) :
                if one == 0 and zero == 0xFF :
                    self.out("s.W%s = self.W%s" % (i, i))
                elif one == 0 :
                    self.out("s.W%s = self.W%s & %s" % (i, i, bin8(zero)))
                elif zero == 0xFF :
                    self.out("s.W%s = self.W%s | %s" % (i, i, bin8(one)))
                else :
                    self.out("s.W%s = (self.W%s | %s) & %s"
                             % (i, i, bin8(one), bin8(zero)))
            self.out("succ.add((%r, s))" % name)
            self.out("return True")
    def gen_mod (self, src_hash) :
        if self.profile :
            self.out("# cython profile=True")
        self.out("# generated on %s" % datetime.datetime.now())
        self.out("src_hash = %r" % src_hash)
        self.out()
        self.out("state_variables = %r" % (self.variables,))
        self.out()
        self.out("from libc.stdio cimport FILE, fopen, fclose, fread, fwrite")
        self.out()
        self.gen_state()
        with open(Path(__file__).parent.parent / "_ui.pyx") as src :
            self.out(src.read())
        with open(Path(__file__).parent / "_states.pyx") as src :
            self.out(src.read())

##
## cython compilation
##

class CythonError (Exception) :
    def __init__ (self, message, output) :
        super().__init__(message)
        self.output = output

class CythonBuild (object) :
    def __init__ (self, path, name) :
        self.path = path
        self.name = name
        self._path = os.getcwd()
        self.output = None
    def run (self, path) :
        try :
            self.output = subprocess.check_output(["python%s.%s"
                                                   % (sys.version_info.major,
                                                      sys.version_info.minor),
                                                   path],
                                                  stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err :
            self.output = err.output
            raise
    def __enter__ (self) :
        os.chdir(self.path)
        return self
    def __exit__ (self, exc_type, exc_value, traceback) :
        try :
            if exc_value is not None :
                if isinstance(self.output, bytes) :
                    raise CythonError("Cython failed", self.output.decode())
                else :
                    raise CythonError("Cython failed", self.output)
            for path in Path(".").glob("statespace_*.*") :
                if path.name.split(".", 1)[0] != self.name :
                    try :
                        path.unlink()
                    except :
                        pass
            shutil.rmtree("build", ignore_errors=True)
        finally :
            os.chdir(self._path)

_setup_py = """
import os.path
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(script_args=["build_ext", "--inplace"],
      ext_modules=cythonize(
          Extension({modname!r}, [{pyx_path!r}],
                    libraries=["DDD"],
                    library_dirs=[os.path.expanduser("~/.local/lib")],
                    include_dirs=[os.path.expanduser("~/.local/include"),
                                  os.path.expanduser("~/work/tools/its/pyddd"),
                                  os.path.expanduser("~/work/tools/its/pyits")],
                    language="c++",
                    extra_compile_args=["-std=c++11"]),
          language_level=3,
          include_path=[os.path.expanduser("~/work/tools/its/pyddd"),
                        os.path.expanduser("~/work/tools/its/pyits")]))
"""

def build (spec, outpath, src_hash, profile=False) :
    modname = "statespace_%s" % datetime.datetime.now().strftime("%s")
    gen = CyGen(spec, Writer(), profile)
    gen.gen_mod(src_hash)
    with CythonBuild(outpath, modname) as cython :
        pyx_path = Path(modname).with_suffix(".pyx")
        with open(pyx_path, mode="w") as out :
            out.write(gen.out.getvalue())
        with open("setup.py", mode="w") as out :
            out.write(_setup_py.format(modname=modname, pyx_path=pyx_path.name))
        cython.run("setup.py")
        mod = sys.modules["statespace"] = importlib.import_module(modname)
        del sys.modules[mod.__name__]
        return mod

def load (spec, rr_path, mod_dir, src_hash, rebuild, profile=False) :
    if not rebuild :
        rr_path = Path(rr_path)
        mod_dir = Path(mod_dir)
        libs = list(mod_dir.glob("statespace_*.so"))
        if libs :
            libs.sort(reverse=True)
            if libs[0].stat().st_mtime > rr_path.stat().st_mtime :
                modname = libs[0].name.split(".", 1)[0]
                cwd = os.getcwd()
                try :
                    os.chdir(libs[0].parent)
                    mod = sys.modules["statespace"] = importlib.import_module(modname)
                    del sys.modules[mod.__name__]
                finally :
                    os.chdir(cwd)
                try :
                    if mod.src_hash == src_hash :
                        return mod
                except AttributeError :
                    pass
    return build(spec, str(mod_dir), src_hash, profile)
