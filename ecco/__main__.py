import argparse, importlib

from . import Model

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true", default=False,
                    help="run in debug mode")
parser.add_argument("model", metavar="PATH", type=str, nargs="?", default=None,
                    help="model to load")
args = parser.parse_args()

def load (path) :
    fmt = path.rsplit(".", 1)[-1].lower()
    try :
        module = importlib.import_module("." + fmt, "ecco")
    except ModuleNotFoundError :
        raise ValueError("unknown model format %r" % fmt)
    return module, getattr(module, "Model", Model)(path, module.parse(path))

if args.model :
    _module, model = load(args.model)
    _globals = globals()
    for _name in getattr(_module, "__extra__", []) :
        _globals[_name] = getattr(_module, _name)
