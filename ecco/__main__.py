import argparse

from . import load

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true", default=False,
                    help="run in debug mode")
parser.add_argument("model", metavar="PATH", type=str, nargs="?", default=None,
                    help="model to load")
args = parser.parse_args()

if args.model :
    _module, model = load(args.model)
    _globals = globals()
    for _name in getattr(_module, "__extra__", []) :
        _globals[_name] = getattr(_module, _name)
