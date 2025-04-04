import argparse
import os
import resource
import psutil

from . import load

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--debug", action="store_true", default=False, help="run in debug mode"
)
parser.add_argument(
    "model", metavar="PATH", type=str, nargs="?", default=None, help="model to load"
)
parser.add_argument(
    "extra", type=str, nargs="*", help="langage-dependent extra arguments"
)
args = parser.parse_args()

if args.model:
    _module, model = load(args.model, *args.extra)
    _globals = globals()
    _extra = getattr(_module, "__extra__", [])
    if isinstance(_extra, dict):
        for _k, _v in _extra.items():
            _globals[_k] = _v
    else:
        for _k in _extra:
            _globals[_k] = getattr(_module, _k)

try:
    percent = max(10, min(100, int(os.environ.get("ECCO_MEMLIMIT", "80")))) / 100.0
except ValueError:
    percent = 0.8
limit = int(psutil.virtual_memory().total * percent)
resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))
