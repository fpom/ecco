import subprocess
import tempfile
from itertools import chain

lark_py = {
    "mrrmodparse.py": ["mrr.lark", "mrrmod.lark"],
    "mrrpatparse.py": ["mrr.lark", "mrrpat.lark"],
}
common = "mrrparse"
common_py = f"{common}.py"


def lark(sources):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".lark") as tmp:
        for src in sources:
            tmp.write(open(src).read())
        tmp.flush()
        return subprocess.check_output(
            ["python", "-m", "lark.tools.standalone", tmp.name], encoding="utf-8"
        ).splitlines(keepends=True)


outfiles = [open(p, "w") for p in lark_py]

with open(common_py, "w") as com:
    for out in outfiles:
        out.write(f"from .{common} import *\n")
    same = True
    for lines in zip(*(lark(sources) for sources in lark_py.values())):
        same = same and lines[0].strip() != "DATA = ("
        if same:
            com.write(lines[0])
        else:
            for out, ln in zip(outfiles, lines):
                out.write(ln)

for out in outfiles:
    out.flush()
    out.close()

subprocess.check_call(["uvx", "ruff", "format", "-q", *chain(lark_py, [common_py])])
