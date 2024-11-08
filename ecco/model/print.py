from abc import ABC, abstractmethod
from typing import Mapping, Callable, Any
from functools import partial

from rich.text import Text

try:
    from IPython.display import display
except Exception:
    display = print


class Printable(ABC):
    @abstractmethod
    def __txt__(
        self, **pmap: Mapping[tuple, str | Callable[[Any, str], Text]]
    ) -> Text: ...

    def __str__(self) -> str:
        return str(self.__txt__())

    def _ipython_display_(self) -> None:
        display(self.__txt__())


def OP_chevron(obj, _):
    return Text(f" {obj} ", "red bold")


class Printer:
    ANY = "any"
    INI = "ini"
    DECL = "decl"
    CMT = "cmt"
    DOM = "dom"
    VAR = "var"
    OP = "op"
    HDR = "hdr"
    CLK = "clk"

    _pmap = {
        (ANY, int): "green",
        (INI, str, "+"): "green bold",
        (INI, str, "-"): "red bold",
        (INI, str, "*"): "yellow bold",
        (INI, int): "green",
        (DECL, str): "blue bold",
        (CMT,): "dim",
        (DOM,): "magenta",
        (VAR,): "blue",
        (OP, str, ">>"): OP_chevron,
        (HDR,): "red bold",
        (CLK,): "magenta bold",
    }

    def __init__(self, pmap: dict):
        self.pm = self._pmap | pmap

    @classmethod
    def join(cls, s, args):
        return Text(s).join(args)

    def __call__(self, obj: Any, ctx: str = ANY) -> Text:
        key = [ctx, type(obj), obj]
        sty = None
        while key:
            try:
                sty = self.pm[tuple(key)]
            except (KeyError, TypeError):
                pass
            if sty is not None:
                break
            key.pop(-1)
        else:
            sty = self._default
        if isinstance(sty, str):
            return Text(str(obj), sty)
        else:
            return sty(obj, ctx)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return Text.assemble(*key)
        elif isinstance(key, Text):
            return key
        else:
            return Text(str(key))

    def __getattr__(self, attr):
        if self.__class__.__dict__.get(attr.upper()) != attr:
            raise AttributeError(f"'Printer' object has no attribute {attr!r}")
        return partial(self, ctx=attr)

    def _default(self, obj, _):
        return Text(str(obj))


__all__ = ["display", "Printable", "Printer", "Text"]
