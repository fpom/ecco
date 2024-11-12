from dataclasses import dataclass, fields
from contextlib import contextmanager
from inspect import isgenerator
from enum import StrEnum
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import (
    Callable,
    Any,
    Iterator,
    Union,
    Self,
    get_origin,
    get_args,
)
from inspect import isclass

from frozendict import frozendict
from rich.abc import RichRenderable
from rich.text import Text
from rich.console import Console
from rich.tree import Tree
from rich.pretty import Pretty
from rich.columns import Columns

try:
    from IPython.display import display
except Exception:
    display = print


class Printable(ABC):
    @abstractmethod
    def __txt__(
        self,
        styles: Mapping[Any, str | Callable[[RichRenderable], RichRenderable]] = {},
    ) -> RichRenderable: ...

    def __str__(self) -> str:
        with (con := Console(color_system=None)).capture() as cap:
            con.print(self.__txt__(), end="")
        return cap.get()

    def _ipython_display_(self) -> None:
        display(self.__txt__())


def OP_chevron(obj, _):
    return Text(f" {obj} ", "red bold")


class Printer:
    styles: dict[Any, str | tuple | Callable[[Text], Text]] = {
        int: "green",
        ("var",): "blue",
    }

    def __init__(self, styles={}):
        self.styles = self.styles | styles
        self.context = []
        for k, v in self.styles.items():
            if isinstance(v, tuple):
                name, *args = v
                handler = getattr(self, f"print_{name}")

                def wrapper(txt):
                    return handler(txt, *args)

                self.styles[k] = wrapper

    def __getattr__(self, ctx):
        @contextmanager
        def ctxmgr():
            self.context.append(ctx)
            yield
            self.context.pop(-1)

        return ctxmgr()

    def __call__(self, obj: Any, *ctx: str) -> Text:
        ctx = tuple(self.context) + ctx
        if isinstance(obj, Text):
            txt = obj
        else:
            txt = Text(str(obj))
        signs = [type(obj), obj]
        for i in range(len(ctx)):
            signs.extend([ctx[i:], ctx[i:] + (type(obj),), ctx[i:] + (obj,)])
            if i > 0:
                signs.extend([ctx[:-i], ctx[:-i] + (type(obj),), ctx[:-i] + (obj,)])
        for s in signs:
            try:
                sty = self.styles[s]
            except (TypeError, KeyError):
                continue
            if callable(sty):
                txt = sty(txt)
            elif isinstance(sty, Text):
                txt = sty
            elif isinstance(sty, str):
                txt.stylize(sty)
            else:
                raise TypeError(f"unexpected style: {sty!r}")
        return txt

    def __truediv__(self, sep):
        def join(*args):
            if len(args) == 1 and (
                isgenerator(args[0]) or isinstance(args[0], (list, tuple))
            ):
                args = args[0]
            return Text(sep).join(a if isinstance(a, Text) else self(a) for a in args)

        return join

    def __getitem__(self, key):
        if isinstance(key, tuple) or isgenerator(key):
            return Text.assemble(*key)
        elif isinstance(key, Text):
            return key
        else:
            return self(key)


@dataclass(frozen=True, eq=True, order=True)
class Record(Printable):
    """Base class for model elements."""

    @classmethod
    def _fields(cls) -> Iterator[tuple[str, bool, type | None, type]]:
        """Enumerate all the fields of a `Record`.

        Yields:
            4-tuples whose items are:

            - `name`: the name of the field
            - `optional`: `True` if the field is optional, `False` otherwise
            - `type`: if not `None`, the field is a container of class `type`
            - `cls`: the type of the field
        """
        for field in fields(cls):
            opt = False
            typ = field.type
            if isclass(typ):
                yield field.name, opt, None, typ
            else:
                if get_origin(typ) is Union:
                    opt = True
                    typ = get_args(typ)[0]
                if (org := get_origin(typ)) in (tuple, frozenset):
                    yield field.name, opt, org, get_args(typ)[0]
                elif org is Mapping:
                    yield field.name, opt, org, get_args(typ)[1]
                else:
                    yield field.name, opt, None, typ

    def __post_init__(self):
        for name, _, typ, _ in self._fields():
            if (val := getattr(self, name)) is None:
                pass
            elif isinstance(val, dict):
                self.__dict__[name] = frozendict(val)
            elif typ in (tuple, frozenset) and not isinstance(val, typ):
                self.__dict__[name] = typ(val)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], **impl: type["Record"]) -> Self:
        """Build a new `Record` instance from its `dict` serialiation.

        Args:
            data: A `dict` holding a previously serialized `Record`.
            impl: Concrete implementation of `Record` classes, in particular of `Expression`.

        Returns:
            A `Record` instance.

        Raises:
            ValueError: If some fields are missing in `data`.
        """
        args = {}
        for name, opt, cont, typ in cls._fields():
            val = data.get(name, None)
            if val is None:
                if not opt:
                    raise ValueError(f"field '{cls.__name__}.{name}' cannot be empty")
                continue
            if issubclass(typ, Record):
                typ = impl.get(typ.__name__, typ)
                if cont is None:
                    args[name] = typ.from_dict(val, **impl)
                else:
                    args[name] = cont(typ.from_dict(v, **impl) for v in val)
            else:
                if cont is None:
                    args[name] = val
                else:
                    args[name] = cont(val)
        return cls(**args)

    def to_dict(self) -> dict[str, object]:
        """Serialize a `Record` into a `dict`.

        Returns:
            A `dict` holding all the data from the `Record`.
        """
        d = {}
        for name, opt, cont, typ in self._fields():
            val = getattr(self, name)
            if val is None and opt:
                continue
            if not issubclass(typ, Record):
                if cont is None:
                    d[name] = val
                else:
                    d[name] = list(val)
            elif cont is None:
                d[name] = val.to_dict()
            else:
                d[name] = [v.to_dict() for v in val]
        return d

    def copy(self, **repl) -> Self:
        """Copy a `Record`, replacing some of its fields.

        Args:
            repl: Fields to be replaced, given by name and value.

        Returns:
            A new `Record` instance.
        """
        args = {}
        for name, _, cont, typ in self._fields():
            val = repl.get(name, getattr(self, name, None))
            if issubclass(typ, Record):
                if cont is None:
                    args[name] = val.copy()
                else:
                    args[name] = cont(v.copy() for v in val)
            else:
                if cont is None:
                    args[name] = val
                else:
                    if cont is Mapping:
                        cont = frozendict
                    args[name] = cont(val)
        return self.__class__(**args)

    def __txt__(self, styles={}, attr=None, index=None, key=None):
        def _print_record(txt):
            name = Text.from_markup(f"[red]({txt})[/]")
            if attr is not None:
                return Text.assemble(Text.from_markup(f"[blue]{attr}[/] = "), name)
            elif index is not None:
                return Text.assemble(Text.from_markup(f"[green]{index}[/]: "), name)
            elif key is not None:
                return Text.assemble(Text.from_markup(f"[yellow]{key}[/]: "), name)
            else:
                return name

        _styles = {("record",): _print_record}
        prn = Printer(_styles | styles)
        root = Tree(prn(self.__class__.__name__, "record"))
        for name, opt, cont, typ in self._fields():
            val = getattr(self, name)
            if issubclass(typ, Record):
                if cont is None:
                    root.add(val.__txt__(styles, name))
                elif isinstance(val, (tuple, list)):
                    node = root.add(
                        Text.from_markup(rf"[blue]{name} [dim]\[][/dim][/blue]")
                    )
                    for i, v in enumerate(val):
                        node.add(v.__txt__(styles, index=i))
                elif isinstance(val, (set, frozenset)):
                    node = root.add(
                        Text.from_markup(rf"[blue]{name} [dim]()[/dim][/blue]")
                    )
                    for v in sorted(val):
                        node.add(v.__txt__(styles))
                elif isinstance(val, (dict, frozendict, Mapping)):
                    node = root.add(
                        Text.from_markup(rf"[blue]{name} [dim]{{}}[/dim][/blue]")
                    )
                    for k, v in sorted(val.items()):
                        node.add(v.__txt__(styles, key=k))
                else:
                    raise TypeError(f"unexpected contained for {name}: {cont}")
            elif opt and val is None:
                pass
            elif isinstance(val, StrEnum):
                root.add(Text.from_markup(f"[blue]{name}[/] = [magenta]{val}[/]"))
            else:
                if isinstance(val, frozenset):
                    val = set(val)
                elif isinstance(val, frozendict):
                    val = dict(val)
                root.add(Columns([Text.from_markup(f"[blue]{name}[/] ="), Pretty(val)]))

        return root


__all__ = ["display", "Printer", "Record"]
