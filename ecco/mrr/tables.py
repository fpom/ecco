import functools
import re

import numpy as np
import pandas as pd

from IPython.display import display


class TableProxy(object):
    _name = re.compile("^[a-z][a-z0-9_]*$", re.I)

    def __init__(self, table):
        self.table = table

    def __contains__(self, key):
        return key in self.table.columns

    def __getitem__(self, key):
        if isinstance(key, str):
            key = [key]
        elif not isinstance(key, slice):
            key = list(key)
        return self.table[key].describe(include="all").fillna("")

    def __delitem__(self, col):
        if isinstance(col, str) and col in self.table.columns:
            self.table.drop(columns=[col], inplace=True)
        elif isinstance(col, tuple):
            for c in col:
                del self[c]
        else:
            raise KeyError("invalid column: %r" % (col,))

    def __setitem__(self, key, val):
        if key in self.table.columns:
            raise KeyError("columns %r exists already" % key)
        if callable(val):
            data = self.table.apply(val, axis=1)
        elif isinstance(val, tuple) and len(val) > 1 and callable(val[0]):
            fun, *args = val
            data = self.table.apply(fun, axis=1, args=args)
        else:
            data = pd.DataFrame({"col": val})
        if isinstance(key, (tuple, list)):
            for k in key:
                if not self._name.match(k):
                    raise ValueError("invalid column name %r" % k)
            data = pd.DataFrame.from_records(data)
            for k, c in zip(key, data.columns):
                self.table[k] = data[c]
        else:
            if not self._name.match(key):
                raise ValueError("invalid column name %r" % key)
            self.table[key] = data

    def _ipython_display_(self):
        display(self[:])

    def _repr_pretty_(self, pp, cycle):
        return pp.text(repr(self[:]))


class NodesTableProxy(TableProxy):
    def __init__(self, compograph, table, name="nodes"):
        super().__init__(table)
        self._c = compograph
        self._n = name

    def __call__(self, col, fun=None):
        if fun is None:
            series = self.table[col].astype(bool)
        else:
            series = self.table[col].apply(fun).astype(bool)
        return self._c._GCS(self.table.index[series])

    def __getattr__(self, name):
        if name in self.table.columns:
            return functools.partial(self.__call__, name)
        else:
            raise AttributeError(f"table {self._n!r} has no column {name!r}")
