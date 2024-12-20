import inspect
import math
import mimetypes
import pathlib
import subprocess
import tempfile

from math import pi
from pathlib import Path
from urllib.parse import quote, urlparse

import igraph as ig
import ipycytoscape as cy
import ipywidgets as ipw
import networkx as nx
import numpy as np
import pandas as pd

from colour import Color
from IPython.display import display
from pandas.api.types import is_numeric_dtype
from jupyter_server import serverapp

from . import bqcm, scm, tikz


def maybeColor(c):
    try:
        return Color(c).hex
    except Exception:
        return None


class Palette(object):
    palettes = {
        # simple
        "red-green": ["#F00", "#0F0"],
        "green-red": ["#0F0", "#F00"],
        "red-green/white": ["#FAA", "#AFA"],
        "green-red/white": ["#AFA", "#FAA"],
        "black-white": ["#000", "#FFF"],
        "white-black": ["#FFF", "#000"],
        "gray": ["#333", "#DDD"],
        "yarg": ["#DDD", "#333"],
        "black": ["#000"],
        "white": ["#FFF"],
        # 12-bit rainbow from https://iamkate.com/data/12-bit-rainbow/
        "rainbow": [
            "#817",
            "#a35",
            "#c66",
            "#e94",
            "#ed0",
            "#9d5",
            "#4d8",
            "#2cb",
            "#0bc",
            "#09c",
            "#36b",
            "#639",
        ],
        # ad-hoc
        "hypergraph": ["#AFA", "#FAA", "#FF8", "#AAF"],
        # pride flags
        "gay pride": ["#FF0000", "#FFA500", "#FFFF00", "#008000", "#0000FF", "#4B0082"],
        "trans pride": ["#00D2FF", "#FFA6B9", "#FFFFFF"],
        # country flags
        "Afghanistan": ["#000000", "#D32011", "#FFFFFF", "#007A36"],
        "Albania": ["#E41E20", "#000000"],
        "Algeria": ["#007229", "#FFFFFF", "#D21034"],
        "Andorra": ["#0018A8", "#FEDF00", "#C7B37F", "#FFFFFF", "#D52B1E"],
        "Angola": ["#CC092F", "#FFCB00", "#000000"],
        "Antigua": ["#CE1126", "#FFFFFF", "#0072C6", "#FCD116", "#000000"],
        "Armenia": ["#D90012", "#0033A0", "#F2A800"],
        "Australia": ["#00008B", "#FFFFFF", "#FF0000"],
        "Austria": ["#ED2939", "#FFFFFF"],
        "Azerbaijan": ["#00B9E4", "#ED2939", "#FFFFFF", "#3F9C35"],
        "Bahamas": ["#000000", "#00778B", "#FFC72C"],
        "Bahrain": ["#FFFFFF", "#CE1126"],
        "Bangladesh": ["#006747", "#DA291C"],
        "Barbados": ["#00267F", "#FFC726", "#000000"],
        "Belarus": ["#C8313E", "#4AA657", "#FFFFFF"],
        "Belgium": ["#000000", "#FDDA24", "#EF3340"],
        "Belize": ["#CE1126", "#003F87", "#FFFFFF", "#289400", "#FFD83C", "#9DD7FF"],
        "Benin": ["#008751", "#FCD116", "#E8112D"],
        "Bhutan": ["#FFD520", "#FFFFFF", "#FF4E12"],
        "Bolivia": ["#D52B1E", "#F9E300", "#007934"],
        "Bosnia and Herzegovina": ["#002395", "#FECB00"],
        "Botswana": ["#75AADB", "#FFFFFF", "#000000"],
        "Brazil": ["#009C3B", "#FFDF00", "#002776", "#FFFFFF"],
        "Brunei": ["#F7E017", "#FFFFFF", "#000000", "#CF1126"],
        "Bulgaria": ["#FFFFFF", "#00966E", "#D62612"],
        "Burkina Faso": ["#EF2B2D", "#FCD116", "#009E49"],
        "Burundi": ["#1EB53A", "#FFFFFF", "#CE1126"],
        "Cambodia": ["#032EA1", "#E00025", "#FFFFFF", "#000000"],
        "Cameroon": ["#007A5E", "#CE1126", "#FCD116"],
        "Canada": ["#FF0000", "#FFFFFF"],
        "Cape Verde": ["#003893", "#FFFFFF", "#F7D116", "#CF2027"],
        "Central African Republic": [
            "#003082",
            "#FFFFFF",
            "#289728",
            "#FFCE00",
            "#D21034",
        ],
        "Chad": ["#002664", "#FECB00", "#C60C30"],
        "Chile": ["#0039A6", "#FFFFFF", "#D52B1E"],
        "Colombia": ["#FCD116", "#003893", "#CE1126"],
        "Comoros": ["#3D8E33", "#FFC61E", "#FFFFFF", "#CE1126", "#3A75C4"],
        "Congo": ["#009543", "#FBDE4A", "#DC241F"],
        "Costa Rica": ["#002B7F", "#FFFFFF", "#CE1126"],
        "Croatia": ["#FF0000", "#FFFFFF", "#0093DD", "#F7DB17", "#171796"],
        "Cuba": ["#002A8F", "#FFFFFF", "#CF142B"],
        "Cyprus": ["#D57800", "#FFFFFF", "#4E5B31"],
        "Czech Republic": ["#11457E", "#FFFFFF", "#D7141A"],
        "Democratic Republic of The Congo": ["#007FFF", "#F7D618", "#CE1021"],
        "Denmark": ["#C60C30", "#FFFFFF"],
        "Djibouti": ["#6AB2E7", "#12AD2B", "#FFFFFF", "#D7141A"],
        "Dominica": ["#006B3F", "#FCD116", "#000000", "#FFFFFF", "#D41C30", "#9461C9"],
        "Dominican Republic": ["#002D62", "#FFFFFF", "#CE1126", "#EAC102", "#008337"],
        "East Timor": ["#000000", "#FFFFFF", "#FFC726", "#DC241F"],
        "Ecuador": ["#FFDD00", "#034EA2", "#ED1C24", "#452C25", "#B87510", "#086F35"],
        "Egypt": ["#CE1126", "#FFFFFF", "#C09300", "#000000"],
        "El Salvador": ["#0F47AF", "#FFFFFF", "#FFCC00", "#1E5B19", "#E60000"],
        "Equatorial Guinea": [
            "#000000",
            "#0073CE",
            "#3E9A00",
            "#FFFFFF",
            "#E32118",
            "#FFD700",
        ],
        "Eritrea": ["#EA0437", "#FFC726", "#12AD2B", "#4189DD"],
        "Estonia": ["#0072CE", "#000000", "#FFFFFF"],
        "Ethiopia": ["#078930", "#FCDD09", "#DA121A", "#0F47AF"],
        "Federated States of Micronesia": ["#75B2DD", "#FFFFFF"],
        "Fiji": ["#002868", "#68BFE5", "#FFFFFF", "#CE1126", "#FFD100", "#00A651"],
        "Finland": ["#002F6C", "#FFFFFF"],
        "France": ["#0055A4", "#FFFFFF", "#EF4135"],
        "Gabon": ["#009E60", "#FCD116", "#3A75C4"],
        "Gambia": ["#CE1126", "#FFFFFF", "#0C1C8C", "#3A7728"],
        "Georgia": ["#FF0000", "#FFFFFF"],
        "Germany": ["#000000", "#DD0000", "#FFCE00"],
        "Ghana": ["#CE1126", "#FCD116", "#000000", "#006B3F"],
        "Greece": ["#0D5EAF", "#FFFFFF"],
        "Grenada": ["#CE1126", "#FCD116", "#007A5E"],
        "Guatemala": ["#4997D0", "#FFFFFF", "#448127", "#F9F0AA", "#6C301E", "#B2B6BA"],
        "Guinea-Bissau": ["#000000", "#CE1126", "#FCD116", "#009E49"],
        "Guinea": ["#CE1126", "#FCD116", "#009460"],
        "Guyana": ["#CE1126", "#000000", "#FCD116", "#FFFFFF", "#009E49"],
        "Haiti": ["#00209F", "#D21034", "#FFFFFF", "#016A16", "#F1B517"],
        "Honduras": ["#0073CF", "#FFFFFF"],
        "Hungary": ["#CD2A3E", "#FFFFFF", "#436F4D"],
        "Iceland": ["#DC1E35", "#FFFFFF", "#02529C"],
        "India": ["#FF9933", "#FFFFFF", "#138808", "#000080"],
        "Indonesia": ["#FF0000", "#FFFFFF"],
        "Iran": ["#239F40", "#FFFFFF", "#DA0000"],
        "Iraq": ["#CE1126", "#FFFFFF", "#007A3D", "#000000"],
        "Ireland": ["#169B62", "#FFFFFF", "#FF883E"],
        "Israel": ["#FFFFFF", "#0038B8"],
        "Italy": ["#008C45", "#F4F5F0", "#CD212A"],
        "Ivory Coast": ["#F77F00", "#FFFFFF", "#009E60"],
        "Jamaica": ["#000000", "#FED100", "#009B3A"],
        "Japan": ["#FFFFFF", "#BC002D"],
        "Jordan": ["#CE1126", "#000000", "#FFFFFF", "#007A3D"],
        "Kazakhstan": ["#00AFCA", "#FEC50C"],
        "Kenya": ["#922529", "#008C51", "#FFFFFF", "#000000"],
        "Kiribati": ["#CE1126", "#FCD116", "#FFFFFF", "#003F87"],
        "Kuwait": ["#000000", "#007A3D", "#FFFFFF", "#CE1126"],
        "Kyrgyzstan": ["#E8112D", "#FFEF00"],
        "Laos": ["#CE1126", "#002868", "#FFFFFF"],
        "Latvia": ["#9E3039", "#FFFFFF"],
        "Lebanon": ["#ED1C24", "#FFFFFF", "#00A651"],
        "Lesotho": ["#00209F", "#FFFFFF", "#000000", "#009543"],
        "Liberia": ["#002868", "#FFFFFF", "#BF0A30"],
        "Libya": ["#E70013", "#000000", "#FFFFFF", "#239E46"],
        "Liechtenstein": ["#002B7F", "#CE1126", "#FFD83D", "#000000"],
        "Lithuania": ["#FDB913", "#006A44", "#C1272D"],
        "Luxembourg": ["#F6343F", "#FFFFFF", "#00A2E1"],
        "Macedonia": ["#D20000", "#FFE600"],
        "Madagascar": ["#FFFFFF", "#FC3D32", "#007E3A"],
        "Malawi": ["#000000", "#CE1126", "#339E35"],
        "Malaysia": ["#010066", "#CC0001", "#FFFFFF", "#FFCC00"],
        "Maldives": ["#D21034", "#007E3A", "#FFFFFF"],
        "Mali": ["#14B53A", "#FCD116", "#CE1126"],
        "Malta": ["#CF142B", "#FFFFFF", "#CCCCCC", "#96877D"],
        "Marshall Islands": ["#003893", "#FFFFFF", "#DD7500"],
        "Mauritania": ["#D01C1F", "#00A95C", "#FFD700"],
        "Mauritius": ["#EA2839", "#1A206D", "#FFD500", "#00A551"],
        "Mexico": ["#006341", "#FFFFFF", "#CE1126"],
        "Moldova": ["#0046AE", "#FFD200", "#CC092F", "#B07F55", "#097754"],
        "Monaco": ["#CE1126", "#FFFFFF"],
        "Mongolia": ["#C4272F", "#F9CF02", "#015197"],
        "Montenegro": ["#C40308", "#D3AE3B", "#1D5E91", "#6D8C3E"],
        "Morocco": ["#C1272D", "#006233"],
        "Mozambique": ["#D21034", "#007168", "#000000", "#FFFFFF", "#FCE100"],
        "Myanmar": ["#FECB00", "#34B233", "#EA2839", "#FFFFFF"],
        "Namibia": ["#FFCE00", "#003580", "#D21034", "#FFFFFF", "#009543"],
        "Nauru": ["#002B7F", "#FFC61E", "#FFFFFF"],
        "Nepal": ["#003893", "#DC143C", "#FFFFFF"],
        "Netherlands": ["#AE1C28", "#FFFFFF", "#21468B"],
        "New Zealand": ["#00247D", "#FFFFFF", "#CC142B"],
        "Nicaragua": ["#0067C6", "#FFFFFF", "#C9A504", "#6FD8F3", "#97C924", "#FF0000"],
        "Niger": ["#E05206", "#FFFFFF", "#0DB02B"],
        "Nigeria": ["#008751", "#FFFFFF"],
        "North Korea": ["#024FA2", "#FFFFFF", "#ED1C27"],
        "Norway": ["#C8102E", "#FFFFFF", "#003087"],
        "Oman": ["#FFFFFF", "#DB161B", "#008000"],
        "Pakistan": ["#01411C", "#FFFFFF"],
        "Palau": ["#0099FF", "#FFFF00"],
        "Palestine": ["#CE1126", "#000000", "#FFFFFF", "#007A3D"],
        "Panama": ["#D21034", "#FFFFFF", "#005293"],
        "Papua New Guinea": ["#000000", "#FFFFFF", "#CE1126", "#FCD116"],
        "Paraguay": ["#D52B1E", "#FFFFFF", "#0038A8", "#000000", "#009A3A", "#FEDF00"],
        "People's Republic of China": ["#DE2910", "#FFDE00"],
        "Peru": ["#D91023", "#FFFFFF"],
        "Philippines": ["#FCD116", "#0038A8", "#CE1126", "#FFFFFF"],
        "Poland": ["#FFFFFF", "#DC143C"],
        "Portugal": ["#006600", "#FF0000", "#FFFF00", "#FFFFFF", "#003399"],
        "Qatar": ["#FFFFFF", "#8D1B3D"],
        "Romania": ["#002B7F", "#FCD116", "#CE1126"],
        "Russia": ["#FFFFFF", "#0033A0", "#DA291C"],
        "Rwanda": ["#00A1DE", "#E5BE01", "#FAD201", "#20603D"],
        "Saint Kitts and Nevis": [
            "#009E49",
            "#FCD116",
            "#CE1126",
            "#FFFFFF",
            "#000000",
        ],
        "Saint Lucia": ["#66CCFF", "#FCD116", "#FFFFFF", "#000000"],
        "Saint Vincent and The Grenadines": ["#0072C6", "#FCD116", "#009E60"],
        "Samoa": ["#002B7F", "#FFFFFF", "#CE1126"],
        "San Marino": [
            "#5EB6E4",
            "#FFFFFF",
            "#F1BF31",
            "#D99F31",
            "#658D5C",
            "#94BB79",
        ],
        "Sao Tome and Principe": ["#D21034", "#12AD2B", "#FFCE00", "#000000"],
        "Saudi Arabia": ["#006C35", "#FFFFFF"],
        "Senegal": ["#00853F", "#FDEF42", "#E31B23"],
        "Serbia": ["#C6363C", "#0C4076", "#FFFFFF", "#EDB92E"],
        "Seychelles": ["#003F87", "#FCD856", "#D62828", "#FFFFFF", "#007A3D"],
        "Sierra Leone": ["#1EB53A", "#FFFFFF", "#0072C6"],
        "Singapore": ["#EF3340", "#FFFFFF"],
        "Slovakia": ["#FFFFFF", "#0B4EA2", "#EE1C25"],
        "Slovenia": ["#FFFFFF", "#005DA4", "#ED1C24", "#FFDD00"],
        "Solomon Islands": ["#0051BA", "#FFFFFF", "#FCD116", "#215B33"],
        "Somalia": ["#4189DD", "#FFFFFF"],
        "South Africa": [
            "#000000",
            "#FFB612",
            "#007A4D",
            "#FFFFFF",
            "#DE3831",
            "#002395",
        ],
        "South Korea": ["#000000", "#FFFFFF", "#CD2E3A", "#0047A0"],
        "South Sudan": [
            "#0F47AF",
            "#FCDD09",
            "#000000",
            "#FFFFFF",
            "#DA121A",
            "#078930",
        ],
        "Spain": ["#AA151B", "#F1BF00", "#0039F0", "#CCCCCC", "#ED72AA", "#058E6E"],
        "Sri Lanka": ["#FFBE29", "#8D153A", "#EB7400", "#00534E"],
        "Sudan": ["#007229", "#D21034", "#FFFFFF", "#000000"],
        "Suriname": ["#377E3F", "#FFFFFF", "#B40A2D", "#ECC81D"],
        "Swaziland": ["#3E5EB9", "#FFD900", "#B10C0C", "#FFFFFF", "#000000"],
        "Sweden": ["#004B87", "#FFCD00"],
        "Switzerland": ["#D52B1E", "#FFFFFF"],
        "Syria": ["#007A3D", "#FFFFFF", "#000000", "#CE1126"],
        "Tajikistan": ["#CC0000", "#FFFFFF", "#006600", "#F8C300"],
        "Tanzania": ["#1EB53A", "#FCD116", "#000000", "#00A3DD"],
        "Thailand": ["#A51931", "#F4F5F8", "#2D2A4A"],
        "Togo": ["#D21034", "#FFFFFF", "#006A4E", "#FFCE00"],
        "Tonga": ["#FFFFFF", "#C10000"],
        "Trinidad and Tobago": ["#DA1A35", "#FFFFFF", "#000000"],
        "Tunisia": ["#E70013", "#FFFFFF"],
        "Turkey": ["#E30A17", "#FFFFFF"],
        "Turkmenistan": ["#00843D", "#FFFFFF", "#D22630", "#FFC72C"],
        "Tuvalu": ["#00247D", "#FFFFFF", "#CF142B", "#FFCE00", "#5B97B1"],
        "Uganda": ["#000000", "#FCDC04", "#D90000", "#FFFFFF", "#9CA69C"],
        "Ukraine": ["#005BBB", "#FFD500"],
        "United Arab Emirates": ["#FF0000", "#00732F", "#FFFFFF", "#000000"],
        "United Kingdom": ["#00247D", "#FFFFFF", "#CF142B"],
        "United States of America": ["#3C3B6E", "#FFFFFF", "#B22234"],
        "Uruguay": ["#FCD116", "#7B3F00", "#FFFFFF", "#0038A8"],
        "Uzbekistan": ["#0099B5", "#CE1126", "#FFFFFF", "#1EB53A"],
        "Vanuatu": ["#000000", "#FDCE12", "#D21034", "#009543"],
        "Vatican City": ["#FFE100", "#FFFFFF", "#CDCDCD", "#FF0000"],
        "Venezuela": ["#FFCC00", "#00247D", "#CF142B", "#FFFFFF"],
        "Vietnam": ["#DA251D", "#FFCD00"],
        "Yemen": ["#CE1126", "#FFFFFF", "#000000"],
        "Zambia": ["#198A00", "#DE2010", "#000000", "#EF7D00"],
        "Zimbabwe": ["#006400", "#FFD200", "#D40000", "#000000", "#FFFFFF", "#FFCC00"],
    }

    def __init__(self, name, mode="lin", sort=True):
        self.colors = self.palettes[name]
        self.name = name
        self.mode = mode
        self.sort = sort

    def __repr__(self):
        return (
            f"{self.__class__.__name__}" f"({self.name!r}, {self.mode!r}, {self.sort})"
        )

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value not in ("abs", "lin", "log"):
            raise ValueError(f"invalid palette mode {value!r}")
        self._mode = value

    @property
    def sort(self):
        return self._sort

    @sort.setter
    def sort(self, value):
        self._sort = bool(value)

    def _ipython_display_(self, width=400):
        w = max(1, width / len(self.colors))
        s = f"background-color: {{col}}; padding: 8px {w}px; color: {{col}};"
        c = "color" if len(self.colors) == 1 else "colors"
        display(
            ipw.VBox(
                [
                    ipw.HTML(f"<b>{self.name}</b>" f" ({len(self.colors)} {c})"),
                    ipw.HTML(
                        "".join(
                            f"<span style=" f"{s.format(col=col)!r}></span>"
                            for col in self.colors
                        )
                    ),
                ]
            )
        )

    def mkcolors(self, values, circular="h", linear="sl"):
        src = pd.DataFrame.from_records(
            (Color(c).hsl for c in self.colors), columns=["h", "s", "l"]
        )
        if self.mode == "abs":
            values = values.astype(float)
            pal = pd.DataFrame(index=values.index, columns=["h", "s", "l"])
            m, M = values.min(), values.max()
            if m == M:
                return [self.colors[0]] * len(values)
            pal["pos"] = (values.astype(float) - m) * (len(self.colors) - 1.0) / (M - m)
        else:
            unique = pd.Series(values.unique())
            if self.sort:
                unique.sort_values(inplace=True)
            count = len(unique)
            pal = pd.DataFrame(index=range(count), columns=["h", "s", "l"])
            if self.mode == "lin":
                pal["pos"] = np.linspace(
                    0.0, len(self.colors) - 1.0, count, dtype=float
                )
            else:
                log = np.logspace(0.0, 1.0, count, base=10, dtype=float)
                pal["pos"] = (log - 1.0) / 9.0
        pal["left"] = np.floor(pal["pos"]).astype(int)
        pal["right"] = np.ceil(pal["pos"]).astype(int)
        P = pal["pos"] = pal["pos"] - pal["left"]
        for col in circular.lower():
            L = pal["left"].map(src[col].get) * 2 * pi
            R = pal["right"].map(src[col].get) * 2 * pi
            pal[col] = np.mod(
                np.arctan2(
                    P * np.sin(R) + (1 - P) * np.sin(L),
                    P * np.cos(R) + (1 - P) * np.cos(L),
                )
                / (2 * pi),
                1.0,
            )
        for col in linear.lower():
            L = pal["left"].map(src[col].get)
            R = pal["right"].map(src[col].get)
            pal[col] = P * R + (1 - P) * L
        colors = [
            Color(hsl=row).hex for row in pal[["h", "s", "l"]].itertuples(index=False)
        ]
        if self.mode == "abs":
            return colors
        else:
            cmap = dict(zip(unique, colors))
            return [cmap[c] for c in values]


for module in (bqcm, scm):
    for name in sorted(getattr(module, "__all__")):
        Palette.palettes[name] = getattr(module, name)


class _Desc(object):
    def __set_name__(self, obj, name):
        if not hasattr(obj, "_opts"):
            obj._opts = {}
        obj._opts[name] = self


class PaletteDesc(_Desc):
    def __init__(self, default, mode="lin", sort=True):
        self.palette = Palette(default, mode, sort)

    def __set_name__(self, obj, name):
        super().__set_name__(obj, name)
        base, end = name.rsplit("_", 1)
        assert end == "palette", f"invalid palette attribute name: {name!r}"
        self.name = name
        self.color = f"{base}_color"

    def __get__(self, obj, type=None):
        return self.palette

    def __set__(self, obj, value):
        if isinstance(value, str):
            value = [value]
        self.palette = Palette(*value)
        setattr(obj, self.color, None)


class _TableDesc(_Desc):
    def __init__(self, table, store, column, default):
        self.table = table
        self.store = store
        self.column = column
        self.default = default

    def __set_name__(self, obj, name):
        super().__set_name__(obj, name)
        if not hasattr(obj, "_ncol"):
            obj._ncol = {}
        if not hasattr(obj, "_ecol"):
            obj._ecol = {}
        if name.startswith("nodes_"):
            obj._ncol[name] = self.column
        elif name.startswith("edges_"):
            obj._ecol[name] = self.column

    def __get__(self, obj, type=None):
        store = getattr(obj, self.store)
        if self.column not in store.columns:
            self.__set__(obj, self.default)
        return store[self.column]


class TableImageDesc(_TableDesc):
    def __init__(self, table, store, column, default=""):
        super().__init__(table, store, column, default)
        self.root = Path("/")
        self.url = None
        self.cwd = cwd = Path.cwd()
        for server in serverapp.list_running_servers():
            root = Path(server["root_dir"])
            if cwd.is_relative_to(root):
                if root.is_relative_to(self.root):
                    self.root = root
                    self.url = Path(server["base_url"]) / "files"

    def geturl(self, location):
        if urlparse(location).scheme:
            url = location
        elif self.url is None:
            raise ValueError(f"invalid URL {location!r}")
        else:
            url = self.url / (self.cwd / Path(location)).relative_to(self.root)
        return f"url({url})"

    def __get__(self, obj, type=None):
        store = getattr(obj, self.store)
        if self.column in store.columns:
            return store[self.column]

    def __set__(self, obj, val):
        store = getattr(obj, self.store)
        table = getattr(obj, self.table)
        if not val:
            # FIXME: restore bg-color
            store.drop(
                columns=[self.column, "background-fit"], errors="ignore", inplace=True
            )
            return
        elif isinstance(val, str) and val in table.columns:
            images = [self.geturl(v) for v in table[val]]
        elif isinstance(val, (list, tuple, pd.Series, np.ndarray)) and len(val) == len(
            table
        ):
            images = [self.geturl(v) for v in val]
        else:
            raise ValueError(f"cannot use {val!r} as image(s)")
        store[self.column] = images
        store["background-fit"] = ["cover"] * len(table)
        # FIXME: save bg-color
        store["background-color"] = ["transparent"] * len(table)


class TableColorDesc(_TableDesc):
    def __init__(self, table, store, column, default):
        super().__init__(table, store, column, default)
        self.last = default

    def __set_name__(self, obj, name):
        super().__set_name__(obj, name)
        base, end = name.rsplit("_", 1)
        assert end == "color", f"invalid color attribute name: {name!r}"
        self.name = name
        self.palette = f"{base}_palette"

    def __set__(self, obj, val):
        store = getattr(obj, self.store)
        table = getattr(obj, self.table)
        if val is None:
            val = self.last
        if isinstance(val, str) and maybeColor(val) is not None:
            obj._legend[self.name] = None
            colors = [Color(val).hex] * len(table)
        elif isinstance(val, str) and val in table.columns:
            obj._legend[self.name] = val
            colors = [maybeColor(c) for c in table[val]]
            if any(c is None for c in colors):
                palette = getattr(obj, self.palette)
                if not is_numeric_dtype(table[val]) and palette.mode == "abs":
                    palette.mode = "lin"
                colors = palette.mkcolors(table[val])
        elif isinstance(val, (list, tuple, pd.Series, np.ndarray)) and len(val) == len(
            table
        ):
            obj._legend[self.name] = None
            colors = [maybeColor(c) for c in val]
            if any(c is None for c in colors):
                pos = colors.index(None)
                raise ValueError(f"invalid color {colors[pos]!r}")
        else:
            raise ValueError(f"cannot use {val!r} as color(s)")
        self.last = val
        store[self.column] = colors


class TableNumberDesc(_TableDesc):
    def __init__(self, table, store, column, default, mini, maxi, cls=None):
        if cls is None:
            cls = type(default)
        self.cls = cls
        super().__init__(table, store, column, default)
        assert mini < maxi, "mini should be strictly less than maxi"
        self.mini = cls(mini)
        self.maxi = cls(maxi)

    def _check(self, n):
        if n < self.mini:
            raise ValueError(f"value should be at least {self.mini}")
        elif n > self.maxi:
            raise ValueError(f"value should be at most {self.maxi}")

    def __set__(self, obj, val):
        store = getattr(obj, self.store)
        table = getattr(obj, self.table)
        if isinstance(val, (int, float)):
            numbers = self.cls(val)
            self._check(numbers)
        elif (isinstance(val, str) and val in table.columns) or (
            isinstance(val, (list, tuple, pd.Series, np.ndarray))
            and len(val) == len(table)
        ):
            if isinstance(val, str) and val in table.columns:
                numbers = table[val].astype(self.cls)
            else:
                numbers = pd.Series(data=val, index=table.index, dtype=self.cls)
            numbers.map(self._check)
        else:
            raise ValueError(f"cannot use {val!r} as {self.cls.__name__}(s)")
        store[self.column] = numbers


class TableTipDesc(_TableDesc):
    tips = {
        "triangle",
        "triangle-tee",
        "circle-triangle",
        "triangle-cross",
        "triangle-backcurve",
        "vee",
        "tee",
        "square",
        "circle",
        "diamond",
        "chevron",
        "none",
    }
    alias = {
        ">": "filled-triangle",
        "<": "filled-triangle",
        "*": "filled-circle",
        "o": "hollow-circle",
        "|": "filled-tee",
        "-": "none",
    }

    def __init__(self, table, store, column, default):
        super().__init__(table, store, column, default)
        base, end = column.rsplit("-", 1)
        assert end == "shape", f"invalid tip column name: {column!r}"
        self.fill_column = f"{base}-fill"

    def _split(self, spec):
        spec = self.alias.get(spec, spec)
        if spec.startswith(("filled-", "hollow-")):
            fill, shape = spec.split("-", 1)
        else:
            fill, shape = "filled", spec
        if shape not in self.tips:
            raise ValueError(f"invalid arrow tip {shape!r}")
        return fill, shape

    def __set__(self, obj, val):
        store = getattr(obj, self.store)
        table = getattr(obj, self.table)
        if isinstance(val, str) and val in table.columns:
            fill, shape = list(zip(*table[val].apply(self._split)))
        elif isinstance(val, str):
            fill, shape = self._split(val)
        elif isinstance(val, (list, tuple, pd.Series, np.ndarray)) and len(val) == len(
            table
        ):
            fill, shape = list(zip(*map(self._split, val)))
        else:
            raise ValueError(f"cannot use {val!r} as arrow tip(s)")
        store[self.fill_column] = fill
        store[self.column] = shape


def in_dict(k, d):
    try:
        return k in d
    except Exception:
        return False


class _TableEnumDesc(_TableDesc):
    values = {}
    alias = {}

    def __init__(self, table, store, column, default, description):
        super().__init__(table, store, column, default)
        self.description = description

    def _check(self, spec):
        val = self.alias.get(spec, spec)
        if val not in self.values:
            raise ValueError(f"invalid {self.description} {spec!r}")
        return val

    def __set__(self, obj, val):
        store = getattr(obj, self.store)
        table = getattr(obj, self.table)
        if isinstance(val, str) and val in table.columns:
            values = table[val].apply(self._check)
        elif isinstance(val, str) or in_dict(val, self.alias):
            values = self._check(val)
        elif isinstance(val, (list, tuple, pd.Series, np.ndarray)) and len(val) == len(
            table
        ):
            values = list(map(self._check, val))
        else:
            raise ValueError(f"cannot use {val!r} as {self.description}")
        store[self.column] = values


class TableShapeDesc(_TableEnumDesc):
    values = {
        "ellipse",
        "triangle",
        "round-triangle",
        "rectangle",
        "round-rectangle",
        "bottom-round-rectangle",
        "cut-rectangle",
        "barrel",
        "rhomboid",
        "diamond",
        "round-diamond",
        "pentagon",
        "round-pentagon",
        "hexagon",
        "round-hexagon",
        "concave-hexagon",
        "heptagon",
        "round-heptagon",
        "octagon",
        "round-octagon",
        "star",
        "tag",
        "round-tag",
        "vee",
    }

    def __init__(self, table, store, column, default):
        super().__init__(table, store, column, default, "shape")


class TableCurveDesc(_TableEnumDesc):
    values = {"bezier", "straight"}

    def __init__(self, table, store, column, default):
        super().__init__(table, store, column, default, "curve")


class TableStyleDesc(_TableEnumDesc):
    values = {"solid", "dotted", "dashed", "double"}
    alias = {"-": "solid", "|": "solid", ":": "dotted", "!": "dashed", "=": "double"}

    def __init__(self, table, store, column, default):
        super().__init__(table, store, column, default, "style")


def _str(txt, **args):
    return str(txt)


class TableStrDesc(_TableDesc):
    def __init__(self, table, store, column, default, convert=_str):
        super().__init__(table, store, column, default)
        self.convert = convert

    def __set__(self, obj, val):
        store = getattr(obj, self.store)
        table = getattr(obj, self.table)
        if isinstance(val, str) and val in table.columns:
            strings = table[val].apply(self.convert)
        elif isinstance(val, str):
            strings = []
            for idx, row in table.iterrows():
                if not isinstance(idx, (tuple, list)):
                    idx = [idx]
                d = dict(zip(table.index.names, idx))
                d.update(row)
                strings.append(self.convert(val.format(**d), **d))
        elif isinstance(val, (list, tuple, pd.Series, np.ndarray)) and len(val) == len(
            table
        ):
            strings = list(map(self.convert, val))
        else:
            raise ValueError(f"cannot use {val!r} as string(s)")
        store[self.column] = strings


class GroupDesc(_Desc):
    def __init__(self, *members):
        self.members = tuple(members)

    def __set_name__(self, obj, name):
        super().__set_name__(obj, name)
        if not hasattr(obj, "_ncol"):
            obj._ncol = {}
        if not hasattr(obj, "_ecol"):
            obj._ecol = {}
        if name.startswith("nodes_"):
            obj._ncol[name] = self.members
        elif name.startswith("edges_"):
            obj._ecol[name] = self.members

    def __get__(self, obj, type=None):
        return tuple(getattr(obj, m) for m in self.members)

    def __set__(self, obj, val):
        for m in self.members:
            setattr(obj, m, val)


class TableDisplayDesc(_TableEnumDesc):
    values = {"element", "none"}
    alias = {True: "element", False: "none"}


class TableVisibilityDesc(_TableEnumDesc):
    values = {"visible", "hidden"}
    alias = {True: "visible", False: "hidden"}


class NodeShowDesc(_Desc):
    modes = {
        "all": {True: [True, True, 1.0], False: [True, True, 1.0]},
        "dim": {True: [True, True, 0.2], False: [True, True, 1.0]},
        "hide": {True: [True, False, 1.0], False: [True, True, 1.0]},
        "drop": {True: [False, True, 1.0], False: [True, True, 1.0]},
    }

    def __init__(self, nodes="nodes", edges="edges"):
        self.mode = "all"
        self.values = True
        self.nodes = nodes
        self.edges = edges
        self.dim = 0.2
        self.invert = False

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, val):
        if val not in self.modes:
            raise ValueError(f"cannot use {val!r} as show mode")
        self._mode = val

    @property
    def dim(self):
        return self.modes["dim"][True][2]

    @dim.setter
    def dim(self, val):
        try:
            val = float(val)
        except Exception:
            raise ValueError(f"cannot use {val!r} as dim level")
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"cannot use {val!r} as dim level")
        self.modes["dim"][True][2] = val

    @property
    def invert(self):
        return self._invert

    @invert.setter
    def invert(self, val):
        self._invert = bool(val)

    def __get__(self, obj, type=None):
        return self.mode, self.values, self.dim, self.invert

    def __set__(self, obj, val):
        if isinstance(val, (str, bool)) and val in ("all", True):
            mode, val = "all", True
        elif isinstance(val, tuple) and 2 <= len(val) <= 4:
            if len(val) == 2:
                mode, val = val
            elif len(val) == 3:
                mode, val, self.invert = val
            else:
                mode, val, self.invert, self.dim = val
        else:
            raise ValueError(f"cannot use {val!r} as show mode and value")
        nodes = getattr(obj, self.nodes)
        edges = getattr(obj, self.edges)
        if isinstance(val, bool):
            values = pd.Series([val] * len(nodes.index), index=nodes.index)
        elif isinstance(val, str) and val in nodes.columns:
            values = nodes[val].astype(bool)
        elif isinstance(val, (tuple, list)) and len(val) == len(nodes.index):
            values = pd.Series([bool(v) for v in val], index=nodes.index)
        else:
            raise ValueError(f"cannot use {val!r} as show value")
        self.mode = mode
        if self.invert:
            values = ~values
        self.values = values
        modes = self.modes[mode]
        hide = set(str(v) for v in values[values].index)
        ei = edges.index.to_frame()
        ev = ei["src"].isin(hide) | ei["dst"].isin(hide)
        for idx, attr in enumerate(("display", "visibility", "opacity")):
            setattr(obj, f"nodes_show_{attr}", [modes[v][idx] for v in values])
            setattr(obj, f"edges_show_{attr}", [modes[v][idx] for v in ev])


class TableValignDesc(_TableEnumDesc):
    values = {"top", "center", "bottom"}

    def __init__(self, table, store, column, default):
        super().__init__(table, store, column, default, "label align")


class TableHalignDesc(_TableEnumDesc):
    values = {"left", "center", "right"}

    def __init__(self, table, store, column, default):
        super().__init__(table, store, column, default, "label align")


class TableAlignDesc(_Desc):
    g2a = {
        "n": "top",
        "s": "bottom",
        "m": "center",
        "c": "center",
        "e": "right",
        "w": "left",
    }
    h = "emcw"
    v = "nmcs"

    def __init__(self, h, v):
        self.hopt = h
        self.vopt = v
        self.a2g = {v: k for k, v in self.g2a.items()}

    def __get__(self, obj, type=None):
        return getattr(obj, self.hopt), getattr(obj, self.vopt)

    def valid(self, txt):
        if not isinstance(txt, str):
            return False
        h = v = None
        for c in txt.lower():
            if h is None and c in self.h:
                h = c
            elif v is None and c in self.v:
                v = c
            else:
                return False
        return h is not None or v is not None

    def split(self, txt):
        h = v = None
        for c in txt.lower():
            if h is None and c in self.h:
                h = self.g2a[c]
            elif v is None and c in self.v:
                v = self.g2a[c]
            else:
                raise ValueError(f"invalid alignment {txt!r}")
        return h, v

    def __set__(self, obj, val):
        table = getattr(obj, obj._opts[self.hopt].table)
        if (isinstance(val, str) and val in table.columns) or (
            isinstance(val, (list, tuple, pd.Series, np.ndarray))
            and len(val) == len(table)
        ):
            if isinstance(val, str):
                align = pd.DataFrame(
                    table[val].map(self.split).tolist(), columns=["h", "v"]
                )
            else:
                align = pd.DataFrame(
                    pd.Series(val).map(self.split).tolist(), columns=["h", "v"]
                )
            setattr(obj, self.hopt, align["h"].tolist())
            setattr(obj, self.vopt, align["v"].tolist())
        elif isinstance(val, str):
            h, v = self.split(val)
            setattr(obj, self.hopt, h or "center")
            setattr(obj, self.vopt, v or "center")
        else:
            raise ValueError(f"cannot use {val!r} as alignment(s)")


class TableWrapDesc(_TableEnumDesc):
    values = {"wrap", "none", "ellipsis"}

    def __init__(self, table, store, column, default):
        super().__init__(table, store, column, default, "text wrap")


class TableAngleDesc(_TableDesc):
    def __set__(self, obj, val):
        store = getattr(obj, self.store)
        table = getattr(obj, self.table)
        if isinstance(val, str) and val in ("autorotate", "auto"):
            angles = "autorotate"
        elif isinstance(val, str) and val in table.columns:
            angles = table[val].astype(float)
        elif isinstance(val, (int, float)):
            angles = float(val)
        elif isinstance(val, (list, tuple, pd.Series, np.ndarray)) and len(val) == len(
            table
        ):
            angles = list(map(float, val))
        else:
            raise ValueError(f"cannot use {val!r} as angle(s)")
        store[self.column] = angles


class LayoutDesc(_Desc):
    values = {
        "random",
        "grid",
        "circle",
        "concentric",
        "breadthfirst",
        "cose",
        "dagre",
        "preset",
        "gv-dot",
        "gv-neato",
        "gv-twopi",
        "gv-circo",
        "gv-fdp",
    }
    # "ig-large", "ig-drl", "ig-fr", "ig-kk",
    # "nx-shell", "nx-spring", "nx-spectral", "nx-spiral"}
    alias = {}

    def __init__(self, default):
        self.value, self.args = self._check(default)

    def _check(self, spec):
        if isinstance(spec, (tuple, list)):
            spec, args = spec
        else:
            args = {}
        val = self.alias.get(spec, spec)
        if val not in self.values:
            raise ValueError(f"invalid graph layout {spec!r}")
        return val, dict(args)

    def __get__(self, obj, type=None):
        return self.value, self.args

    def __set__(self, obj, val):
        self.value, self.args = self._check(val)
        val, args = self.get_layout(self.value, obj)
        if args is None:
            args = self.args
        else:
            args.update((k, v) for k, v in self.args.items() if k not in args)
        obj._cy.set_layout(name=val, **args)

    def get_layout(self, layout, graph):
        for node in graph._cy.graph.nodes:
            node.position = {}
        if layout == "grid":
            return "grid", {"rows": math.ceil(math.sqrt(len(graph.nodes)))}
        elif layout.startswith(("nx-", "gv-", "ig-", "xx-")):
            if layout.startswith("xx-"):
                pos = graph.layout_extra[layout[3:]]
                if callable(pos):
                    pos = pos()
                pos = {str(k): v for k, v in pos.items()}
            else:
                algo = layout[3:]
                n = nx.DiGraph()
                for nid in graph.nodes.index:
                    if layout.startswith("gv-") and nid in graph.groups:
                        n.add_node(f"cluster_{nid}", original_nodes=graph.groups[nid])
                    else:
                        n.add_node(nid)
                n.add_edges_from(graph.edges.index)
                if layout.startswith("nx-"):
                    fun = getattr(nx, f"{algo}_layout")
                    pos = fun(n, scale=80)
                elif layout.startswith("gv-"):
                    pos = nx.nx_pydot.pydot_layout(n, prog=algo)
                    pos = {
                        k[8:]
                        if k.startswith("cluster_") and k[8:] in graph.groups
                        else k: (x, -y)
                        for k, (x, y) in pos.items()
                    }
                else:
                    g = ig.Graph.from_networkx(n)
                    l = g.layout(algo)
                    l.fit_into((150, 150))
                    pos = dict(zip(graph.nodes.index, l.coords))
            for node in graph._cy.graph.nodes:
                x, y = pos[node.data["id"]]
                node.position = {"x": x, "y": y}
            return "preset", {"fit": True}
        else:
            return layout, None


def state_str(txt, **args):
    txt = str(txt)
    if not txt:
        return ""
    icons = []
    try:
        if "has_init" in args["topo"]:
            icons.append("▶")
        if "has_scc" in args["topo"] or "is_hull" in args["topo"]:
            icons.append("⏺")
        if "has_dead" in args["topo"] or "is_dead" in args["topo"]:
            icons.append("⏹")
    except Exception:
        pass
    if icons:
        return f"{txt}\n{''.join(icons)}"
    else:
        return txt


_export_html = "".join(
    """<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body>
<a download="{filename}" href="data:{mimetype};charset=utf-8,{payload}">
Download {filename}
</a>
</body>
</html>""".splitlines()
)


class widget_spec(str):
    "a widget spec name with configuration options"

    def __new__(cls, value, **config):
        return str.__new__(cls, value)

    def __init__(self, value, **config):
        super().__init__()
        self.config = dict(config)


def configure_ui(ui=None, **changes):
    """customise `Graph.ui` by removing or configuring its elements

    Each element is passed as a `key=val` option where `val` may be:

     - `None`: this element is removed, key may be a widget or section name
     - a `dict`: this element is replaced with `widget_spec(key, **val)`, or
       if `key` is a title (as `"Nodes"`) the section is replaced by `val`
    """
    if ui is None:
        ui = Graph.ui
    if isinstance(ui, str):
        if ui in changes:
            if changes[ui] is None:
                return None
            else:
                return widget_spec(ui, **changes[ui])
        else:
            return ui
    elif isinstance(ui, list):
        ret = []
        for item in ui:
            new = configure_ui(item, **changes)
            if new is not None:
                ret.append(new)
        return ret or None
    elif isinstance(ui, dict):
        ret = {}
        for key, val in ui.items():
            if key in changes:
                if changes[key] is None:
                    new = None
                else:
                    new = configure_ui(changes[key], **changes)
            else:
                new = configure_ui(val, **changes)
            if new is not None:
                ret[key] = new
        return ret or None
    else:
        raise ValueError(f"unexpected UI spec: {ui!r}")


class Graph(object):
    """An interactive graph that can be displayed in Jupyter"""

    layout = LayoutDesc("cose")
    layout_extra = {}
    # nodes
    nodes_shape = TableShapeDesc("nodes", "_ns", "shape", "round-rectangle")
    nodes_width = TableNumberDesc("nodes", "_ns", "width", 20.0, 0.0, 100.0)
    nodes_height = TableNumberDesc("nodes", "_ns", "height", 20.0, 0.0, 100.0)
    nodes_image = TableImageDesc("nodes", "_ns", "background-image")
    nodes_fill_color = TableColorDesc("nodes", "_ns", "background-color", "size")
    nodes_fill_palette = PaletteDesc("red-green/white", "abs")
    nodes_fill_opacity = TableNumberDesc(
        "nodes", "_ns", "background-opacity", 1.0, 0.0, 1.0
    )
    nodes_draw_color = TableColorDesc("nodes", "_ns", "border-color", "black")
    nodes_draw_palette = PaletteDesc("black")
    nodes_draw_width = TableNumberDesc("nodes", "_ns", "border-width", 0.0, 0.0, 10.0)
    nodes_draw_style = TableStyleDesc("nodes", "_ns", "border-style", "|")
    nodes_show_display = TableDisplayDesc(
        "nodes", "_ns", "display", "element", "display"
    )
    nodes_show_visibility = TableVisibilityDesc(
        "nodes", "_ns", "visibility", "visible", "visibility"
    )
    nodes_show_opacity = TableNumberDesc("nodes", "_ns", "opacity", 1.0, 0.0, 1.0)
    nodes_show = NodeShowDesc()
    # nodes labels
    nodes_label = TableStrDesc("nodes", "_ns", "label", "{node}", state_str)
    nodes_label_wrap = TableWrapDesc("nodes", "_ns", "text-wrap", "wrap")
    nodes_label_size = TableNumberDesc("nodes", "_ns", "font-size", 8.0, 0.0, 40.0)
    nodes_label_valign = TableValignDesc("nodes", "_ns", "text-valign", "center")
    nodes_label_halign = TableHalignDesc("nodes", "_ns", "text-halign", "center")
    nodes_label_angle = TableAngleDesc("nodes", "_ns", "text-rotation", 0)
    nodes_label_outline_color = TableColorDesc(
        "nodes", "_ns", "text-outline-color", "white"
    )
    nodes_label_outline_palette = PaletteDesc("white")
    nodes_label_outline_opacity = TableNumberDesc(
        "nodes", "_ns", "text-outline-opacity", 0.0, 0.0, 1.0
    )
    nodes_label_outline_width = TableNumberDesc(
        "nodes", "_ns", "text-outline-width", 0.0, 0.0, 10.0
    )
    # edges
    edges_curve = TableCurveDesc("edges", "_es", "curve-style", "bezier")
    edges_draw_color = TableColorDesc("edges", "_es", "line-color", "black")
    edges_draw_palette = PaletteDesc("black")
    edges_draw_width = TableNumberDesc("edges", "_es", "width", 0.6, 0.0, 10.0)
    edges_draw_style = TableStyleDesc("edges", "_es", "line-style", "|")
    # edges labels
    edges_label = TableStrDesc("edges", "_es", "label", "{src}-{dst}")
    edges_label_size = TableNumberDesc("edges", "_es", "font-size", 6.0, 0.0, 40.0)
    edges_label_angle = TableAngleDesc("edges", "_es", "text-rotation", "autorotate")
    edges_label_outline_color = TableColorDesc(
        "edges", "_es", "text-outline-color", "white"
    )
    edges_label_outline_palette = PaletteDesc("white")
    edges_label_outline_opacity = TableNumberDesc(
        "edges", "_es", "text-outline-opacity", 0.75, 0.0, 1.0
    )
    edges_label_outline_width = TableNumberDesc(
        "edges", "_es", "text-outline-width", 3.0, 0.0, 10.0
    )
    edges_show_display = TableDisplayDesc(
        "edges", "_es", "display", "element", "display"
    )
    edges_show_visibility = TableVisibilityDesc(
        "edges", "_es", "visibility", "visible", "visibility"
    )
    edges_show_opacity = TableNumberDesc("edges", "_es", "opacity", 1.0, 0.0, 1.0)
    # tips
    edges_tip_scale = TableNumberDesc("edges", "_es", "arrow-scale", 0.6, 0.0, 5.0)
    edges_target_tip = TableTipDesc("edges", "_es", "target-arrow-shape", ">")
    edges_target_color = TableColorDesc("edges", "_es", "target-arrow-color", "black")
    edges_target_palette = PaletteDesc("black")
    edges_source_tip = TableTipDesc("edges", "_es", "source-arrow-shape", "-")
    edges_source_color = TableColorDesc("edges", "_es", "source-arrow-color", "black")
    edges_source_palette = PaletteDesc("black")
    # meta-properties
    nodes_size = GroupDesc("nodes_width", "nodes_height")
    nodes_color = GroupDesc("nodes_draw_color", "nodes_fill_color")
    nodes_palette = GroupDesc("nodes_draw_palette", "nodes_fill_palette")
    nodes_label_align = TableAlignDesc("nodes_label_halign", "nodes_label_valign")
    edges_tip_color = GroupDesc("edges_target_color", "edges_source_color")
    edges_tip_palette = GroupDesc("edges_target_palette", "edges_source_palette")
    edges_color = GroupDesc("edges_draw_color", "edges_tip_color")
    edges_palette = GroupDesc("edges_draw_palette", "edges_tip_palette")
    #
    ui = [
        {"Graph": [["layout", "reset_view"]]},
        {
            "Nodes": [
                ["nodes_shape", "nodes_size"],
                ["nodes_show"],
                {
                    "Fill": [
                        ["nodes_fill_color", "nodes_fill_palette"],
                        ["nodes_fill_opacity"],
                    ]
                },
                {
                    "Draw": [
                        ["nodes_draw_color", "nodes_draw_palette"],
                        ["nodes_draw_style", "nodes_draw_width"],
                    ]
                },
                {
                    "Labels": [
                        ["nodes_label", "nodes_label_size"],
                        ["nodes_label_align", "nodes_label_angle"],
                        {
                            "Outline": [
                                [
                                    "nodes_label_outline_color",
                                    "nodes_label_outline_palette",
                                ],
                                [
                                    "nodes_label_outline_opacity",
                                    "nodes_label_outline_width",
                                ],
                            ]
                        },
                    ]
                },
            ]
        },
        {
            "Edges": [
                ["edges_draw_color", "edges_draw_palette"],
                ["edges_draw_width", "edges_draw_style"],
                {
                    "Labels": [
                        ["edges_label", "edges_label_size"],
                        ["edges_label_angle"],
                        {
                            "Outline": [
                                [
                                    "edges_label_outline_color",
                                    "edges_label_outline_palette",
                                ],
                                [
                                    "edges_label_outline_opacity",
                                    "edges_label_outline_width",
                                ],
                            ]
                        },
                    ]
                },
                {
                    "Tips": [
                        ["edges_tip_scale"],
                        {
                            "Source": [
                                ["edges_source_tip"],
                                ["edges_source_color", "edges_source_palette"],
                            ]
                        },
                        {
                            "Target": [
                                ["edges_target_tip"],
                                ["edges_target_color", "edges_target_palette"],
                            ]
                        },
                    ]
                },
            ]
        },
        "graph",
        {
            "Inspector": [
                ["select_all", "select_none", "invert_selection"],
                ["inspect_nodes"],
                ["inspect_edges"],
            ]
        },
        {"Export": [["export"]]},
    ]

    def __init__(self, nodes, edges, **args):
        """Build a new `Graph` instance

        # Arguments
         - `nodes`: a `pandas.DataFrame` with the nodes, it must have a column
           `node` with the nodes identites
         - `edges`: a `pandas.DataFrame` with the edges, it must have two
            columns `src` and `dst` with the nodes sources and destination
            as provided in `nodes["node"]`
         - `**args`: varied options listed below

        Most nodes and edges options may be given as a single value that
        is applied to every nodes/edges, or as the name of a column in
        `nodes`/`edges` tables containing the values to apply to each node/edge,
        or as an array of values whose length is the number of
        nodes/edges. When a column or an array is provided, it must
        contain valid values for the expected property, except for the
        color that can be computed in every case.

        ## Graph options

         - `layout (str|tuple[str,dict])`: a layout algorithm, or a pair
           `(algo, options)` where options are the parameters for the chosen
           algorithm
         - `layout_extra (dict)`: additional layout algorithms to be added to
           the UI, mapping names to functions or static positions

        Layout algorithms are mainly provided by
        [Cytoscape.js](https://js.cytoscape.org) or by
        [GraphViz](https://www.graphviz.org) (all those named `gv-...`).
        Available layout algorithms are:

         - `random`: distribute nodes randomly
         - `grid`: distribute nodes on a rectangular grid
         - `circle` and `gv-circo`: distribute the nodes on a circle
         - `concentric` and `gv-twopi`: distribute the nodes on concentric
           circles
         - `breadthfirst`: distribute the nodes hierarchically as a
           breadth-first spanning tree
         - `cose`, `gv-neato`, `gv-fdp`: distribute the nodes using a spring
           simulation
         - `dagre` and `gv-dot`: distribute the nodes hierarchically

        ## Colors

        Colors are obtained from a pair of options `..._color` and
        `..._palette` as follows:

         - a named color (`"red"`, etc.) is applied to evey nodes/edges
         - a HTML colors (`"#C0FFEE"`, etc.) also
         - a column/array of values is used to compute a color for each unique
           value according to the palette (see below)

        A palette is provided as either:

         - a name (`str`) that maps to a series of colors, see `Palette.palettes`
         - an optional mode (one of `"lin"` (default), `"log"`, or `"abs"`) that
           specifies how colors in the palette are interpolated
            - when `mode="lin"` the set of unique values is linearly distributed
              onto the list of colors in the palette, and unique colors are computed
              by interpolation.
            - when `mode="log"` the same principle is applied but the values are
              distributed on a logarithmic scale (values are close one to each other
              at the beginning and get farther as we move to the right)
            - when `mode="abs"` the values must be numbers and are directly used as
              the position within the palette (after being scaled to the appropriate
              range)
         - an optional `bool` (default: `True`) that specifies if values are sorted
           before to generate colors for them

        Consider for example a palette of 4 colors with 5 unique values,
        they can be distributed linearly as:

                  palette:   "#AFA"  "#FAA"   "#FF8"   "#AAF"
                  values:    a       b       c       d       e

        The color for `a` will be `"#AFA"`, the color for `e` will be
        `"#AAF"`, the color for `c` will be 50% `"#FAA"` and 50% `"#FF8"`, the
        color for `b` will be mostly `"#FFA"` with a touch of `"#AFA"`, etc.

        ## Nodes options

         - `nodes_shape`: shape as listed in `TableShapeDesc.values`
         - `nodes_width`: width as `int` or `float`
         - `nodes_height`: height as `int` or `float`
         - `nodes_size`: both width and eight, as `int` or `float`
         - `node_fill_color`: background color
         - `nodes_fill_palette`: palette for background color
         - `nodes_fill_opacity`: background opacity (`float` between `0.0`
           and `1.0`)
         - `nodes_draw_color`: color for the border of nodes
         - `nodes_draw_palette`: palette for `nodes_draw_color`
         - `nodes_draw_style`: line style for the border of nodes,
           as listed in `TableStyleDesc.values`
         - `nodes_draw_width`: line width for the border of nodes
         - `nodes_show`: how to display nodes, specified as:
             * `"all"`: display all nodes
             * `mode, values`: where `values` is a list of Boolean (one for each
                node) or a column in `nodes` table, and `mode` is one of:
                  - `"dim"` dim nodes for which `values` is `True`
                  - `"hide"` hide nodes for which `values` is `True`
                  - `"drop"` drop nodes for which `values` is `True`
             * `mode, values, invert`: as above with `invert` being a `bool` to
                 specify whether `values` should be negated or not
             * `"dim", values, invert, dim` as above with `0.0 <= dim <= 1.0` to set
                 dim level

        A hidden node is still considered to compute layouts while a dropped
        node is (mostly) ignored. GraphViz or PCA layouts make no difference
        between both modes.

        ## Node labels options

         - `nodes_label`: text drawn onto nodes
         - `nodes_label_wrap`: how labels text is wrapped,
           as listed in `TableWrapDesc.values`
         - `nodes_label_size`: size of labels
         - `nodes_label_halign`: horizontal placement of labels wrt nodes,
           as listed in `TableHalignDesc.values`
         - `nodes_label_valign`: vertical placement of labels wrt nodes,
           as listed in `TableValignDesc.values`
         - `node_label_align`: both horizontal and vertical placement of labels
           wrt nodes, given as a one- ot two-chars `str` combining:
           `"n"`orth, `"s"`outh, `"e"`ast, `"w"`est, `"c"`enter, or `"m"`iddle
           (the last two are synonymous). If only one align is given,
           the other default to `"m"`, for instance, `"n"` is equivalent to
           `"nm"` (or `"nc"`). If the first letter is `"c"` or `"m"`, it will
           be used for horizontal alignment, so `"cn"` will fail because it
           specifies twice horizontal alignement. To avoid this, start
           with a character in `"nsew"` that is tight to one direction.
         - `nodes_label_angle`: the rotation of labels, in degrees
         - `nodes_label_outline_width`: width of the outline around node labels
         - `nodes_label_outline_color`: color of the outline around node labels
         - `nodes_label_outline_palette`: palette for `nodes_label_outline_color`
         - `nodes_label_outline_opacity`: opacity of label outline

        ## Edges options

         - `edges_curve`:
            - when `"bezier"`, two opposite edges will be bent automatically to
              avoid their overlapping, which is desired when they have labels
            - when `straight`, two opposite edges will overlap, which may be
              desired when they have no labels
         - `edges_draw_color`, `edges_draw_palette`, `edges_draw_style`,
           and `edges_draw_width`: like `nodes_draw_...` but for edges (which means
           that information is taken from table `edges` if column names are used)
         - `edges_label`, `edges_label_angle`, `edges_label_outline_width`,
           `edges_label_outline_color`, `edges_label_outline_palette`,
           `edges_label_outline_opacity`: like `nodes_label...` but for edges.
           Note that `edges_label_angle="auto"` allows to automatically rotate the
           label following the slope of each edge.

        ## Edges tips

         - `edges_tip_scale`: scaling factor that applies to all edge tips, this is
           a single value (default `0.6`) and cannot be a column/array of values
         - `edges_target_tip`: shape of edge tip at target node, as listed in
           `TableTipDesc.values`. Values may be prefixed with `"filled-"` or
           `"hollow-"` to fill or not the tip drawing. `TableTipDesc.alias`
           provides short names for some edge tips.
         - `edges_target_color` and `edges_target_palette`: color and palette of
           edge tips at target node
         - `edges_source_tip`, `edges_source_color`, `edges_source_palette`: like
           `edges_source_...` but for the tips as edges source

        ## Coumpound options

        Additional properties gather other properties, so assigning them will
        actually assign several other properties, and reading them will return a
        tuple of other properties:

         - `nodes_color`: `nodes_draw_color` and `nodes_fill_color`
         - `nodes_palette`: `nodes_draw_palette` and `nodes_fill_palette`
         - `edges_tip_color`: `edges_target_color` and `edges_source_color`
         - `edges_tip_palette`: `edges_target_palette` and `edges_source_palette`
         - `edges_color`: `edges_draw_color` and `edges_tip_color`
         - `edges_palette`: `edges_draw_palette` and `edges_tip_palette`

        ## User interface

        User interface (UI) option `ui` may be given as a `list` of UI elements
        organised as nested lists and dicts:

         - top-level `list` is a vbox, a vbox arranges its content into
           a `ipywidget.VBox`
         - a `list` in a vbox arranges its content into an `ipywidget.HBox`
         - a `dict` in a vbox arranges its content as an `ipywidget.Accordion`
           whose sections are the keys of the dict, each containing a vbox.
         - no other nesting is allowed
         - atomic elements are:
            - the name of an option listed above (except the coumpound ones)
              to build a widget allowing to tune this option
            - `"graph"`: the interactive graph itself
            - `"reset_view"`: a button to recompute the layout and reset the graph
              display, which is useful when one has zoomed too much and the graph
              is not visible anymore
            - `"select_all"`, `"select_none"`, and `"invert_selection"`: buttons to
              (un)select nodes
            - `"inspect_nodes"` and `"inspect_edges"`: view of `nodes` and `edges`
              tables limited to the selected nodes and the edges connected to them.
              Options `inspect_nodes` and `inspect_edges` may be provided
              (as lists of columns) to restrict the columns displayed here

        All the options presented above, except those related to the building of
        the UI, are also attributes of a `Graph` instance. Which means they can be
        read or assigned to change a graph appearance. In this case, it is
        mandatory to call method `update` to actually update the display after
        changing an option. `Graph` instance also supports item assignement
        to update one option for just one node or edge, see `__setitem__`
        documentation.
        """
        self.nodes = nodes.reset_index()
        self.nodes["node"] = self.nodes["node"].map(str)
        self.nodes.index = self.nodes.pop("node")
        self.edges = edges.reset_index()
        self.edges["src"] = self.edges["src"].map(str)
        self.edges["dst"] = self.edges["dst"].map(str)
        self.edges.index = [self.edges.pop("src"), self.edges.pop("dst")]
        self._ns = pd.DataFrame(index=self.nodes.index)
        self._es = pd.DataFrame(index=self.edges.index)
        self._inspect_nodes_cols = args.pop("inspect_nodes", self.nodes.columns)
        self._inspect_edges_cols = args.pop("inspect_edges", self.edges.columns)
        self.style = [
            {
                "selector": "node.selected",
                "style": {
                    "underlay-color": "black",
                    "underlay-padding": 4,
                    "underlay-opacity": 0.15,
                },
            },
            {
                "selector": "node.hierarchical",
                "style": {
                    "label": "",
                    "border-width": 0.0,
                    "background-color": "#F8F8F2",
                },
            },
        ]
        self._legend = {}
        self.selection = set()
        self.xy = {}
        self.groups = {}
        hierarchy = args.pop("hierarchy", None)
        if hierarchy:
            self.nodes.sort_values(by=hierarchy, inplace=True, na_position="first")
        n = [{"data": {"id": idx}} for idx in self.nodes.index]
        if hierarchy:
            hcol = nodes[hierarchy]
            for node in reversed(n):
                data = node["data"]
                idx = data["id"]
                if hcol[idx]:
                    p = str(hcol[idx])
                    data["parent"] = p
                    self.groups.setdefault(p, set())
                    self.groups[p].add(idx)
                if idx in self.groups:
                    node["classes"] = " ".join(
                        ["hierarchical"] + node.get("classes", "").split()
                    )
        e = [
            {"data": {"id": f"{src}-{dst}", "source": f"{src}", "target": f"{dst}"}}
            for src, dst in self.edges.index
        ]
        self._cy = cy.CytoscapeWidget()
        self._cy.graph.add_graph_from_json({"nodes": n, "edges": e}, directed=True)
        for opt, desc in self._opts.items():
            if isinstance(desc, TableStrDesc) and f"{opt}_str" in args:
                desc.convert = args.pop(f"{opt}_str")
            if opt in args:
                setattr(self, opt, args.pop(opt))
            else:
                getattr(self, opt)
        for opt in list(args):
            if hasattr(self.__class__, opt):
                setattr(self, opt, args.pop(opt))
        self._inspect_nodes = self._inspect_edges = None
        self._widgets = {}
        self._ui = self._make_ui_vbox(self.ui)
        self._cy.on("node", "click", self._on_node_click)
        if args:
            raise TypeError(f"unexpected arguments: {', '.join(args)}")

    def _ipython_display_(self):
        self.update()
        display(self._ui)

    def update(self, layout=False):
        """Update the display after some options have been changed

        Arguments:
         - `layout` (default: `False`): whether update the graph layout, which
           is only needed when option `layout` has been assigned (or when one
           wants to redraw the graph)
        """
        self._cy.set_style(
            [
                {"selector": f"node[id='{n}']", "style": r.to_dict()}
                for n, r in self._ns.iterrows()
            ]
            + [
                {"selector": f"edge[id='{s}-{t}']", "style": r.to_dict()}
                for (s, t), r in self._es.iterrows()
            ]
            + self.style
        )
        if layout:
            self._cy.relayout()
            self.xy = {}
            self._update_xy()

    def __setitem__(self, key, val):
        """Update an option for one specific node or edge

        For instance, to change nodes, we use `"option":node` indexing, and
        for edges, we use `"option":source:target` indexing:

            :::python
            g = Graph(nodes, edges, nodes_fill_color="red", edges_color="red")
            g["color":1] = "#C0FFEE"    # node 1 is turned blue
            g["color":1:2] = "#DEFEC7"  # edge from 1 to 2 is turned  green
            g.update()                  # update display

        `"options"` above is not prefixed by either `"nodes_"` or `"edges_"`
        as this is deduced from the indexing used. So in the second line,
        `"color"` is expanded to `nodes_color` option, and in the third line,
        `"color"` is expanded to `edges_color` option. (Note that both are
        compound options that gather several other ones.)
        """
        if key.step is None:
            prefix = "nodes"
            colnames = self._ncol
            idx = str(key.stop)
            table = self._ns
        else:
            prefix = "edges"
            colnames = self._ecol
            idx = (str(key.stop), str(key.step))
            table = self._es
        todo = [colnames[f"{prefix}_{key.start}"]]
        while todo:
            c = todo.pop(0)
            if isinstance(c, (list, tuple)):
                todo.extend(colnames[x] for x in c)
            else:
                table.loc[idx, c] = val

    #
    # legend
    #
    def legend(self, option="nodes_fill", values=None, vertical=True, span=1, pos=None):
        """Build a `Graph` that shows the legend for a given choice of colors

        Optional arguments:
         - `option` (default: `"nodes_fill"`): the color/palette option whose
           legend must be constructed
         - `values`: the values corresponding to the colors, if the colors
           have been computed from a column, this parameter can be left to
           `None` and the values will be retrieved automatically, otherwise,
           they have to be provided explicitly
         - `vertical` (default: `True`): shall the legend be drawn vertically
           or horizontally
         - `span` (default: `1`): number of columns (or lines if
           `vertical=False`) the legend will be spanned on
         - `pos` (defaut: auto from `vertical`): where to place the labels,
           specified as one direction char from `"nsewcm"`

        Returns: an instance of `Graph` arranged as a grid, that shows
        each value next to its corresponding color.
        """
        color = f"{option}_color"
        table = getattr(self, option.split("_")[0])
        if values is None:
            col = self._legend.get(color, None)
            if col is None:
                raise ValueError("unknown color source," " use parameter 'values'")
            data = table[col]
        elif isinstance(values, str) and values in table.columns:
            data = table[values]
        elif isinstance(values, (list, tuple, pd.Series, np.ndarray)) and len(
            values
        ) == len(table):
            data = pd.DataFrame(list(values), index=table.index)
        else:
            raise ValueError("cannot interpret {values!r} as values")
        data = list(sorted(set(zip(data, getattr(self, color)))))
        nodes = pd.DataFrame.from_records(
            [{"node": n, "label": v, "color": c} for n, (v, c) in enumerate(data)],
            index=["node"],
        )
        edges = pd.DataFrame.from_records(
            [{"src": n, "dst": n + 1} for n in range(len(data) - 1)],
            index=["src", "dst"],
        )
        gridopt = {"fit": True, "condense": True, "avoidOverlapPadding": 5}
        if vertical:
            gridopt["cols"] = span
        else:
            gridopt["rows"] = span
        if pos is None:
            if vertical:
                pos, dx, dy = "e", 5.0, 0.0
            else:
                pos, dx, dy = "s", 0.0, 5.0
        elif pos in ("m", "c"):
            dx = dy = 0.0
        elif pos == "s":
            dx, dy = 0.0, 5.0
        elif pos == "e":
            dx, dy = 5.0, 0.0
        elif pos == "n":
            dx, dy = 0.0, -5.0
        elif pos == "w":
            dx, dy = -5.0, 0.0
        else:
            raise ValueError(f"unexpected value for pos: {pos!r}")
        g = Graph(
            nodes,
            edges,
            nodes_label="label",
            nodes_label_align=pos,
            nodes_fill_color="color",
            nodes_shape="rectangle",
            nodes_size=20.0,
            edges_label="",
            edges_target_tip="-",
            edges_source_tip="-",
            edges_draw_width=0,
            layout=("grid", gridopt),
            ui=["graph"],
        )
        g._ns["text-margin-x"] = dx
        g._ns["text-margin-y"] = dy
        g._cy.graph.nodes.sort(key=lambda n: int(n.data["id"]))
        g._cy.relayout()
        return g

    #
    # ui
    #
    def _select_node(self, node):
        node.classes = " ".join(node.classes.split() + ["selected"])
        self.selection.add(node.data["id"])

    def _unselect_node(self, node):
        node.classes = " ".join(c for c in node.classes.split() if c != "selected")
        self.selection.discard(node.data["id"])

    def _toggle_select_node(self, node):
        classes = node.classes.split()
        if "selected" in classes:
            node.classes = " ".join(c for c in classes if c != "selected")
            self.selection.discard(node.data["id"])
        else:
            node.classes = " ".join(classes + ["selected"])
            self.selection.add(node.data["id"])

    def _update_inspector(self):
        if self._inspect_nodes is not None:
            self._inspect_nodes.clear_output()
            if self.selection:
                with self._inspect_nodes:
                    sub = self.nodes[self.nodes.index.isin(self.selection)]
                    display(sub[self._inspect_nodes_cols])
        if self._inspect_edges is not None:
            self._inspect_edges.clear_output()
            if self.selection:
                with self._inspect_edges:

                    def insel(idx):
                        return idx[0] in self.selection or idx[1] in self.selection

                    sub = self.edges[self.edges.index.map(insel)]
                    display(sub[self._inspect_edges_cols])

    def _on_node_click(self, event):
        nid = event["data"]["id"]
        found = [node for node in self._cy.graph.nodes if nid == node.data["id"]]
        if not found:
            return
        self._toggle_select_node(found[0])
        self._update_inspector()

    def _make_ui_vbox(self, items):
        children = []
        for item in items:
            if isinstance(item, str):
                children.append(self._make_ui_widget(item))
            elif isinstance(item, list):
                children.append(self._make_ui_hbox(item))
            elif isinstance(item, dict):
                children.append(self._make_ui_accordion(item))
            else:
                raise ValueError(f"unexpected UI spec: {item!r}")
        return ipw.VBox(children)

    def _make_ui_hbox(self, items):
        children = []
        for item in items:
            if isinstance(item, str):
                children.append(self._make_ui_widget(item))
            else:
                raise ValueError(f"unexpected UI spec: {item!r}")
        return ipw.HBox(children)

    def _make_ui_accordion(self, items):
        children = []
        titles = []
        for k, v in items.items():
            titles.append(k)
            if isinstance(v, str):
                children.append(self._make_ui_widget(v))
            elif isinstance(v, list):
                children.append(self._make_ui_vbox(v))
            elif isinstance(v, dict):
                children.append(self._make_ui_accordion(v))
            else:
                raise ValueError(f"unexpected UI spec: {v!r}")
        acc = ipw.Accordion(children, selected_index=None)
        for i, t in enumerate(titles):
            acc.set_title(i, t)
        return acc

    def _make_ui_widget(self, name):
        opt = self._opts.get(name, None)
        if isinstance(name, widget_spec):
            cfg = name.config
        else:
            cfg = {}
        make = getattr(self, f"_make_ui_widget_{name}", None)
        if make is not None:
            self._widgets[name] = w = make(name, opt, **cfg)
            return w
        for cls in inspect.getmro(opt.__class__):
            make = getattr(self, f"_make_ui_widget_{cls.__name__}", None)
            if make is not None:
                self._widgets[name] = w = make(name, opt, **cfg)
                return w
        return ipw.Label(f"[{name}]")

    def _make_ui_widget_graph(self, name, opt):
        return self._cy

    def _make_ui_widget_inspect_nodes(self, name, opt):
        self._inspect_nodes = ipw.Output()
        return self._inspect_nodes

    def _make_ui_widget_inspect_edges(self, name, opt):
        self._inspect_edges = ipw.Output()
        return self._inspect_edges

    def _make_ui_widget_reset_view(self, name, opt):
        button = ipw.Button(description="reset layout/view")

        def on_click(event):
            self.update(True)

        button.on_click(on_click)

        return button

    def _make_ui_widget_select_all(self, name, opt):
        button = ipw.Button(description="select all nodes")

        def on_click(event):
            for node in self._cy.graph.nodes:
                self._select_node(node)
            self._update_inspector()

        button.on_click(on_click)
        return button

    def _make_ui_widget_select_none(self, name, opt):
        button = ipw.Button(description="unselect all nodes")

        def on_click(event):
            for node in self._cy.graph.nodes:
                self._unselect_node(node)
            self._update_inspector()

        button.on_click(on_click)
        return button

    def _make_ui_widget_invert_selection(self, name, opt):
        button = ipw.Button(description="invert nodes selection")

        def on_click(event):
            for node in self._cy.graph.nodes:
                self._toggle_select_node(node)
            self._update_inspector()

        button.on_click(on_click)
        return button

    def _make_ui_widget_nodes_show(self, name, opt, columns=[]):
        drop_mode = ipw.Dropdown(
            options=[(str(m), m) for m in NodeShowDesc.modes],
            value="all",
            description="show",
        )
        columns = columns or list(self.nodes.columns)
        drop_col = ipw.Dropdown(
            options=[(str(c), c) for c in columns],
            value=columns[0],
            description="column",
            disabled=True,
        )
        invert_check = ipw.Checkbox(value=False, description="invert", disabled=True)
        dim_entry = ipw.BoundedFloatText(
            value=0.2,
            min=0.0,
            max=1.0,
            step=0.05,
            description="dim level",
            disabled=True,
        )

        def on_mode(event):
            choice = event["new"]
            drop_col.disabled = invert_check.disabled = choice == "all"
            dim_entry.disabled = choice != "dim"
            if choice == "all":
                self.nodes_show = "all"
            else:
                self.nodes_show = (
                    choice,
                    drop_col.value,
                    invert_check.value,
                    dim_entry.value,
                )
            self.update()

        drop_mode.observe(on_mode, names="value")

        def on_col(event):
            self.nodes_show = (
                drop_mode.value,
                event["new"],
                invert_check.value,
                dim_entry.value,
            )
            self.update()

        drop_col.observe(on_col, names="value")

        def on_invert(event):
            self.nodes_show = (
                drop_mode.value,
                drop_col.value,
                event["new"],
                dim_entry.value,
            )
            self.update()

        invert_check.observe(on_invert, names="value")

        def on_dim(event):
            self.nodes_show = (
                drop_mode.value,
                drop_col.value,
                invert_check.value,
                event["new"],
            )
            self.update()

        dim_entry.observe(on_dim, names="value")
        return ipw.HBox([drop_mode, drop_col, invert_check, dim_entry])

    def _make_ui_widget_LayoutDesc(self, name, opt):
        default, _ = getattr(self, name)
        options = list(
            sorted(
                (opt.values | set(f"xx-{x}" for x in self.layout_extra)) - {"preset"}
            )
        )
        libs = {"gv": "GraphViz", "nx": "NetworkX", "ig": "igraph", "xx": "extra"}
        for i, o in enumerate(options):
            if o.startswith(("nx-", "gv-", "ig-", "xx-")):
                code, algo = o.split("-", 1)
                options[i] = (f"{algo} ({libs[code]})", o)
            else:
                options[i] = (o, o)
        drop = ipw.Dropdown(
            options=options, value=default, description=name.split("_")[-1]
        )

        def on_change(event):
            value, args = opt.get_layout(event["new"], self)
            if args is None:
                setattr(self, name, value)
            else:
                setattr(self, name, (value, args))
            self.update(True)

        drop.observe(on_change, names="value")
        return drop

    def _make_ui_widget__TableEnumDesc(self, name, opt):
        default = getattr(self, name)
        unique = default.unique()
        if len(unique) == 1:
            default = unique[0]
        table = getattr(self, opt.table)
        allowed = set(opt.values) | set(opt.alias)
        columns = [col for col in table.columns if table[col].isin(allowed).all()]
        drop = ipw.Dropdown(
            options=["(default)"] + columns,
            value="(default)",
            description=name.split("_")[-1],
        )

        def on_change(event):
            choice = event["new"]
            if choice == "(default)":
                setattr(self, name, default)
            else:
                setattr(self, name, choice)
            self.update()

        drop.observe(on_change, names="value")
        return drop

    def _make_ui_widget_PaletteDesc(self, name, opt):
        palette = getattr(self, name)
        drop_name = ipw.Dropdown(
            options=list(sorted(Palette.palettes, key=str.lower)),
            value=palette.name,
            description="palette",
        )
        drop_mode = ipw.Dropdown(
            options=[("absolute", "abs"), ("linear", "lin"), ("logarithmic", "log")],
            value=palette.mode,
        )
        drop_sort = ipw.Checkbox(
            value=palette.sort, description="sort values", indent=False
        )

        def on_change(event):
            setattr(self, name, (drop_name.value, drop_mode.value, drop_sort.value))
            self.update()

        drop_name.observe(on_change, names="value")
        drop_mode.observe(on_change, names="value")
        drop_sort.observe(on_change, names="value")
        return ipw.HBox([drop_name, drop_mode, drop_sort])

    def _make_ui_widget_TableNumberDesc(self, name, opt):
        table = getattr(self, opt.table)
        default = opt.default
        columns = [col for col in table.columns if is_numeric_dtype(table[col])]
        drop = ipw.Dropdown(
            options=["(default)", "(value)"] + columns,
            value="(default)",
            description=name.split("_")[-1],
        )
        if opt.cls is int:
            entry = ipw.BoundedIntText(
                value=opt.default,
                min=opt.mini,
                max=opt.maxi,
                step=max(1, (opt.maxi - opt.mini) / 100),
                disabled=True,
            )
        else:
            entry = ipw.BoundedFloatText(
                value=opt.default,
                min=opt.mini,
                max=opt.maxi,
                step=(opt.maxi - opt.mini) / 100.0,
                disabled=True,
            )

        def drop_change(event):
            choice = event["new"]
            if choice == "(default)":
                entry.disabled = True
                setattr(self, name, default)
            elif choice == "(value)":
                entry.disabled = False
                setattr(self, name, entry.value)
            else:
                entry.disabled = True
                setattr(self, name, choice)
            self.update()

        drop.observe(drop_change, names="value")

        def entry_change(event):
            setattr(self, name, event["new"])
            self.update()

        entry.observe(entry_change, names="value")
        return ipw.HBox([drop, entry])

    def _make_ui_widget_nodes_size(self, name, opt):
        return self._make_ui_widget_TableNumberDesc(name, self._opts[opt.members[0]])

    def _make_ui_widget_TableColorDesc(self, name, opt):
        table = getattr(self, opt.table)
        default = opt.default
        palette = name.rsplit("_", 1)[0] + "_palette"
        drop = ipw.Dropdown(
            options=["(default)"] + list(sorted(table.columns, key=str.lower)),
            value="(default)",
            description="color",
        )

        def on_change(event):
            choice = event["new"]
            if choice == "(default)":
                setattr(self, name, default)
            else:
                setattr(self, name, choice)
            pal = self._widgets.get(palette, None)
            if pal is not None:
                pal.children[1].value = getattr(self, palette).mode
            self.update()

        drop.observe(on_change, names="value")
        return drop

    def _make_ui_widget_TableStrDesc(self, name, opt):
        table = getattr(self, opt.table)
        default = opt.default
        drop = ipw.Dropdown(
            options=["(default)"] + list(sorted(table.columns, key=str.lower)),
            value="(default)",
            description=name.split("_")[-1],
        )

        def on_change(event):
            choice = event["new"]
            if choice == "(default)":
                setattr(self, name, default)
            else:
                setattr(self, name, choice)
            self.update()

        drop.observe(on_change, names="value")
        return drop

    def _make_ui_widget_nodes_label_align(self, name, opt):
        table = getattr(self, self._opts[opt.hopt].table)
        h, v = getattr(self, name)
        default = h.map(opt.a2g.get) + v.map(opt.a2g.get)
        options = ["(default)"]
        for col in sorted(table.columns, key=str.lower):
            if table[col].apply(opt.valid).all():
                options.append(col)
        drop = ipw.Dropdown(
            options=options, value="(default)", description=name.split("_")[-1]
        )

        def on_change(event):
            choice = event["new"]
            if choice == "(default)":
                setattr(self, name, default)
            else:
                setattr(self, name, choice)
            self.update()

        drop.observe(on_change, names="value")
        return drop

    def _make_ui_widget_TableAngleDesc(self, name, opt):
        table = getattr(self, opt.table)
        default = opt.default
        options = ["(default)", "(auto)", "(value)"]
        options.extend(col for col in table.columns if is_numeric_dtype(table[col]))
        drop = ipw.Dropdown(
            options=options, value="(default)", description=name.split("_")[-1]
        )
        entry = ipw.IntText(value=0, step=1, disabled=True)

        def drop_change(event):
            choice = event["new"]
            if choice == "(value)":
                entry.disabled = False
                setattr(self, name, (float(entry.value) % 360) * pi / 180)
            elif choice == "(default)":
                setattr(self, name, default)
                entry.disabled = True
            elif choice == "(auto)":
                setattr(self, name, "autorotate")
                entry.disabled = True
            else:
                setattr(self, name, (table[choice].astype(float) % 360) * pi / 180)
            self.update()

        drop.observe(drop_change, names="value")

        def entry_change(event):
            setattr(self, name, (float(event["new"]) % 360) * pi / 180)
            self.update()

        entry.observe(entry_change, names="value")
        return ipw.HBox([drop, entry])

    def _make_ui_widget_TableTipDesc(self, name, opt):
        table = getattr(self, opt.table)
        default = opt.default
        allowed = (
            set(opt.tips)
            | {f"hollow-{t}" for t in opt.tips}
            | {f"filled-{t}" for t in opt.tips}
            | set(opt.alias)
        )
        options = ["(default)", "(value)"]
        for col in sorted(table.columns, key=str.lower):
            if table[col].isin(allowed).all():
                options.append(col)
        drop = ipw.Dropdown(
            options=options, value="(default)", description=name.split("_")[-1]
        )
        tips = (
            [a for a in opt.alias if a != "<"]
            + list(sorted(opt.tips))
            + [f"hollow-{t}" for t in sorted(opt.tips) if t != "none"]
        )
        entry = ipw.Dropdown(options=tips, value=default, disabled=True)

        def drop_change(event):
            choice = event["new"]
            if choice == "(default)":
                entry.disabled = True
                setattr(self, name, default)
            elif choice == "(value)":
                entry.disabled = False
                setattr(self, name, entry.value)
            else:
                entry.disabled = True
                setattr(self, name, table[choice])
            self.update()

        drop.observe(drop_change, names="value")

        def entry_change(event):
            setattr(self, name, event["new"])
            self.update()

        entry.observe(entry_change, names="value")
        return ipw.HBox([drop, entry])

    def _make_ui_widget_export(self, name, opt):
        progress = ipw.IntProgress(
            value=0,
            min=0,
            max=len(self.nodes),
            step=1,
            description="captured:",
            orientation="horizontal",
        )
        choice = ipw.Dropdown(
            options=[
                ("PDF", "pdf"),
                ("TikZ", "tikz"),
                ("LaTeX", "tex"),
                ("PNG", "png"),
            ],
            value="pdf",
            description="format",
            disabled=False,
        )
        button = ipw.Button(description="export", disabled=True)
        output = ipw.Output()

        def update():
            progress.value = len(self.xy)
            button.disabled = len(self.xy) != len(self.nodes)

        self._update_xy = update

        def cb(event):
            self.xy[event["data"]["id"]] = (
                event["position"]["x"],
                event["position"]["y"],
            )
            update()

        self._cy.on("node", "mousemove", cb)

        def on_click(event):
            output.clear_output()
            data = quote(self._export(choice.value))
            path = f"graph.{choice.value}"
            mime = mimetypes.guess_type(path, False)[0] or "text/plain"
            html = _export_html.format(filename=path, payload=data, mimetype=mime)
            with output:
                display(ipw.HTML(html))

        button.on_click(on_click)
        return ipw.VBox(
            [
                ipw.Label(
                    "Move mouse over nodes to capture their positions."
                    " When progress bar is full, click 'export'."
                ),
                ipw.HBox([progress, choice, button]),
                output,
            ]
        )

    def _update_xy(self):
        pass

    #
    # export
    #
    def _export(self, fmt):
        handler = getattr(self, f"_export_{fmt}", None)
        if len(self.xy) != len(self.nodes):
            raise ValueError("missing node positions")
        elif handler is None:
            raise ValueError(f"unsupported export format {fmt!r}")
        nodes = self._ns.copy()
        nodes["x"] = nodes.index.map(lambda n: self.xy[n][0])
        nodes["y"] = nodes.index.map(lambda n: self.xy[n][1])
        edges = self._es.copy()
        data = handler(nodes, edges)
        if isinstance(data, str):
            return data.encode("utf-8")
        else:
            return data

    def _export_tikz(self, nodes, edges):
        return tikz.tikz(nodes, edges)

    def _export_tex(self, nodes, edges):
        return tikz.latex(nodes, edges)

    def _export_pdf(self, nodes, edges):
        return self._export_image(nodes, edges, "pdf")

    def _export_png(self, nodes, edges):
        return self._export_image(nodes, edges, "png")

    def _export_image(self, nodes, edges, fmt):
        with tempfile.TemporaryDirectory() as tmp:
            with (pathlib.Path(tmp) / "graph.tex").open("w") as tex:
                tex.write(self._export_tex(nodes, edges))
            try:
                subprocess.check_output(
                    ["lualatex", "-interaction=nonstopmode", "graph.tex"],
                    stdin=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                    encoding="utf-8",
                    errors="replace",
                    timeout=300,
                    cwd=tmp,
                )
            except subprocess.CalledProcessError as err:
                err.tex = (pathlib.Path(tmp) / "graph.tex").read_text()
                err.log = (pathlib.Path(tmp) / "graph.log").read_text()
                raise err
            output = f"graph.{fmt}"
            if fmt != "pdf":
                subprocess.check_output(
                    [
                        "pdftoppm",
                        "-singlefile",
                        f"-{fmt}",
                        "-r",
                        "300",
                        "graph.pdf",
                        "graph",
                    ],
                    stdin=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                    encoding="utf-8",
                    errors="replace",
                    timeout=300,
                    cwd=tmp,
                )
            with (pathlib.Path(tmp) / output).open("rb") as img:
                return img.read()
