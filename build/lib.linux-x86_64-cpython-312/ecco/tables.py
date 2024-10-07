import pathlib, ast, pprint
import pandas as pd

class mkpath (object) :
    def __init__ (self, path) :
        p = pathlib.Path(path)
        while True :
            q = p.with_suffix("")
            if q == p :
                break
            p = q
        self.csv = p.with_suffix(".csv.bz2")
        self.typ = p.with_suffix(".typ")

def read_csv (path, state=None) :
    path = mkpath(path)
    conv = {}
    try :
        dtype = ast.literal_eval(open(path.typ).read().strip())
        for col, typ in list(dtype.items()) :
            if typ == "state" :
                if state is None :
                    dtype[col] = str
                else :
                    del dtype[col]
                    conv[col] = state
    except :
        dtype = None
    return pd.read_csv(path.csv, dtype=dtype, converters=conv, na_filter=False)

def write_csv (df, path, state=None, state_cols=[]) :
    path = mkpath(path)
    if state is None and not state_cols :
        dtypes = {col : typ.name for col, typ in df.dtypes.to_dict().items()}
    else :
        dtypes = {}
        for col, typ in df.dtypes.to_dict().items() :
            if typ.name == "object" and (col in state_cols
                                         or any(isinstance(v, state) for v in df[col])) :
                dtypes[col] = "state"
            else :
                dtypes[col] = typ.name
    with path.typ.open("w") as out :
        pprint.pprint(dtypes, stream=out)
    df.to_csv(path.csv, index=False)
