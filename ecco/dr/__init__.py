from . import drparse, rr
from .. import BaseModel

def parse (path) :
    with open(path) as infile :
        src = drparse.indedent(infile.read())
    return drparse.DeerParser.p_parse(src)

class Model (BaseModel) :
    def rr (self, path=None) :
        """convert model to RR
        Arguments:
         - path: where to save RR (default to the same file as DR with .rr extension)
        """
        if path is None :
            path = self["rr"]
        with open(path, "w") as out :
            return rr.DR2RR.rr(self.spec, out)
    def draw (self) :
        TODO
