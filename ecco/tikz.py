import io, math

from colour import Color

_latex_esc = {
    "\n" : r"\\[-3pt]",
    "\r" : "",
    "\\" : r"{\ifmmode\backslash\else\textbackslash\fi}",
    "$" : r"\$",
    "~" : r"{\ifmmode\sim\else\textasciitilde\fi}",
    "%" : r"\%",
    "#" : r"\#",
    "{" : r"\{",
    "}" : r"\}",
    "&" : r"\&",
    "_" : r"\_",
    "^" : r"{\ifmmode\hat{}\else\textasciicircum\fi}",
}

def esc (txt) :
    chars = []
    for c in txt :
        if c in _latex_esc :
            chars.append(_latex_esc[c])
        elif ord(c) > 256 :
            chars.append(f'{{\\fontspec{{Symbola}}\\symbol{{"{ord(c):X}}}}}')
        else :
            chars.append(c)
    return "".join(chars)

_shape = {
    "ellipse" : "ellipse",
    "triangle" : "regular polygon,regular polygon sides=3",
    "round-triangle" : "regular polygon,regular polygon sides=3,rounded corners",
    "rectangle" : "rectangle",
    "round-rectangle" : "rectangle,rounded corners",
    "bottom-round-rectangle" : "rounded rectangle,rounded rectangle right arc=none,rounded rectangle arc length=120,rotate=90",
    #FIXME
    "cut-rectangle" : "chamfered rectangle",
    "barrel" : "rounded rectangle,rounded rectangle arc length=50,rotate=90,xscale=.43",
    "rhomboid" : "trapezium,trapezium left angle=120,inner sep=.1pt,yscale=.9,xscale=.4",
    "diamond" : "diamond",
    "round-diamond" : "diamond,rounded corners",
    "pentagon" : "regular polygon,regular polygon sides=5",
    "round-pentagon" : "regular polygon,regular polygon sides=5,rounded corners",
    "hexagon" : "regular polygon,regular polygon sides=6",
    "round-hexagon" : "regular polygon,regular polygon sides=6,rounded corners",
    #FIXME
    "concave-hexagon" : "signal",
    "heptagon" : "regular polygon,regular polygon sides=7",
    "round-heptagon" : "regular polygon,regular polygon sides=6,rounded corners",
    "octagon" : "regular polygon,regular polygon sides=8",
    "round-octagon" : "regular polygon,regular polygon sides=8,rounded corners",
    "star" : "star",
    "tag" : "shape=signal",
    "round-tag" : "signal,rounded corners",
    "vee" : "shape=dart,inner sep=1pt,dart tail angle=110,rotate=-90"
}

_align = {# (halign, valign) -> (pos, anchor)
    ("center", "center") : ("center", "center"),
    ("center", "top") : ("north", "south"),
    ("center", "bottom") : ("south", "north"),
    ("left", "center") : ("west", "east"),
    ("left", "top") : ("north west", "south east"),
    ("left", "bottom") : ("south west", "north east"),
    ("right", "center") : ("east", "west"),
    ("right", "top") : ("north east", "south west"),
    ("right", "bottom") : ("south east", "north west")
}

_src_arrow = {
    ("filled", "triangle") : "Stealth[inset=0pt,{opt}]",
    ("hollow", "triangle") : "Stealth[inset=0pt,open,{opt}]",
    ("filled", "triangle-tee") : "Stealth[inset=0pt,{opt}]Square[sep,length=1.5pt,width=4pt,{opt}]",
    ("hollow", "triangle-tee") : "Stealth[inset=0pt,open,{opt}]Square[sep,length=1.5pt,width=4pt,fill=white,{opt}]",
    ("filled", "circle-triangle") : "Circle[{opt}]Stealth[inset=0pt,{opt}]",
    ("hollow", "circle-triangle") : "Circle[open,{opt}]Stealth[inset=0pt,open,{opt}].",
    ("filled", "triangle-cross") : "Stealth[inset=0pt,{opt}]Bar[sep,{opt}]",
    ("hollow", "triangle-cross") : "Stealth[inset=0pt,open,{opt}]Bar[sep,{opt}]",
    ("filled", "triangle-backcurve") : "Stealth[inset=.75pt,{opt}]",
    ("hollow", "triangle-backcurve") : "Stealth[inset=.75pt,open,{opt}]",
    ("filled", "vee") : "Stealth[inset=2pt,{opt}]",
    ("hollow", "vee") : "Stealth[inset=2pt,open,{opt}]",
    ("filled", "tee") : "Square[length=1.5pt,width=4pt,{opt}]",
    ("hollow", "tee") : "Square[length=1.5pt,width=4pt,open,{opt}]",
    ("filled", "square") : "Square[{opt}]",
    ("hollow", "square") : "Square[open,{opt}]",
    ("filled", "circle") : "Circle[{opt}]",
    ("hollow", "circle") : "Circle[open,{opt}]",
    ("filled", "diamond") : "Turned Square[{opt}]",
    ("hollow", "diamond") : "Turned Square[open,{opt}]",
    ("filled", "chevron") : "Straight Barb[{opt}]",
    ("hollow", "chevron") : "Straight Barb[{opt}].Straight Barb[sep=-1pt,{opt}]",
    ("filled", "none") : "",
    ("hollow", "none") : "",
}

_tgt_arrow = {
    ("filled", "triangle") : "Stealth[inset=0pt,{opt}]",
    ("hollow", "triangle") : "Stealth[inset=0pt,open,{opt}]",
    ("filled", "triangle-tee") : "Square[sep,length=1.5pt,width=4pt,{opt}]Stealth[inset=0pt,{opt}]",
    ("hollow", "triangle-tee") : "Square[sep,length=1.5pt,width=4pt,fill=white,{opt}]Stealth[inset=0pt,open,{opt}]",
    ("filled", "circle-triangle") : "Stealth[inset=0pt,{opt}]Circle[{opt}]",
    ("hollow", "circle-triangle") : ".Stealth[inset=0pt,open,{opt}]Circle[open,{opt}]",
    ("filled", "triangle-cross") : "Bar[sep,{opt}]Stealth[inset=0pt,{opt}]",
    ("hollow", "triangle-cross") : "Bar[sep,{opt}]Stealth[inset=0pt,open,{opt}]",
    ("filled", "triangle-backcurve") : "Stealth[inset=.75pt,{opt}]",
    ("hollow", "triangle-backcurve") : "Stealth[inset=.75pt,open,{opt}]",
    ("filled", "vee") : "Stealth[inset=2pt,{opt}]",
    ("hollow", "vee") : "Stealth[inset=2pt,open,{opt}]",
    ("filled", "tee") : "Square[length=1.5pt,width=4pt,{opt}]",
    ("hollow", "tee") : "Square[length=1.5pt,width=4pt,open,{opt}]",
    ("filled", "square") : "Square[{opt}]",
    ("hollow", "square") : "Square[open,{opt}]",
    ("filled", "circle") : "Circle[{opt}]",
    ("hollow", "circle") : "Circle[open,{opt}]",
    ("filled", "diamond") : "Turned Square[{opt}]",
    ("hollow", "diamond") : "Turned Square[open,{opt}]",
    ("filled", "chevron") : "Straight Barb[{opt}]",
    ("hollow", "chevron") : "Straight Barb[sep=-1pt,{opt}].Straight Barb[{opt}]",
    ("filled", "none") : "",
    ("hollow", "none") : "",
}

colors = {}

def newcolor (code, out) :
    global colors
    try :
        rgb = Color(code).hex_l[1:].upper()
    except :
        return
    if rgb not in colors :
        colors[rgb] = cname = f"COL{len(colors)}"
        out.write(f"\\definecolor{{{cname}}}{{HTML}}{{{rgb}}}%\n")

def getcolor (code) :
    global colors
    try :
        rgb = Color(code).hex_l[1:].upper()
    except :
        rgb = None
    return colors.get(rgb, "white")

def tikz (nodes, edges) :
    global colors
    colors = {}
    for coord in "xy" :
        nodes[coord] -= nodes[coord].min()
        nodes[coord] /= 30.0
    nodes["y"] = nodes["y"].max() - nodes["y"]
    out = io.StringIO()
    for table in (nodes, edges) :
        colcol = [col for col in table.columns if col.endswith("-color")]
        for col in table[colcol] :
            for val in table[col].unique() :
                newcolor(val, out)
    out.write("\\sffamily\n"
              "\\begin{tikzpicture}\n")
    for node, row in nodes.iterrows() :
        # node
        out.write(f"\\node["
                  f"inner sep=0pt,"
                  f"outer sep=0pt,"
                  f"{_shape[row['shape']]},"
                  f"minimum width={row['width']}pt,"
                  f"minimum height={row['height']}pt,"
                  f"fill={getcolor(row['background-color'])},"
                  f"fill opacity={row['background-opacity']},")
        if row['border-width'] :
            out.write(f"draw={getcolor(row['border-color'])},"
                      f"line width={row['border-width']}pt,"
                      f"{row['border-style']}")
        out.write(f"]"
                  f" (n{node}) at ({row['x']},{row['y']}) {{}};\n")
        # label
        if str(row["label"]).strip() :
            pos, anchor = _align[tuple(row[["text-halign", "text-valign"]])]
            out.write(f"\\node["
                      f"inner sep=0pt,"
                      f"outer sep=0pt,"
                      f"anchor={anchor},"
                      f"rotate={math.degrees(-row['text-rotation'])},")
            if row["text-wrap"] == "wrap" :
                out.write(f"text width={row['width']}pt,"
                          f"text centered,")
            out.write(f"] at (n{node}.{pos})"
                      f" {{\\fontsize{{{row['font-size']}pt}}{{{row['font-size']}pt}}"
                      f"\\selectfont {esc(row['label'])}}};\n")
    for (src, tgt), row in edges.iterrows() :
        srcopt = [f"color={getcolor(row['source-arrow-color'])}"]
        tgtopt = [f"color={getcolor(row['target-arrow-color'])}"]
        try :
            srcopt.append(f"scale={float(row['arrow-scale']) / 0.6}")
            tgtopt.append(f"scale={float(row['arrow-scale']) / 0.6}")
        except :
            pass
        srctip = _src_arrow[row["source-arrow-fill"],
                            row["source-arrow-shape"]].format(opt=",".join(srcopt))
        tgttip = _tgt_arrow[row["target-arrow-fill"],
                            row["target-arrow-shape"]].format(opt=",".join(tgtopt))
        out.write(f"\\path["
                  f"arrows={{{{{srctip}}}-{{{tgttip}}}}},"
                  f"{row['line-style']},")
        if row["width"] :
                  out.write(f"draw={getcolor(row['line-color'])},"
                            f"line width={row['width']}pt,"
                            f"{row['line-style']},")
        out.write(f"] (n{src}) to [")
        if row["curve-style"] == "bezier" and (tgt, src) in edges.index :
            out.write("bend right=20,")
        out.write(f"]")
        if str(row["label"]).strip() :
            out.write(f" node["
                      f"fill={getcolor(row['text-outline-color'])},"
                      f"fill opacity={row['text-outline-opacity']},"
                      f"inner sep={row['text-outline-width']}pt,")
            if row["text-rotation"] == "autorotate" :
                out.write("sloped,")
            else :
                try :
                    angle = math.degrees(-float(row['text-rotation']))
                except :
                    angle = 0.0
                out.write(f"rotate={angle},")
                out.write(f"] {{\\fontsize{{{row['font-size']}pt}}"
                          f"{{{row['font-size']}pt}}"
                          f"\\selectfont {esc(row['label'])}}}")
        out.write(f" (n{tgt});\n")
    out.write("\\end{tikzpicture}\n")
    return out.getvalue()

def latex (nodes, edges) :
    head = ("\\documentclass{standalone}\n"
            "\\usepackage{fontspec}\n"
            "\\usepackage[rgb]{xcolor}\n"
            "\\usepackage{tikz}\n"
            "\\usetikzlibrary{shapes.geometric}\n"
            "\\usetikzlibrary{shapes.misc}\n"
            "\\usetikzlibrary{shapes.symbols}\n"
            "\\usetikzlibrary{arrows.meta}\n"
            "\\begin{document}\n")
    tail = ("\\end{document}\n")
    return head + tikz(nodes, edges) + tail

if __name__ == "__main__" :
    import pandas as pd
    nodes = pd.read_csv("nodes.csv")
    nodes.index = nodes.pop("node")
    edges = pd.read_csv("edges.csv")
    edges.index = [edges.pop("src"), edges.pop("dst")]
    with open("graph.tex", "w") as tex :
        tex.write(latex(nodes, edges))
