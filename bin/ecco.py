#!/usr/bin/env python3

import argparse, subprocess, shlex, re, webbrowser, sys, multiprocessing, os, signal
from pathlib import Path

parser = argparse.ArgumentParser(prog="ecco",
                                 description="Start ecco from Docker image.")
parser.add_argument("-t", "--tag", type=str, default="latest",
                    help="run specific version (e.g. '0.4', default: 'latest')")
parser.add_argument("-l", "--local", default="franckpommereau/",
                    action="store_const", const="", dest="repo",
                    help="run from local Docker repository")
parser.add_argument("-p", "--port", default=8000, type=int,
                    help="run jupyter on specific port (default: 8000)")
parser.add_argument("-n", "--no-browse",
                    dest="browse", default=True, action="store_false",
                    help="do not launch web brower")
parser.add_argument("-d", "--debug", default=False, action="store_true",
                    help="print Docker output and debugging messages")
parser.add_argument("-u", "--user", default="ecco", type=str,
                    help="run command as USER (default: 'ecco')")
parser.add_argument("-g", "--gui", default=False, action="store_true",
                    help="start Desktop integration GUI")
parser.add_argument("-c", "--chdir", default="/home/ecco", type=str, metavar="DIR",
                    help="set working directory to DIR (default: '/home/ecco')")
parser.add_argument("-m", "--mount", metavar="DIR", default=[],
                    type=str, action="append",
                    help="mount local directory DIR into the Docker container")
parser.add_argument("cmd", default=["jupyter-notebook", "--no-browser",
                                    "--port=8000", "--ip=0.0.0.0"],
                    type=str, nargs="*",
                    help="start a specific command (default: jupyter-notebook)")

args = parser.parse_args()

class Debug (object) :
    def __init__ (self) :
        self.debug = args.debug
        try :
            import colorama
            self.c = {"docker" : colorama.Fore.LIGHTBLACK_EX,
                      "info" : colorama.Fore.BLUE,
                      "warn" : colorama.Fore.RED,
                      "error" : colorama.Fore.RED + colorama.Style.BRIGHT,
                      "log" : colorama.Style.BRIGHT,
                      None : colorama.Style.RESET_ALL}
        except :
            self.c = {"docker" : "",
                      "info" : "",
                      "warn" : "",
                      "error" : "",
                      "log" : "",
                      None : ""}
    def __getitem__ (self, key) :
        return self.c.get(key, "")
    def __call__ (self, *args, kind="docker") :
        if self.debug :
            print(f"{self[kind]}{' '.join(str(a) for a in args)}{self[None]}")
    def log (self, *args, kind="log") :
        print(f"{self[kind]}{' '.join(str(a) for a in args)}{self[None]}")

debug = Debug()

argv = ["docker", "run",
        "-p", f"{args.port}:8000",
        "-u", args.user,
        "-w", args.chdir]

mount = set()
for mnt in args.mount :
    src = Path(mnt).resolve()
    home = Path("/home/ecco")
    dstparts = []
    for part in reversed(src.parts[1:]) :
        dstparts.insert(0, part)
        dstname = "-".join(dstparts)
        if dstname not in mount :
            dst = home / dstname
            mount.add(dstname)
            if str(src) == mnt :
                debug.log("mount:", src, "=>", dst)
            else :
                debug.log("mount:", mnt, "=>", src, "=>", dst)
            break
    else :
        debug.log(f"could not build a mountpoint for {mnt}", kind="error")
        sys.exit(1)
    argv.append("--mount")
    argv.append(f"type=bind,source={src},destination={dst}")

argv.append(f"{args.repo}ecco:{args.tag}")
argv.extend(args.cmd)

_url = re.compile(f"http://127.0.0.1:8000/\S*")
url = None

def gui (pid, url) :
    from PySide2.QtGui import QIcon, QPixmap
    from PySide2.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction
    xpm = b"""/* XPM */
static char * ecco_xpm[] = {
"64 54 188 2",
"  	c None",
". 	c #49CF23",
"+ 	c #40CE0E",
"@ 	c #44CE1C",
"# 	c #46CF1D",
"$ 	c #50D030",
"% 	c #40CE0D",
"& 	c #44CF1A",
"* 	c #3FCE08",
"= 	c #3FCE0B",
"- 	c #46CF1F",
"; 	c #42CE12",
"> 	c #49CF24",
", 	c #45CF1A",
"' 	c #41CE10",
") 	c #42CE13",
"! 	c #3FCE09",
"~ 	c #40CE0C",
"{ 	c #3FCE0C",
"] 	c #43CF13",
"^ 	c #40CE0F",
"/ 	c #4ED02E",
"( 	c #43CE17",
"_ 	c #4ACF27",
": 	c #61D00E",
"< 	c #C8DC0A",
"[ 	c #DFE108",
"} 	c #E1E108",
"| 	c #D3DE09",
"1 	c #8BD40D",
"2 	c #9BD60C",
"3 	c #A9D80B",
"4 	c #7CD20D",
"5 	c #47CF1D",
"6 	c #63D00D",
"7 	c #E8E208",
"8 	c #F5E507",
"9 	c #E4E208",
"0 	c #4BD028",
"a 	c #C9DD0A",
"b 	c #C4DC0A",
"c 	c #C0DB0A",
"d 	c #41CE0E",
"e 	c #93D50C",
"f 	c #CCDD0A",
"g 	c #F9F0A9",
"h 	c #FDFAE7",
"i 	c #FCF8DD",
"j 	c #F7EA72",
"k 	c #44CE19",
"l 	c #6ED10D",
"m 	c #ECE308",
"n 	c #F9F0AB",
"o 	c #FFFFFF",
"p 	c #FFFEFA",
"q 	c #F7EB85",
"r 	c #B0D90B",
"s 	c #72D10D",
"t 	c #FEFCF1",
"u 	c #F6E741",
"v 	c #EDE308",
"w 	c #5AD00E",
"x 	c #43CE16",
"y 	c #53CF0E",
"z 	c #FBF5CC",
"A 	c #4DD02C",
"B 	c #FEFBEE",
"C 	c #E0E108",
"D 	c #F8EE9A",
"E 	c #FBF6D0",
"F 	c #5DD00E",
"G 	c #42CF12",
"H 	c #ACD80B",
"I 	c #FCF6D2",
"J 	c #FEFEF8",
"K 	c #82D30D",
"L 	c #41CE13",
"M 	c #F5E63A",
"N 	c #FEFDF6",
"O 	c #F7EC86",
"P 	c #44CE18",
"Q 	c #A2D70C",
"R 	c #FAF3BC",
"S 	c #F9F0AE",
"T 	c #77D20D",
"U 	c #DCE009",
"V 	c #F7EB80",
"W 	c #A8D80B",
"X 	c #FAF1B5",
"Y 	c #BCDB0A",
"Z 	c #4BCF27",
"` 	c #F1E407",
" .	c #FBF5CB",
"..	c #F7EB7D",
"+.	c #CDDD0A",
"@.	c #F7EB82",
"#.	c #FCF8DC",
"$.	c #7BD76F",
"%.	c #FCF8DE",
"&.	c #F6E862",
"*.	c #D1EECE",
"=.	c #E6E208",
"-.	c #F5E626",
";.	c #F6E85B",
">.	c #E9E308",
",.	c #C9EBC6",
"'.	c #A5D70C",
").	c #ADD80B",
"!.	c #84D879",
"~.	c #55D03C",
"{.	c #C3E9BF",
"].	c #64D352",
"^.	c #68D456",
"/.	c #D6DF09",
"(.	c #D0DE09",
"_.	c #54D03B",
":.	c #C8EBC5",
"<.	c #AFE3AA",
"[.	c #C9EBC5",
"}.	c #A2E09B",
"|.	c #BAE7B5",
"1.	c #F8FCF7",
"2.	c #B1E4AB",
"3.	c #D9F1D6",
"4.	c #3FCE0A",
"5.	c #BBDA0B",
"6.	c #E7E208",
"7.	c #AFD90B",
"8.	c #44CE1A",
"9.	c #C0E8BB",
"0.	c #90DB87",
"a.	c #C7EBC4",
"b.	c #44CF15",
"c.	c #C4EAC0",
"d.	c #6AD458",
"e.	c #D4EFD2",
"f.	c #40CE0B",
"g.	c #97DD8F",
"h.	c #45CF1B",
"i.	c #A8E1A2",
"j.	c #99DD91",
"k.	c #A0DF99",
"l.	c #C7EBC3",
"m.	c #6FD55F",
"n.	c #7CD771",
"o.	c #C2E9BF",
"p.	c #B6E5B2",
"q.	c #66D355",
"r.	c #71D563",
"s.	c #7FD872",
"t.	c #44CF17",
"u.	c #7CD770",
"v.	c #BCE7B8",
"w.	c #62D24F",
"x.	c #D8F0D6",
"y.	c #8FDB84",
"z.	c #A8E1A3",
"A.	c #D6F0D3",
"B.	c #54D03C",
"C.	c #B6E6B1",
"D.	c #B0E3AB",
"E.	c #BDE7B9",
"F.	c #61D24D",
"G.	c #C8EBC4",
"H.	c #CEEDCC",
"I.	c #D8F0D5",
"J.	c #53D136",
"K.	c #7DD771",
"L.	c #70D561",
"M.	c #DFF3DD",
"N.	c #7CD76F",
"O.	c #54D03A",
"P.	c #4BCF28",
"Q.	c #C3E9C0",
"R.	c #CBECC8",
"S.	c #B9E6B4",
"T.	c #98DD91",
"U.	c #B6E5B1",
"V.	c #DAF1D8",
"W.	c #42CE14",
"X.	c #3DCE02",
"Y.	c #4DD02A",
"Z.	c #48CF22",
"`.	c #43CF14",
" +	c #4AD025",
".+	c #46CF1A",
"++	c #48CF20",
"@+	c #45CF19",
"                                                      . + @                                                                     ",
"                                                  # + + + + +                                                                   ",
"                                                  + + + + + + +                                                                 ",
"                                                $ + + + + + + +                                                                 ",
"                                                  + + + + + + + %                                                               ",
"                                                  + + + + + + + +                                                               ",
"                                                    + + + + + + +                                                               ",
"                                                    & + + + + + + *                                                             ",
"                                                      + + + + + + =                 = - ; + + + > ,                             ",
"                                                      % + + + + + +             ' + + + + + + + + + + +                         ",
"                                                        + + + + + '         ) + + + + + + + + + + + + + +                       ",
"                                                        + + + + + +     ' + + + + + + + + + + + + + + + + +                     ",
"                                                        ! + + + + + ~ * + + + + + + + + + + + + + + + + + + + {                 ",
"                                        ] ^ % + ; + /     + + + + + + + + + + + + + + + + + + + + + + + + + + + (               ",
"                                  ' _ + + + + + + + + %   + + + + + + + + + + : < [ } | 1 + : 2 3 3 4 + + + + + + + 5           ",
"                            + ^ + + + + + + + + + + + + + + + + + + + + + + 6 7 8 8 8 8 8 9 8 8 8 8 8 9 2 + + + + + + '         ",
"                        0 + + + + + + + + + + + + + + + + + + + + + + + + + a 8 8 8 8 8 8 8 8 8 8 8 8 8 8 b + + + + + +         ",
"                    ^ + + + + + + + + + + + + + + + + + + + + + + + + + + + [ 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 c + + + + + %       ",
"                  d + + + + + + + + + + + + + + + + + + + + + + + + + + + + [ 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 e + + + + + +     ",
"                + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + f 8 8 8 8 8 8 8 8 g h h i j 8 8 8 9 + + + + + k     ",
"              + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + l m 8 8 8 8 8 8 n o o o o p q 8 8 8 r + + + + +     ",
"            + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + s } 8 8 8 8 8 t o o o o o t u 8 8 v w + + + + x   ",
"          + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + y 9 8 8 8 8 o o o o o o o z 8 8 8 r + + + + +   ",
"        A + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + e 8 8 8 8 B o o o o o o o j 8 8 C + + + + +   ",
"      ) + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 9 8 8 8 D o o o o o o o E 8 8 8 F + + + + G ",
"      + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + H 8 8 8 8 I o o o o o o J 8 8 8 K + + + + ^ ",
"    L + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + y v 8 8 8 M N o o o o o o O 8 8 K + + + + P ",
"    + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + Q 8 8 8 8 R o o o o o o S 8 8 T + + + + + ",
"    + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + U 8 8 8 V o o o o o o z 8 v + + + + + + ",
"  ~ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + W 8 8 8 8 N o o o o o X 8 Y + + + + + Z ",
"  + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + y ` 8 8 8  .o o o o o ..} y + + + + + + ",
"  + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +.8 8 8 @.o o o o #.7 l + + + + + + ) ",
"  + + + + + + + + $.+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + e 8 8 8 8 g p o %.&.K + + + + + + +   ",
"  + + + + + + + + *.+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + =.8 8 8 8 -.;.8 >.+ + + + + + + +   ",
"^ + + + + + + + + ,.+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + '.8 8 8 8 8 8 8 ).+ + + + + + + =   ",
"+ + + + + + !.~.+ {.].^.+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + /.8 8 8 8 8 (.+ + + + + + + + +   ",
"+ + + + _.:.<.[.}.|.1.2.3.4.+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + y 5.7 8 6.7.+ + + + + + + + +     ",
"8.+ + + 9.].+ + :.{.0.a.,.+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + b.    ",
"4.+ + * c.+ + + d.e.+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +       ",
"  + + f.{.+ + + + g.+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + h.      ",
"  + + + i.j.+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +         ",
"  Z + + + k.l.m.+ + + n.o.,._.+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +         ",
"    + + + + _.p.[.i.q.e.r.s.o.+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + %           ",
"    t.+ + + + + + u.r.v.+ + w.+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +             ",
"      % + + + + + + + x.y.+ + z.A.0.+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +               ",
"        + + + + + + + B.C.*.D.E.+ + + F.G.H.I.r.+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +                 ",
"          J.+ + + + + + + + K.e.L.+ r.M.N.+ O.,.+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + P.                  ",
"              k + + + + + + + u.Q.R.2.S.+ + T.U.+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + +                     ",
"                    + + + + + + + + + p.V.M.i.+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + +                       ",
"                      W.+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + )                           ",
"                        X.+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + ~                               ",
"                              ; - + + + + + + + Y.        Z.' + + + + + + + + + + + + + + + + +                                 ",
"                                    { `. +~                     ^ + + + + + + + + + + ) +                                       ",
"                                                                      .+- + + ++@++                                             "};
"""
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    def browse () :
        webbrowser.open(url)
    menu = QMenu()
    action = QAction("Open in browser")
    action.triggered.connect(browse)
    def stop () :
        os.kill(pid, signal.SIGINT)
        app.quit()
    menu.addAction(action)
    quit = QAction("Quit")
    quit.triggered.connect(stop)
    menu.addAction(quit)
    pix = QPixmap()
    pix.loadFromData(xpm)
    icon = QIcon(pix)
    tray = QSystemTrayIcon()
    tray.setIcon(icon)
    tray.setVisible(True)
    tray.setToolTip("ecco")
    tray.setContextMenu(menu)
    app.exec_()

try :
    debug.log("starting Docker")
    debug("running:", " ".join(argv), kind="info")
    sub = subprocess.Popen(argv,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,
                           encoding="utf-8",
                           errors="replace")
    for line in sub.stdout :
        debug(line.strip())
        if url is None :
            match = _url.search(line)
            if match :
                url = match.group()
                debug.log("jupyter-notebook is at:", url)
                if args.gui :
                    proc = multiprocessing.Process(target=gui,
                                                   args=(os.getpid(), url),
                                                   daemon=False)
                    proc.start()
                if args.browse :
                    debug("starting browser...", kind="info")
                    webbrowser.open(url)
                    args.browse = False
except KeyboardInterrupt :
    debug.log("terminating...")
    sub.terminate()
