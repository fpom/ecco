#!/usr/bin/env python3

import argparse, subprocess, shlex, re, webbrowser

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
parser.add_argument("-c", "--chdir", default="/home/ecco", type=str, metavar="DIR",
                    help="set working directory to DIR (default: '/home/ecco')")
parser.add_argument("cmd", default=["jupyter-notebook", "--no-browser",
                                    "--port=8000", "--ip=0.0.0.0"],
                    type=str, nargs="*",
                    help="start a specific command (default: jupyter-notebook)")

args = parser.parse_args()
argv = ["docker", "run",
        "-p", f"{args.port}:8000",
        "-u", args.user,
        "-w", args.chdir,
        f"{args.repo}ecco:{args.tag}"] + args.cmd

url = re.compile(f"http://127.0.0.1:8000/\S*")

class Debug (object) :
    def __init__ (self) :
        self.debug = args.debug
        try :
            import colorama
            self.c = {"docker" : colorama.Fore.LIGHTBLACK_EX,
                      "info" : colorama.Fore.BLUE,
                      "warn" : colorama.Fore.RED,
                      None : colorama.Style.RESET_ALL}
        except :
            self.c = {"docker" : "",
                      "info" : "",
                      "warn" : "",
                      None : ""}
    def __getitem__ (self, key) :
        return self.c.get(key, "")
    def __call__ (self, *args, kind="docker") :
        if self.debug :
            print(f"{self[kind]}{' '.join(str(a) for a in args)}{self[None]}")

debug = Debug()

try :
    debug("running:", " ".join(argv), kind="info")
    sub = subprocess.Popen(argv,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,
                           encoding="utf-8",
                           errors="replace")
    for line in sub.stdout :
        debug(line.strip())
        if args.browse :
            match = url.search(line)
            if match :
                debug("starting browser...", kind="info")
                webbrowser.open(match.group())
                args.browse = False
except KeyboardInterrupt :
    debug("terminating...", kind="warn")
    sub.terminate()
