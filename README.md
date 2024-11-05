# `ecco`: ECological COmputation

`ecco` is a Jupyter library dedicated to the formal modelling and analysis of ecosystems.
It implements methods described and used in several papers:

- _Using discrete systems to exhaustively characterize the dynamics of an integrated
  ecosystem._
  C. Gaucherel, F. Pommereau.
  [DOI:10.1111/2041-210X.13242](https://doi.org/10.1111/2041-210X.13242)
- _Maintaining biodiversity promotes the multifunctionality of social-ecological
  systems: holistic modelling of a mountain system._
  Z. Mao, J. Centanni, F. Pommereau, A. Stokes, C. Gaucherel.
  [DOI:10.1016/j.ecoser.2020.101220](https://doi.org/10.1016/j.ecoser.2020.101220)
- _Discrete-event models for conservation assessment of integrated ecosystems._
  C. Gaucherel, C. Carpentier, I. R. Geijzendorffer, C. Noûs, F. Pommereau.
  [DOI:10.1016/j.ecoinf.2020.101205](https://doi.org/10.1016/j.ecoinf.2020.101205)
- _Understanding Ecosystem Complexity via Application of a Process-Based State Space
  rather than a Potential Surface._
  C. Gaucherel, F. Pommereau, C. Hély.
  [DOI:10.1155/2020/7163920](https://doi.org/10.1155/2020/7163920)

At the heart of `ecco` is the _reaction rules_ modelling language described in [DOI:10.1111/2041-210X.13242](https://doi.org/10.1111/2041-210X.13242) that allows to describe an ecosystem as:

- a set of Boolean entities that define the state of the ecosystem
- a set of rules that define the transitions (aka, states changes)

All together, this yields a state space that `ecco` can compute, analyse, and render graphically in a way that is suitable for interactive exploration and analysis.
`ecco` also features static analysis tools that do not rely on the state space computation.

`ecco` uses symbolic state spaces representation based on decision diagrams, older tools have used explicit state spaces instead.
They have not been publicly released but their features will be progressively ported into `ecco`.
One may find such an early version of `ecco` here: [DOI:10.5281/zenodo.3241370](https://doi.org/10.5281/zenodo.3241370)

## Documentation

`ecco` is documented through Jupyter notebooks that can be found into directory `doc`.

## Trying `ecco` with zero installation

- go to [Play with Docker](https://labs.play-with-docker.com/)
- login to the service, using one of the proposed accounts (you must have one)
- click the _Start_ button
  - if the server is overloaded you get an error message, just go back and click _Start_ again
  - you eventually get a page with a 4 hours countdown, this is the time you have to play with `ecco`
- click _ADD NEW INSTANCE_ on the left
- after a few seconds, you get a shell (large black rectangle filled with gibberish), at the prompt `$ ` type or copy/paste `docker run -p 8000:8000 franckpommereau/ecco jupyterhub`
  - this will display a bunch of progress bars
  - wait until it's all finished (it may take a loooong time)
  - `ecco` ready to use when you see a line like `[I 2023-12-06 16:44:27.283 JupyterHub app:3245] JupyterHub is now running at http://:8000`
- click _OPEN PORT_ at the top of the page and type `8000` at the prompt
  - alternatively, wait for a link `8000` to appear next to the button and click it
- this pops-up a new browser window
  - login as user `ecco` with password `ecco`
  - you get a working Jupyter with `ecco` installed
  - either upload your notebook and files or go to folder `doc` to get some examples
- when you are done with `ecco`, go back to Play with Docker and click _CLOSE SESSION_

## Quickstart on your own machine

`ecco` is available as a Docker image so you don't need to install it.
But of course you need to have Docker installed.

The simplest way to run `ecco` from this Docker image is to download [this script](https://github.com/fpom/ecco/raw/main/bin/ecco.py) and run it.
It will take care if everything.
Use `ecco.py --help` and see below for further information about the available options.

If you wish to do it manually, run `docker run -p 8000:8000 franckpommereau/ecco jupyterhub` then point your browser to [localhost:8000](http://localhost:8000) and login as user `ecco` with password `ecco`.
The notebooks and example models are available in directory `doc`.

The Docker command above runs `ecco` on the top of a multi-user `JupyterHub` installation, you may reconfigure it according to you needs (in particular, to add users and data persistence) for a production usage.

You may also want to run `docker run -p 8000:8000 -u ecco -w /home/ecco franckpommereau/ecco jupyter-notebook --no-browser --port=8000 --ip=0.0.0.0` to start only `Jupyter Notebook` without `JupyterHub`.

To run a specific version of `ecco`, use `docker run ... franckpommereau/ecco:VERSION ...` where version is one of the tags listed on [ecco Github page](https://github.com/fpom/ecco/tags).

## Automated installation

Just use the docker image from the quickstart above, change user `ecco`'s password or remove it's account, and create other accounts according to your needs.

## Manual installation

First, you should install all the dependencies:

- [Python 3](https://www.python.org/) (version tested: 3.7)
- [GraphViz](https://www.graphviz.org/) (version tested: 2.40.1)
- [Jupyter](https://pypi.org/project/jupyter/) (version tested: 1.0.0)
- [NetworkX](https://pypi.org/project/networkx/) (version tested: 2.4)
- [pandas](https://pypi.org/project/pandas/) (version tested: 0.25.3)
- [NumPy](https://pypi.org/project/numpy/) (version tested: 1.18.1)
- [BQPlot](https://pypi.org/project/bqplot/) (version tested: 0.11.9)
- [ipywidgets](https://pypi.org/project/ipywidgets/) (version tested: 7.5.1)
- [colour](https://pypi.org/project/colour/) (version tested: 0.1.5)
- [bitarray](https://pypi.org/project/bitarray/) (version tested: 1.2.0)
- [prince](https://pypi.org/project/prince/) (version tested: 0.7.1)
- [SymPy](https://pypi.org/project/sympy/) (version tested: 1.5)
- [竜 TatSu](https://pypi.org/project/TatSu/) (version tested: 4.4.0)
- [Cython](https://pypi.org/project/Cython/) (version tested: 0.29.14)
- [psutil](https://pypi.org/project/psutil/) (version tested: 5.6.7)
- [pydot](https://pypi.org/project/pydot/) (version tested: 1.4.1)
- [typeguard](https://pypi.org/project/typeguard/)
- [rich[jupyter]](https://pypi.org/project/rich/)
- [pyddd](https://github.com/fpom/pyddd) (latest version)
- [pyits](https://github.com/fpom/pyits) (latest version)
- [pytl](https://github.com/fpom/pytl) (latest version)
- [pyits model-checker](https://forge.ibisc.univ-evry.fr/cthomas/pyits_model_checker) (latest version)
- [ptnet](https://github.com/fpom/cunf-ptnet-py3) (latest version)
- LuaLaTex that is included in most [LaTeX](https://www.latex-project.org) distributions

Then, run `python setup.py install` as usual, or `pip install git+https://github.com/fpom/ecco.git` to install directly from the [GitHub repository](https://github.com/fpom/ecco).

Looking at `docker/Dockerfile` in the distribution will give you the commands to run on a Linux box.

You may want to configure Jupyter notebooks so that files with extensions `.rr` are opened and edited as Python files, which is more convenient than editing them as plain text (editing as YAML could be another good choice).
To do so, add the following to `/etc/jupyter/nbconfig/edit.json`:

```json
{
  "Editor": {
    "file_extension_modes": {
      ".rr": "Python"
    }
  }
}
```

## `ecco` CLI and GUI

You may copy `ecco.py` somewhere in your `PATH`, renaming it to `ecco`.
Then, running just `ecco` will start the Docker container and point your browser onto the freshly started Jupyter Notebook.
Stop the container with `Ctrl-C`.

Since `ecco` runs inside a Docker container, you will not find your files, and everything done inside the container will be lost when it shuts down.
To work around this, use option `--mount` that allows to access a local directory from within the container.
For instance `ecco --mount workdir` creates a directory `workdir` inside the container that gives access to your local directory `workdir` so that everything that is saved to it is also saved outside of the container.
You may use `--mount` has many times as necessary.

Option `--tag` allows to run a specific version of `ecco` and defaults to the lastest.
Option `--port` allows to change the TCP port on which `jupyter-notebook` is available, and it defaults to `8000`.

An experimental option `--gui` provides desktop integration.
It's experimental since it has been tested only under Linux/Gnome so far, but in principle it should work everywhere else.
To use it, you'll have to install package `PySide2` that provides the portable graphical library.
That done, you may run `ecco --gui --no-browse` on you session startup to run a background instance of the container and install an icon in the system tray.
The other options are available too of course, in particular `--mount`.
Right-clicking on the tray icon yields a menu that allows to point your browser onto the running `jupyter-notebook` (which `--no-browse` avoided), or shutting down the container and the GUI.

Lastly, after its own options, `ecco` accepts any command with options to be run inside the container instead of `juputer-notebook`.
Other, more specific, options are available, run `ecco --help` to list them.

## Contact and bug reports

Please use [GitHub issues](https://github.com/fpom/ecco/issues) to report problems or ask questions about `ecco` itself.
For more general questions, feel free to send an email to <franck.pommereau@univ-evry.fr>.

## Licence

Copyright 2020-2021 Franck Pommereau <franck.pommereau@univ-evry.fr>

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program. If not, see [https://www.gnu.org/licenses](https://www.gnu.org/licenses/).
