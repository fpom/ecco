# `ecco`: ECological COmputation

`ecco` is a Jupyter library dedicated to the formal modelling and analysis of ecosystems.
It implements methods described and used in several papers:

 * _Using discrete systems to exhaustively characterize the dynamics of an integrated
   ecosystem._
   C. Gaucherel, F. Pommereau.
   [DOI:10.1111/2041-210X.13242](https://doi.org/10.1111/2041-210X.13242)
 * _Maintaining biodiversity promotes the multifunctionality of social-ecological
   systems: holistic modelling of a mountain system._
   Z. Mao, J. Centanni, F. Pommereau, A. Stokes, C. Gaucherel.
   [DOI:10.1016/j.ecoser.2020.101220](https://doi.org/10.1016/j.ecoser.2020.101220)
 * _Discrete-event models for conservation assessment of integrated ecosystems._
   C. Gaucherel, C. Carpentier, I. R. Geijzendorffer, C. Noûs, F. Pommereau.
   [DOI:10.1016/j.ecoinf.2020.101205](https://doi.org/10.1016/j.ecoinf.2020.101205)
 * _Understanding Ecosystem Complexity via Application of a Process-Based State Space
   rather than a Potential Surface._
   C. Gaucherel, F. Pommereau, C. Hély.
   [DOI:10.1155/2020/7163920](https://doi.org/10.1155/2020/7163920)
   
At the heart of `ecco` is the _reaction rules_ modelling language described in [DOI:10.1111/2041-210X.13242](https://doi.org/10.1111/2041-210X.13242) that allows to describe an ecosystem as:

 * a set of Boolean entities that define the state of the ecosystem
 * a set of rules that define the transitions (aka, states changes)

All together, this yields a state space that `ecco` can compute, analyse, and render graphically in a way that is suitable for interactive exploration and analysis.
`ecco` also features static analysis tools that do not rely on the state space computation.

`ecco` uses symbolic state spaces representation based on decision diagrams, older tools have used explicit state spaces instead.
They have not been publicly released but their features will be progressively ported into `ecco`.
One may find such an early version of `ecco` here: [DOI:10.5281/zenodo.3241370](https://doi.org/10.5281/zenodo.3241370)

## Documentation

`ecco` is documented through Jupyter notebooks that can be found into directory `doc`.

## Contact and bug reports

Please use [GitHub issues](https://github.com/fpom/ecco/issues) to report problems or ask questions about `ecco` itself.
For more general questions, feel free to send an email to <franck.pommereau@univ-evry.fr>.

## Installation

There is not yet an automated installation procedure.
First, you should install all the dependencies:

 * [Python 3](https://www.python.org/) (version tested: 3.7)
 * [Jupyter](https://pypi.org/project/jupyter/) (version tested: 1.0.0)
 * [NetworkX](https://pypi.org/project/networkx/) (version tested: 2.4)
 * [pandas](https://pypi.org/project/pandas/) (version tested: 0.25.3)
 * [NumPy](https://pypi.org/project/numpy/) (version tested: 1.18.1)
 * [BQPlot](https://pypi.org/project/bqplot/) (version tested: 0.11.9)
 * [ipywidgets](https://pypi.org/project/ipywidgets/) (version tested: 7.5.1)
 * [colour](https://pypi.org/project/colour/) (version tested: 0.1.5)
 * [bitarray](https://pypi.org/project/bitarray/) (version tested: 1.2.0)
 * [SymPy](https://pypi.org/project/sympy/) (version tested: 1.5)
 * [竜 TatSu](https://pypi.org/project/TatSu/) (version tested: 4.4.0)
 * [Cython](https://pypi.org/project/Cython/) (version tested: 0.29.14)
 * [psutil](https://pypi.org/project/psutil/) (version tested: 5.6.7)
 * [pyddd](https://github.com/fpom/pyddd) (latest version)
 * [pyits](https://github.com/fpom/pyits) (latest version)
 * [pyits model-checker](https://forge.ibisc.univ-evry.fr/cthomas/pyits_model_checker) (latest version)
 * [pytl](https://github.com/fpom/pytl) (latest version)
 * [ptnet](https://github.com/fpom/cunf-ptnet-py3) (latest version)

Then, `ecco` package directory should be copied somewhere from which Python will be able to import it.

You may want to configure Jupyter notebooks so that files with extensions `.rr` are opened and edited as Python files, which is more convenient than editing them as plain text (editing as YAML could be another good choice).
To do so, add the following to `/etc/jupyter/nbconfig/edit.json`:

```json
{
  "Editor": {
    "file_extension_modes": {
      ".rr": "Python",
    }
  }
}
```

## Licence

Copyright 2020-2021 Franck Pommereau <franck.pommereau@univ-evry.fr>

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program.  If not, see [https://www.gnu.org/licenses](https://www.gnu.org/licenses/).
