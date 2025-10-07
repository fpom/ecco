from pathlib import Path
from ecco.mrr import Model

tests_dir = Path(__file__).parent


def test_termites():
    m = Model.parse(tests_dir / "termites.rr")
    g = m()
    assert len(g) == 1
    #
    assert len(g.lts.init) == 2
    assert len(g.lts.states) == 114
    assert len(g.lts.dead) == 6
    assert len(g.lts.hull) == 44
    assert len(g.lts.tsucc) == len(g.lts.tpred) == 13
    #
    assert len(g.lts.tsucc["C1"](g.lts.states)) == 8
    assert len(g.lts.tsucc["R1"](g.lts.states)) == 4
    assert len(g.lts.tsucc["R2"](g.lts.states)) == 8
    assert len(g.lts.tsucc["R3"](g.lts.states)) == 18
    assert len(g.lts.tsucc["R4"](g.lts.states)) == 2
    assert len(g.lts.tsucc["R5"](g.lts.states)) == 14
    assert len(g.lts.tsucc["R6"](g.lts.states)) == 4
    assert len(g.lts.tsucc["R7"](g.lts.states)) == 48
    assert len(g.lts.tsucc["R8"](g.lts.states)) == 28
    assert len(g.lts.tsucc["R9"](g.lts.states)) == 30
    assert len(g.lts.tsucc["R10"](g.lts.states)) == 16
    assert len(g.lts.tsucc["R11"](g.lts.states)) == 10
    assert len(g.lts.tsucc["R12"](g.lts.states)) == 16
    #
    assert len(g.lts.tpred["C1"](g.lts.states)) == 8
    assert len(g.lts.tpred["R1"](g.lts.states)) == 12
    assert len(g.lts.tpred["R2"](g.lts.states)) == 22
    assert len(g.lts.tpred["R3"](g.lts.states)) == 26
    assert len(g.lts.tpred["R4"](g.lts.states)) == 2
    assert len(g.lts.tpred["R5"](g.lts.states)) == 14
    assert len(g.lts.tpred["R6"](g.lts.states)) == 4
    assert len(g.lts.tpred["R7"](g.lts.states)) == 48
    assert len(g.lts.tpred["R8"](g.lts.states)) == 48
    assert len(g.lts.tpred["R9"](g.lts.states)) == 30
    assert len(g.lts.tpred["R10"](g.lts.states)) == 16
    assert len(g.lts.tpred["R11"](g.lts.states)) == 15
    assert len(g.lts.tpred["R12"](g.lts.states)) == 30


def test_nests():
    m = Model.parse(tests_dir / "nests.mrr")
    g = m()
    assert len(g) == 1
    #
    assert len(g.lts.init) == 1
    assert len(g.lts.states) == 81
    assert len(g.lts.dead) == 0
    assert len(g.lts.hull) == 81
    assert len(g.lts.tsucc) == len(g.lts.tpred) == 7
    #
    assert len(g.lts.tsucc["R1._.i.0"](g.lts.states)) == 45
    assert len(g.lts.tsucc["R1._.i.1"](g.lts.states)) == 45
    assert len(g.lts.tsucc["nest.0.R1"](g.lts.states)) == 36
    assert len(g.lts.tsucc["nest.0.R2"](g.lts.states)) == 9
    assert len(g.lts.tsucc["nest.1.R1"](g.lts.states)) == 36
    assert len(g.lts.tsucc["nest.1.R2"](g.lts.states)) == 9
    assert len(g.lts.tsucc["tick.tick"](g.lts.states)) == 39
    #
    assert len(g.lts.tpred["R1._.i.0"](g.lts.states)) == 72
    assert len(g.lts.tpred["R1._.i.1"](g.lts.states)) == 72
    assert len(g.lts.tpred["nest.0.R1"](g.lts.states)) == 36
    assert len(g.lts.tpred["nest.0.R2"](g.lts.states)) == 9
    assert len(g.lts.tpred["nest.1.R1"](g.lts.states)) == 36
    assert len(g.lts.tpred["nest.1.R2"](g.lts.states)) == 9
    assert len(g.lts.tpred["tick.tick"](g.lts.states)) == 39


def mktests(src):
    src, name = str(src), Path(src).stem
    m = Model.parse(tests_dir / src)
    g = m()
    print(f"def test_{name}():")
    print(f"    m = Model.parse(tests_dir / {src!r})")
    print(f"    g = m()")
    print(f"    assert len(g) == {len(g)}")
    print(f"    #")
    print(f"    assert len(g.lts.init) == {len(g.lts.init)}")
    print(f"    assert len(g.lts.states) == {len(g.lts.states)}")
    print(f"    assert len(g.lts.dead) == {len(g.lts.dead)}")
    print(f"    assert len(g.lts.hull) == {len(g.lts.hull)}")
    print(f"    assert len(g.lts.tsucc) == len(g.lts.tpred) == {len(g.lts.tsucc)}")
    print(f"    #")
    for n, t in g.lts.tsucc.items():
        print(
            f"    assert len(g.lts.tsucc[{n!r}](g.lts.states)) == {len(t(g.lts.states))}"
        )
    print(f"    #")
    for n, t in g.lts.tpred.items():
        print(
            f"    assert len(g.lts.tpred[{n!r}](g.lts.states)) == {len(t(g.lts.states))}"
        )
