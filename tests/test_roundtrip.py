from __future__ import annotations

from pathlib import Path

from confingo import (
    from_dict,
    to_dict,
)
from tests.schemas import (
    Containers,
    Device,
    Session,
    Trainer,
    Training,
)


def test_defaults_round_trip():
    cfg = Training()
    assert from_dict(Training, to_dict(cfg)) == cfg


def test_populated_round_trip():
    cfg = Training(
        device=Device.CUDA,
        seed=123,
        buffer_size=2_000_000,
        output_dir=Path("runs/exp001"),
        trainer=Trainer(lr=1e-4, hidden=(128, 128)),
        sessions=[Session("steady"), Session("startup", 0.5)],
    )
    assert from_dict(Training, to_dict(cfg)) == cfg


def test_containers_round_trip():
    cfg = Containers(
        ints=[1, 2, 3],
        names={"a", "b"},
        frozen=frozenset({4, 5}),
        pair=(7, "x"),
        variadic=(1, 2, 3),
        tags={"a": 1, "b": 2},
        bare_tuple=(1, "two"),
        bare_list=[1, "two"],
        bare_set={1, 2},
        bare_dict={"k": "v"},
        anything=[1, 2],
    )
    assert from_dict(Containers, to_dict(cfg)) == cfg
