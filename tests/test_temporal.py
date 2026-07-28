from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from confingo import ConfigError
from confingo.functional import (
    dumps_json,
    from_dict,
    load_yaml,
    save_yaml,
    to_dict,
)


if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class Temporal:
    ts: dt.datetime = dt.datetime(2020, 1, 1, 0, 0, 0)
    day: dt.date = dt.date(2020, 1, 1)
    at: dt.time = dt.time(0, 0, 0)


def test_from_iso_strings():
    cfg = from_dict(Temporal, {"ts": "2021-06-15T13:45:30", "day": "2021-06-15", "at": "13:45:30"})
    assert cfg.ts == dt.datetime(2021, 6, 15, 13, 45, 30)
    assert cfg.day == dt.date(2021, 6, 15)
    assert cfg.at == dt.time(13, 45, 30)


def test_from_native_objects():
    native = {"ts": dt.datetime(2021, 6, 15, 1, 2, 3), "day": dt.date(2021, 6, 15), "at": dt.time(1, 2, 3)}
    cfg = from_dict(Temporal, native)
    assert cfg.ts == native["ts"]
    assert cfg.day == native["day"]
    assert cfg.at == native["at"]


def test_to_dict_emits_iso_strings():
    cfg = Temporal(ts=dt.datetime(2022, 3, 4, 5, 6, 7), day=dt.date(2022, 3, 4), at=dt.time(5, 6, 7))
    assert to_dict(cfg) == {"ts": "2022-03-04T05:06:07", "day": "2022-03-04", "at": "05:06:07"}


def test_round_trip():
    cfg = Temporal(ts=dt.datetime(2022, 3, 4, 5, 6, 7), day=dt.date(2022, 3, 4), at=dt.time(5, 6, 7))
    assert from_dict(Temporal, to_dict(cfg)) == cfg


def test_dumps_json_no_longer_crashes():
    body = dumps_json(Temporal())
    assert "2020-01-01T00:00:00" in body


def test_bad_datetime_string_rejected():
    with pytest.raises(ConfigError) as info:
        from_dict(Temporal, {"ts": "not-a-datetime"})
    assert any("ISO 8601 datetime" in issue.message for issue in info.value.issues)


def test_datetime_value_rejected_on_date_field():
    with pytest.raises(ConfigError) as info:
        from_dict(Temporal, {"day": dt.datetime(2020, 1, 1, 12, 0)})
    assert any(issue.message == "expected a date, got datetime" for issue in info.value.issues)


def test_yaml_round_trip(tmp_path: Path):
    cfg = Temporal(ts=dt.datetime(2022, 3, 4, 5, 6, 7), day=dt.date(2022, 3, 4), at=dt.time(5, 6, 7))
    path = save_yaml(cfg, tmp_path / "t.yaml")
    assert load_yaml(Temporal, path) == cfg
