"""Tests for common.selectDailyBlocks, which both tiers use to pair forecast blocks with dataset days."""
import numpy as np
import pandas as pd
import pytest

import common

BLOCK = 48


def blocks(issue_days, block_hours=BLOCK):
    """A forecast file with one block_hours-row block per issue day; the value is the block's number."""
    times, values = [], []
    for b, day in enumerate(pd.to_datetime(issue_days)):
        times.extend(pd.date_range(day, periods=block_hours, freq="h"))
        values.extend([b] * block_hours)
    return pd.DataFrame({"block": values}, index=pd.DatetimeIndex(times, name="datetime"))


def test_keeps_whole_blocks_for_the_days_asked_for():
    df = blocks(pd.date_range("2022-01-01", "2022-01-10"))
    out = common.selectDailyBlocks(df, "2022-01-03", "2022-01-05", BLOCK)
    assert len(out) == 3 * BLOCK
    assert out["block"].unique().tolist() == [2, 3, 4]
    assert out.index[0] == pd.Timestamp("2022-01-03") and out.index[-1] == pd.Timestamp("2022-01-06 23:00")


def test_accepts_timestamps_with_hours():
    # The dataset's last day is passed as its last hourly timestamp.
    df = blocks(pd.date_range("2022-01-01", "2022-01-05"))
    out = common.selectDailyBlocks(df, np.datetime64("2022-01-02T00:00"), np.datetime64("2022-01-04T23:00"), BLOCK)
    assert out["block"].unique().tolist() == [1, 2, 3]


def test_weather_starting_after_the_dataset_names_the_missing_days():
    # The tier 1 bug: electricity data from 2019, weather only from 2022.
    df = blocks(pd.date_range("2022-01-01", "2022-01-05"))
    with pytest.raises(ValueError, match=r"2 missing, e\.g\. \[datetime\.date\(2021, 12, 30\)"):
        common.selectDailyBlocks(df, "2021-12-30", "2022-01-03", BLOCK)


def test_a_gap_in_the_middle_is_an_error():
    days = list(pd.date_range("2022-01-01", "2022-01-05"))
    del days[2]
    with pytest.raises(ValueError, match="1 missing"):
        common.selectDailyBlocks(blocks(days), "2022-01-01", "2022-01-05", BLOCK)


def test_sorted_rows_are_rejected():
    # Sorting an overlapping file by time interleaves the blocks.
    df = blocks(pd.date_range("2022-01-01", "2022-01-05")).sort_index(kind="stable")
    with pytest.raises(ValueError, match="consecutive hours"):
        common.selectDailyBlocks(df, "2022-01-01", "2022-01-05", BLOCK)


def test_wrong_block_size_is_rejected():
    df = blocks(pd.date_range("2022-01-01", "2022-01-05"))
    with pytest.raises(ValueError):
        common.selectDailyBlocks(df, "2022-01-01", "2022-01-05", 24)
    with pytest.raises(ValueError, match="multiple of"):
        common.selectDailyBlocks(df.iloc[:-1], "2022-01-01", "2022-01-05", BLOCK)
