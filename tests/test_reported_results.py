"""REPORT.md, the committed output of the full run, against the paper's results."""

from __future__ import annotations

import re

import pytest


def impact(report: str, condition: str) -> tuple[float, float, float]:
    match = re.search(rf"\| {re.escape(condition)} \| ([\d.]+)% \| ([\d.]+)% \| ([\d.]+) \|", report)
    assert match, f"{condition} row missing from REPORT.md"
    return tuple(float(g) for g in match.groups())


def removal(report: str, condition: str) -> tuple[float, float]:
    match = re.search(rf"\| {re.escape(condition)} \| ([\d.]+)% \| ([\d.]+)% \| -?[\d.]+pp \|", report)
    assert match, f"{condition} removal row missing from REPORT.md"
    return float(match.group(1)), float(match.group(2))


def clean(report: str, dataset: str) -> tuple[float, float]:
    match = re.search(rf"\| {dataset} \| ([\d.]+)% \| ([\d.]+)% \| \d+ \|", report)
    assert match, f"{dataset} row missing from REPORT.md"
    return float(match.group(1)), float(match.group(2))


PLACEMENTS = ["eyes", "nose", "mouth", "total"]


def test_recogniser_works_on_unfiltered_faces(report, targets):
    """Table VII: the recognition systems make no errors before filters are applied."""
    assert clean(report, "lfw")[0] >= targets["clean_lfw_accuracy_floor"]


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_filters_raise_the_error_rate(report, placement):
    """Abstract and Table VII: selfie filters degrade recognition."""
    assert impact(report, f"high/{placement}")[1] > clean(report, "lfw")[1]


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_high_coverage_hurts_more_than_low_coverage(report, placement):
    """Table VII: EER grows with facial coverage."""
    assert impact(report, f"high/{placement}")[1] > impact(report, f"low/{placement}")[1]


def test_covering_the_whole_face_hurts_most(report):
    """Table VIII: total-face filters give by far the highest EER of any placement."""
    total = impact(report, "high/total")[1]
    assert all(total >= impact(report, f"high/{p}")[1] for p in PLACEMENTS)


@pytest.mark.parametrize("level", ["low", "medium", "high"])
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_reported_intensity_falls_in_its_table_iv_band(report, targets, level, placement):
    low, high = targets["coverage_bands"][level]
    assert low <= impact(report, f"{level}/{placement}")[2] < high


@pytest.mark.parametrize("placement", ["eyes", "total"])
def test_removal_recovers_most_of_the_lost_accuracy(report, targets, placement):
    """Table X: removal brings high-coverage EER most of the way back to the baseline."""
    before, after = removal(report, f"high/{placement}")
    baseline = clean(report, "lfw")[1]
    assert (before - after) >= targets["removal_recovery_floor"] * (before - baseline)
