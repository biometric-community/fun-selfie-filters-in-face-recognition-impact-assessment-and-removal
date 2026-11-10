"""Filter taxonomy, coverage intensity (Eq. 1, Table IV) and removal, on synthetic images."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from selffilter.eval import best_acc_eer
from selffilter.filters.synth import COVERAGE_LEVELS, PLACEMENTS, apply_filter, remove_with_mask


def faces(value: float = 0.0, size: int = 112) -> torch.Tensor:
    return torch.full((2, 3, size, size), value)


def test_placements_and_coverages_match_the_paper():
    """Sec. III: filters are grouped by placement (Table VIII) and coverage (Table IV)."""
    assert set(PLACEMENTS) == {"eyes", "nose", "mouth", "total"}
    assert set(COVERAGE_LEVELS) == {"low", "medium", "high"}


@pytest.mark.parametrize("placement", sorted(PLACEMENTS))
def test_more_coverage_means_more_intensity(placement):
    """Table IV orders the groups by coverage, so each level must change more of the face."""
    scores = [apply_filter(faces(), placement, level)[2] for level in ("low", "medium", "high")]
    assert scores[0] < scores[1] < scores[2]


@pytest.mark.parametrize("placement", sorted(PLACEMENTS))
@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_coverage_intensity_falls_in_its_table_iv_band(targets, placement, level):
    low, high = targets["coverage_bands"][level]
    assert low <= apply_filter(faces(), placement, level)[2] < high


def test_coverage_intensity_measures_how_much_the_pixels_change():
    """Eq. (1): the average intensity change per facial pixel, so the same sticker scores lower
    on a face whose pixels are already close to the sticker's colours."""
    near = apply_filter(faces(0.2), "total", "high")[2]
    far = apply_filter(faces(-1.0), "total", "high")[2]
    assert far > near


def test_filter_only_changes_pixels_inside_its_mask():
    x = torch.rand(2, 3, 112, 112) * 2 - 1
    out, mask, _ = apply_filter(x, "eyes", "medium")
    outside = mask.expand_as(x) == 0
    assert torch.equal(out[outside], x[outside])
    assert not torch.equal(out[~outside], x[~outside])


def test_removal_only_touches_the_filtered_region_and_moves_towards_the_face():
    """Sec. IV: removal reconstructs the masked region and keeps the rest of the image."""
    ramp = torch.linspace(-0.8, 0.8, 112)
    x = (ramp.view(1, 1, -1, 1) + ramp.view(1, 1, 1, -1)).expand(2, 3, 112, 112) / 2
    filtered, mask, _ = apply_filter(x, "total", "high")
    restored = remove_with_mask(filtered, mask)
    outside = mask.expand_as(x) == 0
    assert torch.equal(restored[outside], filtered[outside])
    assert (restored - x).abs()[~outside].mean() < (filtered - x).abs()[~outside].mean()


def test_eer_is_where_false_match_and_false_non_match_rates_meet():
    """The EER is the error rate where FMR equals FNMR, not the lowest half total error rate."""
    scores = np.array([0.6, 0.7, 0.8, 0.9, 0.0, 0.1, 0.2, 0.65])
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert best_acc_eer(scores, labels)[1] == pytest.approx(0.25, abs=0.01)
