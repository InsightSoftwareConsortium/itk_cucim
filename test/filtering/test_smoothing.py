from pathlib import Path

import itk
import numpy as np
import pytest

from itk_cucim.filtering import smoothing


class TestSmoothing:
    def setup_class(self):
        data = Path(__file__).absolute().parent.parent / "input" / "head_mr.mha"
        self.image = itk.imread(data)

    @pytest.mark.parametrize("radius", [1, (3, 2, 1)])
    def test_median_image_filter(self, radius):

        median_ref = itk.median_image_filter(self.image, radius=radius)
        median_cucim = smoothing.cucim_median_image_filter(self.image, radius=radius)

        comparison = itk.comparison_image_filter(
            median_ref, median_cucim, verify_input_information=True
        )
        assert np.sum(comparison) == 0.0
