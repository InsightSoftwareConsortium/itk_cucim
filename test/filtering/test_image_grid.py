from pathlib import Path

import itk
import numpy as np
import pytest

from itk_cucim.filtering import image_grid


class TestImageGrid:
    def setup_class(self):
        data = Path(__file__).absolute().parent.parent / "input" / "head_mr.mha"
        # uint8 data
        image_u8 = itk.imread(data)
        self.image = image_u8
        # float32 data
        self.image_f32 = image_u8.astype(np.float32)

    @pytest.mark.parametrize("floating", [False, True])
    @pytest.mark.parametrize("shrink_factors", [1, 2, 3, 7, (4, 3, 2)])
    def test_median_image_filter(self, shrink_factors, floating):
        if floating:
            image = self.image_f32
        else:
            image = self.image
        kwargs = dict(shrink_factors=shrink_factors)
        shrink_ref = itk.bin_shrink_image_filter(image, **kwargs)
        shrink_cucim = image_grid.cucim_bin_shrink_image_filter(
            image, **kwargs
        )

        comparison = itk.comparison_image_filter(
            shrink_ref, shrink_cucim, verify_input_information=True
        )
        if floating:
            assert np.sum(comparison) == 0.0
        else:
            # ignore difference in integer rounding
            assert np.max(comparison) <= 1.0
