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

    def _compare_bin_shrink(self, image, float_tol=1e-3, **kwargs):
        floating = np.dtype(image.dtype).kind == 'f'
        shrink_ref = itk.bin_shrink_image_filter(image, **kwargs)
        shrink_cucim = image_grid.cucim_bin_shrink_image_filter(
            image, **kwargs
        )
        comparison = itk.comparison_image_filter(
            shrink_ref, shrink_cucim, verify_input_information=True
        )
        if floating:
            assert np.sum(np.abs(comparison)) <= float_tol
        else:
            # ignore difference in integer rounding
            assert np.max(comparison) <= 1.0

    @pytest.mark.parametrize("floating", [False, True])
    @pytest.mark.parametrize("shrink_factors", [1, 2, 3, 7, (4, 3, 2)])
    def test_bin_shrink_filter(self, shrink_factors, floating):
        if floating:
            image = self.image_f32
        else:
            image = self.image
        kwargs = dict(shrink_factors=shrink_factors)
        self._compare_bin_shrink(image, float_tol=0., **kwargs)

    def test_bin_shrink_filter_numpy_input(self):
        rng = np.random.default_rng()
        image = rng.standard_normal((512, 256), dtype=np.float32)
        self._compare_bin_shrink(image, float_tol=1e-3, shrink_factors=(4, 2))
