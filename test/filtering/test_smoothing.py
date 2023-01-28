from pathlib import Path

import itk
import numpy as np
import pytest

from itk_cucim.filtering import smoothing


class TestSmoothing:
    def setup_class(self):
        data = Path(__file__).absolute().parent.parent / "input" / "head_mr.mha"
        # uint8 data
        image_u8 = itk.imread(data)
        self.image = image_u8

        Caster = itk.CastImageFilter[itk.itkImagePython.itkImageUC3,
                                     itk.itkImagePython.itkImageF3].New()
        # float32 data
        self.image_f32 = Caster(image_u8)

    def _compare_median(self, image, radius):
        median_ref = itk.median_image_filter(image, radius=radius)
        median_cucim = smoothing.cucim_median_image_filter(image, radius=radius)

        comparison = itk.comparison_image_filter(
            median_ref, median_cucim, verify_input_information=True
        )
        assert np.sum(comparison) == 0.0

    @pytest.mark.parametrize("radius", [1, (3, 2, 1)])
    @pytest.mark.parametrize("floating", [False, True])
    def test_median_image_filter(self, radius, floating):
        if floating:
            image = self.image_f32
        else:
            image = self.image
        self._compare_median(image, radius)

    def test_median_image_filter_numpy_input(self):
        rng = np.random.default_rng()
        image = rng.standard_normal((512, 256), dtype=np.float32)
        self._compare_median(image, radius=2)

    def _compare_discrete_gaussian(self, image, **kwargs):
        floating = np.dtype(image.dtype).kind == 'f'
        gaussian_ref = itk.discrete_gaussian_image_filter(image, **kwargs)
        gaussian_cucim = smoothing.cucim_discrete_gaussian_image_filter(
            image, **kwargs
        )

        comparison = itk.comparison_image_filter(
            gaussian_ref, gaussian_cucim, verify_input_information=True
        )

        if not floating:
            # values may differ by up to 1 due to integer rounding differences
            gaussian_ref = np.asarray(gaussian_ref, dtype=np.float32)
            gaussian_cucim = np.asarray(gaussian_cucim, dtype=np.float32)
            assert np.max(np.abs(gaussian_ref - gaussian_cucim)) <= 1
        else:
            assert np.max(comparison) < 1e-3

    @pytest.mark.parametrize("variance", [1, 4, (3, 2, 1)])
    @pytest.mark.parametrize("use_image_spacing", [False, True])
    @pytest.mark.parametrize("floating", [False, True])
    def test_discrete_gaussian_image_filter(self, variance, use_image_spacing, floating):
        if floating:
            image = self.image_f32
        else:
            image = self.image
        kwargs = dict(
            variance=variance,
            use_image_spacing=use_image_spacing
        )
        self._compare_discrete_gaussian(image, **kwargs)

    def test_discrete_gaussian_image_filter_numpy_input(self):
        rng = np.random.default_rng()
        image = rng.standard_normal((512, 256), dtype=np.float32)
        kwargs = dict(
            variance=(3, 2),
            use_image_spacing=False,
        )
        self._compare_discrete_gaussian(image, **kwargs)
