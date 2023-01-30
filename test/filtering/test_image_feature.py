from pathlib import Path

import itk
import numpy as np
import pytest

from itk_cucim.filtering import image_feature


class TestImageFeatures:
    def setup_class(self):
        data = Path(__file__).absolute().parent.parent / "input" / "head_mr.mha"
        # uint8 data
        image_u8 = itk.imread(data)
        self.image_u8 = image_u8

        Caster = itk.CastImageFilter[itk.itkImagePython.itkImageUC3,
                                     itk.itkImagePython.itkImageF3].New()
        # float32 data
        self.image_f32 = Caster(image_u8)

    def _compare_discrete_gaussian_derivative(self, image, **kwargs):
        floating = np.dtype(image.dtype).kind == 'f'
        gaussian_ref = itk.discrete_gaussian_derivative_image_filter(
            image, **kwargs
        )
        gaussian_cucim = image_feature.cucim_discrete_gaussian_derivative_image_filter(  # noqa
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
            assert np.max(comparison) < 1e-2

    # TODO: resolve failures for `floating = False` case
    #       (a subset of cases currently fail with a difference of 255)
    @pytest.mark.parametrize("floating", [True])
    @pytest.mark.parametrize(
        "use_image_spacing, normalize",
         [(False, False), (True, False), (True, True)]
    )
    @pytest.mark.parametrize("variance", [1, 4, (3, 2, 1)])
    @pytest.mark.parametrize("order", [1, 2, 3, 4, (2, 1, 3)])
    def test_discrete_gaussian_derivative_image_filter(
        self, variance, order, use_image_spacing, normalize, floating
    ):
        if floating:
            image = self.image_f32
        else:
            image = self.image_u8
        kwargs = dict(
            variance=variance,
            order=order,
            use_image_spacing=use_image_spacing,
            normalize_across_scale=normalize,
        )
        self._compare_discrete_gaussian_derivative(image, **kwargs)

    def test_discrete_gaussian_derivative_image_filter_numpy_input(self):
        rng = np.random.default_rng()
        image = rng.standard_normal((512, 256), dtype=np.float32)
        kwargs = dict(
            variance=(3, 1),
            order=(2, 1),
            use_image_spacing=False,
            normalize_across_scale=False,
        )
        self._compare_discrete_gaussian_derivative(image, **kwargs)
