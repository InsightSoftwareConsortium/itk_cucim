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
        self.image = image_u8

        Caster = itk.CastImageFilter[itk.itkImagePython.itkImageUC3,
                                     itk.itkImagePython.itkImageF3].New()
        # float32 data
        self.image_f32 = Caster(image_u8)

    @pytest.mark.parametrize(
        "use_image_spacing, normalize_across_scale",
         [(False, False), (True, False), (True, True)]
    )
    @pytest.mark.parametrize("variance", [1, 4, (3, 2, 1)])
    @pytest.mark.parametrize("order", [1, 2, 3, 4, (2, 1, 3)])
    def test_discrete_gaussian_image_filter_f32(
        self, variance, order, use_image_spacing, normalize_across_scale
    ):
        kwargs = dict(
            variance=variance,
            order=order,
            use_image_spacing=use_image_spacing,
            normalize_across_scale=normalize_across_scale,
        )
        gaussian_ref = itk.discrete_gaussian_derivative_image_filter(
            self.image_f32, **kwargs
        )
        gaussian_cucim = image_feature.cucim_discrete_gaussian_derivative_image_filter(  # noqa
            self.image_f32, **kwargs
        )

        comparison = itk.comparison_image_filter(
            gaussian_ref, gaussian_cucim, verify_input_information=True
        )

        assert np.max(comparison) < 1e-2
