from pathlib import Path

import itk
import numpy as np
import pytest

from itk_cucim.filtering import distance_map


class TestDistanceMap:
    def setup_class(self):
        data = Path(__file__).absolute().parent.parent / "input" / "horse.png"
        # uint8 data
        binary_horse = itk.imread(data)[..., 0] > 0
        self.image = itk.image_view_from_array(binary_horse.view(np.uint8))

    def _compare_signed_maurer_distance(self, image, **kwargs):
        shrink_ref = itk.signed_maurer_distance_map_image_filter(
            image, **kwargs
        )
        shrink_cucim = distance_map.cucim_signed_maurer_distance_map_image_filter(  # noqa
            image, **kwargs
        )
        # using comparison_image_filter mainly to verify the input information
        comparison = itk.comparison_image_filter(
            shrink_ref, shrink_cucim, verify_input_information=True
        )
        # use NumPy's assert_allclose to check relative tolerance
        # Empirically found a need to use a larger tolerance when
        # squared_distance is True.
        squared_distance = kwargs.get('squared_distance', False)
        rtol = 1e-4 if squared_distance else 1e-6
        np.testing.assert_allclose(
            shrink_ref, shrink_cucim, atol=4e-4, rtol=rtol
        )

    @pytest.mark.parametrize("squared_distance", [False, True])
    @pytest.mark.parametrize("inside_is_positive", [False, True])
    @pytest.mark.parametrize("spacing", [None, (2, 2), (1.5, 3.3)])
    def test_signed_maurer_distance_map_image_filter(self, spacing, inside_is_positive, squared_distance):
        if spacing is None:
            image = self.image
        else:
            image = itk.image_duplicator(self.image)
            image.SetSpacing(spacing)
        kwargs = dict(
            squared_distance=squared_distance,
            inside_is_positive=inside_is_positive,
        )
        self._compare_signed_maurer_distance(image, **kwargs)

    def test_signed_maurer_distance_map_image_filter_nonzero_background(self):
        image = self.image
        kwargs = dict(
            squared_distance=False,
            inside_is_positive=False,
            background_value=1,
        )
        with pytest.raises(ValueError):
            distance_map.cucim_signed_maurer_distance_map_image_filter(
                image, **kwargs
            )

    def test_signed_maurer_distance_map_image_filter_numpy_input(self):
        image = itk.array_view_from_image(self.image)
        self._compare_signed_maurer_distance(image, squared_distance=False)
