from itk_cucim.filtering import smoothing
import itk
import numpy as np

from pathlib import Path

def test_median_image_filter():
    image = itk.imread(Path(__file__).absolute().parent.parent / 'input' / 'head_mr.mha')

    median_ref = itk.median_image_filter(image, radius=[1,2,1])
    median_cucim = smoothing.cucim_median_image_filter(image, radius=[1,2,1])

    comparison = itk.comparison_image_filter(median_ref, median_cucim, verify_input_information=True)
    assert np.sum(comparison) == 0.0

    median_ref = itk.median_image_filter(image, radius=[3,2,1])
    median_cucim = smoothing.cucim_median_image_filter(image, radius=[3,2,1])

    comparison = itk.comparison_image_filter(median_ref, median_cucim, verify_input_information=True)
    assert np.sum(comparison) == 0.0
