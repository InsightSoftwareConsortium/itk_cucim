"""cuCIM accelerated filters for the ITKSmoothing module."""
import math
import warnings

import cucim.skimage
import cupy as cp
import itk
import numpy as np

from ._discrete_gaussian import discrete_gaussian_derivative_filter


def cucim_discrete_gaussian_derivative_image_filter(*args, **kwargs):
    input_image = args[0]
    ref_filt = itk.DiscreteGaussianDerivativeImageFilter.New(*args, **kwargs)
    wrapper = itk.PyImageFilter.New(input_image)

    def generate_output_information(wrapper):
        ref_filt.UpdateOutputInformation()
        ref_output = ref_filt.GetOutput()
        wrapper_output = wrapper.GetOutput()
        # Copy image metadata as computed by the reference CPU filter
        wrapper_output.CopyInformation(ref_output)
    wrapper.SetPyGenerateOutputInformation(generate_output_information)

    def generate_data(wrapper):
        input_image = wrapper.GetInput()
        input_array = itk.array_view_from_image(input_image)
        cu_input_array = cp.array(input_array)

        output_image = wrapper.GetOutput()
        output_image.SetBufferedRegion(output_image.GetRequestedRegion())
        output_image.Allocate()
        output_array = itk.array_view_from_image(output_image)

        maximum_error = ref_filt.GetMaximumError()
        maximum_kernel_width = ref_filt.GetMaximumKernelWidth()
        use_image_spacing = ref_filt.GetUseImageSpacing()
        normalize_across_scale = ref_filt.GetNormalizeAcrossScale()
        variance = ref_filt.GetVariance()
        sigma = tuple(math.sqrt(v) for v in reversed(variance))
        order = tuple(reversed(ref_filt.GetOrder()))

        if use_image_spacing:
            spacing = tuple(input_image.GetSpacing())
        else:
            spacing = (1.0, ) * input_array.ndim
        cu_output_array = discrete_gaussian_derivative_filter(
            cu_input_array,
            sigma=sigma,
            order=order,
            spacing=spacing,
            normalize_across_scale=normalize_across_scale,
            max_error=maximum_error,
            max_half_width=maximum_kernel_width - 1,
        )
        output_array[:] = cu_output_array.get()[:]
    wrapper.SetPyGenerateData(generate_data)

    wrapper.Update()

    return wrapper.GetOutput()
