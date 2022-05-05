"""cuCIM accelerated filters for the ITKSmoothing module."""

import itk
import cucim.skimage
import numpy as np
import cupy as cp

def cucim_median_image_filter(*args, **kwargs):
    input_image = args[0]
    ref_filt = itk.MedianImageFilter.New(*args, **kwargs)
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

        radius = ref_filt.GetRadius()
        footprint = cp.ones([r*2+1 for r in reversed(radius)])

        cu_output_array = cucim.skimage.filters.median(cu_input_array, footprint, mode='nearest')
        output_array[:] = cu_output_array.get()[:]
    wrapper.SetPyGenerateData(generate_data)

    wrapper.Update()

    return wrapper.GetOutput()
