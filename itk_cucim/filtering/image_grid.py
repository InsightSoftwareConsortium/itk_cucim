"""cuCIM accelerated filters for the ITKImageGrid module."""
import cupy as cp
import itk
from cucim.skimage.transform import downscale_local_mean
from itk.support import helpers


@helpers.accept_array_like_xarray_torch
def cucim_bin_shrink_image_filter(*args, **kwargs):
    input_image = args[0]
    ref_kwargs = kwargs.copy()
    if 'shrink_factor' in ref_kwargs:
        ref_kwargs['ShrinkFactor'] = ref_kwargs.pop('shrink_factor')
    if 'shrink_factors' in ref_kwargs:
        ref_kwargs['ShrinkFactors'] = ref_kwargs.pop('shrink_factors')
    ref_filt = itk.BinShrinkImageFilter.New(*args, **ref_kwargs)
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

        shrink_factors = tuple(reversed(ref_filt.GetShrinkFactors()))
        expected_shape = tuple(
            max(s // f, 1) for s, f in zip(input_array.shape, shrink_factors)
        )
        cu_output_array = downscale_local_mean(
            cu_input_array,
            shrink_factors,
        )
        # Note: downscale_local_mean pads the shape up to a multiple of the
        #       shrink factor, so we need to truncate to the expected shape.
        out_slices = tuple(slice(s) for s in expected_shape)
        output_array[:] = cu_output_array[out_slices].get()[:]
    wrapper.SetPyGenerateData(generate_data)

    wrapper.Update()

    return wrapper.GetOutput()
