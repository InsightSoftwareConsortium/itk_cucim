import cupy as cp
import itk
import numpy as np
from itk.support import helpers

from cucim.core.operations.morphology import distance_transform_edt
from cucim.skimage.morphology import binary_erosion


def _signed_euclidean_distance_map(
    image,
    spacing=None,
    squared_distance=False,
    inside_is_positive=False,
):
    """Signed Euclidean distance transform.

    Parameters
    ----------
    image : cupy.ndarray
        The binary image for which to compute the signed distance transform.
    spacing : tuple of float
        The dimension of a pixel along each axis.
    squared_distance : bool
        If ``True``, the squared Euclidean distance is returned.
    inside_is_positive : bool
        If ``True``, the distances inside the object are positive while those
        outside are negative. The default behavior is the opposite.

    Returns
    -------
    signed_distance : cupy.ndarray
        The signed distance (in pixels or as determined by spacing).

    Notes
    -----
    This function is designed to give output equivalent to ITK's
    `SignedMaurerDistanceTransformImageFilter` [1]_. Note that ``spacing[i]``
    is the spacing along axis ``i``. This is reversed relative to the order
    returned by the ``GetSpacing`` method of ``itk.Image``.

    Note that while the output of this function is equivalent to Maurer's
    algorithm as implemented in ITK, the underlying implementation used here
    involves the subtraction of two unsigned distance transforms computed via
    the parallel banding blus (PBA+) algorithm [2]_, [3]_.

    References
    ----------
    .. [1] C. R. Maurer, Rensheng Qi and V. Raghavan, "A linear time algorithm
        for computing exact Euclidean distance transforms of binary images in
        arbitrary dimensions," in IEEE Transactions on Pattern Analysis and
        Machine Intelligence, vol. 25, no. 2, pp. 265-270, Feb. 2003.
        :DOI:`10.1109/TPAMI.2003.1177156`
    .. [2] Thanh-Tung Cao, Ke Tang, Anis Mohamed, and Tiow-Seng Tan. 2010.
        Parallel Banding Algorithm to compute exact distance transform with the
        GPU. In Proceedings of the 2010 ACM SIGGRAPH symposium on Interactive
        3D Graphics and Games (I3D ’10). Association for Computing Machinery,
        New York, NY, USA, 83–90.
        :DOI:`10.1145/1730804.1730818`
    .. [3] https://www.comp.nus.edu.sg/~tants/pba.html
    """
    if image.dtype == np.uint8:
        # can avoid copy from uint8->bool
        image = image.view(bool)
    else:
        # copy=False to omit copy of images that are already boolean
        image = image.astype(bool, copy=False)


    footprint = cp.ones((3, ) * image.ndim, dtype=bool)
    if inside_is_positive:
        image_in = ~image
    else:
        # boundary: erode by one pixel to get an equivalent result to ITK
        image_in = binary_erosion(image, footprint=footprint)

    distance_kwargs = dict(return_distances=True, return_indices=False)
    if spacing is not None:
        spacing = tuple(spacing)
        if any(s != 1.0 for s in spacing):
            distance_kwargs['sampling'] = spacing

    # distance transform of the eroded image
    distance = distance_transform_edt(image_in, **distance_kwargs)

    # now compute a second unsigned distance transform
    if inside_is_positive:
        # boundary: erode by one pixel to get an equivalent result to ITK
        image_in = binary_erosion(image, footprint=footprint)
    else:
        image_in = ~image
    distances_inv = distance_transform_edt(image_in, **distance_kwargs)

    if squared_distance:
        distances_inv *= distances_inv
        distance *= distance

    # subtract the unsigned transforms to get the signed result
    distances_inv -= distance
    return distances_inv


@helpers.accept_array_like_xarray_torch
def cucim_signed_maurer_distance_map_image_filter(*args, **kwargs):
    input_image = args[0]

    # TODO: remove requirement to cast to float32
    #   Output dtype is float32 for the distance transform.
    #   PyImageFilter currently supports 2D, 3D or 4D images of dtypes:
    #     float32, float64, uint8, uint16, int16
    #   with restriction that input and output dtypes must match.
    #   To get float32 output, we must cast the input to float32 here.
    float32_type = itk.Image[itk.F, input_image.ndim]
    input_image = itk.cast_image_filter(
        input_image, ttype=(type(input_image), float32_type)
    )
    args = (input_image, ) + args[1:]

    ref_filt = itk.SignedMaurerDistanceMapImageFilter.New(*args, **kwargs)
    wrapper = itk.PyImageFilter.New(input_image)
    if ref_filt.GetBackgroundValue() != 0:
        raise NotImplementedError(
            "only background_value=0 is currently supported"
        )

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
        # TODO: need boolean input for cuCIM implementation
        cu_input_array = cp.array(input_array).astype(bool, copy=False)

        output_image = wrapper.GetOutput()
        output_image.SetBufferedRegion(output_image.GetRequestedRegion())
        output_image.Allocate()
        output_array = itk.array_view_from_image(output_image)

        if ref_filt.GetUseImageSpacing():
            spacing = tuple(input_image.GetSpacing())[::-1]
        else:
            spacing = None
        cu_output_array = _signed_euclidean_distance_map(
            cu_input_array,
            spacing=spacing,
            squared_distance=ref_filt.GetSquaredDistance(),
            inside_is_positive=ref_filt.GetInsideIsPositive(),
        )
        output_array[:] = cu_output_array.get()[:]
    wrapper.SetPyGenerateData(generate_data)

    wrapper.Update()

    return wrapper.GetOutput()
