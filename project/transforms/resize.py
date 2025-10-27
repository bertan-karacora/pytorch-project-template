import logging
import math

import torchvision.transforms.v2 as tv_transforms


_LOGGER = logging.getLogger(__name__)
MODES_INTERPOLATION = tv_transforms.InterpolationMode


def resize_shape_to_multiple_of_base(shape, size_resized_min, base):
    height, width = shape[2:]
    scale = size_resized_min / min(height, width)

    def resize_dim(dim, scale, base, size_resized_min):
        scaled_in_base = scale * dim / base
        resized = round(scaled_in_base) * base

        if resized < size_resized_min:
            resized = (math.ceil(scaled_in_base)) * base

        return resized

    height_resized = resize_dim(height, scale, base, size_resized_min)
    width_resized = resize_dim(width, scale, base, size_resized_min)

    shape_resized = shape[:2] + (height_resized, width_resized)

    return shape_resized


class ResizeToMultipleOfBase(tv_transforms.Transform):
    """Resize input tensor to a shape where height and width are multiples of base."""

    def __init__(self, base, size_resized_min, mode_interpolation=None, use_antialiasing=True):
        super().__init__()

        self.base = base
        self.mode_interpolation = mode_interpolation or MODES_INTERPOLATION.BILINEAR
        self.size_resized_min = size_resized_min
        self.use_antialiasing = use_antialiasing

    def _transform(self, inpt):
        shape_resized = resize_shape_to_multiple_of_base(inpt.shape, size_resized_min=self.size_resized_min, base=self.base)
        output = tv_transforms.functional.resize(inpt, shape_resized[2:], interpolation=self.mode_interpolation, antialias=self.use_antialiasing)
        return output
