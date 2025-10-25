from typing import Tuple

from .tensor import Tensor
from .tensor_functions import rand
from typing import Optional
from .tensor_functions import Function, tensor

from .autodiff import Context


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    raise NotImplementedError("Need to implement for Task 4.3")


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Pooled tensor
    """
    # TODO: Implement for Task 4.3.
    raise NotImplementedError("Need to implement for Task 4.3")


class Max(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Forward of max should be max reduction"""
        # TODO: Implement for Task 4.4.
        raise NotImplementedError("Need to implement for Task 4.4")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward of max should be argmax (see argmax function)"""
        # TODO: Implement for Task 4.4.
        raise NotImplementedError("Need to implement for Task 4.4")


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction over a dimension"""
    # TODO: Implement for Task 4.4.
    raise NotImplementedError("Need to implement for Task 4.4")


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor"""
    # TODO: Implement for Task 4.4.
    raise NotImplementedError("Need to implement for Task 4.4")


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor"""
    # TODO: Implement for Task 4.4.
    raise NotImplementedError("Need to implement for Task 4.4")


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor
    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations """
    # TODO: Implement for Task 4.4.
    raise NotImplementedError("Need to implement for Task 4.4")


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Pooled tensor
    """
    # TODO: Implement for Task 4.4.
    raise NotImplementedError("Need to implement for Task 4.4")


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise

    Args:
    ----
        input: input tensor
        rate: probability [0, 1) of dropping out each position
        ignore: skip dropout, i.e. do nothing at all

    Returns:
    -------
        tensor with random positions dropped out
    """
    # TODO: Implement for Task 4.4.
    raise NotImplementedError("Need to implement for Task 4.4")