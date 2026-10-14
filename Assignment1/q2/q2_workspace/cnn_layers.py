"""Q2 implementations: NumPy only, no automatic differentiation.

All spatial tensors use NCHW. Implement only the four functions below.
The geometry is fixed: 3x3 cross-correlation, stride 1, zero padding 1;
2x2 max pooling, stride 2, no padding. Do not generalize the API.

Inputs are finite float32 or float64 arrays. Inputs belonging to a call have
matching dtypes. Preserve that dtype; never modify input arrays in place.
Support arbitrary positive batch/channel sizes and H,W >= 2. Pooling requires
even H,W. Noncontiguous NumPy views are valid inputs. A cache is an opaque,
per-forward object: choose its structure, but do not use global state.
A backward call returns fresh gradients, does not update parameters, and does
not divide by the batch size. The incoming gradient already incorporates any
loss reduction. Separate outstanding caches must work independently.
"""
from typing import Any
import numpy as np


def conv2d_forward(x: np.ndarray, weight: np.ndarray,
                   bias: np.ndarray) -> tuple[np.ndarray, Any]:
    """Apply 3x3 cross-correlation with stride 1 and zero padding 1.

    Parameters
    ----------
    x : ndarray, shape (N, C_in, H, W)
    weight : ndarray, shape (C_out, C_in, 3, 3)
        Do not flip the kernel. All channels contribute to each output filter.
    bias : ndarray, shape (C_out,)
        Add one scalar to every output position in the corresponding channel.

    Returns
    -------
    out : ndarray, shape (N, C_out, H, W)
        Same floating dtype as x.
    cache : object
        Information your conv2d_backward needs. Its structure is your choice.
        Do not update weights or biases in this function.
    """
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    dtype = x.dtype

    # No padding on each spatial side
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))

    # Cross-correlation as a sum over the 9 kernel offsets (i, j)
    out = np.zeros((N, C_out, H, W), dtype=dtype)
    for i in range(3):
        for j in range(3):
            patch = xp[:, :, i:i + H, j:j + W]  
            out += np.einsum('nchw,oc->nohw', patch, weight[:, :, i, j], optimize=True)
    out += bias.reshape(1, C_out, 1, 1)

    # Copy the weights 
    cache = (xp, weight.copy(), H, W)
    return out.astype(dtype, copy=False), cache


def conv2d_backward(dout: np.ndarray, cache: Any
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backpropagate through the matching conv2d_forward call.

    Parameters
    ----------
    dout : ndarray, shape (N, C_out, H, W)
        Upstream derivative of a scalar loss with respect to the output.
    cache : object returned by your conv2d_forward

    Returns
    -------
    dx : ndarray, shape (N, C_in, H, W)
    dweight : ndarray, shape (C_out, C_in, 3, 3)
    dbias : ndarray, shape (C_out,)
        Sum contributions over the batch and spatial axes; do not average.
        Return (dx, dweight, dbias), in that order, in the input dtype.
        Overlapping receptive fields must accumulate their input gradients.
    """
    xp, weight, H, W = cache
    dtype = xp.dtype
    dbias = dout.sum(axis=(0, 2, 3))
    dweight = np.zeros_like(weight)
    dxp = np.zeros_like(xp)

    for i in range(3):
        for j in range(3):
            patch = xp[:, :, i:i + H, j:j + W]
            dweight[:, :, i, j] = np.einsum('nohw,nchw->oc', dout, patch, optimize=True)

            dxp[:, :, i:i + H, j:j + W] += np.einsum('nohw,oc->nchw', dout, weight[:, :, i, j], optimize=True)

    # Remove the padding border
    dx = dxp[:, :, 1:-1, 1:-1].copy()
    return dx.astype(dtype, copy=False), dweight.astype(dtype, copy=False), dbias.astype(dtype, copy=False)


def maxpool2d_forward(x: np.ndarray) -> tuple[np.ndarray, Any]:
    """Apply 2x2 max pooling with stride 2 and no padding.

    Parameters
    ----------
    x : ndarray, shape (N, C, H, W)
        H and W must be even. Values may be negative or repeated.

    Returns
    -------
    out : ndarray, shape (N, C, H//2, W//2), same dtype as x
    cache : object
        Information your maxpool2d_backward needs. At tied maxima, select
        only the first maximum in row-major order within the 2x2 window:
        top-left, top-right, bottom-left, bottom-right. Cache this selection.
    """
    N, C, H, W = x.shape
    if H % 2 or W % 2:
        raise ValueError('maxpool2d_forward requires even H and W')
    H2, W2 = H // 2, W // 2

    # Split each spatial axis into (block, offset)
    windows = (x.reshape(N, C, H2, 2, W2, 2)
                .transpose(0, 1, 2, 4, 3, 5)
                .reshape(N, C, H2, W2, 4))

    index = np.argmax(windows, axis=-1)   
    out = np.take_along_axis(windows, index[..., None], axis=-1)[..., 0]

    cache = (index, x.shape, x.dtype)
    return out.astype(x.dtype, copy=False), cache


def maxpool2d_backward(dout: np.ndarray, cache: Any) -> np.ndarray:
    """Route each upstream gradient to its selected maximum.

    Parameters
    ----------
    dout : ndarray, shape (N, C, H//2, W//2)
    cache : object returned by your maxpool2d_forward

    Returns
    -------
    dx : ndarray, shape (N, C, H, W), same dtype as the forward input
        One selected position receives the upstream gradient per window;
        all other positions receive zero. Do not split a tied gradient or
        send it to every tied maximum. Do not average or update any arrays.
    """
    index, shape, dtype = cache
    N, C, H, W = shape
    H2, W2 = H // 2, W // 2

    dwindows = np.zeros((N, C, H2, W2, 4), dtype=dtype)
    np.put_along_axis(dwindows, index[..., None], dout[..., None].astype(dtype), axis=-1)

    # Undo the window reshape from the forward pass
    dx = (dwindows.reshape(N, C, H2, W2, 2, 2)
                  .transpose(0, 1, 2, 4, 3, 5)
                  .reshape(N, C, H, W))
    return dx
