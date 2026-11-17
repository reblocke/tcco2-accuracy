"""Evaluate a saved draw batch with the native normal mixture."""

import ctypes

import numpy as np
from scipy.special import logsumexp

from tcco2_accuracy.core.future_setting import PredictiveKernel


def direct_batch_density(library):
    # Native evaluator has the same unnormalized normal components as draw_logpdf.
    native = ctypes.CDLL(str(library)).mixture
    vec = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    native.argtypes = [ctypes.c_int, ctypes.c_int, *([vec] * 5)]
    native.restype = ctypes.c_int

    def batch(b, order, offsets):
        k = PredictiveKernel(b[None, :, :], order=order)
        v = np.empty(len(offsets))
        if native(
            k.mean.size,
            len(v),
            k.mean.ravel(),
            (1 / k.sd).ravel(),
            k.log_factor.ravel(),
            offsets,
            v,
        ):
            raise MemoryError()
        tiny = v < np.finfo(float).tiny * k.mean.size
        v[~tiny] = np.log(v[~tiny]) - np.log(len(b))
        for j in np.flatnonzero(tiny):
            v[j] = logsumexp(k.draw_logpdf(offsets[j])) - np.log(len(b))
        return v

    return batch
