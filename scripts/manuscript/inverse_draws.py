"""Read a four-chain selection and authenticate its archived NPY in a single sequential pass."""

import hashlib

import numpy as np


def select_saved(path, expected_sha, retained, count):
    header = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        header.shape != (4, retained, 5)
        or header.dtype != np.float64
        or not header.flags.c_contiguous
        or not 16 <= count <= retained
    ):
        raise ValueError("Unexpected saved chain layout or draw count")
    offset, row_bytes = header.offset, header.strides[1]
    del header
    within = np.linspace(0, retained - 1, count, dtype=int)
    indices = (np.arange(4)[:, None] * retained + within).ravel()
    result = np.empty((len(indices), 5))
    check = hashlib.sha256()
    with path.open("rb") as stream:
        check.update(stream.read(offset))
        step = 262144
        for start in range(0, 4 * retained, step):
            stop = min(start + step, 4 * retained)
            block = stream.read((stop - start) * row_bytes)
            if len(block) != (stop - start) * row_bytes:
                raise ValueError("Incomplete saved chain")
            check.update(block)
            left, right = indices.searchsorted([start, stop])
            result[left:right] = np.frombuffer(block, dtype=np.float64).reshape(-1, 5)[
                indices[left:right] - start
            ]
        if stream.read(1) or check.hexdigest() != expected_sha:
            raise ValueError("Saved chain differs from its checksum")
    return result.reshape(4, count, 5)
