from __future__ import annotations

import sys
import time
from typing import Dict, Iterable

import numpy as np

from .aggregation import weighted_fedavg


def deterministic_updates(client_count: int, seed: int) -> tuple[list[Dict[str, np.ndarray]], list[int]]:
    rng = np.random.default_rng(seed)
    updates = [
        {
            "linear.weight": rng.normal(0, 0.05, size=(10, 32)).astype(np.float32),
            "linear.bias": rng.normal(0, 0.05, size=(10,)).astype(np.float32),
            "large.segmented": rng.normal(0, 0.01, size=(10003,)).astype(np.float32),
        }
        for _ in range(client_count)
    ]
    counts = [100 + 7 * index for index in range(client_count)]
    return updates, counts


def tenseal_weighted_fedavg(updates, sample_counts, poly_modulus_degree=8192, scale_bits=40):
    try:
        import tenseal as ts
    except ImportError as exc:
        raise RuntimeError(
            "TenSEAL is not installed in the Python currently running this script. "
            f"Executable: {sys.executable}. Create rq1-lee-he, then run the E0 "
            "command through 'conda run -n rq1-lee-he python ...'."
        ) from exc

    started = time.perf_counter()
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    context.global_scale = 2**scale_bits
    key_seconds = time.perf_counter() - started
    slots = poly_modulus_degree // 2
    total = float(sum(sample_counts))
    output = {}
    encryption_seconds = aggregation_seconds = decryption_seconds = 0.0
    ciphertext_bytes = 0

    for key in updates[0]:
        chunks = []
        for update, count in zip(updates, sample_counts):
            flat = np.asarray(update[key], dtype=np.float64).ravel() * (count / total)
            encrypted = []
            for start in range(0, flat.size, slots):
                tick = time.perf_counter()
                value = ts.ckks_vector(context, flat[start : start + slots])
                encryption_seconds += time.perf_counter() - tick
                ciphertext_bytes += len(value.serialize())
                encrypted.append(value)
            chunks.append(encrypted)

        tick = time.perf_counter()
        aggregate = chunks[0]
        for client in chunks[1:]:
            aggregate = [left + right for left, right in zip(aggregate, client)]
        aggregation_seconds += time.perf_counter() - tick

        tick = time.perf_counter()
        decrypted = np.concatenate([np.asarray(value.decrypt()) for value in aggregate])
        decryption_seconds += time.perf_counter() - tick
        output[key] = decrypted[: updates[0][key].size].reshape(updates[0][key].shape)

    timing = {
        "key_generation_seconds": key_seconds,
        "encryption_seconds": encryption_seconds,
        "aggregation_seconds": aggregation_seconds,
        "decryption_seconds": decryption_seconds,
        "ciphertext_bytes": ciphertext_bytes,
    }
    return output, timing
