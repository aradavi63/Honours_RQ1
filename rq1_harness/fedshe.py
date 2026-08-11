from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np

from .aggregation import weighted_fedavg

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS_PATH = ROOT / "repos" / "FedSHE" / "ModDict.json"


def fedshe_plain_weighted_fedavg(updates, sample_counts):
    """FedSHE plaintext control using the study's weighted FedAvg contract."""
    return weighted_fedavg(updates, sample_counts)


def load_fedshe_ckks_parameters(
    security_level: str = "128",
    multiplication_depth: str = "0",
    polynomial_degree: str = "16384",
    path: Path = PARAMETERS_PATH,
) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        parameter_tree = json.load(stream)
    try:
        return parameter_tree[security_level][multiplication_depth][polynomial_degree]
    except KeyError as exc:
        raise ValueError(
            "FedSHE has no CKKS parameters for "
            f"security={security_level}, depth={multiplication_depth}, "
            f"degree={polynomial_degree}"
        ) from exc


def _ciphertext_size(ciphertext) -> int:
    for method_name in ("to_bytes", "toBytes"):
        method = getattr(ciphertext, method_name, None)
        if method is not None:
            return len(method())
    return int(sys.getsizeof(ciphertext))


def fedshe_ckks_weighted_fedavg(
    updates: list[Mapping[str, np.ndarray]],
    sample_counts: list[int],
    security_level: str = "128",
    multiplication_depth: str = "0",
    polynomial_degree: str = "8192",
    round_decimals: int = 3,
):
    """Segmented Pyfhel CKKS aggregation adapted from FedSHE's SegCKKS.py.

    FedSHE divides an equal-client sum after decryption. Scaling each client by
    its sample fraction before encryption preserves that result for equal client
    sizes and also implements correct weighted FedAvg for unequal sizes.
    """
    try:
        from Pyfhel import Pyfhel
    except ImportError as exc:
        raise RuntimeError(
            "Pyfhel is required for the FedSHE CKKS backend. Use the "
            "rq1-fedshe-ckks Linux environment; the pinned Pyfhel build is not "
            "available in the current Windows environment."
        ) from exc

    reference = weighted_fedavg(updates, sample_counts)
    parameters = load_fedshe_ckks_parameters(
        security_level, multiplication_depth, polynomial_degree
    )
    started = time.perf_counter()
    he = Pyfhel()
    status = he.contextGen(**parameters)
    if status is False:
        raise RuntimeError(f"Pyfhel rejected FedSHE CKKS parameters: {parameters}")
    he.keyGen()
    key_seconds = time.perf_counter() - started

    slots = int(he.get_nSlots())
    if slots < 1:
        raise RuntimeError("Pyfhel returned an invalid CKKS slot count")
    total = float(sum(sample_counts))
    output: dict[str, np.ndarray] = {}
    encryption_seconds = aggregation_seconds = decryption_seconds = 0.0
    ciphertext_bytes = 0

    for key, expected in reference.items():
        clients = []
        for update, count in zip(updates, sample_counts):
            flat = np.asarray(update[key], dtype=np.float64).ravel() * (count / total)
            encrypted_blocks = []
            for block_id in range(math.ceil(flat.size / slots)):
                block = flat[block_id * slots : (block_id + 1) * slots]
                tick = time.perf_counter()
                plaintext = he.encodeFrac(block)
                ciphertext = he.encryptPtxt(plaintext)
                encryption_seconds += time.perf_counter() - tick
                ciphertext_bytes += _ciphertext_size(ciphertext)
                encrypted_blocks.append(ciphertext)
            clients.append(encrypted_blocks)

        tick = time.perf_counter()
        aggregate = clients[0]
        for client in clients[1:]:
            for block_id, ciphertext in enumerate(client):
                aggregate[block_id] += ciphertext
        aggregation_seconds += time.perf_counter() - tick

        tick = time.perf_counter()
        decrypted = np.concatenate(
            [np.asarray(he.decryptFrac(ciphertext)) for ciphertext in aggregate]
        )
        decryption_seconds += time.perf_counter() - tick
        # FedSHE's dec_vector explicitly rounds every decrypted value to three
        # decimal places. Keep that behavior visible because it can affect
        # aggregation correctness and downstream model utility.
        decrypted = np.round(decrypted, decimals=round_decimals)
        output[key] = decrypted[: expected.size].reshape(expected.shape)

    timing = {
        "key_generation_seconds": key_seconds,
        "encryption_seconds": encryption_seconds,
        "aggregation_seconds": aggregation_seconds,
        "decryption_seconds": decryption_seconds,
        "ciphertext_bytes": ciphertext_bytes,
    }
    return output, timing
