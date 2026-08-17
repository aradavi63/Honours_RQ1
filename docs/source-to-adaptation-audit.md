# Source-to-adaptation audit

This audit records how the pinned reference repositories relate to the RQ1 shared harness. Its purpose is to prevent an adaptation from being described as copied code or a full reproduction. The authoritative machine-readable register is `experiments/source_adaptation_audit.yaml`.

## Relationship labels

- **Runtime dependency:** the harness reads or imports an artefact from the submodule while running.
- **Algorithm adaptation:** the harness independently implements selected operations found in the reference.
- **Conceptual baseline:** the reference supports the threat model, but the harness method is a simpler or standard baseline.
- **Not reproduced:** important reference behavior is deliberately outside the current controlled experiment.

## Summary

| Mechanism | Exact reference anchor | Harness anchor | Relationship | Safe description |
|---|---|---|---|---|
| FedAvg | `FedAvg/utils.py::average_weights`; `FedAvg/fed_avg.py::FedAvg` | `aggregation.py::weighted_fedavg`; `training.py::train_client` | Independent controlled reimplementation | Common weighted-FedAvg baseline informed by the pinned reference |
| Lee CKKS | `network_node.py::encryptWeights/aggregateEncryptedWeights`; `server_node.py::create_ckks_context/decryptWeights` | `e0.py::tenseal_weighted_fedavg` | Structural TenSEAL adaptation | Lee-inspired server-initiated CKKS aggregation adapter |
| FedSHE CKKS | `SegCKKS.py`; `ModDict.json`; `server.py::Server.FedAvg` | `fedshe.py::fedshe_ckks_weighted_fedavg` | Parameter and algorithm adaptation; direct parameter-file dependency | Adapted segmented FedSHE CKKS using pinned parameters and published rounding |
| Label poisoning | `poison_data.py`; `label_replacement.py`; `class_flipping_methods.py`; `attack_timing.py` | `poisoning.py`; `training.py::train_client`; `run_e2.py` | Controlled threat-model adaptation | Targeted label-flipping attack adapted from DataPoisoning_FL |
| Gradient inversion | `OP-GIA/IG/inverting.py`; `GradientReconstructor.reconstruct`; `reconstruction_costs` | `inversion.py`; `run_e3.py` | Simplified gradient-matching baseline | DLG/iDLG-style baseline informed by GIA/OP-GIA, not OP-GIA reproduction |
| Update membership | `FedMIA/main.py::get_cos_score`; `mia_attack_auto.py::lira_attack_ldh_cosine` | `membership.py`; `run_e4a.py` | FedMIA-style algorithm adaptation | Spatial-temporal update attack informed by FedMIA |
| Black-box membership | `FedMIA/main.py::get_all_losses`; `mia_attack_auto.py::common_attack` | `membership.py::per_sample_scores`; `run_e4b.py` | Standard baseline with reference overlap | Standard loss/confidence black-box membership baseline |

## What is directly reused

The only current runtime read from attack/mechanism source code is FedSHE's pinned `ModDict.json`. No harness module imports executable Python functions from the six submodules. The submodules otherwise provide fixed reference implementations against which the independent adapters can be inspected and validated.

## Material changes that must be disclosed

1. **FedAvg:** the references generally average clients equally; the harness uses sample-count-weighted model deltas.
2. **Lee:** the harness retains CKKS encryption/addition/decryption but omits Lee's route simulation, GUI and distributed noise cancellation. It is not the complete protocol.
3. **FedSHE:** segmentation, Pyfhel operations, pinned parameters and three-decimal decryption rounding are retained. Pre-encryption sample weighting and common metrics are added.
4. **Poisoning:** local source-label replacement is retained, but the label pair, schedules, deterministic selection, model and metrics are controlled by this study.
5. **Gradient inversion:** the harness uses a small squared-gradient-matching optimiser. It does not implement OP-GIA's complete reconstruction machinery, priors, regularisation or defences.
6. **FedMIA:** per-sample gradient cosine, cross-client comparison, multi-round evidence and Gaussian non-target modelling are retained conceptually. The full shadow-client likelihood-ratio pipeline is not reproduced.
7. **Black-box membership:** this is a standard final-model loss/confidence baseline, not FedMIA itself.

## Thesis wording

Use: “The pinned repositories were used as reference implementations. Their relevant mechanisms and threat models were adapted into a common harness so that model, dataset, partitions, seeds and evaluation were held constant.”

Do not use: “All original repository attacks were run unchanged” or “the harness reproduces OP-GIA/FedMIA.”

## Validation still required

- Run a small original-repository smoke test for each feasible submodule.
- Define one behavior-level equivalence test per adapter, rather than expecting numerical equality across different models and datasets.
- Record environment, Git commit and package provenance in every final result.
- Revisit this audit if any adapter changes before confirmatory experiments.
