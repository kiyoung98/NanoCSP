![NanoCSP](assets/nanocsp.png)

# NanoCSP

A slowrun benchmark for crystal structure prediction on the [MP20
polymorph split](https://arxiv.org/abs/2509.12178). Inspired by [NanoGPT
Slowrun](https://github.com/qlabs-eng/slowrun).

## Setting

- **Task**: given a composition, generate a crystal structure (lattice +
  fractional coordinates).
- **Dataset**: MP20 polymorph split — polymorphs of the same composition
  stay on the same side, so models must reproduce per-composition diversity.
- **Metric**: [METRe](https://arxiv.org/abs/2509.12178) — fraction of test
  references with at least one matching generation under pymatgen
  `StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10°)`. Higher is better.
- **Budget**: single RTX 3090, 12 h wall-clock from `python train.py`
  invocation to exit.

## Reproduce

```bash
git clone https://github.com/kiyoung98/NanoCSP.git && cd NanoCSP
pip install -r requirements.txt
python prepare_data.py
python train.py
python evaluate.py --samples_dir runs/<run>/test_samples
```

## World Record History

PRs that beat the current METRe under the 12 h budget are appended below.

| # | METRe ↑ | Description | Date | Time | Script | Contributors |
| - | - | - | - | - | - | - |
| 1 | 67.93% | Baseline: reproduce [DiffCSP](https://arxiv.org/abs/2309.04475), DDPM diffusion on CSPNet | 05/13/26 | 711 min | [Script](https://github.com/kiyoung98/NanoCSP/blob/86751daac2632c924199107bbfa85103b5b81c15/train.py) | [@kiyoung98](https://github.com/kiyoung98) |
| 2 | 69.83% | Reproduce [FlowMM](https://arxiv.org/abs/2406.04713), Riemannian flow matching | 05/13/26 | 719 min | [Script](https://github.com/kiyoung98/NanoCSP/blob/e9cb9f50cfaea7b73dcaa87bd0b6333c51bedf85/train.py) | [@kiyoung98](https://github.com/kiyoung98) |
| 3 | 73.15% | Reproduce [OMatG](https://github.com/FERMat-ML/OMatG), linear ODE flow matching | 05/13/26 | 700 min | [Script](https://github.com/kiyoung98/NanoCSP/blob/07d27b49594a15a77874f995d59bd6cdf4235a9b/train.py) | [@kiyoung98](https://github.com/kiyoung98) |

## Submission

Submit by opening a PR.
