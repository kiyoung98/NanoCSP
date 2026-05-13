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

## Submission

Submit by opening a PR.
