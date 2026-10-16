# DMADSAC — Article 2 Reproducibility Package

Public reproducibility materials for:

**Dynamic and Fixed Discrete Soft Actor-Critic Versus a Transparent Memory-Aware Urgency Heuristic: A Paired Five-Seed, 330-Day Study-Planning Simulation**

Target venue: IEEE Transactions on Learning Technologies.

## Overview

This repository contains the curated public research package for Article 2. It compares:

- Dynamic-DSAC
- Static-DSAC
- MAU-Heuristic

in a paired five-seed, 330-day synthetic study-planning simulation.

Official master seeds:

- `14040504`
- `1123252065`
- `1529594671`
- `997781685`
- `1719214619`

All learning, recall, workload, and stress outcomes reported by the study are simulator-derived. They should not be interpreted as validated effects on real learners.

## Repository structure

- `src/final330_pair/` — curated archival Dynamic-DSAC and Static-DSAC source package
- `src/heuristic/` — MAU-Heuristic implementation and validation code
- `protocols/final330/` — locked standardized assessment and validation protocol materials
- `protocols/heuristic_5seed/` — assessment/validation materials used with the heuristic benchmark
- `environment/` — Python and software-environment records
- `experiments/memory_sensitivity_v1/` — public-safe read-only memory-model sensitivity runner
- `results/memory_sensitivity_v1/` — sensitivity outputs
- `provenance/memory_sensitivity_v1/` — sanitized input hashes and run-status evidence
- `PUBLIC_PACKAGE_MANIFEST.csv` — SHA256 inventory of the public release package

## Main study results

At D330, mean equal-book expected recall was approximately:

- Dynamic-DSAC: `8.332`
- Static-DSAC: `7.447`
- MAU-Heuristic: `19.227`

Dynamic minus Static was approximately `+0.885` with a hierarchical 95% CI of `[-0.355, 2.139]`.

The MAU comparison is not dose matched. MAU also generated substantially more total study exposure, so its recall contrast must not be interpreted as evidence of superior efficiency per unit study time.

## Memory-model sensitivity

The locked assessment implementation was first reproduced with a maximum absolute question-level difference of zero.

The study then evaluated six post-hoc engineering perturbations around the locked assessment memory equation. Across these perturbations, the aggregate ordering remained:

`MAU > Dynamic > Static`

These perturbations are engineering robustness checks around the simulator measurement model. They are not validated human cognitive models.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/msosmanpour/DMADSAC.git
cd DMADSAC
```

### 2. Inspect the recorded environment

```bash
python --version
cat environment/PYTHON_VERSION.txt
```

On Windows PowerShell, `Get-Content environment/PYTHON_VERSION.txt` can be used instead of `cat`.

`environment/PIP_FREEZE.txt` records the package environment used for the curated research archive. Exact cross-platform installation is not guaranteed.

### 3. Inspect the public-safe sensitivity runner

```bash
python experiments/memory_sensitivity_v1/ARTICLE2_MEMORY_SENSITIVITY_PUBLIC_V1.py --help
```

The runner accepts:

- `--dsac-seed14040504-root`
- `--dsac-multiseed-root`
- `--mau-root`
- optional `--out-dir`

### 4. Re-score the sensitivity analysis

For Bash/macOS/Linux:

```bash
python experiments/memory_sensitivity_v1/ARTICLE2_MEMORY_SENSITIVITY_PUBLIC_V1.py \
  --dsac-seed14040504-root "<DSAC_SEED_14040504_ROOT>" \
  --dsac-multiseed-root "<DSAC_MULTI_SEED_ROOT>" \
  --mau-root "<MAU_5SEED_ROOT>" \
  --out-dir "<OUTPUT_DIRECTORY>"
```

For Windows PowerShell:

```powershell
python experiments/memory_sensitivity_v1/ARTICLE2_MEMORY_SENSITIVITY_PUBLIC_V1.py `
  --dsac-seed14040504-root "<DSAC_SEED_14040504_ROOT>" `
  --dsac-multiseed-root "<DSAC_MULTI_SEED_ROOT>" `
  --mau-root "<MAU_5SEED_ROOT>" `
  --out-dir "<OUTPUT_DIRECTORY>"
```

The angle-bracket values above are placeholders, not commands to run verbatim.

The runner verifies the SHA256 of the required question-level assessment inputs before analysis.

The large historical question-level source inputs are not currently distributed in this public repository. Their logical identities, hashes, and byte sizes are recorded in:

`provenance/memory_sensitivity_v1/INPUT_PROVENANCE_PUBLIC.csv`

Therefore, the repository supports source inspection, protocol verification, output auditing, and re-scoring when the verified inputs are available, but it is not currently a completely self-contained re-execution package.

## What can be reproduced or audited from this repository?

### Publicly supported

- inspection of the archived Dynamic/Static source
- inspection of the MAU heuristic source
- inspection of locked assessment protocols
- inspection of software-environment records
- verification of published sensitivity outputs
- verification of input SHA256 provenance
- execution of the public-safe sensitivity runner when matching inputs are supplied
- integrity checking of the public package

### Not guaranteed from this repository alone

- bit-for-bit replay of every historical 330-day execution
- restoration of historical SQLite database states
- restoration of large training checkpoints
- reconstruction of the production Azhin Hooshmand platform
- replay of private forensic audit logs
- reproduction from question-level inputs that are not distributed here

## Public/private boundary

This repository intentionally excludes:

- production databases
- operational Azhin Hooshmand user data
- credentials and secrets
- historical SQLite databases
- large model checkpoints
- workstation-specific path inventories
- private forensic audit records
- internal advisor correspondence
- manuscript working history
- Article 3 materials

The complete forensic archive is retained privately by the author.

## Authorship and software contribution

The public software and research-code package was authored by Mohammad Shovan Osmanpour with AI-assisted drafting and coding support from OpenAI ChatGPT. The doctoral supervisor and advisor listed on the associated manuscript did not contribute source code to this repository. Their manuscript roles and authorship are governed separately by the doctoral-program and journal authorship requirements.

## Important source-provenance note

The files under `src/final330_pair/` are curated archival reproducibility copies.

Names such as `FULL_...` and `STATIC_...` identify the curated research package; the filenames alone must not be treated as proof of the exact historical filesystem path loaded during execution.

## Integrity

Use `PUBLIC_PACKAGE_MANIFEST.csv` to verify the SHA256 and byte size of files in the curated public package.

The manifest must be regenerated whenever the public package changes.

## Citation

Citation metadata are provided in `CITATION.cff`.

After the associated journal article is published, the final article DOI should be added as the preferred citation.

## License

See `LICENSE`.

## Article separation

This repository contains Article 2 materials only.

Article 3 code, runtime provenance, executions, results, and forensic evidence are maintained separately and are not part of this repository.
