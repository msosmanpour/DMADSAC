# DMADSAC — Article 2 Reproducibility Package

This repository is the public curated reproducibility package for:

**Dynamic and Fixed Discrete Soft Actor-Critic Versus a Transparent
Memory-Aware Urgency Heuristic: A Paired Five-Seed, 330-Day
Study-Planning Simulation**

Target journal: IEEE Transactions on Learning Technologies.

## Methods

The study compares:

- Dynamic-DSAC
- Static-DSAC
- MAU-Heuristic

using a paired five-seed, 330-day synthetic study-planning simulation.

Official seeds:

- 14040504
- 1123252065
- 1529594671
- 997781685
- 1719214619

All reported learning and stress outcomes are simulator-derived and are not
validated estimates of effects on real students.

## Public reproducibility contents

The repository contains curated Article 2 materials including:

- `src/final330_pair/` — archived Dynamic/Static source package;
- `src/heuristic/` — MAU heuristic source package;
- `protocols/` — locked protocol materials that are safe for public release;
- `environment/` — software-environment records;
- `experiments/memory_sensitivity_v1/` — public-safe sensitivity code;
- `results/memory_sensitivity_v1/` — compact sensitivity outputs;
- `provenance/memory_sensitivity_v1/` — sanitized input provenance.

## Memory-model sensitivity

The locked assessment model was reproduced exactly at question level before
the perturbation analysis.

The sensitivity analysis re-scores already generated D330 anchor
question-level assessment records. It does not retrain a policy and does not
rerun the 330-day scheduler.

This is a robustness analysis of the simulator measurement model, not
empirical validation of a human cognitive model.

## Reproducibility boundary

This public repository intentionally excludes:

- production databases;
- production Azhin Hooshmand data;
- credentials or secrets;
- raw historical SQLite databases;
- large checkpoints;
- absolute local-machine paths;
- private forensic audit logs;
- historical execution evidence containing workstation paths;
- internal advisor correspondence;
- manuscript working history;
- Article 3 materials.

A complete forensic archive is retained privately by the authors.

## Article separation

This repository contains Article 2 material only.

Article 3 source, execution provenance, experimental runs, and results are
not part of this repository.
