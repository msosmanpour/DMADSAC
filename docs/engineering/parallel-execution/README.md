# CPU parallelism and memory-aware scheduling: archived evidence

This folder documents **real completed research-engineering executions** using Python 3.12 on a Windows laptop (12 logical CPU threads, approximately 16 GB RAM). These are engineering records from a synthetic learner benchmark, **not an independently reproduced public demonstration of the full experiment**.

## What ran concurrently

- Three locked seeds, each with **MAU** and **EarliestDue**, were dispatched through a `ThreadPoolExecutor(max_workers=workers)` whose worker tasks launch separate **Python subprocesses** in independent cloned projects. The thread pool coordinates the subprocesses; it is **not** GPU or multithreaded Python numerical execution.
- Receipt intervals verify a **maximum of 4 simultaneously running heuristic subprocesses** across six method-seed tasks. All 6 received an exit code of 0 and have 330-day, 33,000-learner-day receipts.
- Memory-heavy **Dynamic-DSAC** and **Static-DSAC** were executed sequentially; the receipts verify maximum overlapping DSAC process count **1** across six tasks. This prevented knowingly co-scheduling two ~8.3–8.5 GiB peak-working-set jobs on a 16 GB laptop.
- An earlier Seed-2 run recorded MAU/EarliestDue wall-clock overlap of **4.867 hours**, even though the larger overall V35 run ended partially after a separate DSAC memory-pressure stop. Do **not** label that entire V35 run successful.

## Measured evidence and audit trail

`archived_execution_metrics.csv` contains the per-process timestamp intervals (UTC), measured elapsed hours, peak process working-set and source algorithm hashes derived from checked receipts. `evidence_summary.json` retains the original archive digests and verified concurrency properties; `verified_orchestrator_excerpt.txt` shows the actual concurrency logic taken from the pinned V41 orchestration code.

These measurements show genuine overlap and a resource-aware scheduling design; **no controlled serial comparison was run**, so a numerical speedup factor or CPU utilization improvement is **not claimed**. Low-RAM guards, per-method database isolation, integrity receipts, compact evidence packaging and failure-preserving design were used in the original workflow. Historical runtime source hash provenance remains qualified for some earlier research steps.

## Reproducibility and privacy boundary

This folder intentionally contains **no source databases, private learner data, Django configuration, model checkpoints, Windows user paths, credentials, raw logs, or executable simulation projects**. The complete archives remain private and can be checked by their SHA256 values in `evidence_summary.json`. Public metrics alone cannot independently verify that those private original files exist: request reviewed evidence or a controlled demonstration if needed.

## Live demonstration (not the archived study)

Run `python parallel_demo.py` from this directory for a tiny, self-contained **illustration of the same subprocess/concurrency mechanism**. Its synthetic toy work does not reproduce the 330-day benchmark and should not be described as original study evidence.
