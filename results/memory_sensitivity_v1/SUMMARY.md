# Article 2 read-only memory-model sensitivity

This analysis re-scores already generated D330 anchor question-level records. It does not rerun any scheduler or policy.

- Locked-model maximum absolute question-level reproduction difference: 0
- Locked aggregate: Dynamic=8.332, Static=7.447, MAU=19.227

## Aggregate sensitivity

| Scenario | Dynamic | Static | MAU | Dyn-Static | MAU-Static | MAU-Dynamic | Rank |
|---|---:|---:|---:|---:|---:|---:|---|
| LOCKED | 8.332 | 7.447 | 19.227 | +0.885 | +11.780 | +10.895 | MAU > Dynamic > Static |
| SHORT_RETENTION_0p5x | 4.578 | 4.100 | 12.381 | +0.477 | +8.281 | +7.803 | MAU > Dynamic > Static |
| LONG_RETENTION_2x | 13.655 | 12.348 | 26.225 | +1.307 | +13.877 | +12.570 | MAU > Dynamic > Static |
| LONG_TAIL_SHAPE_0p75 | 9.536 | 8.616 | 19.630 | +0.920 | +11.014 | +10.094 | MAU > Dynamic > Static |
| STEEP_TAIL_SHAPE_1p25 | 7.757 | 6.913 | 19.135 | +0.844 | +12.222 | +11.378 | MAU > Dynamic > Static |
| LOW_REPETITION_REVIEW | 4.674 | 4.196 | 11.895 | +0.478 | +7.699 | +7.221 | MAU > Dynamic > Static |
| HIGH_REPETITION_REVIEW | 11.062 | 9.837 | 22.044 | +1.225 | +12.206 | +10.981 | MAU > Dynamic > Static |

Interpretation boundary: these are post-hoc engineering perturbations around the locked memory equation, not empirically validated cognitive models. They test measurement-model robustness while holding realized exposure histories fixed.
