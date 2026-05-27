# Integration Contract — `marts.v_attrition_features`

Schema dumped from `pa-warehouse-prod.marts.v_attrition_features` at **2026-05-27 12:49 UTC** via BigQuery `INFORMATION_SCHEMA` (`docs/_schema_dump.py`). Ground-truth contract for the retention-prediction loader (Story 1.1.8).

**Row count in schema dump:** 10 columns

| # | column_name | data_type | is_nullable |
|---|-------------|-----------|-------------|
| 1 | `employee_id` | `STRING` | YES |
| 2 | `snapshot_date` | `DATE` | YES |
| 3 | `tenure_months` | `INT64` | YES |
| 4 | `performance_tier` | `STRING` | YES |
| 5 | `compa_ratio` | `FLOAT64` | YES |
| 6 | `is_critical_role` | `BOOL` | YES |
| 7 | `successor_count` | `INT64` | YES |
| 8 | `age_at_window_close` | `INT64` | YES |
| 9 | `gender` | `STRING` | YES |
| 10 | `voluntary_exit_label` | `BOOL` | YES |

---

_Timestamp reflects the actual moment of last regeneration (UTC). To verify the current mart schema matches this contract, re-run `python docs/_schema_dump.py` — any column-level diff surfaces in `git status`. Story 1.1.8 (cross-project contract test) automates this comparison in CI once the Story 1.1 loader ships._
