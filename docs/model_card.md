# Model Card — retention-prediction champion (v0.5)

*Model cards follow the [Mitchell et al. 2019](https://arxiv.org/abs/1810.03993) structure. This card pairs with [`data_card.md`](data_card.md) (the data) and [`methodology.md`](methodology.md) (the derivations).*

> **⚠ Synthetic data. Decision support, not decision making.** This model surfaces
> candidates for *additional human review*. It must never drive an employment
> decision. See [Out-of-Scope Use](#out-of-scope--forbidden-use).

---

## Model details

| | |
|---|---|
| **Name / registry URI** | `rp-champion` → `models:/rp-champion/Production` (MLflow) |
| **Version** | v0.5 (Loop 2) |
| **Type** | Gradient-Boosted Trees — XGBoost `XGBClassifier`, wrapped in `RetentionModel` |
| **Cohort** | `hybrid` (HRIS features + survey signal) |
| **Selection rule** | Calibration gate (Brier ≤ base-rate Brier `β·(1−β)`) → highest AUC-PR among eligible cells. See [`champion.py`](../src/retention/evaluation/champion.py). |
| **Hyperparameters** | Conservative defaults — `n_estimators=100`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`. **No class re-weighting** (`scale_pos_weight` unset) — this is deliberate: it is *why* GBM stays calibrated and clears the gate, where LR/EBM (balanced weights) fail it. Hyperparameter tuning is deferred to Loop 3. |
| **Output** | `predict_proba(X)[:, 1]` = P(voluntary exit). Operating threshold chosen by F2 (recall-weighted) sweep on validation. |
| **License / owner** | MIT · Nicolas Duarte |

---

## Intended use

- **Primary use:** rank employees by modeled voluntary-exit risk so a People
  Analytics / HRBP team can **prioritise human-led retention conversations** —
  decision *support*.
- **Primary users:** People Analytics teams, HRBPs, with analyst oversight.
- **Scope:** a single organisation's HRIS + engagement-survey snapshot matching
  the [integration contract](integration_contract.md).

## Out-of-scope / forbidden use

This model must **not** be used for, and is not validated for:
- **EEOC-covered employment decisions** (hiring, firing, promotion, compensation).
- **Automated employment decision tools** under **NYC Local Law 144 (AEDT)** without an independent bias audit.
- **EU AI Act Annex III** high-risk employment/worker-management systems.
- Any pipeline that **automates** an action on an employee from the model's output without a human in the loop.

---

## Training & evaluation data

- **Source:** synthetic `marts.v_attrition_features` from [pa-warehouse](https://github.com/nicolasduarte-data/pa-warehouse) — **no real employee data**. See [`data_card.md`](data_card.md).
- **Size / grain:** 1,278 employees, **one row per employee**, 13 features (10 HRIS + 3 survey-signal). Base rate ≈ 18.8 % voluntary exit (**39 positive examples**).
- **Split — read this:** a **single cross-section** (every row shares one `snapshot_date`), so the 70/15/15 train/val/test split is a **deterministic, leak-free holdout partitioned by `employee_id` — NOT a temporal split.** One row per employee means no employee straddles splits, so the estimate is honest, but it does **not** test month-T → month-T+1 generalisation. True temporal validation awaits time-series snapshots.
- **Leakage controls:** preprocessors fit on train only; operating point and champion selected on validation; the test set is touched exactly once. A 4-layer leakage gate (allowlist / semantic / empirical / pattern) guards the feature set.

---

## Performance

*Held-out **test** set, champion (GBM × hybrid). Validation comparison reported with 5,000-resample paired bootstrap CIs.*

| Metric | Value | Note |
|---|---|---|
| AUC-PR (test) | **0.274** | vs no-skill baseline ≈ 0.188 |
| ECE (test) | **0.048** | calibration error |
| Brier (test) | 0.173 | *marginal* — just above the base-rate bar; reported, not hidden |
| Precision@10 % | ≈ 0.35 | the operational metric (hybrid GBM) |
| Nested 5×5 CV AUC-PR | **0.269 ± 0.024** | unbiased estimate (sample std) |

**Honest comparison verdict:** across the six model×cohort cells, **every 95 %
bootstrap CI overlaps** — at n = 39 positives, **no pairwise model difference is
statistically significant.** GBM×hybrid is the *modal* bootstrap winner (44 % of
resamples), not a majority one; the hybrid cohort wins ~74 %. The survey signal
helps *directionally* (lift positive for all three models, never negative) but the
data cannot certify the magnitude. Full receipts in [`methodology.md`](methodology.md).

**Calibration:** GBM is the only family that clears the calibration gate
(Brier ≤ `β·(1−β)`). LR/EBM inflate probabilities via balanced class weights and
fail it — so their rankings may be usable but their probabilities are not safe to
operate. Test calibration here is *marginal*, which is **why a production
deployment should add an explicit calibration step (isotonic / Platt) before
trusting any threshold or expected-value figure.**

**Business framing (decision support, not a guarantee):** with replacement cost
≈ 1.5 × salary and a \$2,000 retention conversation (SHRM 2024), the program
breaks even once those conversations succeed **~6–7 % of the time** — see the
expected-value sensitivity sweep.

---

## Ethical considerations & fairness

- **Fairness is NOT yet audited.** A protected attribute (`gender`) is present in
  the feature set. A formal fairness audit (Fairlearn, calibration parity per
  Chouldechova 2017, compound-attribute slices) is **Loop 3a (v0.6)** and is a
  prerequisite before any non-research use.
- **Associational, not causal (Rung 1).** The model ranks *who* is likely to
  leave. It does **not** establish what *causes* leaving, nor that intervening
  *causes* retention. Treat feature associations as signals for human inquiry, not
  as levers with guaranteed effect.
- **Human-in-the-loop is mandatory.** Outputs are inputs to a conversation, never
  an automated decision.

## Caveats & limitations

- **Single cross-section** — not temporal validation (see above).
- **Underpowered** — n = 39 positive examples; comparative claims are directional, not certified.
- **Synthetic data** — behavior on real production HRIS data is unverified.
- **Marginal test calibration** — add explicit calibration before operational use.
- **No fairness audit yet** — Loop 3a.

---

## Maintenance & reproduction

- **Retrain / regenerate:** `make evaluate` re-runs the full notebook (train 6 cells → nested CV → comparison → champion selection → test confirmation → persist → register).
- **Artifacts:** `reports/models/champion.pkl` (self-describing: model + threshold + provenance + val/test metrics + seed) and the MLflow registry entry `rp-champion/Production`.
- **Load:** `retention.evaluation.champion.load_champion()` (file store) or `mlflow.sklearn.load_model("models:/rp-champion/Production")` (registry).
- **Determinism:** `SEED=42` throughout; `make repro` for `PYTHONHASHSEED`-pinned runs.
- **Refresh cadence:** re-evaluate when pa-warehouse ships a new mart migration (especially when time-series snapshots land — they unlock true temporal validation and retire the single-cross-section caveat).

*Card version: v0.5 · 2026-05-30. Update on every champion change.*
