# Communities and Crime — DeepSeek-V4-Pro ReAct test

## Dataset

- Source: UCI Machine Learning Repository, Communities and Crime (dataset 183)
- Rows: 1,994
- Predictors presented to the agent: 122
- Target: `ViolentCrimesPerPop` (normalized 0–1)
- Removed as non-predictive metadata: `state`, `county`, `community`, `communityname`, `fold`
- Split: 1,196 train / 399 validation / 399 untouched test rows

## Agent run

- Decision model: `deepseek-v4-pro`
- Mode: tool-driven ReAct with thinking enabled
- Tool calls: 39
- Discarded attempts: 3
- Fully evaluated candidates: 2
- Selected candidate: 3

The agent rejected three intermediate models after their t-statistic tools found
insignificant predictors. The two final candidates both completed adjusted R²,
F-statistic, t-statistic, RMSE, and VIF tools before comparison.

## Candidate comparison

| Candidate | Variables | Adjusted R² | F-statistic | F p-value | Validation RMSE | Max VIF | Insignificant predictors |
|---|---|---:|---:|---:|---:|---:|---|
| 3 (selected) | `PctUnemployed`, `racepctblack`, `PctFam2Par`, `agePct12t21`, `PctIlleg` | 0.5984 | 357.0934 | 7.81e-234 | 0.1557 | 5.2909 | None |
| 5 | `medIncome`, `PctVacantBoarded`, `PctPersDenseHous`, `PctUsePubTrans`, `PctPolicBlack` | 0.4670 | 210.4222 | 7.20e-161 | 0.1786 | 1.4022 | None |

### Selected candidate t-statistics

| Variable | Coefficient | t-statistic | p-value |
|---|---:|---:|---:|
| `PctUnemployed` | 0.0858 | 2.9525 | 0.00321 |
| `racepctblack` | 0.1023 | 3.4848 | 0.000510 |
| `PctFam2Par` | -0.2969 | -6.9627 | 5.51e-12 |
| `agePct12t21` | -0.1423 | -5.0031 | 6.49e-7 |
| `PctIlleg` | 0.4277 | 9.9930 | 1.24e-22 |

## Untouched test-set result

- RMSE: **0.148338**
- MSE (derived from RMSE): **0.022004**
- R²: **0.540581**
- Test rows: **399**

The target is normalized to the range 0–1. The final MSE is below the earlier
hard requirement of 5, though cross-dataset comparisons should account for the
target scale.

Full tool observations and decision memory are stored in `agent_result.json`.
The final actual-vs-predicted and residual plot is stored in
`best_model_test_diagnostics.svg`.
