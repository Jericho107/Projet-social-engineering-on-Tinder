# Validation Matrix

| Claim | Executable evidence | Failure path | Status |
|---|---|---|---|
| Prediction target is not used as a feature | `src/governance.py` | target leakage injection | implemented |
| Post-outcome partner fields are blocked | feature audit | `dec_o` / post-outcome injection | implemented |
| Target remains binary | target contract | invalid class value | implemented |
| Probabilities remain bounded | probability contract | value outside [0, 1] | implemented |
| Statistical and predictive interpretations remain separated | project analysis + README boundary | interpretation review | implemented |
| Streamlit runtime resolves repository assets correctly | root path fix | runtime smoke path | implemented |
| CI validates governance rules | GitHub Actions + pytest/Ruff | leakage reverse test | implemented |
| Causal attraction claims | no causal design in repository | not applicable | not claimed |

## Review principle

Association, predictive importance and causality are treated as different statements. The repository only supports the level of interpretation justified by the analysis design.
