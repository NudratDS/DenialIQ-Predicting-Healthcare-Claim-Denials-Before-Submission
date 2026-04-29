# DenialIQ: 120K Medical Claims with X12 Denial Codes | RCM AI

[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://kaggle.com/datasets/nudratabbas/denialiq)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-green)](https://creativecommons.org/licenses/by/4.0/)
[![Version](https://img.shields.io/badge/Version-2.0-orange)](https://github.com/NudratDS/DenialIQ)
[![Synthetic](https://img.shields.io/badge/Data-Synthetic%20%7C%20HIPAA--Safe-purple)](https://github.com/NudratDS/DenialIQ)

> 120,000 realistic synthetic medical claims labeled with real X12 835 denial codes, denial categories, appeal success probability, and recovery actions. Built for RCM AI, denial prediction ML, and LLM fine-tuning.

---
<p align="center">
  <img src="https://githubusercontent.com" alt="EDA Overview" width="100%">
</p>


## The Problem

**28% of medical claims are denied on first submission.**

- Average cost to rework one denied claim: **$25–$118**
- ~50% of denied claims are **never reworked** → pure revenue loss
- US healthcare loses **$262B+/year** to claim denials

RCM teams need ML models that flag high-risk claims **before submission**.  
This dataset was built to train exactly those models.

---

## Dataset Files

| File | Rows | Description |
|------|------|-------------|
| `claims_main.csv` | 120,000 | Claims with 25 fields including CPT, ICD-10, payer, outcome |
| `denial_labels.csv` | ~33,600 | Denied claims with appeal labels + recovery actions |
| `payer_rules.csv` | ~350 | Payer × CPT prior auth rules + historical denial rates |
| `train_test_split.csv` | 120,000 | 70/15/15 stratified split |
| `llm_finetune.jsonl` | ~67,200 | Balanced JSONL for LLM classification fine-tuning |
| `data_dictionary.csv` | 48 | Full schema reference for all files |

---

## Key Features

```
✅ Real X12 835 denial codes (CO-4, CO-11, PR-204, OA-23, etc.)
✅ 7 denial categories (medical_necessity, coding_error, auth_missing, ...)
✅ Appeal success probability per denial
✅ CPT-level pricing (realistic Medicare fee schedule anchors)
✅ 6 payer types × 13 specialties
✅ Prior auth logic tied to actual auth-required CPT codes
✅ Documentation completeness score correlated with outcome
✅ LLM-ready JSONL output
✅ Full data dictionary
✅ Train/val/test split included
✅ Version tagged for reproducibility
```

---

## Schema — claims_main.csv

```
claim_id                    UUID
claim_submission_date       YYYY-MM-DD
payer_type                  Medicare_FFS | Medicare_Advantage | Medicaid_Managed |
                            Commercial_PPO | Commercial_HMO | Commercial_EPO
provider_specialty          13 specialties (Cardiology, Ortho, BH, Oncology, etc.)
cpt_code                    AMA CPT procedure code
modifier                    25 | 59 | GT | GQ | 76 | 79 | null
place_of_service_code       CMS POS code (11=Office, 22=Outpatient, 23=ER, 02=Telehealth)
primary_icd10_dx            ICD-10-CM code
secondary_icd10_dx          Pipe-delimited (up to 4)
prior_auth_required         bool
prior_auth_obtained         bool
documentation_completeness  float 0–1
claim_amount_usd            float (CPT-level pricing, payer-adjusted)
outcome                     paid | denied | partial_pay | pending
denial_reason_code          X12 835 code
denial_category             7-class label
```

---

## Use Cases

### 1. Denial Prediction (Pre-Submission)
```python
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

claims = pd.read_csv("claims_main.csv")
splits = pd.read_csv("train_test_split.csv")
df = claims.merge(splits, on="claim_id")

# Binary target
df["target"] = (df["outcome"] == "denied").astype(int)

train = df[df["split"] == "train"]
# ... encode + train XGBoost
```

### 2. Denial Category Classification (Multi-Class)
```python
denied = claims[claims["outcome"] == "denied"]
# Predict: medical_necessity | coding_error | auth_missing | ...
```

### 3. LLM Fine-Tuning
```python
import json
records = [json.loads(line) for line in open("llm_finetune.jsonl")]
# Each record has: cpt_code, payer_type, prior_auth_obtained,
#                  doc_completeness, label, denial_category
```

### 4. Appeal Success Prediction
```python
denial_labels = pd.read_csv("denial_labels.csv")
# Target: appeal_success_probability (regression)
# or appealable (binary classification)
```

---

## Denial Rate Distribution

```
Outcome         Rate
──────────────────────
paid            52.0%
denied          28.0%    ← aligned with CMS/HFMA benchmarks
partial_pay     13.0%
pending          7.0%
```

## Denial Category Breakdown (of denied claims)

```
medical_necessity    28%
coding_error         22%
auth_missing         18%
eligibility          12%
duplicate             8%
timely_filing         7%
bundling              5%
```

---

## Quick Start

```bash
git clone https://github.com/NudratDS/DenialIQ
cd DenialIQ
pip install pandas numpy tqdm
python denialiq_generator_v2.py
```

Or run directly on Kaggle:  
→ [kaggle.com/datasets/nudratabbas/denialiq](https://kaggle.com/datasets/nudratabbas/denialiq)

---

## Kaggle Notebook

**"Predicting Claim Denials Before Submission | XGBoost + SHAP | RCM"**

Covers:
- Full EDA with 6 visualizations
- Feature engineering (auth_risk, doc_score, payer_denial_rate)
- Logistic Regression baseline → XGBoost
- SHAP beeswarm + bar charts
- Precision-recall curve with optimal threshold annotation
- **ROI calculator** — converts model accuracy into monthly $ saved
- Inference demo — single claim → denial probability

→ [View on Kaggle](https://kaggle.com/nudratabbas)

---

## Dataset Metadata

```
Version:          2.0
Generated:        2026-04-28
Synthetic:        True (HIPAA-safe, no real patient data)
Seed:             42 (fully reproducible)
License:          CC BY 4.0
Denial benchmark: CMS/HFMA 2023 denial rate reports
CPT pricing:      Medicare fee schedule anchors (2× billed multiplier)
X12 codes:        ASC X12 835 transaction standard
```

---

## Custom Healthcare Datasets

Need a custom healthcare dataset for your AI model?  
I build clinical NLP, RCM, and medical coding datasets for healthtech startups.

**→ DM on LinkedIn:** [linkedin.com/in/nudrat-abbas-664378324](https://linkedin.com/in/nudrat-abbas-664378324)  
**→ Twitter/X:** [@NudratDS](https://x.com/NudratDS)

---

## Citation

```bibtex
@dataset{denialiq2026,
  author    = {Nudrat Abbas},
  title     = {DenialIQ: 120K Medical Claims with X12 Denial Codes},
  year      = {2026},
  version   = {2.0},
  publisher = {Kaggle},
  url       = {https://kaggle.com/datasets/nudratabbas/denialiq}
}
```

---

⭐ **Star this repo** if it helped you — it keeps me building more open healthcare datasets.

[![GitHub stars](https://img.shields.io/github/stars/NudratDS/DenialIQ?style=social)](https://github.com/NudratDS/DenialIQ)
