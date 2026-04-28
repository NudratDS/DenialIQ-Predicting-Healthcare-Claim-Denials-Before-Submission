# DenialIQ: Predicting Healthcare Claim Denials Before Submission

Hospitals and provider groups lose billions every year due to preventable claim denials.

Most of these denials are not random. They follow patterns.

This project explores a simple question:

Can we identify high-risk claims before they are submitted?

---

## What This Repository Does

This repository contains a structured dataset and a machine learning pipeline designed to simulate real-world revenue cycle challenges.

It focuses on predicting claim denials using features commonly available at the time of submission.

The goal is not just model accuracy, but decision-making.

---

## What’s Inside

- Synthetic medical claims dataset with denial outcomes
- Features covering payer behavior, coding patterns, and documentation quality
- End-to-end notebook for denial prediction using XGBoost
- Model explainability using SHAP to understand denial drivers

Notebook and full dataset will be added shortly.

---

## Why This Matters

Claim denials are one of the biggest sources of revenue leakage in healthcare.

Common causes include:
- Missing prior authorization
- Incorrect coding or modifiers
- Payer-specific policy rules
- Incomplete documentation

Most systems detect denials after submission.

This project focuses on preventing them before they happen.

---

## Example Use Cases

- Flag high-risk claims before submission
- Assist billing teams in improving documentation quality
- Train machine learning models for denial prediction
- Build decision-support tools for revenue cycle teams
- Integrate into RCM automation workflows

---

## Approach

We simulate a real-world claims environment using structured data:

- Payer types including Medicare, Medicaid, and commercial plans
- CPT and ICD coding combinations
- Prior authorization signals
- Documentation completeness indicators

A baseline model is built using gradient boosting and evaluated using practical metrics such as recall for denied claims.

SHAP is used to surface feature importance and decision logic.

---

## Who This Is For

- Healthcare data scientists
- Revenue cycle management teams
- Healthtech founders building RCM tools
- ML engineers working on tabular healthcare data

---

## What’s Coming Next

- Full dataset release
- Expanded feature set
- Prior authorization dataset with approval and denial labels
- LLM-based appeal generation workflows

---

## About Me

I work on healthcare data problems that directly impact cost, risk, and outcomes.

This includes:
- Clinical NLP
- Revenue cycle analytics
- Predictive modeling for healthcare operations

If you’re building something in this space and need structured data or collaboration, feel free to reach out.

---

## Contact

Open an issue or connect via LinkedIn.
