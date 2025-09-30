Read the full detailed report: https://medium.com/@yashaswini.dinesh/crisp-dm-record-cardiac-failure-death-rate-prediction-a5eafc2afe3e
Methodology (CRISP_DM)

Phase 1 — Business & Data Understanding

Objective: prioritize care by identifying high-risk patients (Recall-first).

Metrics: AUPRC (class imbalance), Recall, Precision, F1, AUROC; Brier score for calibration.

EDA: schema, missingness, target balance, distributions, key relationships.

Phase 2 — Data Preparation

Pipeline (ColumnTransformer):

Numerics: median impute → Yeo–Johnson → RobustScaler.

Binary flags: mode impute; retain 0/1.

Leakage control: remove time from features.

Phase 3 — Modeling

Baselines → Logistic Regression (L2, class_weight='balanced'), Decision Tree, Random Forest.

5-fold stratified OOF CV for fair comparison (same preprocessing in all folds).

Phase 4 — Evaluation

Choose model based on CV AUPRC.

Adjust threshold to achieve Recall ≈ 0.80 (from OOF PR curve); confirm on test.

Present test ROC/PR, confusion matrices (default vs tuned), calibration (Brier + reliability).

Fairness slices: by sex and age groups at the tuned operating point.

Cost curve: analyze trade-offs (example cost = 5×FN + 1×FP).

Phase 5 — Deployment

Export a single scikit-learn pipeline (.joblib) + threshold in config.json.

FastAPI service: /predict provides probabilities & labels using the tuned threshold.

Model Card for governance & operations.
