"""
========================================================================================
ACTIVITY PHASE RECOGNITION PIPELINE 

TASKS EVALUATED:
1. Binary: Baseline vs. 2-Back Part 1 (B vs. 2B1)
2. Binary: Baseline vs. 2-Back Part 2 (B vs. 2B2)
3. Binary: Baseline vs. 2-Back Whole  (B vs. 2BW)
4. Binary: Baseline vs. Speech/Rest   (B vs. SR)


MODALITIES & FEATURE SETS:
- HRV (PPG): All Features & Top 20% Features
- EDA:       All Features & Top 20% Features
- Fused:     All Features & Top 20% Features

VALIDATION:
- Leave-One-Participant-Out (LOPO): each fold tests all windows from one participant
  and trains on windows from all remaining participants
- In-fold median imputation
- In-fold feature selection (SelectKBest via ANOVA F-value)
- Random Forest Classifier (100 estimators, max_depth=12)
========================================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
)

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# 1. PATH CONFIGURATION & DATA LOADING
# -------------------------------------------------------------------------
DATA_DIR    = r"C:\Users\Admin\Downloads\Step1"
ROSTER_PATH = r"C:\Users\Admin\Downloads\Selected_Participants_Combined_Phase_Features\participants_145_self_report.csv"

# Load participant roster (144 participants)
roster = pd.read_csv(ROSTER_PATH)
roster_pts = set(roster["Participant"].astype(str).str.strip().str.upper().unique())

def load_phase_df(filename):

    filepath = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(filepath)
    df["Participant"] = df["Participant"].astype(str).str.strip().str.upper()
    df = df[df["Participant"].isin(roster_pts)].copy()
    return df

b_raw  = load_phase_df("Baseline.csv")
tb_raw = load_phase_df("2backPart12.csv")
sr_raw = load_phase_df("SpeechRest.csv")

# -------------------------------------------------------------------------
# 2. FEATURE IDENTIFICATION (PPG, EDA, FUSED)
# -------------------------------------------------------------------------
meta_cols = {
    "Participant", "Phase", "EDA_Source_File", "Window_Index", "Window_Start_sec",
    "Window_End_sec", "Condition", "Window_Start", "Window_End", "Phase_Label"
}

common_cols = set(b_raw.columns).intersection(set(tb_raw.columns)).intersection(set(sr_raw.columns)) - meta_cols

# Separate EDA features from PPG/HRV features
eda_features = sorted([c for c in common_cols if not (c.startswith("HRV_") or c.startswith("HR_") or c.startswith("PPI_"))])
ppg_features = sorted([c for c in common_cols if (c.startswith("HRV_") or c.startswith("HR_") or c.startswith("PPI_"))])
fused_features = eda_features + ppg_features

print(f"Features Identified: EDA = {len(eda_features)}, PPG = {len(ppg_features)}, Fused = {len(fused_features)}")

# -------------------------------------------------------------------------
# 3. CONSTRUCT EVALUATION TASKS
# -------------------------------------------------------------------------
b_raw["Phase_Label"]  = "Baseline"
sr_raw["Phase_Label"] = "SpeechRest"
tb_raw["Phase_Label"] = "2BackWhole"

tb_p1 = tb_raw[tb_raw["Phase"] == "2BackPart1"].copy()
tb_p1["Phase_Label"] = "2BackPart1"

tb_p2 = tb_raw[tb_raw["Phase"] == "2BackPart2"].copy()
tb_p2["Phase_Label"] = "2BackPart2"

tasks = {
    "Baseline vs 2BackPart1": pd.concat([b_raw, tb_p1], ignore_index=True),
    "Baseline vs 2BackPart2": pd.concat([b_raw, tb_p2], ignore_index=True),
    "Baseline vs 2BackWhole": pd.concat([b_raw, tb_raw], ignore_index=True),
    "Baseline vs SpeechRest": pd.concat([b_raw, sr_raw], ignore_index=True),

}

modalities = {
    "HRV": ppg_features,
    "EDA": eda_features,
    "Fused": fused_features
}

# -------------------------------------------------------------------------
# 4. CROSS-VALIDATION LOOP (LOPO on Participants; retain all windows)
# -------------------------------------------------------------------------
def main():
    results = []

    for task_name, task_df in tasks.items():
        print(f"\n================================================================================")
        print(f"TASK: {task_name} (Total Windows: {len(task_df)})")
        print(f"================================================================================")

        unique_labels = sorted(task_df["Phase_Label"].unique())
        label_to_idx  = {lbl: i for i, lbl in enumerate(unique_labels)}
        y_all         = task_df["Phase_Label"].map(label_to_idx).values
        groups        = task_df["Participant"].values
        is_multiclass = len(unique_labels) > 2

        for mod_name, feat_cols in modalities.items():
            X_all = task_df[feat_cols].apply(
    pd.to_numeric, errors="coerce"
).to_numpy(dtype=np.float64)

# Convert Inf and -Inf to NaN
            X_all[~np.isfinite(X_all)] = np.nan
            n_feats = len(feat_cols)
            k_top20 = max(1, int(np.round(n_feats * 0.20)))

            feature_sets = [
                ("All Features", n_feats),
                (f"Top 20% ({k_top20})", k_top20)
            ]

            for f_label, k_feat in feature_sets:
                # Keep every window. Leave all windows from one participant out together.
                logo = LeaveOneGroupOut()
                fold_acc, fold_bacc, fold_f1, fold_auc = [], [], [], []

                for fold_idx, (tr_idx, te_idx) in enumerate(logo.split(X_all, y_all, groups=groups)):
                    X_tr, y_tr = X_all[tr_idx].copy(), y_all[tr_idx].copy()
                    X_te, y_te = X_all[te_idx].copy(), y_all[te_idx].copy()

                    # In-fold median imputation
                    
# ---------------------------------------------------------
# Robust in-fold preprocessing: NaN, Inf, huge values
# ---------------------------------------------------------

# Convert all non-finite values to NaN
                    X_tr[~np.isfinite(X_tr)] = np.nan
                    X_te[~np.isfinite(X_te)] = np.nan

                    # Prevent overflow inside sklearn's float32 conversion
                    FLOAT32_LIMIT = 1e30

                    X_tr = np.clip(X_tr, -FLOAT32_LIMIT, FLOAT32_LIMIT)
                    X_te = np.clip(X_te, -FLOAT32_LIMIT, FLOAT32_LIMIT)

                    # Calculate medians ONLY from training data
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        col_meds = np.nanmedian(X_tr, axis=0)

                    # Entirely missing training features -> median = 0
                    col_meds = np.where(np.isfinite(col_meds), col_meds, 0.0)

                    # Impute training data
                    nan_tr = np.isnan(X_tr)
                    X_tr[nan_tr] = np.take(col_meds, np.where(nan_tr)[1])

                    # Impute test data using training medians
                    nan_te = np.isnan(X_te)
                    X_te[nan_te] = np.take(col_meds, np.where(nan_te)[1])

                    # Final safety check
                    assert np.isfinite(X_tr).all(), "Non-finite values remain in X_tr"
                    assert np.isfinite(X_te).all(), "Non-finite values remain in X_te"

                    # In-fold feature selection if Top 20%
                    if k_feat < n_feats:
                        selector = SelectKBest(score_func=f_classif, k=k_feat)
                        X_tr = selector.fit_transform(X_tr, y_tr)
                        X_te = selector.transform(X_te)

                    # Random Forest Classifier
                    clf = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=12,
                        random_state=42,
                        class_weight="balanced",
                        n_jobs=-1
                    )
                    clf.fit(X_tr, y_tr)

                    preds = clf.predict(X_te)
                    probs = clf.predict_proba(X_te)

                    fold_acc.append(accuracy_score(y_te, preds))
                    fold_bacc.append(balanced_accuracy_score(y_te, preds))
                    fold_f1.append(f1_score(y_te, preds, average="weighted", zero_division=0))

                    try:
                        if is_multiclass:
                            fold_auc.append(roc_auc_score(y_te, probs, multi_class="ovr", average="weighted"))
                        else:
                            fold_auc.append(roc_auc_score(y_te, probs[:, 1]))
                    except Exception:
                        fold_auc.append(np.nan)

                mean_acc, std_acc   = np.mean(fold_acc), np.std(fold_acc)
                mean_bacc, std_bacc = np.mean(fold_bacc), np.std(fold_bacc)
                mean_f1, std_f1     = np.mean(fold_f1), np.std(fold_f1)
                mean_auc, std_auc   = np.nanmean(fold_auc), np.nanstd(fold_auc)

                print(
                    f"  {mod_name:6s} | {f_label:14s} | "
                    f"Acc: {mean_acc:.3f} ± {std_acc:.3f} | "
                    f"F1: {mean_f1:.3f} ± {std_f1:.3f} | "
                    f"AUC: {mean_auc:.3f} ± {std_auc:.3f}"
                )

                results.append({
                    "Comparison_Task": task_name,
                    "Modality": mod_name,
                    "Feature_Selection": f_label,
                    "Number_Features": k_feat,
                    "Accuracy_Mean": mean_acc,
                    "Accuracy_Std": std_acc,
                    "Balanced_Accuracy_Mean": mean_bacc,
                    "Balanced_Accuracy_Std": std_bacc,
                    "Weighted_F1_Mean": mean_f1,
                    "Weighted_F1_Std": std_f1,
                    "ROC_AUC_Mean": mean_auc,
                    "ROC_AUC_Std": std_auc
                })

    df_out = pd.DataFrame(results)
    out_path = "Activity_Phase_Recognition_Performance_LOPO_AllWindows.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved Activity Recognition results to {out_path}")

if __name__ == "__main__":
    main()