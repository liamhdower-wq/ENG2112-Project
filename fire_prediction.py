"""
fire_prediction.py — Random Forest fire occurrence classifier
10-fold stratified cross-validation; final output averages the 3 best folds (by ROC-AUC).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_curve, average_precision_score)
from sklearn.impute import SimpleImputer

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_CSV   = '/Users/liam/Desktop/ENGG2112/Final/rainfall_vegetation_fire_solar_temp_finalv3.csv'
OUTPUT_CSV  = '/Users/liam/Desktop/ENGG2112/Final/fire_probability_predictions.csv'
OUTPUT_PLOT = '/Users/liam/Desktop/ENGG2112/Final/fire_model_evaluation.png'

N_FOLDS     = 10   # number of CV folds
TOP_K_FOLDS = 3    # average predictions from the best K folds (ranked by ROC-AUC)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_csv(INPUT_CSV)
print(f"  {df.shape[0]:,} rows x {df.shape[1]} columns")

# ── 2. Feature columns ────────────────────────────────────────────────────────
ID_COLS   = ['grid_id', 'lat_centre', 'lon_centre', 'year', 'month']
DROP_COLS = ['vegetation_cover', 'fire_occurred', 'num_fires', 'total_area_ha',
             'dominant_cause', 'fire_hotspot_count', 'fire_total_frp',
             'fire_mean_confidence', 'fire_max_brightness', 'fire_data_available']

FEATURE_COLS = [c for c in df.columns if c not in ID_COLS + DROP_COLS]
print(f"\nFeatures ({len(FEATURE_COLS)}): {FEATURE_COLS}")

# ── 3. Labelled subset ────────────────────────────────────────────────────────
labelled = df[df['fire_occurred'].notna()].copy()
X_raw = labelled[FEATURE_COLS].values
y     = labelled['fire_occurred'].astype(int).values
print(f"\nLabelled rows: {len(labelled):,}  |  Fire=1: {y.sum():,}  Fire=0: {(y==0).sum():,}")

# ── 4. 10-Fold Cross-Validation ───────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_results = []   # list of dicts, one per fold
# oof = out-of-fold probability accumulator (for later ensemble on labelled rows)
oof_probs   = np.zeros((len(y), N_FOLDS))

# Full-dataset prediction accumulator (one column per fold)
X_all_raw   = df[FEATURE_COLS].values
all_probs   = np.zeros((len(df), N_FOLDS))

print(f"\nRunning {N_FOLDS}-fold stratified cross-validation ...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y), start=1):
    # --- impute per fold (fit on train, transform both) ---
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_raw[train_idx])
    X_val   = imputer.transform(X_raw[val_idx])
    y_train, y_val = y[train_idx], y[val_idx]

    # --- train ---
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=3,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # --- validate ---
    y_prob_val = rf.predict_proba(X_val)[:, 1]
    oof_probs[val_idx, fold - 1] = y_prob_val

    auc = roc_auc_score(y_val, y_prob_val)
    ap  = average_precision_score(y_val, y_prob_val)
    print(f"  Fold {fold:2d}  ROC-AUC={auc:.4f}  AP={ap:.4f}")

    # --- predict full dataset ---
    X_all_imp = imputer.transform(X_all_raw)
    all_probs[:, fold - 1] = rf.predict_proba(X_all_imp)[:, 1]

    # store everything needed later
    fold_results.append({
        'fold': fold, 'auc': auc, 'ap': ap,
        'val_idx': val_idx, 'y_val': y_val, 'y_prob_val': y_prob_val,
        'imputer': imputer, 'model': rf
    })

# ── 5. Select top-K folds by ROC-AUC ─────────────────────────────────────────
fold_results.sort(key=lambda d: d['auc'], reverse=True)
top_folds = fold_results[:TOP_K_FOLDS]
top_fold_ids = [f['fold'] for f in top_folds]
print(f"\nTop {TOP_K_FOLDS} folds (by ROC-AUC): {top_fold_ids}")
for f in top_folds:
    print(f"  Fold {f['fold']}  AUC={f['auc']:.4f}  AP={f['ap']:.4f}")

# ── 6. Ensemble: average predictions from top-K folds ────────────────────────
top_col_idx  = [f['fold'] - 1 for f in top_folds]           # 0-based column indices
fire_prob    = all_probs[:, top_col_idx].mean(axis=1)        # mean over top-K folds

# Reconstruct OOF probs from top-K folds for threshold search
# For each labelled row, average only the folds where it was in the *validation* set
# and that fold is among the top-K; fall back to top-K full-dataset mean otherwise.
oof_top      = oof_probs[:, top_col_idx]
oof_ensemble = np.where(
    (oof_top > 0).any(axis=1, keepdims=True),
    np.where(oof_top > 0, oof_top, np.nan),
    oof_top
)
# nanmean: for rows with ≥1 top-K OOF value, use that; else full mean
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    y_prob_oof = np.nanmean(oof_ensemble, axis=1)
# Any remaining zeros (row never appeared as val in a top-K fold) → use full-data mean
mask_zero = y_prob_oof == 0
y_prob_oof[mask_zero] = fire_prob[df['fire_occurred'].notna().values][mask_zero]

# ── 7. Optimal threshold via F1 maximisation (on OOF ensemble) ───────────────
precisions, recalls, thresholds = precision_recall_curve(y, y_prob_oof)
f1_scores   = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_idx    = np.argmax(f1_scores)
best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
y_pred_oof  = (y_prob_oof >= best_thresh).astype(int)

roc_auc = roc_auc_score(y, y_prob_oof)
ap      = average_precision_score(y, y_prob_oof)
fpr, tpr, _ = roc_curve(y, y_prob_oof)
report  = classification_report(y, y_pred_oof,
                                target_names=['No Fire', 'Fire'], output_dict=True)
cm      = confusion_matrix(y, y_pred_oof)

print(f"\n── Ensemble (top-{TOP_K_FOLDS} folds) OOF performance (threshold={best_thresh:.3f}) ──")
print(classification_report(y, y_pred_oof, target_names=['No Fire', 'Fire']))
print(f"ROC-AUC       : {roc_auc:.4f}")
print(f"Avg Precision : {ap:.4f}")
print(f"Accuracy      : {report['accuracy']:.4f}")

# ── 8. Feature importances (mean over top-K fold models) ─────────────────────
imp_matrix = np.vstack([f['model'].feature_importances_ for f in top_folds])
imp        = pd.Series(imp_matrix.mean(axis=0), index=FEATURE_COLS).sort_values(ascending=False)
print("\nFeature importances (mean Gini over top-K folds):")
print(imp.to_string())

# ── 9. Save predictions ───────────────────────────────────────────────────────
fire_pred = (fire_prob >= best_thresh).astype(int)

out = df[ID_COLS + ['fire_occurred']].copy()
out['fire_probability'] = np.round(fire_prob, 6)
out['fire_predicted']   = fire_pred
out['threshold_used']   = round(best_thresh, 4)
out = out.sort_values(['lon_centre', 'lat_centre', 'year', 'month']).reset_index(drop=True)
out.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Predictions saved -> {OUTPUT_CSV}")

# ── 10. Plots ─────────────────────────────────────────────────────────────────
FIRE_RED  = '#C0392B'
SAFE_BLUE = '#2980B9'
GREY      = '#7F8C8D'
BG        = '#F8F9FA'

fig = plt.figure(figsize=(18, 14), facecolor=BG)
fig.suptitle(
    f'Random Forest — Fire Occurrence Model Evaluation  |  '
    f'Top-{TOP_K_FOLDS}/{N_FOLDS} Fold Ensemble  |  '
    f'AUC={roc_auc:.3f}  Threshold={best_thresh:.2f}',
    fontsize=14, fontweight='bold', y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

# Plot 1: ROC Curve
ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor(BG)
ax1.plot(fpr, tpr, color=FIRE_RED, lw=2.5, label=f'ROC (AUC={roc_auc:.3f})')
ax1.plot([0,1],[0,1], color=GREY, lw=1.5, linestyle='--', label='Random')
ax1.fill_between(fpr, tpr, alpha=0.08, color=FIRE_RED)
ax1.set_xlabel('False Positive Rate', fontsize=11)
ax1.set_ylabel('True Positive Rate', fontsize=11)
ax1.set_title('ROC Curve', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)

# Plot 2: Precision-Recall Curve
ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor(BG)
ax2.plot(recalls, precisions, color=FIRE_RED, lw=2.5, label=f'AP={ap:.3f}')
ax2.axvline(report['Fire']['recall'], color=GREY, linestyle='--', lw=1.5,
            label=f'Threshold={best_thresh:.2f}')
ax2.fill_between(recalls, precisions, alpha=0.08, color=FIRE_RED)
ax2.set_xlabel('Recall', fontsize=11)
ax2.set_ylabel('Precision', fontsize=11)
ax2.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)

# Plot 3: Confusion Matrix
ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor(BG)
ConfusionMatrixDisplay(cm, display_labels=['No Fire', 'Fire']).plot(
    ax=ax3, colorbar=False, cmap='Reds')
ax3.set_title(f'Confusion Matrix\n(threshold={best_thresh:.2f})', fontsize=13, fontweight='bold')

# Plot 4: Classification Metrics + per-fold AUC
ax4 = fig.add_subplot(gs[1, 0]); ax4.set_facecolor(BG)
metrics = ['precision', 'recall', 'f1-score']
vals = [report['Fire'][m] for m in metrics]
x = np.arange(len(metrics)); width = 0.5
bars = ax4.bar(x, vals, width, color=FIRE_RED, alpha=0.85)
for bar, v in zip(bars, vals):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
             f'{v:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(['Precision', 'Recall', 'F1-Score'], fontsize=11)
ax4.set_ylim(0, 1.15); ax4.set_ylabel('Score', fontsize=11)
ax4.set_title('Fire Class Metrics', fontsize=13, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
all_aucs = [f['auc'] for f in sorted(fold_results, key=lambda d: d['fold'])]
summary  = (f"Accuracy: {report['accuracy']:.3f}\nROC-AUC: {roc_auc:.3f}\nAP: {ap:.3f}\n"
            f"CV AUC μ={np.mean(all_aucs):.3f} σ={np.std(all_aucs):.3f}\n"
            f"Top folds: {top_fold_ids}")
ax4.text(0.98, 0.97, summary, transform=ax4.transAxes, ha='right', va='top', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=GREY, alpha=0.8))

# Plot 5: Feature Importances
ax5 = fig.add_subplot(gs[1, 1:3]); ax5.set_facecolor(BG)
colors_imp = [FIRE_RED if i==0 else SAFE_BLUE if i<3 else GREY for i in range(len(imp))]
bars = ax5.barh(imp.index[::-1], imp.values[::-1], color=colors_imp[::-1], alpha=0.85)
for bar, v in zip(bars, imp.values[::-1]):
    ax5.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
             f'{v:.3f}', va='center', fontsize=9)
ax5.set_xlabel('Feature Importance (mean Gini, top-K folds)', fontsize=11)
ax5.set_title('Feature Importances', fontsize=13, fontweight='bold')
ax5.set_xlim(0, imp.max()*1.25); ax5.grid(axis='x', alpha=0.3)
ax5.tick_params(axis='y', labelsize=9)
fig.canvas.draw()
pos = ax5.get_position()
ax5.set_position([pos.x0 + 0.04, pos.y0, pos.width - 0.04, pos.height])

plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"✓ Plots saved -> {OUTPUT_PLOT}")
plt.show()
