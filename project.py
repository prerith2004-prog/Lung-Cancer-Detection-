import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.metrics import (
classification_report, confusion_matrix, roc_auc_score,
roc_curve, precision_recall_curve, average_precision_score,
f1_score, accuracy_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

─────────────────────────────────────────────────────────────
XGBoost — pure-NumPy implementation (no external lib needed)
─────────────────────────────────────────────────────────────

class XGBoostClassifier:
"""
Lightweight XGBoost-style gradient boosted classifier.
Uses log-loss gradient/hessian updates — same algorithm as XGBoost.
"""
def init(self, n_estimators=200, max_depth=4, learning_rate=0.05,
subsample=0.8, colsample=0.8, reg_lambda=1.0,
min_child_weight=1, random_state=42):
self.n_estimators = n_estimators
self.max_depth = max_depth
self.learning_rate = learning_rate
self.subsample = subsample
self.colsample = colsample
self.reg_lambda = reg_lambda
self.min_child_weight = min_child_weight
self.random_state = random_state
self.trees_ = []
self.feature_indices_= []
self.base_score_ = 0.5

# ── internal tree node ──────────────────────────────────
class _Node:
    __slots__ = ('feature','threshold','left','right','value','is_leaf')
    def __init__(self):
        self.feature=self.threshold=self.left=self.right=self.value=None
        self.is_leaf=False

def _build(self, X, grad, hess, depth):
    node = self._Node()
    G, H = grad.sum(), hess.sum()
    if depth == 0 or len(grad) < self.min_child_weight * 2:
        node.is_leaf = True
        node.value   = -G / (H + self.reg_lambda)
        return node
    best_gain, best_feat, best_thresh = -np.inf, None, None
    n_feats = X.shape[1]
    for f in range(n_feats):
        vals = np.unique(X[:, f])
        if len(vals) < 2:
            continue
        # Subsample thresholds for speed
        all_thresh = (vals[:-1] + vals[1:]) / 2
        if len(all_thresh) > 10:
            idx = np.round(np.linspace(0, len(all_thresh)-1, 10)).astype(int)
            thresholds = all_thresh[idx]
        else:
            thresholds = all_thresh
        for t in thresholds:
            l = X[:, f] <= t; r = ~l
            if l.sum() < self.min_child_weight or r.sum() < self.min_child_weight:
                continue
            GL,HL = grad[l].sum(), hess[l].sum()
            GR,HR = grad[r].sum(), hess[r].sum()
            gain = (GL**2/(HL+self.reg_lambda) +
                    GR**2/(HR+self.reg_lambda) -
                    G**2 /(H +self.reg_lambda)) / 2
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, f, t
    if best_feat is None:
        node.is_leaf = True
        node.value   = -G / (H + self.reg_lambda)
        return node
    mask = X[:, best_feat] <= best_thresh
    node.feature   = best_feat
    node.threshold = best_thresh
    node.left  = self._build(X[mask],  grad[mask],  hess[mask],  depth-1)
    node.right = self._build(X[~mask], grad[~mask], hess[~mask], depth-1)
    return node

def _predict_node(self, node, x):
    if node.is_leaf:
        return node.value
    return self._predict_node(
        node.left if x[node.feature] <= node.threshold else node.right, x)

def fit(self, X, y):
    rng  = np.random.default_rng(self.random_state)
    X, y = np.array(X, dtype=float), np.array(y, dtype=float)
    n, p = X.shape
    # log-odds base score
    pos = y.mean().clip(1e-6, 1-1e-6)
    F   = np.full(n, np.log(pos / (1 - pos)))
    n_col = max(1, int(p * self.colsample))
    for _ in range(self.n_estimators):
        prob  = 1 / (1 + np.exp(-F))
        grad  = prob - y               # dL/dF
        hess  = prob * (1 - prob)      # d²L/dF²
        row_idx = rng.choice(n, int(n * self.subsample), replace=False)
        col_idx = rng.choice(p, n_col, replace=False)
        tree = self._build(X[np.ix_(row_idx, col_idx)],
                           grad[row_idx], hess[row_idx], self.max_depth)
        updates = np.array([self._predict_node(tree, X[i, col_idx]) for i in range(n)])
        F += self.learning_rate * updates
        self.trees_.append(tree)
        self.feature_indices_.append(col_idx)
    self.base_F_ = F
    return self

def predict_proba(self, X):
    X = np.array(X, dtype=float)
    n = X.shape[0]
    F = np.zeros(n)
    for tree, cols in zip(self.trees_, self.feature_indices_):
        F += self.learning_rate * np.array(
            [self._predict_node(tree, X[i, cols]) for i in range(n)])
    prob1 = 1 / (1 + np.exp(-F))
    return np.column_stack([1 - prob1, prob1])

def predict(self, X):
    return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# sklearn compatibility
def get_params(self, deep=True):
    return dict(n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.learning_rate, subsample=self.subsample,
                colsample=self.colsample, reg_lambda=self.reg_lambda,
                min_child_weight=self.min_child_weight, random_state=self.random_state)

def set_params(self, **p):
    for k, v in p.items(): setattr(self, k, v)
    return self
─────────────────────────────────────────────────────────────
1. DATA GENERATION
─────────────────────────────────────────────────────────────

def generate_lung_cancer_data(n_samples=1500, seed=42):
"""
Synthetic lung cancer dataset modelled on clinical features
(similar to UCI Lung Cancer / SEER dataset structure).
Features: demographics, symptoms, imaging, biopsy markers.
"""
rng = np.random.default_rng(seed)
n_mal = int(n_samples * 0.55) # slightly imbalanced
n_ben = n_samples - n_mal

def make_class(n, malignant):
    m = int(malignant)
    return {
        # Demographics
        'Age':              rng.normal(63 + 5*m,  10,  n).clip(25, 90),
        'Smoking_Years':    rng.normal(25 + 10*m, 12,  n).clip(0,  60),
        'Pack_Years':       rng.normal(30 + 15*m, 15,  n).clip(0,  80),
        'Passive_Smoke':    rng.binomial(1, 0.3 + 0.2*m, n),

        # Symptoms (0-10 severity)
        'Cough_Severity':   rng.normal(4 + 3*m,   2, n).clip(0, 10),
        'Dyspnea':          rng.normal(3 + 3*m,   2, n).clip(0, 10),
        'Chest_Pain':       rng.normal(2 + 4*m,   2, n).clip(0, 10),
        'Hemoptysis':       rng.normal(1 + 3*m,   1, n).clip(0, 10),
        'Weight_Loss':      rng.normal(2 + 3*m,   2, n).clip(0, 10),
        'Fatigue':          rng.normal(4 + 2*m,   2, n).clip(0, 10),

        # Imaging features
        'Nodule_Size_mm':   rng.normal(8 + 18*m,  8, n).clip(1,  60),
        'Nodule_Count':     rng.poisson(1 + 2*m,     n).clip(0,  10),
        'Spiculation':      rng.binomial(1, 0.1 + 0.7*m, n),
        'Calcification':    rng.binomial(1, 0.5 - 0.3*m, n),
        'Ground_Glass':     rng.binomial(1, 0.2 + 0.4*m, n),
        'Consolidation':    rng.binomial(1, 0.1 + 0.5*m, n),
        'PET_SUV':          rng.normal(2 + 8*m,    3, n).clip(0,  20),

        # Lab / biopsy markers
        'CEA_Level':        rng.lognormal(0.5 + 1.5*m, 0.8, n).clip(0, 50),
        'CYFRA21_1':        rng.lognormal(0.3 + 1.2*m, 0.7, n).clip(0, 30),
        'NSE_Level':        rng.lognormal(0.2 + 1.0*m, 0.6, n).clip(0, 25),
        'LDH_Level':        rng.normal(220 + 80*m,  60, n).clip(100, 600),

        # Genetic / history
        'Family_History':   rng.binomial(1, 0.15 + 0.25*m, n),
        'EGFR_Mutation':    rng.binomial(1, 0.05 + 0.30*m, n),
        'KRAS_Mutation':    rng.binomial(1, 0.02 + 0.20*m, n),
        'ALK_Rearrangement':rng.binomial(1, 0.02 + 0.08*m, n),
        'COPD_History':     rng.binomial(1, 0.20 + 0.20*m, n),
        'Asbestos_Exposure':rng.binomial(1, 0.05 + 0.15*m, n),

        # Spirometry
        'FEV1_Pct':         rng.normal(75 - 15*m,  15, n).clip(20, 120),
        'FVC_Pct':          rng.normal(80 - 10*m,  12, n).clip(30, 120),
        'DLCO_Pct':         rng.normal(80 - 20*m,  15, n).clip(20, 120),

        'Label': np.ones(n, dtype=int) * int(malignant)
    }

df = pd.concat([
    pd.DataFrame(make_class(n_mal, True)),
    pd.DataFrame(make_class(n_ben, False))
], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

return df
─────────────────────────────────────────────────────────────
2. DATA CLEANING
─────────────────────────────────────────────────────────────

def clean_data(df):
print("\n[Cleaning] Starting data cleaning...")
initial = df.shape

df = df.drop_duplicates()
df = df.dropna()

# Clip obvious physiological impossibilities
df['Age']           = df['Age'].clip(18, 90)
df['Nodule_Size_mm']= df['Nodule_Size_mm'].clip(0, 60)
df['PET_SUV']       = df['PET_SUV'].clip(0, 25)

# IQR-based outlier clipping per continuous column (preserves rows)
continuous = ['CEA_Level','CYFRA21_1','NSE_Level','LDH_Level',
              'Pack_Years','Smoking_Years']
for col in continuous:
    Q1, Q3 = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(Q1, Q3)

print(f"  Initial : {initial}  →  After : {df.shape}")
return df.reset_index(drop=True)
─────────────────────────────────────────────────────────────
3. EDA
─────────────────────────────────────────────────────────────

def run_eda(df, output_dir='.'):
print("\n[EDA] Generating EDA plots...")
import os

fig = plt.figure(figsize=(20, 15))
fig.patch.set_facecolor('#0a0d14')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

tc  = '#e2e8f0'
mal = '#f97316'   # orange — malignant
ben = '#22d3ee'   # cyan   — benign
acc = '#a78bfa'

def style_ax(ax, title):
    ax.set_facecolor('#141824')
    ax.set_title(title, color=tc, fontsize=10, pad=8, fontweight='bold')
    ax.tick_params(colors=tc, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor('#2d3748')

labels = {0: 'Benign', 1: 'Malignant'}

# 3a. Class distribution
ax = fig.add_subplot(gs[0, 0])
counts = df['Label'].value_counts().sort_index()
bars = ax.bar(['Benign','Malignant'], counts.values,
              color=[ben, mal], width=0.5, edgecolor='none')
for b, v in zip(bars, counts.values):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+8,
            f'{v:,}', ha='center', va='bottom', color=tc, fontsize=9)
style_ax(ax, 'Class Distribution')
ax.set_ylabel('Count', color=tc, fontsize=8)

# 3b. Age distribution
ax = fig.add_subplot(gs[0, 1])
for lbl, clr in [(0,ben),(1,mal)]:
    ax.hist(df[df['Label']==lbl]['Age'], bins=30, alpha=0.7,
            color=clr, label=labels[lbl], density=True)
ax.legend(fontsize=7, facecolor='#141824', labelcolor=tc)
style_ax(ax, 'Age Distribution')
ax.set_xlabel('Age (years)', color=tc, fontsize=8)

# 3c. Nodule Size
ax = fig.add_subplot(gs[0, 2])
for lbl, clr in [(0,ben),(1,mal)]:
    ax.hist(df[df['Label']==lbl]['Nodule_Size_mm'], bins=40, alpha=0.7,
            color=clr, label=labels[lbl], density=True)
ax.legend(fontsize=7, facecolor='#141824', labelcolor=tc)
style_ax(ax, 'Nodule Size (mm)')
ax.set_xlabel('Size (mm)', color=tc, fontsize=8)

# 3d. Pack Years vs Nodule Size scatter
ax = fig.add_subplot(gs[1, 0])
for lbl, clr in [(0,ben),(1,mal)]:
    sub = df[df['Label']==lbl]
    ax.scatter(sub['Pack_Years'], sub['Nodule_Size_mm'],
               alpha=0.3, s=10, color=clr, label=labels[lbl])
ax.legend(fontsize=7, facecolor='#141824', labelcolor=tc)
style_ax(ax, 'Pack Years vs Nodule Size')
ax.set_xlabel('Pack Years', color=tc, fontsize=8)
ax.set_ylabel('Nodule Size (mm)', color=tc, fontsize=8)

# 3e. CEA Level box plot
ax = fig.add_subplot(gs[1, 1])
data_bp = [df[df['Label']==0]['CEA_Level'], df[df['Label']==1]['CEA_Level']]
bp = ax.boxplot(data_bp, patch_artist=True,
                medianprops=dict(color='white', linewidth=2))
bp['boxes'][0].set_facecolor(ben+'88')
bp['boxes'][1].set_facecolor(mal+'88')
ax.set_xticklabels(['Benign','Malignant'], color=tc, fontsize=8)
style_ax(ax, 'CEA Level by Class')
ax.set_ylabel('CEA (ng/mL)', color=tc, fontsize=8)

# 3f. PET SUV distribution
ax = fig.add_subplot(gs[1, 2])
for lbl, clr in [(0,ben),(1,mal)]:
    ax.hist(df[df['Label']==lbl]['PET_SUV'], bins=40, alpha=0.7,
            color=clr, label=labels[lbl], density=True)
ax.legend(fontsize=7, facecolor='#141824', labelcolor=tc)
style_ax(ax, 'PET SUV Distribution')
ax.set_xlabel('SUV Max', color=tc, fontsize=8)

# 3g. Symptom severity radar (mean per class)
ax = fig.add_subplot(gs[2, 0])
syms = ['Cough_Severity','Dyspnea','Chest_Pain',
        'Hemoptysis','Weight_Loss','Fatigue']
for lbl, clr in [(0,ben),(1,mal)]:
    means = df[df['Label']==lbl][syms].mean()
    ax.barh(syms, means.values, alpha=0.7, color=clr, label=labels[lbl])
ax.legend(fontsize=7, facecolor='#141824', labelcolor=tc)
style_ax(ax, 'Avg Symptom Severity')
ax.set_xlabel('Mean Score (0–10)', color=tc, fontsize=8)
ax.tick_params(axis='y', labelsize=7)

# 3h. Genetic marker prevalence
ax = fig.add_subplot(gs[2, 1])
markers = ['EGFR_Mutation','KRAS_Mutation','ALK_Rearrangement',
           'Family_History','Spiculation']
x = np.arange(len(markers))
w = 0.35
for i, (lbl, clr) in enumerate([(0,ben),(1,mal)]):
    rates = df[df['Label']==lbl][markers].mean().values * 100
    ax.bar(x + i*w, rates, width=w, color=clr, alpha=0.85, label=labels[lbl])
ax.set_xticks(x + w/2)
ax.set_xticklabels(['EGFR','KRAS','ALK','Fam.Hist','Spic.'], color=tc, fontsize=7)
ax.legend(fontsize=7, facecolor='#141824', labelcolor=tc)
style_ax(ax, 'Genetic Markers & Risk Factors (%)')
ax.set_ylabel('%', color=tc, fontsize=8)

# 3i. Spirometry FEV1 vs DLCO
ax = fig.add_subplot(gs[2, 2])
for lbl, clr in [(0,ben),(1,mal)]:
    sub = df[df['Label']==lbl]
    ax.scatter(sub['FEV1_Pct'], sub['DLCO_Pct'],
               alpha=0.3, s=10, color=clr, label=labels[lbl])
ax.legend(fontsize=7, facecolor='#141824', labelcolor=tc)
style_ax(ax, 'FEV1% vs DLCO%')
ax.set_xlabel('FEV1 (%)', color=tc, fontsize=8)
ax.set_ylabel('DLCO (%)', color=tc, fontsize=8)

fig.suptitle('Lung Cancer Detection — Exploratory Data Analysis',
             fontsize=14, color=tc, fontweight='bold', y=0.99)

path = os.path.join(output_dir, 'eda_report.png')
plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"  EDA saved → {path}")
─────────────────────────────────────────────────────────────
4. FEATURE SELECTION
─────────────────────────────────────────────────────────────

def select_features(df):
print("\n[Feature Selection] Applying multi-method feature selection...")

feature_cols = [c for c in df.columns if c != 'Label']
X = df[feature_cols].values
y = df['Label'].values

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Method 1: Mutual Information
mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
mi_rank   = pd.Series(mi_scores, index=feature_cols).rank(ascending=False)

# Method 2: RFE with Random Forest
rfe_estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rfe = RFE(rfe_estimator, n_features_to_select=20, step=2)
rfe.fit(X_scaled, y)
rfe_selected = set(np.array(feature_cols)[rfe.support_])

# Method 3: RF feature importance
rf_sel = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_sel.fit(X_scaled, y)
fi = pd.Series(rf_sel.feature_importances_, index=feature_cols)
fi_top = set(fi.nlargest(20).index)

# Union of top features from all methods
mi_top     = set(mi_rank.nsmallest(20).index)
selected   = list(mi_top | rfe_selected | fi_top)

print(f"  Mutual Info top-20     : {len(mi_top)}")
print(f"  RFE selected           : {len(rfe_selected)}")
print(f"  RF Importance top-20   : {len(fi_top)}")
print(f"  Union selected features: {len(selected)}")
print(f"  Features: {sorted(selected)}")

return selected, scaler, fi
─────────────────────────────────────────────────────────────
5. FEATURE ENGINEERING
─────────────────────────────────────────────────────────────

def engineer_features(df):
print("\n[Features] Engineering clinical composite features...")
df = df.copy()

# Composite risk score
df['Smoking_Risk']    = df['Smoking_Years'] * df['Pack_Years'] / 100
df['Imaging_Score']   = (df['Nodule_Size_mm'] / 10 +
                          df['PET_SUV'] / 5 +
                          df['Spiculation'] * 2 +
                          df['Ground_Glass'] +
                          df['Consolidation'])
df['Marker_Score']    = (np.log1p(df['CEA_Level']) +
                          np.log1p(df['CYFRA21_1']) +
                          np.log1p(df['NSE_Level']))
df['Symptom_Burden']  = (df['Cough_Severity'] + df['Dyspnea'] +
                          df['Chest_Pain']     + df['Hemoptysis'] +
                          df['Weight_Loss']    + df['Fatigue']) / 6
df['Genetic_Load']    = (df['EGFR_Mutation'] + df['KRAS_Mutation'] +
                          df['ALK_Rearrangement'] + df['Family_History'])
df['Pulm_Function']   = (df['FEV1_Pct'] + df['FVC_Pct'] + df['DLCO_Pct']) / 3
df['CEA_log']         = np.log1p(df['CEA_Level'])
df['CYFRA_log']       = np.log1p(df['CYFRA21_1'])
df['Age_Nodule']      = df['Age'] * df['Nodule_Size_mm'] / 1000
df['Nodule_PET']      = df['Nodule_Size_mm'] * df['PET_SUV']

new = ['Smoking_Risk','Imaging_Score','Marker_Score','Symptom_Burden',
       'Genetic_Load','Pulm_Function','CEA_log','CYFRA_log',
       'Age_Nodule','Nodule_PET']
print(f"  Added {len(new)} engineered features")
return df
─────────────────────────────────────────────────────────────
6. ENSEMBLE MODEL TRAINING
─────────────────────────────────────────────────────────────

def build_ensemble(X_train, X_test, y_train, y_test):
print("\n[Ensemble] Training individual models...")

# ── Individual models ────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_leaf=2,
    class_weight='balanced', n_jobs=-1, random_state=42)

xgb = XGBoostClassifier(
    n_estimators=60, max_depth=3, learning_rate=0.15,
    subsample=0.8, colsample=0.8, reg_lambda=1.5, random_state=42)

gb = GradientBoostingClassifier(
    n_estimators=80, max_depth=3, learning_rate=0.1,
    subsample=0.8, min_samples_leaf=5, random_state=42)

results = {}

for name, model in [('Random Forest', rf),
                     ('XGBoost',       xgb),
                     ('Gradient Boost',gb)]:
    print(f"\n  ── {name} ──")
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results[name] = _score(name, y_test, y_pred, y_proba, model)

# ── Soft Voting Ensemble ─────────────────────────────────
print("\n  ── Soft Voting Ensemble ──")
# Use trained models — manual soft vote
proba_avg = (results['Random Forest']['y_proba'] +
             results['XGBoost']['y_proba'] +
             results['Gradient Boost']['y_proba']) / 3
y_vote = (proba_avg >= 0.5).astype(int)
results['Voting Ensemble'] = _score('Voting Ensemble', y_test, y_vote, proba_avg, None)

# ── Stacking Ensemble (LR meta-learner) ──────────────────
print("\n  ── Stacking Ensemble ──")
skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = [('Random Forest', rf), ('XGBoost', xgb), ('Gradient Boost', gb)]

# Build OOF meta-features
meta_train = np.zeros((len(y_train), len(models)))
for j, (_, mdl) in enumerate(models):
    oof = np.zeros(len(y_train))
    for train_idx, val_idx in skf.split(X_train, y_train):
        mdl.fit(X_train[train_idx], y_train[train_idx])
        oof[val_idx] = mdl.predict_proba(X_train[val_idx])[:, 1]
    meta_train[:, j] = oof

meta_test = np.column_stack([
    m.predict_proba(X_test)[:, 1] for _, m in models])

meta_lr = LogisticRegression(C=1.0, max_iter=500)
meta_lr.fit(meta_train, y_train)
y_stack_proba = meta_lr.predict_proba(meta_test)[:, 1]
y_stack       = (y_stack_proba >= 0.5).astype(int)
results['Stacking Ensemble'] = _score('Stacking Ensemble', y_test, y_stack,
                                       y_stack_proba, None)

return results, rf   # return rf for feature importance

def _score(name, y_true, y_pred, y_proba, model):
auc = roc_auc_score(y_true, y_proba)
ap = average_precision_score(y_true, y_proba)
f1 = f1_score(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
print(f" Accuracy : {acc:.4f} | ROC-AUC : {auc:.4f} | "
f"Avg Prec : {ap:.4f} | F1 : {f1:.4f}")
print(classification_report(y_true, y_pred,
target_names=['Benign','Malignant'], digits=4))
return {'model': model, 'y_pred': y_pred, 'y_proba': y_proba,
'auc': auc, 'ap': ap, 'f1': f1, 'acc': acc}

─────────────────────────────────────────────────────────────
7. EVALUATION PLOTS
─────────────────────────────────────────────────────────────

def plot_results(results, y_test, feature_cols, rf_model, output_dir='.'):
import os
tc = '#e2e8f0'
bg = '#0a0d14'
panel = '#141824'
palette = ['#22d3ee','#f97316','#a78bfa','#facc15','#34d399']

# ── Plot 1: ROC / PR / Confusion / Model comparison ─────
fig = plt.figure(figsize=(20, 12))
fig.patch.set_facecolor(bg)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

def sax(ax, title):
    ax.set_facecolor(panel)
    ax.set_title(title, color=tc, fontsize=11, fontweight='bold')
    ax.tick_params(colors=tc)
    for sp in ax.spines.values(): sp.set_edgecolor('#2d3748')
    ax.xaxis.label.set_color(tc)
    ax.yaxis.label.set_color(tc)

# ROC
ax = fig.add_subplot(gs[0, 0])
ax.plot([0,1],[0,1],'--',color='#4a5568',lw=1)
for (name, res), clr in zip(results.items(), palette):
    fpr,tpr,_ = roc_curve(y_test, res['y_proba'])
    ax.plot(fpr,tpr,color=clr,lw=2,label=f"{name} ({res['auc']:.3f})")
ax.legend(fontsize=7.5, facecolor=panel, labelcolor=tc)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
sax(ax, 'ROC Curves')

# Precision-Recall
ax = fig.add_subplot(gs[0, 1])
for (name, res), clr in zip(results.items(), palette):
    p,r,_ = precision_recall_curve(y_test, res['y_proba'])
    ax.plot(r,p,color=clr,lw=2,label=f"{name} (AP={res['ap']:.3f})")
ax.legend(fontsize=7.5, facecolor=panel, labelcolor=tc)
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
sax(ax, 'Precision-Recall Curves')

# Confusion matrix — stacking ensemble
best = 'Stacking Ensemble'
cm   = confusion_matrix(y_test, results[best]['y_pred'])
ax   = fig.add_subplot(gs[0, 2])
im   = ax.imshow(cm, cmap='YlOrRd', aspect='auto')
ax.set_xticks([0,1]); ax.set_xticklabels(['Benign','Malignant'], color=tc)
ax.set_yticks([0,1]); ax.set_yticklabels(['Benign','Malignant'], color=tc)
for i in range(2):
    for j in range(2):
        ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center',
                fontsize=14,fontweight='bold',
                color='white' if cm[i,j]>cm.max()/2 else '#1a1d27')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
sax(ax, f'Confusion Matrix ({best})')
plt.colorbar(im, ax=ax, shrink=0.8)

# Model Comparison bar chart
ax = fig.add_subplot(gs[1, 0])
metrics = ['acc','auc','f1','ap']
m_labels= ['Accuracy','ROC-AUC','F1','Avg Prec']
x = np.arange(len(metrics))
w = 0.15
for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metrics]
    ax.bar(x + i*w, vals, width=w, color=palette[i], alpha=0.85, label=name)
ax.set_xticks(x + w*2)
ax.set_xticklabels(m_labels, color=tc, fontsize=8)
ax.set_ylim(0.7, 1.02)
ax.legend(fontsize=6.5, facecolor=panel, labelcolor=tc)
sax(ax, 'Model Performance Comparison')
ax.set_ylabel('Score', color=tc)

# Feature importance
ax = fig.add_subplot(gs[1, 1:])
fi   = pd.Series(rf_model.feature_importances_, index=feature_cols).nlargest(18).sort_values()
clrs = plt.cm.YlOrRd(np.linspace(0.35, 0.85, len(fi)))
bars = ax.barh(fi.index, fi.values, color=clrs, edgecolor='none')
for bar, v in zip(bars, fi.values):
    ax.text(v + 0.001, bar.get_y()+bar.get_height()/2,
            f'{v:.4f}', va='center', fontsize=7.5, color=tc)
ax.set_xlabel('Importance', color=tc)
sax(ax, 'Top 18 Feature Importances (Random Forest)')

fig.suptitle('Lung Cancer Detection — Ensemble Model Evaluation',
             fontsize=14, color=tc, fontweight='bold', y=1.01)

path = os.path.join(output_dir, 'model_results.png')
plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=bg)
plt.close()
print(f"\n  Model results saved → {path}")

# ── Plot 2: Threshold analysis for best model ─────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(bg)

best_proba = results['Stacking Ensemble']['y_proba']
thresholds = np.linspace(0.1, 0.9, 80)
f1s, sens, spec = [], [], []
for t in thresholds:
    yp = (best_proba >= t).astype(int)
    f1s.append(f1_score(y_test, yp, zero_division=0))
    cm_ = confusion_matrix(y_test, yp)
    sens.append(cm_[1,1] / (cm_[1,1]+cm_[1,0]+1e-9))
    spec.append(cm_[0,0] / (cm_[0,0]+cm_[0,1]+1e-9))

ax = axes[0]
ax.set_facecolor(panel)
ax.plot(thresholds, f1s,   color='#facc15', lw=2, label='F1 Score')
ax.plot(thresholds, sens,  color='#f97316', lw=2, label='Sensitivity (Recall)')
ax.plot(thresholds, spec,  color='#22d3ee', lw=2, label='Specificity')
ax.axvline(0.5, color='#6b7280', lw=1, linestyle='--', label='Default (0.5)')
best_t = thresholds[np.argmax(f1s)]
ax.axvline(best_t, color='#a78bfa', lw=1.5, linestyle='--',
           label=f'Best F1 threshold ({best_t:.2f})')
ax.legend(fontsize=8, facecolor=panel, labelcolor=tc)
ax.set_xlabel('Classification Threshold', color=tc)
ax.set_ylabel('Score', color=tc)
ax.set_title('Threshold Analysis — Stacking Ensemble', color=tc, fontweight='bold')
ax.tick_params(colors=tc)
for sp in ax.spines.values(): sp.set_edgecolor('#2d3748')

# Score distribution
ax = axes[1]
ax.set_facecolor(panel)
ax.hist(best_proba[y_test==0], bins=40, alpha=0.7, color='#22d3ee',
        label='Benign', density=True)
ax.hist(best_proba[y_test==1], bins=40, alpha=0.7, color='#f97316',
        label='Malignant', density=True)
ax.axvline(0.5, color='white', lw=1.5, linestyle='--', label='Threshold 0.5')
ax.legend(fontsize=8, facecolor=panel, labelcolor=tc)
ax.set_xlabel('Predicted Probability (Malignant)', color=tc)
ax.set_ylabel('Density', color=tc)
ax.set_title('Prediction Score Distribution', color=tc, fontweight='bold')
ax.tick_params(colors=tc)
for sp in ax.spines.values(): sp.set_edgecolor('#2d3748')

fig.tight_layout()
path2 = os.path.join(output_dir, 'threshold_analysis.png')
plt.savefig(path2, dpi=150, bbox_inches='tight', facecolor=bg)
plt.close()
print(f"  Threshold analysis saved → {path2}")
─────────────────────────────────────────────────────────────
8. MAIN PIPELINE
─────────────────────────────────────────────────────────────

def main():
import os
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("  LUNG CANCER DETECTION — ENSEMBLE PIPELINE")
print("=" * 60)

# Step 1: Generate data
print("\n[Data] Generating clinical dataset...")
df = generate_lung_cancer_data(n_samples=2000, seed=42)
print(f"  Shape: {df.shape} | "
      f"Malignant: {df['Label'].sum()} | Benign: {(df['Label']==0).sum()}")

# Step 2: Clean
df = clean_data(df)

# Step 3: EDA
run_eda(df, output_dir)

# Step 4: Feature engineering
df = engineer_features(df)

# Step 5: Feature selection
selected_feats, scaler, fi_series = select_features(df)

# Step 6: Prepare train/test split
X = scaler.fit_transform(df[selected_feats].values)
y = df['Label'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print(f"\n[Split] Train: {X_train.shape} | Test: {X_test.shape}")

# Step 7: Train ensemble
results, rf_model = build_ensemble(X_train, X_test, y_train, y_test)

# Step 8: Plots
plot_results(results, y_test, selected_feats, rf_model, output_dir)

# Summary
print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)
print(f"\n  Output directory : ./{output_dir}/")
print(f"  ├── eda_report.png         (8-panel clinical EDA)")
print(f"  ├── model_results.png      (ROC / PR / CM / Feature importance)")
print(f"  └── threshold_analysis.png (threshold tuning + score dist.)")
print()
print(f"  {'Model':<22}  {'Accuracy':>9}  {'ROC-AUC':>9}  {'F1':>9}")
print(f"  {'-'*22}  {'-'*9}  {'-'*9}  {'-'*9}")
for name, res in results.items():
    print(f"  {name:<22}  {res['acc']:>9.4f}  {res['auc']:>9.4f}  {res['f1']:>9.4f}")
best = max(results, key=lambda k: results[k]['auc'])
print(f"\n  ✓ Best Model: {best}  (AUC={results[best]['auc']:.4f})\n")

if name == 'main':
main()

Close
