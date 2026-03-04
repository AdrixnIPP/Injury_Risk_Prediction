import os
import random
import csv
from os import path

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

import sklearn.metrics as metrics
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, confusion_matrix

# ==========================================
# FILES
# ==========================================
DATA_DEMO_GZ = "day_demo_2012_2019.csv.gz"


# ==========================================
# Utility metrics
# ==========================================
def getFScore(beta, PR, RE):
    if PR == 0 and RE == 0:
        return 0
    return (1 + (beta * beta)) * (PR * RE) / ((beta * beta * PR) + RE)

def getPerformanceMeasurements(y_test, y_prob, in_thresh):
    thresh = 0.5 if in_thresh == -1 else in_thresh
    cm = confusion_matrix(y_test, y_prob >= thresh)

    TP = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]
    TN = cm[0][0]

    if (TP + FP) == 0 or (TP + FN) == 0 or ((TN + FP) == 0) or (TN + FN) == 0:
        PR = 0
        mcc = 0
    else:
        PR = TP / (TP + FP)
        mcc = ((TP * TN) - (FP * FN)) / (np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))

    RE = TP / (TP + FN) if (TP + FN) else 0
    SP = TN / (TN + FP) if (TN + FP) else 0
    acc = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) else 0

    F05 = getFScore(0.5, PR, RE)
    F1 = getFScore(1, PR, RE)
    F2 = getFScore(2, PR, RE)

    return PR, RE, SP, F1, F2, acc, mcc, cm, TP, FP, TN, FN

def getStats(y_test, y_prob, in_thresh):
    auc = roc_auc_score(y_test, y_prob)

    fpr = np.linspace(0, 1, 101)
    fpr_org, tpr_org, threshold = metrics.roc_curve(y_test, y_prob)
    tpr = np.interp(fpr, fpr_org, tpr_org)

    if in_thresh is None:
        n_thresh = len(threshold)
        relist = np.zeros(n_thresh)
        splist = np.zeros(n_thresh)

        for th in range(n_thresh):
            _, re, sp, *_ = getPerformanceMeasurements(y_test, y_prob, threshold[th])
            relist[th] = re
            splist[th] = sp

        idx = np.argmin(np.abs(relist - splist))
        best_thresh = threshold[idx]
        stats = {"thresh": best_thresh, "auc": auc, "fpr": fpr, "tpr": tpr}
    else:
        stats = {"thresh": in_thresh, "auc": auc, "fpr": fpr, "tpr": tpr}

    return stats


# ==========================================
# Normalization (z-score per athlete based on healthy events)
# + safe fallbacks
# ==========================================
def getMeanStd(data: pd.DataFrame):
    """
    Means/stds computed on healthy events (injury==0) per athlete.
    If an athlete has no healthy rows, we fall back to global stats.
    """
    data_num = data.drop(columns=["Date"], errors="ignore").copy()

    healthy = data_num[data_num["injury"] == 0].copy()
    if healthy.empty:
        # fallback: use all rows
        healthy = data_num.copy()

    mean_df = healthy.groupby("Athlete ID").mean(numeric_only=True)
    std_df = healthy.groupby("Athlete ID").std(numeric_only=True)

    std_df.replace(to_replace=0.0, value=0.01, inplace=True)

    global_mean = mean_df.mean(numeric_only=True)
    global_std = std_df.mean(numeric_only=True).replace(0.0, 0.01)

    return mean_df, std_df, global_mean, global_std

def normalize_row(row: pd.Series, mean_df, std_df, global_mean, global_std):
    athlete_id = row["Athlete ID"]
    exclude = {"injury", "Date", "Athlete ID"}
    feat_cols = [c for c in row.index if c not in exclude]

    if athlete_id in mean_df.index:
        mu = mean_df.loc[athlete_id]
        su = std_df.loc[athlete_id]
    else:
        mu = global_mean
        su = global_std

    out = row.copy()
    out[feat_cols] = (row[feat_cols] - mu[feat_cols]) / su[feat_cols]
    return out


# ==========================================
# Data loading
# ==========================================
def loadData():
    df = pd.read_csv(DATA_DEMO_GZ, compression="gzip")
    df["Athlete ID"] = df["Athlete ID"].astype(float).astype(int)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(["Athlete ID", "Date"])
    return df


# ==========================================
# Calibration curve + ROC
# ==========================================
def plotCalibrationCurve(y_test, y_pred, filename):
    plt.figure(figsize=(7, 7))
    frac_pos, mean_pred = calibration_curve(y_test, y_pred, n_bins=10, strategy="quantile")
    plt.plot(mean_pred, frac_pos, "s-", label="XGBoost")
    plt.plot([0, 1], [0, 1], ls="--", c="0.3")
    plt.ylabel("Fréquence observée (positifs)")
    plt.xlabel("Probabilité prédite")
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.legend(loc="lower right")
    plt.title("Courbe de calibration (reliability)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plotROC(val_fpr, val_tpr, val_auc, test_fpr, test_tpr, test_auc, filename):
    plt.figure(figsize=(9, 6))
    plt.plot(val_fpr, val_tpr, lw=2, label=f"Validation (AUC = {val_auc:.4f})")
    plt.plot(test_fpr, test_tpr, lw=2, label=f"Test (AUC = {test_auc:.4f})")
    plt.plot([0, 1], [0, 1], lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ==========================================
# ACWR computation + plot
# ==========================================
def compute_acute_km_day(df_day: pd.DataFrame) -> pd.Series:
    cols = [c for c in df_day.columns if c.startswith("total km.")]
    if not cols:
        return pd.Series(np.nan, index=df_day.index)
    return df_day[cols].sum(axis=1, numeric_only=True)

def add_acwr(df_day: pd.DataFrame, chronic_window_days: int = 28) -> pd.DataFrame:
    out = df_day.copy()
    if "Date" not in out.columns:
        return out

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.sort_values(["Athlete ID", "Date"])
    out["Acute_7d_km"] = compute_acute_km_day(out)

    def _per_athlete(g):
        g = g.sort_values("Date").copy()
        g["Chronic_28d_km"] = g["Acute_7d_km"].rolling(chronic_window_days, min_periods=7).mean()
        g["ACWR"] = g["Acute_7d_km"] / g["Chronic_28d_km"]
        return g

    out = out.groupby("Athlete ID", group_keys=False).apply(_per_athlete)
    return out

def plot_acwr_for_athlete(df_day: pd.DataFrame, athlete_id: int, filename: str):
    g = df_day[df_day["Athlete ID"] == athlete_id].copy()
    g = g.dropna(subset=["Date"]).sort_values("Date")
    if g.empty or "ACWR" not in g.columns:
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(g["Date"], g["Acute_7d_km"], label="Charge aiguë (7j km)")
    ax1.plot(g["Date"], g["Chronic_28d_km"], label="Charge chronique (moyenne 28j)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Kilomètres")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(g["Date"], g["ACWR"], linestyle="--", label="ACWR")
    ax2.set_ylabel("ACWR")
    ax2.axhline(1.5, linestyle=":", linewidth=1)
    ax2.legend(loc="upper right")

    plt.title(f"Suivi ACWR — Athlète {athlete_id}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ==========================================
# Model training / bagging
# ==========================================
def trainModel(params, dftrain, df_val, mean_df, std_df, global_mean, global_std, calibrate_method):
    y_train = np.array(dftrain["injury"]).astype(int)
    y_val = np.array(df_val["injury"]).astype(int)

    data_train = dftrain.copy()
    data_val = df_val.copy()

    # Normalize row-wise
    data_train = data_train.apply(lambda r: normalize_row(r, mean_df, std_df, global_mean, global_std), axis=1)
    data_val = data_val.apply(lambda r: normalize_row(r, mean_df, std_df, global_mean, global_std), axis=1)

    # Build X matrices
    X_train = data_train.drop(columns=["injury", "Date", "Athlete ID"], errors="ignore").to_numpy()
    X_val = data_val.drop(columns=["injury", "Date", "Athlete ID"], errors="ignore").to_numpy()

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        learning_rate=0.01,
        max_depth=random.choice(params["XGBDepthList"]),
        n_estimators=random.choice(params["XGBEstimatorsList"]),
        importance_type="total_gain",
        eval_metric="auc",
        verbosity=1,
    )
    model.fit(X_train, y_train)

    calib_model = CalibratedClassifierCV(model, method=calibrate_method, cv="prefit")
    calib_model.fit(X_val, y_val)

    return model, calib_model

def getBalancedSubset(X_train: pd.DataFrame, samplesPerClass: int):
    # Keep athletes who have both classes
    stats = (
        X_train[["Athlete ID", "injury"]]
        .groupby(["Athlete ID", "injury"])
        .size()
        .reset_index(name="counts")
    )
    stats = stats.groupby("Athlete ID").size().reset_index(name="counts")
    stats = stats[stats["counts"] >= 2]
    athleteList = stats["Athlete ID"].unique()

    if len(athleteList) == 0:
        # fallback: global balanced sample
        inj = X_train[X_train["injury"] == 1]
        ok = X_train[X_train["injury"] == 0]
        n = min(len(inj), len(ok), samplesPerClass)
        return pd.concat([inj.sample(n, replace=True), ok.sample(n, replace=True)], ignore_index=True)

    samplesPerAthlete = max(1, int(np.floor(samplesPerClass / len(athleteList))))

    healthySet = []
    injurySet = []
    for athlete in athleteList:
        inj = X_train[(X_train["Athlete ID"] == athlete) & (X_train["injury"] == 1)]
        ok = X_train[(X_train["Athlete ID"] == athlete) & (X_train["injury"] == 0)]
        if len(inj) == 0 or len(ok) == 0:
            continue
        injurySet.append(inj.sample(samplesPerAthlete, replace=True))
        healthySet.append(ok.sample(samplesPerAthlete, replace=True))

    balanced = pd.concat(injurySet + healthySet, ignore_index=True)
    return balanced

def predict_proba_bagging(modelList, X: np.ndarray) -> np.ndarray:
    probs = []
    for model in modelList:
        probs.append(model.predict_proba(X)[:, 1])
    return np.mean(np.vstack(probs), axis=0)

def applyBagging(modelList, X_test, mean_df, std_df, global_mean, global_std, in_thresh, filename):
    y_test = np.array(X_test["injury"]).astype(int)

    dfX = X_test.copy()
    dfX = dfX.apply(lambda r: normalize_row(r, mean_df, std_df, global_mean, global_std), axis=1)
    X = dfX.drop(columns=["injury", "Date", "Athlete ID"], errors="ignore").to_numpy()

    y_prob_bag = predict_proba_bagging(modelList, X)
    plotCalibrationCurve(y_test, y_prob_bag, filename)

    stats = getStats(y_test, y_prob_bag, None if in_thresh is None else in_thresh)
    thresh = stats["thresh"]
    return thresh, stats["auc"], stats["fpr"], stats["tpr"], y_test, y_prob_bag

def writeResults(outdir, row):
    resultsFilename = f"./{outdir}/results.csv"
    header = ["N_test_athletes", "nbags", "exp", "test_thresh", "val_auc", "test_auc"]
    write_header = not path.exists(resultsFilename)

    with open(resultsFilename, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

def runExperiment(params, exp):
    outdir = f"demo_day_{params['samplesPerClass']}"
    os.makedirs(outdir, exist_ok=True)

    df = loadData()
    df = add_acwr(df, chronic_window_days=28)

    athletes = sorted(df["Athlete ID"].unique().tolist())
    test_athletes = athletes[-params["nTestAthletes"]:]
    X_test = df[df["Athlete ID"].isin(test_athletes)].copy()

    X_trainval = df[~df["Athlete ID"].isin(test_athletes)].copy()

    # stats for normalization computed on trainval
    mean_df, std_df, global_mean, global_std = getMeanStd(X_trainval)

    # Validation subset (balanced)
    X_val = getBalancedSubset(X_trainval, params["samplesPerClass"])

    modelList = []
    for _ in range(params["nbags"]):
        X_train_bag = getBalancedSubset(X_trainval, params["samplesPerClass"])
        _, calib_model = trainModel(
            params, X_train_bag, X_val,
            mean_df, std_df, global_mean, global_std,
            params["calibrationType"]
        )
        modelList.append(calib_model)

    # Create val_samples matching test class counts (if possible)
    n_ok = len(X_test[X_test["injury"] == 0])
    n_inj = len(X_test[X_test["injury"] == 1])
    ok_pool = X_trainval[X_trainval["injury"] == 0]
    inj_pool = X_trainval[X_trainval["injury"] == 1]
    n_ok = min(n_ok, len(ok_pool))
    n_inj = min(n_inj, len(inj_pool))

    val_samples = pd.concat(
        [
            ok_pool.sample(n_ok, replace=False) if n_ok > 0 else ok_pool.sample(min(100, len(ok_pool)), replace=True),
            inj_pool.sample(n_inj, replace=False) if n_inj > 0 else inj_pool.sample(min(100, len(inj_pool)), replace=True),
        ],
        ignore_index=True,
    )

    val_thresh, val_auc, val_fpr, val_tpr, _, _ = applyBagging(
        modelList, val_samples, mean_df, std_df, global_mean, global_std,
        in_thresh=None, filename=f"./{outdir}/calibrate_validation_{exp}.png"
    )

    test_thresh, test_auc, test_fpr, test_tpr, y_test, y_prob = applyBagging(
        modelList, X_test, mean_df, std_df, global_mean, global_std,
        in_thresh=val_thresh, filename=f"./{outdir}/calibrate_test_{exp}.png"
    )

    plotROC(val_fpr, val_tpr, val_auc, test_fpr, test_tpr, test_auc, f"./{outdir}/ROC_{exp}.png")
    writeResults(outdir, [params["nTestAthletes"], params["nbags"], exp, test_thresh, val_auc, test_auc])

    # quick ACWR plot for first test athlete
    if len(test_athletes) > 0:
        aid = int(test_athletes[0])
        plot_acwr_for_athlete(df, aid, f"./{outdir}/ACWR_athlete_{aid}_exp{exp}.png")

    return val_auc, test_auc

def main():
    params = {
        "nTestAthletes": 10,
        "nbags": 9,
        "calibrationType": "sigmoid",
        "nExp": 5,
        "samplesPerClass": 2048,
        "XGBEstimatorsList": [256, 512],
        "XGBDepthList": [2, 3],
    }

    for exp in range(params["nExp"]):
        print(f"Demo Day | exp={exp+1}/{params['nExp']}")
        val_auc, test_auc = runExperiment(params, exp)
        print(f"AUC Val={val_auc:.4f} | AUC Test={test_auc:.4f}")

if __name__ == "__main__":
    main()
