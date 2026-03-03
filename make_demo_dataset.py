import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

INFILE = "day_approach_maskedID_timeseries.csv"
OUTFILE = "day_demo_2012_2019.csv"

# --- 1) Charger
df = pd.read_csv(INFILE)

# S'assurer que l'ID est stable
df["Athlete ID"] = df["Athlete ID"].astype(float).astype(int)

# --- 2) Générer des dates cohérentes papier (2012–2019)
# Le papier indique une période de 7 ans (2012–2019). :contentReference[oaicite:3]{index=3}
START = pd.Timestamp("2012-01-01")
END = pd.Timestamp("2019-12-31")

def assign_dates_over_period(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values(by=g.columns.tolist(), kind="mergesort").reset_index(drop=True)
    # dates espacées sur toute la période, sans supposer 1 ligne par jour
    g["Date"] = pd.date_range(start=START, end=END, periods=len(g))
    return g

df = df.groupby("Athlete ID", group_keys=False).apply(assign_dates_over_period)

# --- 3) Modifs "méthode B" (démo réaliste)
# Idée: on simule un peu plus de charge + un peu moins de récup, avec bruit
# et sans casser la structure des colonnes.
def apply_demo_transform(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Colonnes candidates
    km_cols = [c for c in out.columns if c.startswith("total km.")]
    z34_cols = [c for c in out.columns if c.startswith("km Z3-4.")]
    z5_cols = [c for c in out.columns if c.startswith("km Z5-T1-T2.")]
    sessions_cols = [c for c in out.columns if c.startswith("nr. sessions.")]
    rec_cols = [c for c in out.columns if c.startswith("perceived recovery.")]
    ex_cols = [c for c in out.columns if c.startswith("perceived exertion.")]
    ts_cols = [c for c in out.columns if c.startswith("perceived trainingSuccess.")]
    strength_cols = [c for c in out.columns if c.startswith("strength training.")]
    alt_cols = [c for c in out.columns if c.startswith("Hours alternative.")]

    # 3.1 volume (km) : +10% à +25% selon une variabilité
    if km_cols:
        factor = rng.normal(loc=1.15, scale=0.07, size=len(out))
        factor = np.clip(factor, 1.00, 1.30)
        out[km_cols] = out[km_cols].mul(factor, axis=0)

    # 3.2 intensité : un peu plus de Z5 (ex: bloc d'intervalles), +0% à +20%
    if z5_cols:
        factor = rng.normal(loc=1.08, scale=0.08, size=len(out))
        factor = np.clip(factor, 0.95, 1.25)
        out[z5_cols] = out[z5_cols].mul(factor, axis=0)

    # 3.3 récupération perçue : légère baisse (fatigue), -0% à -15%
    if rec_cols:
        delta = rng.normal(loc=-0.06, scale=0.03, size=len(out))
        out[rec_cols] = out[rec_cols].add(delta, axis=0)

    # 3.4 effort perçu : légère hausse, +0 à +0.10
    if ex_cols:
        delta = rng.normal(loc=0.04, scale=0.03, size=len(out))
        out[ex_cols] = out[ex_cols].add(delta, axis=0)

    # 3.5 un peu de bruit léger partout (évite les patterns trop “propres”)
    for cols in [km_cols, z34_cols, z5_cols, rec_cols, ex_cols, ts_cols]:
        if cols:
            noise = rng.normal(loc=0.0, scale=0.02, size=(len(out), len(cols)))
            out[cols] = out[cols].to_numpy() + noise

    # --- 4) Clipping dans des plages “typiques” Table 1 (day approach) :contentReference[oaicite:4]{index=4}
    # Table 1 (day): (valeurs typiques)
    # nr. sessions [0,2]
    # total distance [0,25]
    # km Z3-4 [0,15]
    # km Z5-T1-T2 [0,10]
    # strength training [0,1]
    # hours alternative [0,3]
    # perceived exertion/success/recovery [0,1]
    def clip_cols(cols, lo, hi):
        if cols:
            out[cols] = out[cols].clip(lo, hi)

    clip_cols(sessions_cols, 0, 2)
    clip_cols(km_cols, 0.0, 25.0)
    clip_cols(z34_cols, 0.0, 15.0)
    clip_cols(z5_cols, 0.0, 10.0)
    clip_cols(strength_cols, 0.0, 1.0)
    clip_cols(alt_cols, 0.0, 3.0)
    clip_cols(ex_cols, 0.0, 1.0)
    clip_cols(ts_cols, 0.0, 1.0)
    clip_cols(rec_cols, 0.0, 1.0)

    return out

df_demo = apply_demo_transform(df)

# --- 5) Sauvegarde
df_demo.to_csv(OUTFILE, index=False)
print(f"✅ Saved demo dataset: {OUTFILE} (rows={len(df_demo)})")