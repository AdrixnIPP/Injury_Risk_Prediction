import pickle
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import shap

from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve


# =========================
# FILES
# =========================
DATA = "day_demo_2012_2019.csv"
MODEL_PATH = "xgb_model.pkl"
CALIB_PATH = "calib_model.pkl"


# =========================
# i18n (EN/FR) — FR completed + richer
# =========================
I18N = {
    "en": {
        "lang_label": "Language",
        "title": "Injury Risk Prediction Dashboard",
        "caption": (
            "Source: Lövdal, S. S., den Hartigh, R. J. R., & Azzopardi, G. (2021). "
            "Injury Prediction in Competitive Runners With Machine Learning. "
            "International Journal of Sports Physiology and Performance, 16(10), 1522–1531. "
            "https://doi.org/10.1123/ijspp.2020-0518"
        ),
        "filters": "Filters",
        "athletes": "Athletes",
        "years": "Year",
        "time_window": "Time window",
        "summary": "Selection summary",
        "context": "Context & purpose",
        "visual": "Monitoring views",
        "actions": "Decisions & explanations",
        "risk_trend_title": "Injury risk score over time (per athlete)",
        "risk_trend_help": (
            "Use this to spot rising trends or unstable periods. "
            "A sustained increase often suggests a load/recovery imbalance to investigate."
        ),
        "acwr_title": "ACWR over time (Acute:Chronic Workload Ratio)",
        "acwr_help": (
            "ACWR compares recent load (7 days) vs the athlete’s habitual load (28-day baseline). "
            "Spikes can signal sudden load changes that should be managed."
        ),
        "risk_vs_load_title": "Risk vs short-term load (Acute 7-day km)",
        "risk_vs_load_help": (
            "Shows how the risk score behaves relative to recent volume. "
            "Clusters at high load + high risk help prioritize monitoring and recovery actions."
        ),
        "heatmap_title": "Long-term heatmap (seasonality & trends)",
        "heatmap_help": (
            "Useful to understand season blocks (build/compete/off) and long-term patterns. "
            "Helps staff contextualize risk/load by quarter or year."
        ),
        "focus_athlete": "Focus athlete",
        "pick_day": "Pick a day",
        "drivers_title": "Main drivers (why this score?)",
        "drivers_help": (
            "These explanations come from SHAP values. "
            "Positive contributions push the risk up; negative contributions push it down. "
            "Use them as coaching signals (load/recovery balance), not as medical causality."
        ),
        "drivers_howto": (
            "**How to read this:**\n"
            "- **Direction:** whether the factor increased or decreased the risk score on that day.\n"
            "- **Magnitude:** how strong the influence was compared to other variables.\n"
            "- **Actionable use:** identify controllable levers (volume, intensity mix, session count, recovery signals).\n"
            "\n"
            "> Important: SHAP explains the model’s decision, not the biological cause of injuries."
        ),
        "contrib_heatmap_title": "SHAP contribution heatmap (recent days)",
        "contrib_heatmap_help": (
            "Red pushes risk up, blue pushes it down. "
            "Use it to quickly see which factors repeatedly drive risk during the recent period."
        ),
        "ml_audit": "🧠 Machine Learning details (model audit)",
        "raw_data": "Raw data (selected athletes)",
        "about": """
### Motivation
Staying injury-free is one of the most important factors for success in sport, yet injuries are hard to predict. Wearables and data science make it possible to monitor training load and detect early warning signals.

This project builds on a study that tested machine learning injury prediction in competitive runners using detailed training logs. The dataset includes 74 high-level runners tracked over seven years, combining objective GPS metrics (distance, duration) with subjective feedback (perceived effort, recovery, training success).

The best-performing approach uses day-by-day information from the 7 days preceding an injury. The model family is **bagged XGBoost**, which captures non-linear interactions between load, intensity and recovery.

### What is this dashboard for?
This dashboard translates the research into a practical staff tool:
- track **risk score** trajectories per athlete,
- monitor **workload indicators** (including ACWR),
- and use **SHAP** to understand what is driving risk up or down for each athlete/day.

### Why XGBoost & SHAP?
- **XGBoost** is strong on tabular data and handles complex interactions typical of training/injury dynamics.
- **SHAP** improves trust and usability by explaining model outputs (beyond a black-box risk number).

> ⚠️ The risk score is not a medical diagnosis. It is a monitoring signal to support decisions.
""",
        "guidance": """
**Practical guidance**
- Green (<0.30): keep progression & monitor recovery  
- Amber (0.30–0.60): reduce intensity, add easy days, emphasize recovery  
- Red (>0.60): reduce load & intensity, monitor symptoms closely
""",
    },

    "fr": {
        "lang_label": "Langue",
        "title": "Dashboard de prédiction du risque de blessure",
        "caption": (
            "Source : Lövdal, S. S., den Hartigh, R. J. R., & Azzopardi, G. (2021). "
            "Injury Prediction in Competitive Runners With Machine Learning. "
            "International Journal of Sports Physiology and Performance, 16(10), 1522–1531. "
            "https://doi.org/10.1123/ijspp.2020-0518"
        ),
        "filters": "Filtres",
        "athletes": "Athlètes",
        "years": "Année",
        "time_window": "Fenêtre temporelle",
        "summary": "Résumé de la sélection",
        "context": "Contexte & objectif",
        "visual": "Vues de monitoring",
        "actions": "Décisions & explications",
        "risk_trend_title": "Évolution du score de risque de blessure dans le temps (par athlète)",
        "risk_trend_help": (
            "Ce graphique sert à repérer les tendances à la hausse et les périodes instables. "
            "Une hausse durable peut indiquer un déséquilibre charge / récupération à investiguer."
        ),
        "acwr_title": "Évolution de l’ACWR (ratio charge aiguë / chronique) dans le temps",
        "acwr_help": (
            "L’ACWR compare la charge récente (7 jours) à la charge habituelle (baseline sur 28 jours). "
            "Des pics peuvent signaler une montée trop rapide de charge à encadrer."
        ),
        "risk_vs_load_title": "Risque vs charge récente (volume sur 7 jours)",
        "risk_vs_load_help": (
            "Ce graphique met en regard le score de risque et le volume récent. "
            "Les zones « forte charge + risque élevé » aident à prioriser la vigilance, l’ajustement de charge et la récup."
        ),
        "heatmap_title": "Heatmap long terme (saisonnalité & tendances)",
        "heatmap_help": (
            "Utile pour lire les cycles de saison (construction / compétition / récupération) et les tendances de fond. "
            "Aide le staff à contextualiser risque et charge par trimestre ou par année."
        ),
        "focus_athlete": "Athlète focus",
        "pick_day": "Choisir un jour",
        "drivers_title": "Facteurs principaux (pourquoi ce score ?)",
        "drivers_help": (
            "Ces explications proviennent des valeurs SHAP. "
            "Une contribution positive augmente le risque ; une contribution négative le diminue. "
            "À utiliser comme signal d’aide à la décision (équilibre charge/récup), pas comme causalité médicale."
        ),
        "drivers_howto": (
            "**Comment lire ces facteurs :**\n"
            "- **Sens (hausse/baisse)** : indique si la variable pousse le score de risque vers le haut ou vers le bas ce jour-là.\n"
            "- **Amplitude** : mesure l’importance relative par rapport aux autres variables.\n"
            "- **Usage terrain** : identifier des leviers actionnables (volume, répartition intensités, nombre de séances, signaux de récup).\n"
            "\n"
            "> Important : SHAP explique la décision du modèle, pas la cause biologique de la blessure."
        ),
        "contrib_heatmap_title": "Heatmap des contributions SHAP (jours récents)",
        "contrib_heatmap_help": (
            "Rouge = pousse le risque à la hausse. Bleu = pousse le risque à la baisse. "
            "Permet de voir rapidement quels facteurs reviennent souvent quand le risque monte."
        ),
        "ml_audit": "🧠 Détails Machine Learning (audit modèle)",
        "raw_data": "Données brutes (athlètes sélectionnés)",
        "about": """
### Motivation
Être le plus possible « injury-free » est un facteur clé de performance sportive, mais la blessure reste difficile à prévoir. Les wearables et la data permettent aujourd’hui de mieux suivre la charge d’entraînement et d’identifier des signaux précoces.

Ce projet s’appuie sur une étude qui teste la prédiction de blessures chez des coureurs compétitifs via du machine learning et des carnets d’entraînement détaillés. Le jeu de données contient 74 coureurs de haut niveau suivis sur 7 ans, avec un mix de données objectives GPS (distance, durée) et de données subjectives (effort perçu, récupération, ressenti de réussite de séance).

L’approche la plus performante exploite des informations **jour par jour** sur les **7 jours précédant** une blessure. La famille de modèles utilisée est **XGBoost en bagging**, capable de capturer des interactions non linéaires entre charge, intensité et récupération.

### Objectif du dashboard interactif
Ce dashboard transforme le travail de recherche en outil opérationnel pour un staff :
- suivre l’**évolution du score de risque** par athlète,
- monitorer des indicateurs de **charge** (dont l’ACWR),
- et comprendre les **facteurs explicatifs** via SHAP (ce qui tire le risque vers le haut ou vers le bas pour un jour et un athlète donné).

### Pourquoi XGBoost & SHAP ?
- **XGBoost** est très performant sur des données tabulaires et gère bien les interactions complexes typiques de la dynamique charge/blessure.
- **SHAP** apporte de la transparence : on ne se contente pas d’un score, on peut expliquer *pourquoi* le modèle alerte, ce qui est crucial pour l’adhésion des coachs.

> ⚠️ Le score de risque n’est pas un diagnostic médical : c’est un signal de monitoring pour aider à la décision.
""",
        "guidance": """
**Recommandations terrain**
- Vert (<0.30) : poursuivre la progression & surveiller la récup  
- Orange (0.30–0.60) : réduire l’intensité, ajouter du facile, focus récupération  
- Rouge (>0.60) : réduire charge & intensité, surveiller les symptômes de près
""",
    }
}


# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="Injury Risk Dashboard", page_icon="🏃", layout="wide")


# =========================
# VISUAL THEME (improved)
# =========================
st.markdown(
    """
    <style>
      .stApp { background: #E9E9EE; }
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

      /* Container cards */
      div[data-testid="stContainer"] {
        background: #F6F6F8;
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 18px;
        padding: 16px 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
      }

      h1, h2, h3 { letter-spacing: -0.02em; }
      h2 { margin-top: 0.6rem; }
      div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

      /* Sidebar spacing */
      section[data-testid="stSidebar"] > div { padding-top: 1.0rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

def polish_chart(chart: alt.Chart):
    return (
        chart.configure_view(strokeOpacity=0, fill="transparent")
        .properties(background="transparent")
        .configure_axis(
            labelFontSize=12, titleFontSize=13,
            grid=True, gridOpacity=0.18, tickSize=3,
            labelColor="#222", titleColor="#222",
        )
        .configure_legend(titleFontSize=12, labelFontSize=11, orient="bottom")
        .configure_title(fontSize=16, anchor="start", fontWeight=600, color="#222")
    )


# =========================
# LOADERS
# =========================
@st.cache_resource(show_spinner=False)
def load_models():
    xgb_model = pickle.load(open(MODEL_PATH, "rb"))
    calib_model = pickle.load(open(CALIB_PATH, "rb"))
    explainer = shap.TreeExplainer(xgb_model)
    return xgb_model, calib_model, explainer

@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["Athlete ID"] = df["Athlete ID"].astype(float).astype(int)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    return df


# =========================
# FEATURES / NORMALIZATION
# =========================
def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ["injury", "Date", "Athlete ID", "Year"]]

def getMeanStd(data: pd.DataFrame):
    tmp = data[data["injury"] == 0].copy()
    tmp = tmp.drop(columns=["Date", "Year"], errors="ignore")

    mean_df = tmp.groupby("Athlete ID").mean(numeric_only=True).drop(columns=["injury"], errors="ignore")
    std_df  = tmp.groupby("Athlete ID").std(numeric_only=True).drop(columns=["injury"], errors="ignore")

    std_df.replace(0, 0.01, inplace=True)

    global_mean = mean_df.mean()
    global_std  = std_df.mean().replace(0, 0.01)
    return mean_df, std_df, global_mean, global_std

def normalize_row(row: pd.Series, mean_df, std_df, global_mean, global_std) -> pd.Series:
    athlete_id = row["Athlete ID"]
    feat_cols = [c for c in row.index if c not in ["injury", "Date", "Athlete ID", "Year"]]

    if athlete_id in mean_df.index:
        mu, su = mean_df.loc[athlete_id], std_df.loc[athlete_id]
    else:
        mu, su = global_mean, global_std

    out = row.copy()
    out[feat_cols] = (row[feat_cols] - mu[feat_cols]) / su[feat_cols]
    return out

def rename_feature(col: str) -> str:
    if "." not in col:
        return col
    base, last = col.rsplit(".", 1)
    if not last.isdigit():
        return col
    lag = int(last)

    mapping = {
        "total km": "Distance totale (km)",
        "perceived recovery": "Récupération perçue",
        "perceived exertion": "Effort perçu",
        "perceived trainingSuccess": "Réussite de séance (ressenti)",
        "km Z3-4": "Distance intensité modérée (Z3-4)",
        "km Z5-T1-T2": "Distance haute intensité",
        "nr. sessions": "Nombre de séances",
        "strength training": "Renforcement",
    }
    label = mapping.get(base, base)
    return f"{label} (J-{lag})"

def compute_acute_km(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in df.columns if c.startswith("total km.")]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].sum(axis=1, numeric_only=True)

def add_acwr(panel: pd.DataFrame, chronic_window_days: int = 28) -> pd.DataFrame:
    out = panel.copy()
    out = out.sort_values(["Athlete ID", "Date"])
    out["Acute_7d_km"] = compute_acute_km(out)

    def _per_athlete(g):
        g = g.sort_values("Date").copy()
        g["Chronic_28d_km"] = g["Acute_7d_km"].rolling(chronic_window_days, min_periods=7).mean()
        g["ACWR"] = g["Acute_7d_km"] / g["Chronic_28d_km"]
        return g

    return out.groupby("Athlete ID", group_keys=False).apply(_per_athlete)

def risk_band(score: float):
    if score < 0.30:
        return "Green", "Low", "✅"
    if score < 0.60:
        return "Amber", "Moderate", "⚠️"
    return "Red", "High", "🛑"

def coach_message(score: float, lang: str) -> str:
    if lang == "fr":
        if score < 0.30:
            return "Charge globalement équilibrée : continuer progressivement et surveiller la récupération."
        if score < 0.60:
            return "Risque en hausse : réduire l’intensité / ajouter du facile / accentuer la récup (sommeil, jours off)."
        return "Risque élevé : réduire charge & intensité, prioriser la récup, surveiller les signaux et alerter si besoin."
    else:
        if score < 0.30:
            return "Training looks balanced. Keep building gradually and keep an eye on recovery."
        if score < 0.60:
            return "Risk is rising. Consider reducing intensity and adding recovery (sleep, easy days)."
        return "High risk. Reduce load & intensity. Prioritize recovery and monitor symptoms closely."


# =========================
# Sidebar language + filters
# =========================
st.sidebar.header("Settings")
lang_choice = st.sidebar.selectbox(
    I18N["en"]["lang_label"] + " / " + I18N["fr"]["lang_label"],
    ["Français", "English"],
    index=0
)
lang = "fr" if lang_choice == "Français" else "en"
T = I18N[lang]

xgb_model, calib_model, explainer = load_models()
df = load_dataset()

st.sidebar.header(T["filters"])

all_athletes = sorted(df["Athlete ID"].dropna().unique().tolist())
selected_athletes = st.sidebar.multiselect(
    T["athletes"],
    options=all_athletes,
    default=[all_athletes[0]] if all_athletes else [],
)

years = sorted([int(y) for y in df["Year"].dropna().unique()])
selected_years = st.sidebar.multiselect(
    T["years"],
    options=years,
    default=years,
)

time_window = st.sidebar.radio(T["time_window"], ["30 days", "90 days", "All"], index=0)

if not selected_athletes:
    st.info("Sélectionne au moins un athlète pour démarrer." if lang == "fr" else "Select at least one athlete to start.", icon="ℹ️")
    st.stop()

panel_raw = df[df["Athlete ID"].isin(selected_athletes)].copy()
if selected_years:
    panel_raw = panel_raw[panel_raw["Year"].isin(selected_years)]
panel_raw = panel_raw.sort_values(["Athlete ID", "Date"])

if time_window != "All" and panel_raw["Date"].notna().any():
    days = 30 if time_window == "30 days" else 90

    def cut_window(g):
        max_date = g["Date"].max()
        return g[g["Date"] >= (max_date - pd.Timedelta(days=days))]

    panel_raw = panel_raw.groupby("Athlete ID", group_keys=False).apply(cut_window)


# =========================
# Header + About
# =========================
st.title(T["title"])
st.caption(T["caption"])

with st.container(border=True):
    st.markdown(T["about"])

with st.container(border=True):
    st.subheader(T["summary"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(T["athletes"], len(selected_athletes))
    c2.metric(T["years"], "All" if len(selected_years) == len(years) else ", ".join(map(str, selected_years[:3])) + ("..." if len(selected_years) > 3 else ""))
    c3.metric("Lignes" if lang == "fr" else "Rows", len(panel_raw))
    if panel_raw["Date"].notna().any() and len(panel_raw) > 0:
        c4.metric("Période" if lang == "fr" else "Time span", f"{panel_raw['Date'].min().date()} → {panel_raw['Date'].max().date()}")
    else:
        c4.metric("Période" if lang == "fr" else "Time span", "Dates indisponibles" if lang == "fr" else "Dates not available")


# =========================
# Compute risk + SHAP + ACWR
# =========================
@st.cache_data(show_spinner=True, ttl=60 * 60 * 2)
def compute_panel(panel_raw: pd.DataFrame):
    out_rows = []
    shap_store = {}

    for aid, g in panel_raw.groupby("Athlete ID"):
        g = g.sort_values("Date").copy()

        mean_df, std_df, global_mean, global_std = getMeanStd(g)
        g_norm = g.apply(lambda r: normalize_row(r, mean_df, std_df, global_mean, global_std), axis=1)

        X = g_norm[get_feature_cols(g_norm)].copy()

        risk = calib_model.predict_proba(X.to_numpy())[:, 1]
        shap_values = explainer.shap_values(X)

        g["Risk"] = risk
        out_rows.append(g)

        shap_store[int(aid)] = {
            "dates": g["Date"].reset_index(drop=True),
            "X": X.reset_index(drop=True),
            "risk": pd.Series(risk),
            "injury": g["injury"].reset_index(drop=True) if "injury" in g.columns else None,
            "shap_values": shap_values,
        }

    panel = pd.concat(out_rows, ignore_index=True) if out_rows else panel_raw.copy()
    panel = add_acwr(panel, chronic_window_days=28)
    return panel, shap_store

panel, shap_store = compute_panel(panel_raw)


# =========================================================
# SECTION — Context
# =========================================================
st.header(T["context"])
with st.container(border=True):
    st.markdown(
        "Ce dashboard sert à **monitorer** la charge et le risque, et à **prioriser** les actions terrain (ajustement charge, récup, vigilance médicale)."
        if lang == "fr" else
        "This dashboard is a monitoring tool to support staff decisions (load adjustment, recovery focus, medical vigilance)."
    )


# =========================================================
# SECTION — Visual monitoring
# =========================================================
st.header(T["visual"])

if panel.empty or panel["Date"].isna().all():
    st.warning("Aucune date valide dans les données filtrées." if lang == "fr" else "No valid dates found in the filtered data.")
else:
    # Quick KPI: best / worst (latest)
    latest = panel.sort_values("Date").groupby("Athlete ID").tail(1)
    best = latest.loc[latest["Risk"].idxmin()]
    worst = latest.loc[latest["Risk"].idxmax()]

    with st.container(border=True):
        st.subheader("Synthèse rapide" if lang == "fr" else "Quick summary")
        a, b, c = st.columns(3)

        _, label_b, icon_b = risk_band(float(best["Risk"]))
        _, label_w, icon_w = risk_band(float(worst["Risk"]))

        a.metric(
            "Athlète le moins à risque (dernier point)" if lang == "fr" else "Lowest risk (latest)",
            f"Athlete {int(best['Athlete ID'])}",
            delta=f"{icon_b} {label_b} ({best['Risk']:.2f})",
        )
        b.metric(
            "Athlète le plus à risque (dernier point)" if lang == "fr" else "Highest risk (latest)",
            f"Athlete {int(worst['Athlete ID'])}",
            delta=f"{icon_w} {label_w} ({worst['Risk']:.2f})",
        )
        c.metric(
            "Moyenne risque (dernier point)" if lang == "fr" else "Average risk (latest)",
            f"{latest['Risk'].mean():.2f}",
        )

    # --- Risk trend
    with st.container(border=True):
        st.subheader(T["risk_trend_title"])
        st.caption(T["risk_trend_help"])

        risk_chart = (
            alt.Chart(panel)
            .mark_line()
            .encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("Risk:Q", title="Risque (0→1)" if lang == "fr" else "Risk (0→1)", scale=alt.Scale(zero=False)),
                color=alt.Color("Athlete ID:N", title=T["athletes"]),
                tooltip=["Date:T", "Athlete ID:N", "Risk:Q", "Acute_7d_km:Q", "ACWR:Q"],
            )
            .properties(height=380)
        )
        st.altair_chart(polish_chart(risk_chart), use_container_width=True)

    # --- ACWR time-series (with reference lines)
    with st.container(border=True):
        st.subheader(T["acwr_title"])
        st.caption(T["acwr_help"])

        acwr_df = panel.dropna(subset=["ACWR", "Date"]).copy()

        base = alt.Chart(acwr_df).encode(
            x=alt.X("Date:T", title="Date"),
            color=alt.Color("Athlete ID:N", title=T["athletes"]),
        )

        line = base.mark_line().encode(
            y=alt.Y("ACWR:Q", title="ACWR", scale=alt.Scale(zero=False)),
            tooltip=["Date:T", "Athlete ID:N", "Acute_7d_km:Q", "Chronic_28d_km:Q", "ACWR:Q"],
        )

        # reference thresholds (common heuristic bands)
        ref_1_3 = alt.Chart(pd.DataFrame({"y": [1.3]})).mark_rule(strokeDash=[6, 4]).encode(y="y:Q")
        ref_1_5 = alt.Chart(pd.DataFrame({"y": [1.5]})).mark_rule(strokeDash=[2, 2]).encode(y="y:Q")
        ref_0_8 = alt.Chart(pd.DataFrame({"y": [0.8]})).mark_rule(strokeDash=[6, 4]).encode(y="y:Q")

        chart = (line + ref_0_8 + ref_1_3 + ref_1_5).properties(height=340)
        st.altair_chart(polish_chart(chart), use_container_width=True)

    # --- Risk vs acute load
    with st.container(border=True):
        st.subheader(T["risk_vs_load_title"])
        st.caption(T["risk_vs_load_help"])

        scatter_df = panel.dropna(subset=["Acute_7d_km", "Risk"]).copy()

        scatter = (
            alt.Chart(scatter_df)
            .mark_circle(size=90, opacity=0.65)
            .encode(
                x=alt.X("Acute_7d_km:Q", title="Charge aiguë (7j km)" if lang == "fr" else "Acute load (7d km)"),
                y=alt.Y("Risk:Q", title="Risque" if lang == "fr" else "Risk", scale=alt.Scale(zero=False)),
                color=alt.Color("Athlete ID:N", title=T["athletes"]),
                tooltip=["Date:T", "Athlete ID:N", "Acute_7d_km:Q", "ACWR:Q", "Risk:Q"],
            )
            .properties(height=380)
        )
        st.altair_chart(polish_chart(scatter), use_container_width=True)

    # --- Long-term heatmap (back)
    with st.container(border=True):
        st.subheader(T["heatmap_title"])
        st.caption(T["heatmap_help"])

        agg_mode = st.selectbox("Agrégation" if lang == "fr" else "Aggregation", ["Trimestre", "Année"], index=0)
        metric = st.selectbox("Métrique" if lang == "fr" else "Metric",
                              ["Risque (moyenne)", "Charge aiguë 7j (moyenne)", "ACWR (moyenne)"],
                              index=0)
        view_mode = st.selectbox("Lignes" if lang == "fr" else "Rows", ["Par athlète", "Moyenne groupe"], index=0)

        long_df = panel.dropna(subset=["Date"]).copy()

        if agg_mode == "Trimestre":
            long_df["Period"] = long_df["Date"].dt.to_period("Q").astype(str).str.replace("Q", "-Q", regex=False)
            period_title = "Trimestre" if lang == "fr" else "Quarter"
        else:
            long_df["Period"] = long_df["Date"].dt.year.astype(str)
            period_title = "Année" if lang == "fr" else "Year"

        if metric.startswith("Risque"):
            value_col, value_title = "Risk", ("Risque moyen" if lang == "fr" else "Average risk")
        elif metric.startswith("Charge"):
            value_col, value_title = "Acute_7d_km", ("Charge aiguë moyenne (7j km)" if lang == "fr" else "Avg acute load (7d km)")
        else:
            value_col, value_title = "ACWR", ("ACWR moyen" if lang == "fr" else "Average ACWR")

        if view_mode == "Moyenne groupe":
            agg = long_df.groupby(["Period"], as_index=False)[value_col].mean().rename(columns={value_col: "Value"})
            agg["Row"] = "Moyenne groupe" if lang == "fr" else "Group average"
        else:
            agg = long_df.groupby(["Athlete ID", "Period"], as_index=False)[value_col].mean().rename(columns={value_col: "Value"})
            agg["Row"] = agg["Athlete ID"].astype(int).astype(str)

        # chronological sort
        if agg_mode == "Trimestre":
            parts = agg["Period"].str.split("-Q", expand=True)
            agg["_pkey"] = parts[0].astype(int) * 10 + parts[1].astype(int)
            period_sort = agg[["Period", "_pkey"]].drop_duplicates().sort_values("_pkey")["Period"].tolist()
            agg = agg.drop(columns=["_pkey"])
        else:
            period_sort = sorted(agg["Period"].unique(), key=lambda x: int(x))

        heatmap = (
            alt.Chart(agg)
            .mark_rect()
            .encode(
                x=alt.X("Period:N", sort=period_sort, title=period_title, axis=alt.Axis(labelAngle=0, labelLimit=120)),
                y=alt.Y("Row:N", title=("Athlète" if lang == "fr" else "Athlete") if view_mode == "Par athlète" else "", sort="ascending"),
                color=alt.Color("Value:Q", title=value_title, scale=alt.Scale(scheme="viridis")),
                tooltip=[
                    alt.Tooltip("Row:N", title="Athlète" if lang == "fr" else "Athlete"),
                    alt.Tooltip("Period:N", title=period_title),
                    alt.Tooltip("Value:Q", title=value_title, format=".3f"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(polish_chart(heatmap), use_container_width=True)


# =========================================================
# SECTION — Decisions + SHAP explanations
# =========================================================
st.header(T["actions"])

with st.container(border=True):
    st.markdown(T["guidance"])

if shap_store:
    cols = st.columns([2, 2, 3], vertical_alignment="center")

    with cols[0]:
        focus_athlete = st.selectbox(T["focus_athlete"], options=sorted(shap_store.keys()), index=0)

    pack = shap_store[int(focus_athlete)]
    dates = pack["dates"]
    X_focus = pack["X"]
    risk_focus = pack["risk"]
    shap_values_focus = pack["shap_values"]

    with cols[1]:
        date_idx = st.slider(T["pick_day"], 0, max(0, len(dates) - 1), max(0, len(dates) - 1))

    risk_val = float(risk_focus.iloc[date_idx]) if len(risk_focus) else float("nan")
    _, band_label, band_icon = risk_band(risk_val)

    with cols[2]:
        st.markdown(f"**Date :** {str(dates.iloc[date_idx])}" if lang == "fr" else f"**Date:** {str(dates.iloc[date_idx])}")
        st.markdown(f"### {band_icon} {band_label} (score : **{risk_val:.2f}**)" if lang == "fr" else f"### {band_icon} {band_label} (score: **{risk_val:.2f}**)")
        st.write(coach_message(risk_val, lang))

    # SHAP drivers
    with st.container(border=True):
        st.subheader(T["drivers_title"])
        st.caption(T["drivers_help"])
        st.markdown(T["drivers_howto"])

        sv = shap_values_focus[date_idx]
        feat_names = [rename_feature(c) for c in X_focus.columns]
        topk = np.argsort(np.abs(sv))[::-1][:10]

        drivers = pd.DataFrame(
            {"Facteur": [feat_names[i] for i in topk], "Impact": [float(sv[i]) for i in topk]}
        )

        if lang == "fr":
            drivers["Sens"] = np.where(drivers["Impact"] >= 0, "Augmente le risque", "Réduit le risque")
        else:
            drivers["Sens"] = np.where(drivers["Impact"] >= 0, "Increases risk", "Decreases risk")

        drivers["Impact_abs"] = drivers["Impact"].abs()
        drivers = drivers.sort_values("Impact_abs", ascending=True)

        bar = (
            alt.Chart(drivers)
            .mark_bar()
            .encode(
                x=alt.X("Impact:Q", title="Contribution au score de risque" if lang == "fr" else "Contribution to risk score"),
                y=alt.Y("Facteur:N", title="", sort=None),
                color=alt.Color("Sens:N", title=""),
                tooltip=["Facteur:N", "Impact:Q", "Sens:N"],
            )
            .properties(height=380)
        )
        st.altair_chart(polish_chart(bar), use_container_width=True)

    # Recent SHAP heatmap (readable)
    with st.container(border=True):
        st.subheader(T["contrib_heatmap_title"])
        st.caption(T["contrib_heatmap_help"])

        N = st.slider("Nombre de jours affichés" if lang == "fr" else "Days shown",
                      min_value=7, max_value=30, value=14, step=1)
        N = min(N, len(dates))
        start_idx = max(0, len(dates) - N)

        sv_window = shap_values_focus[start_idx:len(dates), :]
        mean_abs = np.mean(np.abs(sv_window), axis=0)
        top_feat_idx = np.argsort(mean_abs)[::-1][:8]

        heat = pd.DataFrame(
            sv_window[:, top_feat_idx],
            columns=[rename_feature(X_focus.columns[i]) for i in top_feat_idx],
        )
        heat["Date"] = dates.iloc[start_idx:len(dates)].dt.strftime("%Y-%m-%d").tolist()

        heat_long = heat.melt(id_vars=["Date"], var_name="Facteur", value_name="Contribution")

        cap = float(np.quantile(np.abs(heat_long["Contribution"]), 0.95)) if len(heat_long) else 1.0
        heat_long["Contribution_cap"] = heat_long["Contribution"].clip(-cap, cap)

        heatmap_recent = (
            alt.Chart(heat_long)
            .mark_rect()
            .encode(
                x=alt.X("Facteur:N", title="Facteurs principaux" if lang == "fr" else "Main drivers",
                        sort=None, axis=alt.Axis(labelAngle=-20, labelLimit=180)),
                y=alt.Y("Date:N", title="Date", sort=None, axis=alt.Axis(labelLimit=120)),
                color=alt.Color(
                    "Contribution_cap:Q",
                    title="Contribution",
                    scale=alt.Scale(scheme="redblue", reverse=True, domain=[-cap, cap]),
                ),
                tooltip=[
                    alt.Tooltip("Date:N", title="Date"),
                    alt.Tooltip("Facteur:N", title="Facteur" if lang == "fr" else "Factor"),
                    alt.Tooltip("Contribution:Q", title="Contribution brute" if lang == "fr" else "Raw contribution", format=".4f"),
                ],
            )
            .properties(height=max(320, 22 * N))
        ).configure_mark(opacity=0.95)

        st.altair_chart(polish_chart(heatmap_recent), use_container_width=True)


# =========================================================
# ML Audit
# =========================================================
with st.expander(T["ml_audit"], expanded=False):
    threshold = st.slider(
        "Seuil de classification" if lang == "fr" else "Classification threshold",
        0.05, 0.95, 0.50, 0.05
    )

    if "injury" not in panel.columns:
        st.info("Colonne 'injury' introuvable." if lang == "fr" else "No 'injury' column found.")
    else:
        y_true = panel["injury"].astype(int).to_numpy()
        y_score = panel["Risk"].to_numpy()
        y_pred = (y_score >= threshold).astype(int)

        auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("AUC", f"{auc:.3f}" if np.isfinite(auc) else "n/a")
        c2.metric("Accuracy", f"{acc:.3f}")
        c3.metric("Precision", f"{prec:.3f}")
        c4.metric("Recall", f"{rec:.3f}")
        c5.metric("F1", f"{f1:.3f}")

        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=["Vrai : pas blessé", "Vrai : blessé"], columns=["Prédit : pas blessé", "Prédit : blessé"])
        st.dataframe(cm_df, use_container_width=True)

        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
            roc_chart = alt.Chart(roc_df).mark_line().encode(
                x=alt.X("FPR:Q", title="Taux de faux positifs"),
                y=alt.Y("TPR:Q", title="Taux de vrais positifs"),
            ).properties(height=260, title="Courbe ROC")
            st.altair_chart(polish_chart(roc_chart), use_container_width=True)

            frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
            cal_df = pd.DataFrame({"Probabilité prédite": mean_pred, "Fréquence observée": frac_pos})
            cal_chart = alt.Chart(cal_df).mark_line(point=True).encode(
                x=alt.X("Probabilité prédite:Q", title="Probabilité prédite"),
                y=alt.Y("Fréquence observée:Q", title="Fréquence observée"),
            ).properties(height=260, title="Courbe de calibration")
            st.altair_chart(polish_chart(cal_chart), use_container_width=True)

with st.expander(T["raw_data"], expanded=False):
    st.dataframe(panel.sort_values(["Athlete ID", "Date"]), use_container_width=True)