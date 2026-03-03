import pickle
import pandas as pd
import shap

# Load model
model = pickle.load(open("xgb_model.pkl", "rb"))

# Load data
df = pd.read_csv("day_approach_maskedID_timeseries.csv")

# Features
X = df.drop(columns=['injury', 'Date', 'Athlete ID'])

# Create explainer 
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)

# save shap values
pickle.dump(shap_values, open("shap_values.pkl", "wb"))

# mapping des noms de features
def rename_feature(col: str) -> str:
    # Ex: "total km.5" -> base="total km", lag="5"
    # Ex: "nr. sessions.4" -> base="nr. sessions", lag="4"
    # Ex: "nr. sessions" -> pas de lag
    
    if "." not in col:
        return col

    base, last = col.rsplit(".", 1)   # split uniquement sur le dernier point

    # si le suffixe n'est pas un int, on ne touche pas
    if not last.isdigit():
        return col

    lag = int(last)

    mapping = {
        "total km": "Total distance (km)",
        "perceived recovery": "Perceived recovery",
        "perceived exertion": "Perceived exertion",
        "perceived trainingSuccess": "Training success",
        "perceived trainingSuccess.1": "Training success",  # au cas où
        "km Z3-4": "Distance zone 3-4 (km)",
        "km Z5-T1-T2": "High intensity distance (km)",
        "nr. sessions": "Number of sessions",
        "strength training": "Strength training",
    }

    label = mapping.get(base, base)
    return f"{label} ({lag}d ago)"


# appliquer rename
X_renamed = X.copy()
X_renamed.columns = [rename_feature(c) for c in X.columns]

# Summary plot
shap.summary_plot(shap_values, X)

# compute risk score
calib_model = pickle.load(open("calib_model.pkl", "rb"))

risk_scores = calib_model.predict_proba(X)[:,1]

# dataframe final
results = X_renamed.copy()
results["risk_score"] = risk_scores

# ajouter athlete id et date
results["Athlete ID"] = df["Athlete ID"]
results["Date"] = df["Date"]

# save
results.to_csv("injury_risk_predictions.csv", index=False)