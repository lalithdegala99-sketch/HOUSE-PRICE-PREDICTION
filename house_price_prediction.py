"""House Price Prediction - Lalith Kumar Degala
Tools: Python, pandas, numpy, scikit-learn, matplotlib, seaborn
NOTE: uses a synthetic dataset so it runs anywhere. To use real data,
replace the DATA section with: df = pd.read_csv("your_kaggle_file.csv")
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------- 1. DATA ----------
rng = np.random.default_rng(42)
n = 1500
df = pd.DataFrame({
    "area_sqft": rng.integers(500, 4000, n),
    "bedrooms": rng.integers(1, 6, n),
    "bathrooms": rng.integers(1, 4, n),
    "age_years": rng.integers(0, 40, n),
    "location_score": rng.integers(1, 11, n),
})
df["price_lakhs"] = (
    df.area_sqft * 0.035 + df.bedrooms * 4 + df.bathrooms * 3
    - df.age_years * 0.6 + df.location_score * 9
    + rng.normal(0, 8, n)
).round(2)

# ---------- 2. SPLIT ----------
X = df.drop(columns="price_lakhs")
y = df["price_lakhs"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- 3. MODELS ----------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
}
results, preds = {}, {}
for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    preds[name] = p
    results[name] = {
        "R2": r2_score(y_test, p),
        "MAE": mean_absolute_error(y_test, p),
        "RMSE": np.sqrt(mean_squared_error(y_test, p)),
    }
res = pd.DataFrame(results).T.round(3)
print(res)
res.to_csv("/mnt/user-data/outputs/model_results.csv")

sns.set_theme(style="whitegrid")

# ---------- 4. CHARTS ----------
# Chart 1: correlation heatmap
plt.figure(figsize=(7, 5.5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/1_correlation_heatmap.png", dpi=150)
plt.close()

# Chart 2: actual vs predicted (best model)
best = res["R2"].idxmax()
plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds[best], alpha=0.5, color="#0a66c2")
lims = [y_test.min(), y_test.max()]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")
plt.title(f"Actual vs Predicted - {best} (R2 = {res.loc[best, 'R2']:.2f})")
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/2_actual_vs_predicted.png", dpi=150)
plt.close()

# Chart 3: feature importance
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(7, 4.5))
imp.plot(kind="barh", color="#0a66c2")
plt.title("What Drives House Price? (Feature Importance)")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/3_feature_importance.png", dpi=150)
plt.close()

# Chart 4: model comparison
plt.figure(figsize=(6, 4.5))
res["R2"].plot(kind="bar", color=["#9aa5b1", "#0a66c2"], rot=0)
plt.ylim(0, 1)
plt.ylabel("R2 Score")
plt.title("Model Comparison")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/4_model_comparison.png", dpi=150)
plt.close()
print("Charts saved. Best model:", best)
