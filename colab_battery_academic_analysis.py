# ============================================================
# Smartphone Battery & Academic Performance Analysis
# Paste this WHOLE script into ONE Google Colab cell and press Run.
# No file upload needed - it generates the raw data, cleans it, and plots.
# ============================================================
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings; warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="deep")

# ------------------------------------------------------------
# 1. GENERATE RANDOM (SYNTHETIC) DATA
# ------------------------------------------------------------
rng = np.random.default_rng(42)
n = 60
ids    = [f"S{i:03d}" for i in range(1, n + 1)]
age    = rng.integers(17, 25, n)
gender = rng.choice(["Male", "Female"], n)
year   = np.clip(age - 16, 1, 4)
screen = np.clip(rng.normal(5.5, 1.8, n), 1.5, 11).round(1)
charges = np.clip((screen / 2.2 + rng.normal(1.2, .6, n)).round(), 1, 6).astype(int)
battery = np.clip(rng.normal(86, 8, n), 60, 100).round(0).astype(int)
night  = rng.choice([0, 1], n, p=[.55, .45])
study  = np.clip(rng.normal(3.5, 1.3, n) - 0.15 * (screen - 5.5), 0.5, 8).round(1)
attend = np.clip(rng.normal(82, 9, n) - 0.8 * (screen - 5.5), 50, 100).round(0).astype(int)
cgpa   = np.clip(6.5 + 0.45 * study - 0.18 * (screen - 5.5) - 0.25 * night
                 + 0.012 * (attend - 80) + rng.normal(0, .45, n), 3, 10).round(2)

clean0 = pd.DataFrame({
    "student_id": ids, "age": age, "gender": gender, "year_of_study": year,
    "daily_screen_time_hrs": screen, "daily_charge_cycles": charges,
    "battery_health_pct": battery,
    "charges_overnight": np.where(night == 1, "Yes", "No"),
    "daily_study_hrs": study, "attendance_pct": attend, "cgpa": cgpa})

# ---- make it messy (this is the RAW data) ----
raw = clean0.copy().astype(object)
raw["gender"] = [rng.choice([g, g.lower(), g[0], g.upper() + " "]) for g in raw["gender"]]
raw["charges_overnight"] = [rng.choice([v, v.lower(), "Y" if v == "Yes" else "N"]) for v in raw["charges_overnight"]]
raw["daily_screen_time_hrs"] = [f"{v} hrs" if rng.random() < .1 else v for v in raw["daily_screen_time_hrs"]]
raw["battery_health_pct"]    = [f"{v}%" if rng.random() < .15 else v for v in raw["battery_health_pct"]]
for col, k in [("daily_screen_time_hrs", 4), ("battery_health_pct", 3), ("daily_study_hrs", 3),
               ("cgpa", 3), ("attendance_pct", 3), ("age", 2)]:
    for i in rng.choice(n, k, replace=False):
        raw.loc[i, col] = np.nan                       # missing values
raw.loc[7, "daily_screen_time_hrs"] = 45.0              # outlier
raw.loc[21, "battery_health_pct"]   = 250               # impossible
raw.loc[33, "cgpa"]                 = 14.2              # above 10 scale
raw.loc[48, "age"]                  = 2                 # typo
raw = pd.concat([raw, raw.iloc[[5, 17, 40]]], ignore_index=True)   # duplicates
raw = raw.sample(frac=1, random_state=1).reset_index(drop=True)

print("RAW DATA:", raw.shape)
display(raw.head(10))

# ------------------------------------------------------------
# 2. INSPECT PROBLEMS IN RAW DATA
# ------------------------------------------------------------
print("\nDuplicate rows:", raw.duplicated().sum())
print("Gender values:", raw["gender"].unique())
print("Overnight values:", raw["charges_overnight"].unique())
print("\nMissing values:\n", raw.isna().sum()[raw.isna().sum() > 0])

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
m = raw.isna().sum(); m = m[m > 0].sort_values()
ax[0].barh(m.index, m.values, color="#d9534f"); ax[0].set_title("Missing values per column (RAW)")
ax[1].scatter(range(len(raw)), pd.to_numeric(raw["cgpa"], errors="coerce"), s=25)
ax[1].axhline(10, color="red", ls="--", label="Max valid CGPA = 10")
ax[1].set_title("Raw CGPA - outlier above scale"); ax[1].legend()
plt.tight_layout(); plt.show()

# ------------------------------------------------------------
# 3. CLEAN THE DATA
# ------------------------------------------------------------
df = raw.drop_duplicates().copy()
df["gender"] = df["gender"].astype(str).str.strip().str.lower().str[0].map({"m": "Male", "f": "Female"})
df["charges_overnight"] = df["charges_overnight"].astype(str).str.strip().str.lower().str[0].map({"y": "Yes", "n": "No"})
num_cols = ["age", "daily_screen_time_hrs", "battery_health_pct", "daily_study_hrs", "cgpa", "attendance_pct"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")
df.loc[(df.age < 15) | (df.age > 40), "age"] = np.nan
df.loc[df.daily_screen_time_hrs > 16, "daily_screen_time_hrs"] = np.nan
df.loc[df.battery_health_pct > 100, "battery_health_pct"] = np.nan
df.loc[df.cgpa > 10, "cgpa"] = np.nan
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
for c in ["age", "battery_health_pct", "attendance_pct"]:
    df[c] = df[c].round().astype(int)
df = df.sort_values("student_id").reset_index(drop=True)

print("\nCLEANED DATA:", df.shape, "| missing:", int(df.isna().sum().sum()))
display(df.head())
display(df.describe().round(2))

# Save both files in Colab (left sidebar > Files). Uncomment to download:
raw.to_csv("smartphone_battery_academic_raw.csv", index=False)
df.to_csv("smartphone_battery_academic_cleaned.csv", index=False)
# from google.colab import files; files.download("smartphone_battery_academic_cleaned.csv")

# ------------------------------------------------------------
# 4. VISUALIZATIONS
# ------------------------------------------------------------
# 4a. Distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
cols = ["daily_screen_time_hrs", "daily_study_hrs", "cgpa", "battery_health_pct", "attendance_pct", "daily_charge_cycles"]
for ax, c in zip(axes.ravel(), cols):
    sns.histplot(df[c], kde=True, ax=ax); ax.set_title(c); ax.set_xlabel("")
plt.suptitle("Distributions of key variables", fontsize=14, y=1.02); plt.tight_layout(); plt.show()

# 4b. Category counts
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, c in zip(axes, ["gender", "year_of_study", "charges_overnight"]):
    sns.countplot(data=df, x=c, ax=ax, palette="Set2"); ax.set_title(c)
plt.tight_layout(); plt.show()

# 4c. Correlation heatmap
corr = df.select_dtypes("number").corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=.5)
plt.title("Correlation matrix"); plt.tight_layout(); plt.show()
print("Correlation with CGPA:\n", corr["cgpa"].drop("cgpa").sort_values().round(3))

# 4d. Screen time & study time vs CGPA
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, x in zip(axes, ["daily_screen_time_hrs", "daily_study_hrs"]):
    sns.regplot(data=df, x=x, y="cgpa", ax=ax, scatter_kws={"alpha": .7, "s": 45}, line_kws={"color": "red"})
    r, p = stats.pearsonr(df[x], df["cgpa"])
    ax.set_title(f"{x} vs CGPA (r = {r:.2f}, p = {p:.3f})")
plt.tight_layout(); plt.show()

# 4e. Battery & charging habits
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
sns.regplot(data=df, x="battery_health_pct", y="cgpa", ax=axes[0], scatter_kws={"alpha": .7}, line_kws={"color": "red"})
r, p = stats.pearsonr(df["battery_health_pct"], df["cgpa"])
axes[0].set_title(f"Battery health vs CGPA (r = {r:.2f}, p = {p:.2f})")
sns.barplot(data=df, x="daily_charge_cycles", y="cgpa", ax=axes[1], errorbar="sd", palette="viridis")
axes[1].set_title("Mean CGPA by daily charge cycles")
sns.boxplot(data=df, x="charges_overnight", y="cgpa", ax=axes[2], palette="Set2")
sns.stripplot(data=df, x="charges_overnight", y="cgpa", ax=axes[2], color="black", alpha=.4, size=3)
axes[2].set_title("CGPA: overnight charging vs not")
plt.tight_layout(); plt.show()

# 4f. Screen-time groups
df["screen_group"] = pd.cut(df["daily_screen_time_hrs"], bins=[0, 4, 7, 12],
                            labels=["Low (<4h)", "Medium (4-7h)", "High (>7h)"])
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
sns.boxplot(data=df, x="screen_group", y="cgpa", ax=axes[0], palette="Blues"); axes[0].set_title("CGPA by screen-time group")
grp = df.groupby("screen_group", observed=True)[["cgpa", "daily_study_hrs", "attendance_pct"]].mean()
grp.plot(kind="bar", ax=axes[1], rot=0); axes[1].set_title("Group averages")
plt.tight_layout(); plt.show()

# 4g. Pairplot
sns.pairplot(df[["daily_screen_time_hrs", "daily_study_hrs", "battery_health_pct", "attendance_pct", "cgpa"]],
             kind="reg", plot_kws={"line_kws": {"color": "red"}, "scatter_kws": {"alpha": .5, "s": 20}}, height=2)
plt.suptitle("Pairwise relationships", y=1.02); plt.show()

# ------------------------------------------------------------
# 5. STATISTICAL TESTS & REGRESSION
# ------------------------------------------------------------
yes = df.loc[df.charges_overnight == "Yes", "cgpa"]; no = df.loc[df.charges_overnight == "No", "cgpa"]
t, p = stats.ttest_ind(yes, no, equal_var=False)
print(f"\nOvernight chargers mean CGPA = {yes.mean():.2f} | Others = {no.mean():.2f} | Welch t = {t:.2f}, p = {p:.3f}")
f, p2 = stats.f_oneway(*[g["cgpa"].values for _, g in df.groupby("screen_group", observed=True)])
print(f"ANOVA across screen-time groups: F = {f:.2f}, p = {p2:.4f}")

features = ["daily_screen_time_hrs", "daily_study_hrs", "battery_health_pct", "daily_charge_cycles", "attendance_pct"]
X = df[features].astype(float); Xs = (X - X.mean()) / X.std()
A = np.column_stack([np.ones(len(Xs)), Xs.values]); y = df["cgpa"].values
beta, *_ = np.linalg.lstsq(A, y, rcond=None); pred = A @ beta
r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
coef = pd.Series(beta[1:], index=features).sort_values()
print(f"\nRegression R-squared = {r2:.3f}\n{coef.round(3)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
coef.plot(kind="barh", ax=axes[0], color=["#d9534f" if v < 0 else "#5cb85c" for v in coef])
axes[0].axvline(0, color="black", lw=.8)
axes[0].set_title("Standardized coefficients (effect on CGPA)")
axes[1].scatter(y, pred, alpha=.7)
lims = [min(y.min(), pred.min()) - .2, max(y.max(), pred.max()) + .2]
axes[1].plot(lims, lims, "r--"); axes[1].set_xlabel("Actual CGPA"); axes[1].set_ylabel("Predicted CGPA")
axes[1].set_title(f"Actual vs predicted (R² = {r2:.2f})")
plt.tight_layout(); plt.show()

print("""
CONCLUSIONS
- Study time is the strongest positive predictor of CGPA.
- Higher screen time is linked to lower CGPA (less study time & attendance).
- Battery health shows almost no relationship with CGPA.
- Charging habits have weak / borderline effects.
- Data is synthetic: correlation is not causation.
""")
