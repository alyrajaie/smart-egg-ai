# app.py
# SMART EGG INCUBATOR (LUX TREND ANALYZER) - Research UI (Blue) + 2-color hist + clean sidebar
import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve


# =========================
# Page setup + CSS (Full Blue + White Card)
# =========================
st.set_page_config(page_title="SMART EGG INCUBATOR (LUX TREND ANALYZER)", layout="wide")

PBS_BLUE = "#0b2d5c"
CARD_BORDER = "#d7e6ff"

CSS = f"""
<style>
/* Full background in blue */
.stApp {{
    background: {PBS_BLUE};
}}

/* Main content looks like a white academic panel */
.block-container {{
    background: #ffffff;
    border-radius: 14px;
    padding: 18px 22px 22px 22px;
    margin-top: 14px;
    margin-bottom: 18px;
}}

/* Titles */
h1, h2, h3 {{
    color: {PBS_BLUE};
    font-family: 'Segoe UI', Arial, sans-serif;
}}

/* Sidebar background */
section[data-testid="stSidebar"] {{
    background: {PBS_BLUE};
}}
section[data-testid="stSidebar"] * {{
    color: #ffffff !important;
}}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stSelectbox label {{
    font-weight: 650;
}}

/* IMPORTANT: fix blank inputs (text was white) */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] textarea {{
    color: {PBS_BLUE} !important;
    background: #ffffff !important;
    border-radius: 10px !important;
}}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
    color: {PBS_BLUE} !important;
    background: #ffffff !important;
    border-radius: 10px !important;
}}

/* Metrics cards */
div[data-testid="stMetric"] {{
    background: #f6faff;
    border: 1px solid {CARD_BORDER};
    padding: 14px 14px 10px 14px;
    border-radius: 12px;
}}

/* Dataframes */
div[data-testid="stDataFrame"] {{
    border: 1px solid {CARD_BORDER};
    border-radius: 12px;
    overflow: hidden;
}}

/* Buttons */
.stDownloadButton button, .stButton button {{
    background-color: {PBS_BLUE};
    color: white;
    border-radius: 10px;
    border: 0px;
}}
.stDownloadButton button:hover, .stButton button:hover {{
    background-color: #134a92;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =========================
# Header with Logo + Title
# =========================
LOGO_FILE = "16.-Politeknik-Banting-Selangor.png"  # put in same folder with app.py
logo_path = Path(LOGO_FILE)

c1, c2, c3 = st.columns([1.3, 6.7, 1.0], vertical_alignment="center")

with c1:
    # NO message if missing (as requested)
    if logo_path.exists():
        st.image(str(logo_path), width=120)

with c2:
    st.markdown(
        "<h1 style='margin-bottom:0px;'>SMART EGG INCUBATOR (LUX TREND ANALYZER)</h1>"
        "<div style='color:#2a4a7a;font-size:15px;font-weight:650;'>"
        "AI-based forecasting using Lux Trend (Day 1–18) + Temperature + Humidity"
        "</div>",
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        "<div style='text-align:right;font-size:52px;line-height:1;'>🥚</div>"
        "<div style='text-align:right;color:#2a4a7a;font-weight:800;'>Egg AI</div>",
        unsafe_allow_html=True
    )


# =========================
# Sidebar Settings
# =========================
st.sidebar.header("Settings")

thr = st.sidebar.number_input("Rule Threshold slope_7_18 (lux/day)", value=-13.0, step=0.5)

has_label_toggle = st.sidebar.checkbox("Enable ML (if label exists)", value=True)
test_size = st.sidebar.slider("ML Test size (if label exists)", 0.1, 0.5, 0.3, 0.05)

show_per_egg = st.sidebar.checkbox("Show per-egg view", value=True)
show_rule = st.sidebar.checkbox("Show Rule-based results", value=True)

# Image calibration moved to Advanced (clean UI)
with st.sidebar.expander("Advanced: Image → Lux Calibration (Proof Only)", expanded=False):
    calib_scale = st.number_input("Lux per intensity (scale)", value=20.0, step=1.0)
    calib_offset = st.number_input("Lux offset", value=0.0, step=10.0)
    roi_mode = st.selectbox("ROI mode", ["Center", "Full image"])
    roi_percent = st.slider("ROI size (center %)", 10, 100, 40, 5)


# =========================
# Helpers
# =========================
def to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    wide = df_long.pivot(index="egg_id", columns="day", values="lux").reset_index()
    if "label_hatch" in df_long.columns:
        lab = df_long.groupby("egg_id")["label_hatch"].first().reset_index()
        wide = wide.merge(lab, on="egg_id", how="left")
    return wide


def normalize_wide_columns(df_wide: pd.DataFrame) -> pd.DataFrame:
    for d in range(1, 19):
        col = f"lux_day{d:02d}"
        if col in df_wide.columns and d not in df_wide.columns:
            df_wide[d] = pd.to_numeric(df_wide[col], errors="coerce")
    return df_wide


def add_features(df_wide: pd.DataFrame) -> pd.DataFrame:
    for d in range(1, 19):
        if d in df_wide.columns:
            df_wide[d] = pd.to_numeric(df_wide[d], errors="coerce")

    needed = [1, 7, 18]
    missing = [d for d in needed if d not in df_wide.columns]
    if missing:
        st.error(f"Missing required day columns: {missing}. Need at least Day 1, 7, 18 lux.")
        st.stop()

    df_wide["delta_1_7"] = df_wide[7] - df_wide[1]
    df_wide["delta_7_18"] = df_wide[18] - df_wide[7]

    df_wide["slope_7_18_rule"] = (df_wide[18] - df_wide[7]) / 11.0

    def poly_slope(row, start_day, end_day):
        x = np.arange(start_day, end_day + 1, dtype=float)
        y = row.loc[start_day:end_day].values.astype(float)
        slope, _ = np.polyfit(x, y, 1)
        return slope

    have_1_7 = all(d in df_wide.columns for d in range(1, 8))
    have_7_18 = all(d in df_wide.columns for d in range(7, 19))

    df_wide["slope_1_7"] = df_wide.apply(lambda r: poly_slope(r, 1, 7), axis=1) if have_1_7 else (df_wide[7] - df_wide[1]) / 6.0
    df_wide["slope_7_18"] = df_wide.apply(lambda r: poly_slope(r, 7, 18), axis=1) if have_7_18 else df_wide["slope_7_18_rule"]

    return df_wide


def estimate_lux_from_image(img: Image.Image, scale: float, offset: float, roi_mode: str, roi_percent: int) -> dict:
    gray = img.convert("L")
    arr = np.array(gray).astype(np.float32)

    h, w = arr.shape
    if roi_mode == "Center":
        p = roi_percent / 100.0
        rh = int(h * p)
        rw = int(w * p)
        y0 = (h - rh) // 2
        x0 = (w - rw) // 2
        roi = arr[y0:y0 + rh, x0:x0 + rw]
    else:
        roi = arr

    intensity_mean = float(np.mean(roi))
    est_lux = intensity_mean * scale + offset

    return {
        "intensity_mean": intensity_mean,
        "est_lux": est_lux
    }


# =========================
# Upload CSV
# =========================
st.markdown("---")
uploaded = st.file_uploader(
    "Upload CSV (LONG: egg_id, day, lux, label_hatch, temp_c, rh_pct) OR WIDE: egg_id + day columns",
    type=["csv"]
)

if not uploaded:
    st.info("Please upload a CSV file to start.")
    st.stop()

df = pd.read_csv(uploaded)

if {"egg_id", "day", "lux"}.issubset(df.columns):
    df_long = df.copy()
    df_wide = to_wide(df_long)
else:
    df_wide = df.copy()
    df_long = None

if "egg_id" not in df_wide.columns:
    st.error("Missing required column: egg_id")
    st.stop()

df_wide["egg_id"] = df_wide["egg_id"].astype(str)
df_wide = normalize_wide_columns(df_wide)
df_wide = add_features(df_wide)

has_label = ("label_hatch" in df_wide.columns) and df_wide["label_hatch"].notna().any()

# =========================
# Sidebar: Dataset Summary (ONLY show if exists)
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Summary")

# Avg temp/rh ONLY if column exists in LONG file
if df_long is not None and "temp_c" in df_long.columns:
    st.sidebar.write(f"Avg Temperature: **{df_long['temp_c'].mean():.2f} °C**")
if df_long is not None and "rh_pct" in df_long.columns:
    st.sidebar.write(f"Avg Humidity: **{df_long['rh_pct'].mean():.2f} %RH**")

# Lux min/max for hatched eggs (ONLY if label exists)
if has_label:
    if df_long is not None and "label_hatch" in df_long.columns:
        hatched = df_long[df_long["label_hatch"] == 1]["lux"]
        if len(hatched) > 0:
            st.sidebar.write(f"Lux (Hatch=1) Min–Max: **{hatched.min():.2f} – {hatched.max():.2f}**")
    else:
        day_cols = [d for d in range(1, 19) if d in df_wide.columns]
        hatched_rows = df_wide[df_wide["label_hatch"] == 1]
        vals = hatched_rows[day_cols].values.flatten()
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            st.sidebar.write(f"Lux (Hatch=1) Min–Max: **{float(vals.min()):.2f} – {float(vals.max()):.2f}**")


# =========================
# Per-egg view
# =========================
if show_per_egg:
    st.markdown("---")
    st.subheader("Per Egg View")

    egg_list = df_wide["egg_id"].sort_values().tolist()
    selected_egg = st.selectbox("Select egg_id", egg_list)

    row = df_wide[df_wide["egg_id"] == selected_egg].iloc[0]
    days_available = [d for d in range(1, 19) if d in df_wide.columns and pd.notna(row[d])]
    lux_series = [row[d] for d in days_available]

    a, b, c = st.columns(3)
    a.metric("slope_7_18 (Rule)", f"{row['slope_7_18_rule']:.2f}")
    b.metric("slope_7_18 (Polyfit)", f"{row['slope_7_18']:.2f}")
    c.metric("delta_7_18", f"{row['delta_7_18']:.2f}")

    fig_egg = plt.figure()
    plt.plot(days_available, lux_series, marker="o")
    plt.xlabel("Day")
    plt.ylabel("Lux")
    plt.title(f"Egg {selected_egg}: Lux Trend (Day 1–18)")
    st.pyplot(fig_egg)


# =========================
# Rule-based section + 2-color histogram when label exists
# =========================
if show_rule:
    st.markdown("---")
    st.subheader("Rule-based Results (Threshold)")

    df_wide["pred_rule"] = np.where(df_wide["slope_7_18_rule"] < thr, 1, 0)
    df_wide["status_rule"] = np.where(df_wide["pred_rule"] == 1, "Likely Hatch", "Unlikely")

    m1, m2, m3 = st.columns(3)
    m1.metric("Total eggs", len(df_wide))
    m2.metric("Likely Hatch (Rule)", int((df_wide["pred_rule"] == 1).sum()))
    m3.metric("Unlikely (Rule)", int((df_wide["pred_rule"] == 0).sum()))

    st.dataframe(
        df_wide[["egg_id", "slope_7_18_rule", "delta_7_18", "status_rule"]]
        .rename(columns={"slope_7_18_rule": "slope_7_18"})
        .sort_values("egg_id")
    )

    st.write("Histogram: slope_7_18 (Rule)")
    fig = plt.figure()

    data_all = df_wide["slope_7_18_rule"].dropna()

    if has_label:
        # 2-color histogram (as you requested)
        d1 = df_wide[df_wide["label_hatch"] == 1]["slope_7_18_rule"].dropna()
        d0 = df_wide[df_wide["label_hatch"] == 0]["slope_7_18_rule"].dropna()
        plt.hist(d0, bins=35, alpha=0.7, label="Hatch = 0")
        plt.hist(d1, bins=35, alpha=0.7, label="Hatch = 1")
        plt.legend()
    else:
        plt.hist(data_all, bins=35)

    plt.axvline(thr)
    plt.xlabel("slope_7_18 (lux/day)")
    plt.ylabel("count")
    plt.title("Distribution of slope_7_18 (Rule)")
    st.pyplot(fig)

    # Trend
    if df_long is not None:
        st.write("Trend: Average Lux by Day")
        grouped = df_long.groupby("day")["lux"].mean().reset_index()
        fig2 = plt.figure()
        plt.plot(grouped["day"], grouped["lux"], marker="o")
        plt.xlabel("day")
        plt.ylabel("avg lux")
        plt.title("Average Lux Trend (All Eggs)")
        st.pyplot(fig2)


# =========================
# ML Logistic Regression (optional)
# =========================
if has_label_toggle and has_label:
    st.markdown("---")
    st.subheader("Explainable AI: Logistic Regression")

    features = ["slope_1_7", "slope_7_18", "delta_1_7", "delta_7_18"]
    X = df_wide[features].copy()
    y = df_wide["label_hatch"].astype(int)

    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]
    y = y[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    a, b, c = st.columns(3)
    a.metric("Accuracy (test)", f"{(y_pred == y_test).mean():.3f}")
    b.metric("ROC-AUC (test)", f"{auc:.3f}")
    c.metric("Test samples", f"{len(y_test)}")

    st.write("Confusion Matrix (test):")
    st.write(cm)

    st.write("Classification Report (test):")
    st.text(classification_report(y_test, y_pred))

    st.write("ROC Curve (test):")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    figroc = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    st.pyplot(figroc)

    # Auto explanation table (clean)
    coef_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_[0]})
    coef_df["Direction"] = np.where(coef_df["Coefficient"] < 0, "More negative → higher hatch probability", "More positive → lower hatch probability")
    coef_df = coef_df.sort_values(by="Coefficient", key=lambda s: s.abs(), ascending=False)

    st.write("Auto Explanation (Table):")
    st.dataframe(coef_df)


# =========================
# Image Module (Proof only)
# =========================
st.markdown("---")
st.subheader("Image-based Lux Estimation (Proof Only)")

img_up = st.file_uploader("Upload egg candling image (JPG/PNG)", type=["png", "jpg", "jpeg"], key="eggimg")
if img_up:
    img = Image.open(img_up)
    st.image(img, caption="Uploaded candling image", use_container_width=True)

    res = estimate_lux_from_image(img, scale=calib_scale, offset=calib_offset, roi_mode=roi_mode, roi_percent=roi_percent)
    x1, x2 = st.columns(2)
    x1.metric("Mean intensity (0–255)", f"{res['intensity_mean']:.2f}")
    x2.metric("Estimated Lux", f"{res['est_lux']:.2f}")
st.markdown("---")
st.subheader("Generate Report")

# Prepare summary numbers
total_eggs = len(df_wide)
rule_likely = int((df_wide["pred_rule"] == 1).sum()) if "pred_rule" in df_wide.columns else None
rule_unlikely = int((df_wide["pred_rule"] == 0).sum()) if "pred_rule" in df_wide.columns else None

lux_min = None
lux_max = None
if has_label and df_long is not None and "label_hatch" in df_long.columns:
    hatched = df_long[df_long["label_hatch"] == 1]["lux"]
    if len(hatched) > 0:
        lux_min = float(hatched.min())
        lux_max = float(hatched.max())

# ML metrics if available
ml_text = ""
if has_label_toggle and has_label:
    ml_text = f"""
    <h3>Machine Learning (Logistic Regression)</h3>
    <ul>
      <li><b>Test size</b>: {test_size}</li>
      <li><b>Accuracy (test)</b>: {(y_pred == y_test).mean():.3f}</li>
      <li><b>ROC-AUC (test)</b>: {auc:.3f}</li>
      <li><b>Confusion Matrix</b>: {cm.tolist()}</li>
    </ul>
    """

# Button to generate downloadable HTML report
if st.button("Generate Report"):
    report_html = f"""
    <html>
    <head>
      <meta charset="utf-8">
      <title>Smart Egg AI Report</title>
      <style>
        body {{ font-family: Arial; margin: 30px; }}
        h1 {{ color: #0b2d5c; }}
        h2 {{ color: #134a92; }}
        .card {{ border: 1px solid #d7e6ff; padding: 14px; border-radius: 10px; margin-bottom: 12px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #eef5ff; }}
      </style>
    </head>
    <body>
      <h1>SMART EGG INCUBATOR (LUX TREND ANALYZER) - REPORT</h1>
      <p><b>File:</b> {uploaded.name} <br/>
         <b>Generated:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
      </p>

      <div class="card">
        <h2>Dataset Summary</h2>
        <ul>
          <li><b>Total eggs</b>: {total_eggs}</li>
          <li><b>Rule Threshold slope_7_18</b>: {thr}</li>
          <li><b>Rule Likely Hatch</b>: {rule_likely if rule_likely is not None else "N/A"}</li>
          <li><b>Rule Unlikely</b>: {rule_unlikely if rule_unlikely is not None else "N/A"}</li>
          <li><b>Lux (Hatch=1) Min–Max</b>: {f"{lux_min:.2f} – {lux_max:.2f}" if (lux_min is not None) else "N/A"}</li>
        </ul>
      </div>

      {ml_text}

      <div class="card">
        <h2>Interpretation (Panel-friendly)</h2>
        <ul>
          <li><b>slope_7_18</b> represents daily lux change from day 7 to day 18.</li>
          <li>More negative slope means lux decreases faster (egg becomes darker), often associated with embryo development.</li>
          <li>Rule-based decision uses threshold: if slope_7_18 &lt; {thr} → Likely Hatch.</li>
          <li>Logistic Regression provides explainable probability using slope and delta features.</li>
        </ul>
      </div>
    </body>
    </html>
    """

    st.success("Report generated. Download below.")
    st.download_button(
        "Download Report (HTML)",
        data=report_html.encode("utf-8"),
        file_name="SmartEgg_AI_Report.html",
        mime="text/html"
    )
st.success("Ready. Upload another CSV anytime.")