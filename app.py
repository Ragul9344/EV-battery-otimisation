import io
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="EV Battery Thermal AI", page_icon="🔋", layout="wide")
FEATURES = ["battery_temp_C", "current_A", "voltage_V", "ambient_temp_C"]
TARGET = "next_temp_C"
DEMO_PATH = Path(__file__).parent / "data" / "demo_battery_thermal.csv"

st.title("🔋 AI-Based Predictive EV Battery Thermal Management")
st.caption("Student prototype • temperature forecasting • cooling decision support")
st.warning("Research/demo tool only. It is not a certified battery-management system. Do not connect this dashboard directly to a real EV battery or use it as a safety controller.")

with st.sidebar:
    st.header("1. Choose data")
    mode = st.radio("Dataset", ["Built-in demo dataset", "Upload your CSV", "Upload NASA .mat file"])
    uploaded = None
    if mode != "Built-in demo dataset":
        uploaded = st.file_uploader("Choose dataset file", type=["csv"] if mode == "Upload your CSV" else ["mat"])
    st.divider()
    st.header("2. Forecast settings")
    horizon = st.slider("Forecast steps ahead (dataset rows)", 1, 10, 1)
    threshold = st.slider("Illustrative temperature alert (°C)", 30, 60, 40)
    assumed_cooling_effect = st.slider("Assumed cooling effect for scenario only (°C per step)", 0.0, 3.0, 0.8, 0.1)

def synthetic_data(n=240, seed=12):
    """Synthetic demo data, not measured EV data."""
    rng = np.random.default_rng(seed)
    ambient = rng.uniform(20, 38, n)
    current = rng.uniform(1, 18, n)
    voltage = rng.uniform(3.2, 4.15, n)
    temp = np.zeros(n)
    temp[0] = ambient[0] + 2
    for i in range(n - 1):
        heat = 0.018 * current[i] ** 2 + 0.20 * max(ambient[i] - 25, 0)
        cooling = 0.10 * (temp[i] - ambient[i])
        temp[i + 1] = np.clip(temp[i] + 0.22 * heat - 0.06 * cooling + rng.normal(0, 0.18), ambient[i] - 1, 65)
    df = pd.DataFrame({"battery_temp_C": temp, "current_A": current, "voltage_V": voltage, "ambient_temp_C": ambient})
    df[TARGET] = df["battery_temp_C"].shift(-1)
    return df.dropna().reset_index(drop=True)

def field(obj, name):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.void) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    if isinstance(obj, dict):
        return obj.get(name)
    return None

def numbers(value):
    if value is None:
        return np.array([])
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.array([])

def nasa_mat_to_frame(raw_bytes):
    """Best-effort reader for common NASA PCoE battery .mat structures."""
    from scipy.io import loadmat
    mat = loadmat(io.BytesIO(raw_bytes), squeeze_me=True, struct_as_record=False)
    root = next((v for k, v in mat.items() if not k.startswith("__")), None)
    cycles = field(root, "cycle")
    if cycles is None:
        raise ValueError("Could not find the NASA cycle structure in this .mat file.")
    rows = []
    for cycle in np.atleast_1d(cycles):
        kind = field(cycle, "type")
        if isinstance(kind, np.ndarray):
            kind = kind.item() if kind.size == 1 else kind
        if isinstance(kind, bytes):
            kind = kind.decode(errors="ignore")
        if isinstance(kind, str) and kind.lower() == "impedance":
            continue
        ambient_values = numbers(field(cycle, "ambient_temperature"))
        ambient = float(ambient_values[0]) if ambient_values.size else np.nan
        d = field(cycle, "data")
        temps = numbers(field(d, "Temperature_measured"))
        currents = numbers(field(d, "Current_measured"))
        volts = numbers(field(d, "Voltage_measured"))
        count = min(len(temps), len(currents), len(volts))
        for i in range(count):
            if np.isfinite(temps[i]) and np.isfinite(currents[i]) and np.isfinite(volts[i]):
                rows.append({"battery_temp_C": float(temps[i]), "current_A": float(currents[i]), "voltage_V": float(volts[i]), "ambient_temp_C": ambient if np.isfinite(ambient) else float(temps[i] - 2)})
    frame = pd.DataFrame(rows)
    if len(frame) < 20:
        raise ValueError("Could not extract enough charge/discharge measurements. Try a NASA PCoE B0005/B0006/B0007/B0018 .mat file.")
    frame[TARGET] = frame["battery_temp_C"].shift(-1)
    return frame.dropna().reset_index(drop=True)

def validate_frame(df):
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    clean = df[required].apply(pd.to_numeric, errors="coerce").dropna()
    if len(clean) < 20:
        raise ValueError("At least 20 valid rows are required.")
    return clean.reset_index(drop=True)

try:
    if mode == "Built-in demo dataset":
        data = pd.read_csv(DEMO_PATH) if DEMO_PATH.exists() else synthetic_data()
        data_label = "Synthetic demonstration data — not real battery measurements"
    elif uploaded is None:
        st.info("Upload a CSV or NASA .mat file to continue, or choose the built-in demo dataset.")
        st.stop()
    elif mode == "Upload your CSV":
        data = pd.read_csv(uploaded)
        data_label = "User-uploaded CSV: " + uploaded.name
    else:
        data = nasa_mat_to_frame(uploaded.getvalue())
        data_label = "NASA MATLAB data: " + uploaded.name
    data = validate_frame(data)
except Exception as exc:
    st.error("Could not load dataset: " + str(exc))
    st.markdown("CSV columns required: battery_temp_C, current_A, voltage_V, ambient_temp_C, next_temp_C.")
    st.markdown("For NASA data, use a compatible PCoE battery .mat file.")
    st.stop()

st.info(f"Active dataset: {data_label} | Usable rows: {len(data):,}")
if len(data) < 30:
    st.warning("Small dataset: forecast metrics may be unstable.")
X, y = data[FEATURES], data[TARGET]
test_count = max(1, int(len(data) * 0.2))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_count, shuffle=False)
model = RandomForestRegressor(n_estimators=120, random_state=42, min_samples_leaf=2)
model.fit(X_train, y_train)
test_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, test_pred)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest battery temperature", f"{data['battery_temp_C'].iloc[-1]:.1f} °C")
m2.metric("Test MAE", f"{mae:.2f} °C")
m3.metric("Dataset rows", f"{len(data):,}")
m4.metric("Alert threshold", f"{threshold} °C")

left, right = st.columns(2)
with left:
    st.subheader("Temperature sequence")
    p = data.reset_index().rename(columns={"index": "row"})
    st.plotly_chart(px.line(p, x="row", y=["battery_temp_C", TARGET], labels={"value": "Temperature (°C)", "row": "Dataset row", "variable": "Series"}), use_container_width=True)
with right:
    st.subheader("Test prediction vs actual")
    p = pd.DataFrame({"Actual": y_test.to_numpy(), "Predicted": test_pred})
    p["Test row"] = np.arange(len(p))
    st.plotly_chart(px.line(p, x="Test row", y=["Actual", "Predicted"], labels={"value": "Temperature (°C)", "variable": "Series"}), use_container_width=True)

st.subheader("Predict and compare cooling scenarios")
latest = data.iloc[-1]
base = {c: float(latest[c]) for c in FEATURES}
forecasts = []
for step in range(1, horizon + 1):
    pred = float(model.predict(pd.DataFrame([base])[FEATURES])[0])
    forecasts.append({"Forecast step": step, "AI forecast": pred, "Illustrative cooled scenario": pred - assumed_cooling_effect, "Alert": "CHECK" if pred >= threshold else "Normal"})
    base["battery_temp_C"] = pred
forecast = pd.DataFrame(forecasts)
a, b = st.columns(2)
a.metric("Next-step AI forecast", f"{forecast['AI forecast'].iloc[0]:.1f} °C")
b.metric("Illustrative cooled scenario", f"{forecast['Illustrative cooled scenario'].iloc[0]:.1f} °C")
st.plotly_chart(px.line(forecast, x="Forecast step", y=["AI forecast", "Illustrative cooled scenario"], markers=True, labels={"value": "Temperature (°C)", "variable": "Scenario"}), use_container_width=True)
st.dataframe(forecast, use_container_width=True, hide_index=True)
if (forecast["Alert"] == "CHECK").any():
    st.error("Forecast reaches the chosen alert threshold. Review the battery using an approved BMS.")
else:
    st.success("Forecast stays below the chosen illustrative alert threshold.")

st.subheader("Model feature importance")
importance = pd.DataFrame({"Input": FEATURES, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
st.plotly_chart(px.bar(importance, x="Importance", y="Input", orientation="h"), use_container_width=True)
with st.expander("Dataset format, assumptions and safety"):
    st.markdown("""
    **Required CSV columns:** battery_temp_C, current_A, voltage_V, ambient_temp_C, next_temp_C.

    The NASA PCoE dataset can include measured battery temperature, voltage, current and ambient temperature during charge/discharge cycles. This model predicts the next recorded sample; it does not guarantee a fixed number of minutes ahead. Cooling effect is an adjustable what-if assumption, not a learned value, because the NASA dataset does not directly label fan interventions.

    This educational prototype does not control hardware or replace a certified battery-management system. Real deployment requires pack-specific data, validated sensors, manufacturer limits and safety testing.
    """)
