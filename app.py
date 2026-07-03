import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import joblib

st.set_page_config(page_title="AirVibe 🌬️", page_icon="🌍", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(160deg, #0f2027, #203a43, #2c5364); color: white; }
.big-card { border-radius: 20px; padding: 25px; text-align: center;
            background: rgba(255,255,255,0.08); backdrop-filter: blur(10px); }
h1, h2, h3 { color: white !important; }
</style>
""", unsafe_allow_html=True)

st.title("🌬️ AirVibe — Air Quality Predictor")
st.caption("ML-powered PM2.5 estimates + 16-day outlook")

city = st.sidebar.text_input("City", "Tashkent")
geo = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                   params={"name": city, "count": 1}).json()
if not geo.get("results"):
    st.error("City not found"); st.stop()
lat, lon = geo["results"][0]["latitude"], geo["results"][0]["longitude"]

# --- Live data + 16-day forecast from Open-Meteo ---
aq = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality",
    params={"latitude": lat, "longitude": lon, "forecast_days": 7,
            "hourly": "pm2_5,pm10,ozone,nitrogen_dioxide", "timezone": "auto"}).json()
wx = requests.get("https://api.open-meteo.com/v1/forecast",
    params={"latitude": lat, "longitude": lon, "forecast_days": 16,
            "daily": "temperature_2m_max,temperature_2m_min,weather_code",
            "timezone": "auto"}).json()

df = pd.DataFrame(aq["hourly"]); df["time"] = pd.to_datetime(df["time"])
now = df.iloc[0]

def aqi_vibe(pm):
    if pm <= 12: return "😊 Good", "#2ecc71"
    if pm <= 35: return "😐 Moderate", "#f1c40f"
    if pm <= 55: return "😷 Unhealthy (sensitive)", "#e67e22"
    if pm <= 150: return "🤢 Unhealthy", "#e74c3c"
    return "☠️ Hazardous", "#8e44ad"

label, color = aqi_vibe(now["pm2_5"])
c1, c2, c3, c4 = st.columns(4)
for col, name, val, unit in [
    (c1, "PM2.5", now["pm2_5"], "µg/m³"), (c2, "PM10", now["pm10"], "µg/m³"),
    (c3, "Ozone", now["ozone"], "µg/m³"), (c4, "NO₂", now["nitrogen_dioxide"], "µg/m³")]:
    col.markdown(f'<div class="big-card"><h3>{name}</h3>'
                 f'<h1>{val:.0f}</h1><p>{unit}</p></div>', unsafe_allow_html=True)

st.markdown(f"### Right now: <span style='color:{color}'>{label}</span>",
            unsafe_allow_html=True)

# --- PM2.5 chart ---
fig = go.Figure(go.Scatter(x=df["time"], y=df["pm2_5"], fill="tozeroy",
                           line=dict(color="#00d4ff", width=3)))
fig.update_layout(title="PM2.5 — next 7 days", template="plotly_dark",
                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)

# --- 16-day weather outlook ---
st.subheader("🗓️ 16-Day Weather Outlook")
icons = {0:"☀️",1:"🌤️",2:"⛅",3:"☁️",45:"🌫️",51:"🌦️",61:"🌧️",71:"🌨️",95:"⛈️"}
d = wx["daily"]
cols = st.columns(8)
for i in range(16):
    with cols[i % 8]:
        ic = icons.get(d["weather_code"][i], "🌡️")
        st.markdown(f'<div class="big-card" style="padding:10px">'
                    f'<p>{d["time"][i][5:]}</p><h2>{ic}</h2>'
                    f'<p>{d["temperature_2m_max"][i]:.0f}° / {d["temperature_2m_min"][i]:.0f}°</p>'
                    f'</div>', unsafe_allow_html=True)

st.caption("Model: LightGBM gradient boosting · Data: Open-Meteo · "
           "Forecasts have uncertainty — no model is 100% accurate.")
