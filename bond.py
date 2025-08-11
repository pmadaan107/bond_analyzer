# yield_spread_forecast_app.py (International)
# -------------------------------------------------------------
# Yield Spread Forecasting with AR/VAR and Risk/"Recession" Probability
# -------------------------------------------------------------
# New: COUNTRY SUPPORT
# - Choose a country (US, CA, UK, DE, FR, JP, AU, IN)
# - Pull monthly long-term (≈10Y) and short-term (≈3M) rates from FRED's
#   OECD-harmonized series via CSV endpoints (no API key, Python 3.12+ safe)
# - Build 10Y–3M spread (proxy for 10Y–2Y when 2Y isn't available)
# - Macro covariates from OECD series on FRED: Unemployment, CPI YoY, IP YoY
# - For the US ONLY: use NBER USREC as the recession target for a logistic model
# - For other countries: use a generic "risk" probability: probability that the
#   yield curve is inverted within a forward window (logistic model)
# -------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tools.tools import add_constant
import plotly.graph_objects as go
from datetime import date

# =============================
# ---------- UI SETUP ---------
# =============================
st.set_page_config(page_title="Yield Spread Forecasting (AR/VAR) – International", layout="wide")

HERO_TITLE = "🌍 Yield Spread Forecasting — AR/VAR + Risk Probability"
HERO_SUB = "Pick a country • OECD/FRED monthly data • CFA L2 concepts: term structure, econometrics"

st.markdown(
    """
    <style>
    :root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3; --muted:#9db1d6; }
    html, body, [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--text); }
    [data-testid="stSidebar"] { background: #0c1220; }
    .big { font-size: 1.6rem; font-weight:700; }
    .muted { color: var(--muted); }
    .card { background: var(--panel); padding: 16px; border:1px solid var(--border); border-radius: 16px; }
    .hr { height:1px; background:var(--border); margin: 1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"<div class='big'>{HERO_TITLE}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='muted'>{HERO_SUB}</div>", unsafe_allow_html=True)
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# =============================
# --------- DATA LAYER --------
# =============================
# FRED CSV helper (works on Streamlit Cloud without pandas_datareader)

def fred_series(series_id: str, start=None, end=None) -> pd.DataFrame:
    """Robust FRED loader.
    First try the stable 'downloaddata' CSV (DATE,VALUE). If that fails, fall back to fredgraph.csv.
    """
    # 1) Preferred: downloaddata endpoint
    url1 = f"https://fred.stlouisfed.org/series/{series_id}/downloaddata/{series_id}.csv"
    try:
        s = pd.read_csv(url1, parse_dates=["DATE"]).rename(columns={"VALUE": series_id})
        s = s.set_index("DATE").sort_index()
    except Exception:
        # 2) Fallback: fredgraph.csv?id=SERIES plus explicit start/end
        url2 = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        s = pd.read_csv(url2, parse_dates=["DATE"]).rename(columns={series_id: series_id})
        s = s.set_index("DATE").sort_index()

    # Coerce to numeric and clean
    s[series_id] = pd.to_numeric(s[series_id], errors="coerce")
    if start:
        s = s[s.index >= pd.to_datetime(start)]
    if end:
        s = s[s.index <= pd.to_datetime(end)]
    return s

# Country → OECD/FRED series map (monthly)
# Long-term yields (10Y): IRLTLT01{ISO3}M156N (percent per annum)
# Short-term rates (~3M): IR3TIB01{ISO3}M156N
# Unemployment rate: LRUNTTTT{ISO3}M156S
# CPI YoY: CPALTT01{ISO3}M657N (growth rate, same period prev year, %)
# Industrial Production YoY: PRMNTO01{ISO3}M657N (%, total industry)
COUNTRIES = {
    "United States": {
        "iso3": "USA",
        "LT": "DGS10",        # keep US from DGS10 (daily) but we'll resample to M
        "ST": "DGS2",         # 2Y for closer match; fallback to IR3TIB01USA... if needed
        "UNRATE": "UNRATE",
        "CPI_YoY": "CPIAUCSL",  # we'll transform to YoY
        "INDPRO": "INDPRO",
        "USREC": "USREC",
        "is_us": True,
    },
    "Canada":      {"iso3":"CAN"},
    "United Kingdom": {"iso3":"GBR"},
    "Germany":     {"iso3":"DEU"},
    "France":      {"iso3":"FRA"},
    "Japan":       {"iso3":"JPN"},
    "Australia":   {"iso3":"AUS"},
    "India":       {"iso3":"IND"},
}


def load_country_data(country: str, start: str, end: str) -> pd.DataFrame:
    cfg = COUNTRIES[country]
    if cfg.get("is_us"):
        # US: pull daily series then aggregate to monthly means
        lt = fred_series(cfg["LT"], start, end)  # DGS10 daily
        st_ = fred_series(cfg["ST"], start, end) # DGS2  daily
        un = fred_series(cfg["UNRATE"], start, end)
        cpi = fred_series(cfg["CPI_YoY"], start, end)   # level
        ip = fred_series(cfg["INDPRO"], start, end)     # level
        rec = fred_series(cfg["USREC"], start, end)     # monthly 0/1

        df = lt.join(st_, how="outer").join(un, how="outer").join(cpi, how="outer").join(ip, how="outer").join(rec, how="outer")
        df = df.sort_index()
        # Resample to monthly average for rates/levels
        m = df.resample("M").mean()
        m["SPREAD"] = m[cfg["LT"]] - m[cfg["ST"]]
        # Transform to YoY
        m["CPI_YoY"] = m[cfg["CPI_YoY"]].pct_change(12) * 100
        m["INDPRO_YoY"] = m[cfg["INDPRO"]].pct_change(12) * 100
        # Standardize column names
        out = m.rename(columns={cfg["LT"]:"LT", cfg["ST"]:"ST", cfg["UNRATE"]:"UNRATE", cfg["CPI_YoY"]:"CPI", cfg["INDPRO"]:"INDPRO", cfg["USREC"]:"USREC"})
        out = out[["SPREAD","UNRATE","CPI_YoY","INDPRO_YoY","USREC"]]
        return out.dropna(subset=["SPREAD"]).ffill()

    # Non-US: use OECD monthly identifiers hosted on FRED
    iso3 = cfg["iso3"]
    lt_id = f"IRLTLT01{iso3}M156N"
    st_id = f"IR3TIB01{iso3}M156N"
    un_id = f"LRUNTTTT{iso3}M156S"
    cpi_yoy_id = f"CPALTT01{iso3}M657N"
    ip_yoy_id = f"PRMNTO01{iso3}M657N"

    lt = fred_series(lt_id, start, end)
    st_ = fred_series(st_id, start, end)
    un = fred_series(un_id, start, end)
    cpi = fred_series(cpi_yoy_id, start, end)
    ip = fred_series(ip_yoy_id, start, end)

    df = lt.join(st_, how="outer").join(un, how="outer").join(cpi, how="outer").join(ip, how="outer")
    df = df.sort_index()

    # Build unified frame
    out = pd.DataFrame(index=df.index)
    out["LT"] = df[lt_id]
    out["ST"] = df[st_id]
    out["UNRATE"] = df[un_id]
    out["CPI_YoY"] = df[cpi_yoy_id]
    out["INDPRO_YoY"] = df[ip_yoy_id]
    out["SPREAD"] = out["LT"] - out["ST"]
    # No official recession series for generic countries
    out["USREC"] = np.nan
    return out.dropna(subset=["SPREAD"]).ffill()

# =============================
# --------- MODELING -----------
# =============================

def fit_ar(model_df: pd.DataFrame, max_lags: int = 12):
    y = model_df["SPREAD"].dropna()
    best_aic, best_p, best_model = np.inf, None, None
    for p in range(1, max_lags + 1):
        try:
            m = AutoReg(y, lags=p, old_names=False).fit()
            if m.aic < best_aic:
                best_aic, best_p, best_model = m.aic, p, m
        except Exception:
            pass
    return best_model, best_p, best_aic


def fit_var(model_df: pd.DataFrame, max_lags: int = 6):
    X = model_df[["SPREAD", "UNRATE", "CPI_YoY", "INDPRO_YoY"]].dropna()
    if len(X) < 24:
        raise ValueError("Not enough data points for VAR; try an earlier start date.")
    var = VAR(X)
    sel = var.select_order(maxlags=max_lags)
    p = sel.aic or sel.bic or sel.fpe or sel.hqic
    p = int(p) if p is not None else 2
    res = var.fit(maxlags=p)
    return res, p


def forecast_ar(model, steps: int = 12):
    fc = model.get_forecast(steps=steps)
    mean = fc.predicted_mean
    conf = fc.conf_int(alpha=0.05)
    return mean, conf


def forecast_var(model, steps: int = 12):
    fc = model.forecast_interval(model.endog, steps=steps, alpha=0.05)
    mean = pd.DataFrame(fc[0], columns=model.names)
    lower = pd.DataFrame(fc[1], columns=model.names)
    upper = pd.DataFrame(fc[2], columns=model.names)
    return mean, lower, upper


def fit_us_recession_logit(model_df: pd.DataFrame, lag_months: int = 6):
    d = model_df.dropna(subset=["USREC"]).copy()
    d["REC_FWD"] = d["USREC"].rolling(window=lag_months, min_periods=1).max().shift(-lag_months)
    d = d.dropna()
    X = add_constant(d[["SPREAD", "UNRATE", "CPI_YoY", "INDPRO_YoY"]], has_constant="add")
    y = d["REC_FWD"]
    logit = sm.Logit(y, X).fit(disp=False)
    d["PROB"] = logit.predict(X)
    return logit, d.rename(columns={"PROB":"REC_PROB"})


def fit_inversion_risk_logit(model_df: pd.DataFrame, lag_months: int = 6):
    """Generic risk proxy for non-US: probability of an inversion (SPREAD<0) within next L months."""
    d = model_df.copy()
    inv = (d["SPREAD"] < 0).astype(int)
    d["INV_FWD"] = inv.rolling(window=lag_months, min_periods=1).max().shift(-lag_months)
    d = d.dropna()
    X = add_constant(d[["SPREAD", "UNRATE", "CPI_YoY", "INDPRO_YoY"]], has_constant="add")
    y = d["INV_FWD"]
    logit = sm.Logit(y, X).fit(disp=False)
    d["INV_PROB"] = logit.predict(X)
    return logit, d

# =============================
# --------- VISUALS ------------
# =============================

def make_spread_chart(hist: pd.Series, fc_mean: pd.Series = None, fc_lower=None, fc_upper=None, title="Term Spread"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="Historical"))
    if fc_mean is not None:
        fig.add_trace(go.Scatter(x=fc_mean.index, y=fc_mean.values, mode="lines", name="Forecast"))
        if fc_lower is not None and fc_upper is not None:
            fig.add_trace(go.Scatter(x=fc_mean.index, y=fc_upper.values, mode="lines", name="Upper 95%", line=dict(dash="dot")))
            fig.add_trace(go.Scatter(x=fc_mean.index, y=fc_lower.values, mode="lines", name="Lower 95%", line=dict(dash="dot"), fill="tonexty"))
    fig.update_layout(template="plotly_dark", height=420, title=title, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def make_prob_chart(prob_s: pd.Series, thresh: float, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prob_s.index, y=prob_s.values, mode="lines", name="Probability"))
    fig.add_hline(y=thresh, line_dash="dash", annotation_text=f"Threshold {thresh:.0%}")
    fig.update_layout(template="plotly_dark", height=300, title=title)
    return fig


def turning_points(s: pd.Series, thresh: float):
    cross_up = (s.shift(1) < thresh) & (s >= thresh)
    cross_dn = (s.shift(1) >= thresh) & (s < thresh)
    return s[cross_up].index.to_list(), s[cross_dn].index.to_list()

# =============================
# -------- SIDEBAR -------------
# =============================
st.sidebar.header("⚙️ Controls")
country = st.sidebar.selectbox("Country", list(COUNTRIES.keys()), index=0)
start = st.sidebar.date_input("Start date", value=date(1990,1,1))
end = st.sidebar.date_input("End date", value=date.today())
model_type = st.sidebar.radio("Model Type", ["AR", "VAR"], index=0)
max_lags = st.sidebar.slider("Max lags (order selection)", 2, 18, 12)
steps = st.sidebar.slider("Forecast horizon (months)", 3, 24, 12)
recess_lag = st.sidebar.slider("Forward window (months)", 3, 18, 12)
prob_thresh = st.sidebar.slider("Turn-point threshold", 0.1, 0.9, 0.5, 0.05)

with st.sidebar.expander("ℹ️ Data notes"):
    st.write(
        """
        **Sources:** FRED CSV endpoints. For non-US, OECD-harmonized monthly series:
        • LT: IRLTLT01{ISO3}M156N  • ST: IR3TIB01{ISO3}M156N  • UNRATE: LRUNTTTT{ISO3}M156S
        • CPI YoY: CPALTT01{ISO3}M657N  • IP YoY: PRMNTO01{ISO3}M657N
        US uses DGS10/DGS2/UNRATE/CPIAUCSL/INDPRO/USREC.
        """
    )

# =============================
# --------- MAIN FLOW ---------
# =============================

with st.spinner(f"Loading {country} data…"):
    model_df = load_country_data(country, start.isoformat(), end.isoformat())

col1, col2, col3 = st.columns([2,1,1])
with col1:
    st.markdown("### 📊 Data Preview")
    st.dataframe(model_df.tail(12))
with col2:
    st.markdown("### 🔧 Model Fit")
    if model_type == "AR":
        ar_model, p_sel, aic = fit_ar(model_df, max_lags=max_lags)
        st.metric("AR order (AIC)", p_sel)
        st.caption(f"AIC: {aic:.2f}")
    else:
        try:
            var_model, p_sel = fit_var(model_df, max_lags=min(max_lags, 12))
            st.metric("VAR order (AIC)", p_sel)
        except Exception as e:
            st.error(str(e))
            var_model = None
with col3:
    st.markdown("### 🧭 Horizon")
    st.metric("Forecast months", steps)
    st.metric("Forward window", recess_lag)

# Forecast
hist_spread = model_df["SPREAD"].copy()
if model_type == "AR":
    mean, conf = forecast_ar(ar_model, steps=steps)
    fc_index = pd.date_range(hist_spread.index[-1] + pd.offsets.MonthEnd(0), periods=steps, freq="M")
    fc_mean = pd.Series(mean.values, index=fc_index, name="Forecast")
    fc_lower = pd.Series(conf.iloc[:, 0].values, index=fc_index, name="Lower95")
    fc_upper = pd.Series(conf.iloc[:, 1].values, index=fc_index, name="Upper95")
else:
    if var_model is not None:
        mean, lower, upper = forecast_var(var_model, steps=steps)
        fc_index = pd.date_range(hist_spread.index[-1] + pd.offsets.MonthEnd(0), periods=steps, freq="M")
        fc_mean = pd.Series(mean["SPREAD"].values, index=fc_index, name="Forecast")
        fc_lower = pd.Series(lower["SPREAD"].values, index=fc_index, name="Lower95")
        fc_upper = pd.Series(upper["SPREAD"].values, index=fc_index, name="Upper95")
    else:
        fc_mean = fc_lower = fc_upper = None

st.plotly_chart(
    make_spread_chart(hist_spread, fc_mean, fc_lower, fc_upper, title=f"{country}: Term Spread (LT–ST)"),
    use_container_width=True,
)

# Probability models
if country == "United States":
    logit_model, prob_df = fit_us_recession_logit(model_df, lag_months=recess_lag)
    prob_series = prob_df["REC_PROB"]
    title = "Recession Probability (NBER, forward window)"
else:
    logit_model, prob_df = fit_inversion_risk_logit(model_df, lag_months=recess_lag)
    prob_series = prob_df["INV_PROB"]
    title = "Yield Curve Inversion Risk (forward window)"

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Turning-Point Probability")
st.plotly_chart(make_prob_chart(prob_series, prob_thresh, title), use_container_width=True)

ups, downs = turning_points(prob_series, prob_thresh)
colA, colB = st.columns(2)
with colA:
    st.markdown("**Crossed ABOVE threshold** (risk rising)")
    st.write([d.strftime("%Y-%m-%d") for d in ups] or "– None –")
with colB:
    st.markdown("**Crossed BELOW threshold** (risk easing)")
    st.write([d.strftime("%Y-%m-%d") for d in downs] or "– None –")

st.markdown("</div>", unsafe_allow_html=True)

# Downloadables

def to_csv(df: pd.DataFrame):
    return df.to_csv(index=True).encode("utf-8")

c1, c2, c3 = st.columns(3)
with c1:
    st.download_button("⬇️ Download dataset (CSV)", data=to_csv(model_df), file_name=f"{country}_dataset.csv", mime="text/csv")
with c2:
    if fc_mean is not None:
        fc_df = pd.DataFrame({"Forecast": fc_mean, "Lower95": fc_lower, "Upper95": fc_upper})
        st.download_button("⬇️ Download forecast (CSV)", data=to_csv(fc_df), file_name=f"{country}_forecast.csv", mime="text/csv")
with c3:
    st.download_button("⬇️ Download probabilities (CSV)", data=to_csv(prob_series.to_frame("PROB")), file_name=f"{country}_prob.csv", mime="text/csv")

st.caption("Data via FRED/OECD; for non-US countries, probability targets inverted-curve risk rather than official recession dates.")

