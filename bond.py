# bond_yield_curve_explorer.py
import math
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Bond Valuation & Yield Curve Explorer", layout="wide")

# ======================
# Utility: Canada curve
# ======================
def try_fetch_canada_curve() -> Tuple[Dict[float,float], str]:
    """
    Tries to fetch a Canada spot/zero-ish curve.
    1) Primary (preferred): BoC 'zero-coupon' style endpoints (if available).
    2) Fallback (proxy): BoC Valet 'bond_yields' group (benchmark yields; not pure spots).
    Returns (tenor->rate_decimal, label_date). If all fail, returns ({}, "").
    """
    # --- Attempt 1: (placeholder) zero-coupon group (if BoC exposes; may change)
    # If you have a confirmed zero-coupon endpoint, plug it here.
    # For portability, we proceed to fallback proxy immediately.
    try:
        url = "https://www.bankofcanada.ca/valet/observations/group/bond_yields/json?recent=1"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            obs = data["observations"][-1]
            date_label = obs["d"]

            # Map common Gov. of Canada benchmarks (proxy for spots):
            # Keys are (tenor_years, series_id in BoC Valet)
            # These IDs can evolve; adjust if you know exact codes in your environment.
            candidates = [
                (0.5, "V39053"),   # 6M T-bill
                (1.0, "V39054"),   # 1Y
                (2.0, "V39062"),   # 2Y
                (3.0, "V39057"),   # 3Y
                (5.0, "V39058"),   # 5Y
                (7.0, "V39059"),   # 7Y
                (10.0, "V39056"),  # 10Y
                (20.0, "V39060"),  # 20Y
                (30.0, "V39061"),  # 30Y
            ]
            curve = {}
            for t, sid in candidates:
                if sid in obs and "v" in obs[sid]:
                    # Values are in percent; convert to decimal
                    curve[t] = float(obs[sid]["v"]) / 100.0
            # Keep only if we got at least a few points
            if len(curve) >= 4:
                return dict(sorted(curve.items())), date_label
    except Exception:
        pass

    return {}, ""  # give control to manual entry in UI

def interp_linear(curve: Dict[float,float], t: float) -> float:
    xs = np.array(sorted(curve.keys()))
    ys = np.array([curve[k] for k in xs])
    if t <= xs[0]: return float(ys[0])
    if t >= xs[-1]: return float(ys[-1])
    i = np.searchsorted(xs, t)
    x0, x1 = xs[i-1], xs[i]
    y0, y1 = ys[i-1], ys[i]
    w = (t - x0) / (x1 - x0)
    return float((1 - w) * y0 + w * y1)

# ======================
# Bond math (no bumps)
# ======================
def cashflow_times(maturity_years: float, freq: int) -> np.ndarray:
    n = int(round(maturity_years * freq))
    return np.array([(k+1)/freq for k in range(n)], dtype=float)

def ytm_price(face: float, coupon_rate: float, ytm: float, maturity_years: float, freq: int) -> float:
    c = coupon_rate * face / freq
    t = cashflow_times(maturity_years, freq)
    y = ytm / freq
    disc = 1.0 / (1.0 + y) ** (t * freq)
    cf = np.full_like(t, c); cf[-1] += face
    return float(np.sum(cf * disc))

def macaulay_mod_duration_convexity(face: float, coupon_rate: float, ytm: float,
                                    maturity_years: float, freq: int) -> Tuple[float,float,float,float]:
    """
    Returns: (Price, MacaulayDuration_years, ModifiedDuration_years, Convexity_years2)
    Discrete comp formulas; no bumps.
    """
    c = coupon_rate * face / freq
    t = cashflow_times(maturity_years, freq)
    k = (t * freq).astype(int)
    y = ytm / freq
    cf = np.full_like(t, c); cf[-1] += face
    disc = 1.0 / (1.0 + y) ** k

    P = float(np.sum(cf * disc))
    # Macaulay (years)
    D_mac = float(np.sum((k * cf * disc)) / (P * freq))
    # Modified (years)
    D_mod = D_mac / (1.0 + y)
    # Discrete convexity (years^2)
    # C = (1 / (P * m^2)) * sum( CF_k * k*(k+1) / (1+y)^{k+2} )
    numer = np.sum(cf * (k * (k + 1)) / (1.0 + y) ** (k + 2))
    C = float(numer / (P * (freq ** 2)))
    return P, D_mac, D_mod, C

# Curve-based pricing (continuous comp for stability)
def price_from_spot_curve(face: float, coupon_rate: float, maturity_years: float, freq: int,
                          spot_curve: Dict[float,float]) -> Tuple[float, pd.DataFrame]:
    c = coupon_rate * face / freq
    t = cashflow_times(maturity_years, freq)
    cf = np.full_like(t, c); cf[-1] += face
    s = np.array([interp_linear(spot_curve, ti) for ti in t])  # decimal
    df = np.exp(-s * t)                                       # continuous comp DF
    pv = cf * df
    price = float(np.sum(pv))
    table = pd.DataFrame({
        "Time (yrs)": t,
        "Cash Flow": cf,
        "Spot Rate": s,
        "Discount Factor": df,
        "PV": pv
    })
    return price, table

def fisher_weil_measures(face: float, coupon_rate: float, maturity_years: float, freq: int,
                         spot_curve: Dict[float,float]) -> Tuple[float,float,float]:
    """
    Returns: (Price, Fisher–Weil Duration [years], FW Convexity [years^2])
    Using continuous comp spot DFs: DF = exp(-s(t)*t).
    FW Dur = (1/P) * sum( t * CF_t * DF_t )
    FW Convexity ~ (1/P) * sum( t^2 * CF_t * DF_t ) under small parallel shifts of s(t)
    """
    c = coupon_rate * face / freq
    t = cashflow_times(maturity_years, freq)
    cf = np.full_like(t, c); cf[-1] += face
    s = np.array([interp_linear(spot_curve, ti) for ti in t])
    df = np.exp(-s * t)
    pv = cf * df
    P = float(np.sum(pv))
    FW_D = float(np.sum(t * pv) / P)
    FW_C = float(np.sum((t**2) * pv) / P)
    return P, FW_D, FW_C

def ytm_from_price(face, coupon_rate, maturity_years, freq, target_price, tol=1e-12, iters=200):
    lo, hi = -0.99, 2.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        p = ytm_price(face, coupon_rate, mid, maturity_years, freq)
        if abs(p - target_price) < tol:
            return mid
        if p > target_price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

# ======================
# Sidebar: Bond inputs
# ======================
with st.sidebar:
    st.subheader("Bond Inputs")
    face = st.number_input("Face Value", 0.0, 10_000_000.0, 100.0, step=100.0)
    coupon_pct = st.number_input("Coupon Rate (annual %)", 0.0, 100.0, 5.00, step=0.25)
    coupon = coupon_pct / 100.0
    maturity = st.number_input("Maturity (years)", 0.25, 50.0, 10.0, step=0.25)
    freq = st.selectbox("Coupon Frequency", [1,2,4], index=1)

# ======================
# Top banner: Canada spots
# ======================
st.markdown("<h1 style='margin-bottom:0'>🇨🇦 Canada Spot Curve & Bond Explorer</h1>", unsafe_allow_html=True)
st.caption("Live Canada curve (when available) + closed-form duration/convexity. No bumping.")

# Data source selector
st.divider()
st.subheader("Current Canada Spot Curve")
src = st.radio("Source", ["Live fetch (Bank of Canada, proxy)", "Manual input"], horizontal=True)

can_curve = {}
can_date = ""
if src == "Live fetch (Bank of Canada, proxy)":
    can_curve, can_date = try_fetch_canada_curve()
    if not can_curve:
        st.warning("Couldn’t fetch live data right now. Paste spot points below or try again.")
else:
    st.info("Enter spot rates (annual %) by maturity. Example: 0.5:4.2, 2:4.5, 5:4.7, 10:4.8, 30:5.0")
manual = st.text_input("Manual spot map (years:percent, comma-separated)", value="")
if manual.strip():
    pairs = manual.split(",")
    for p in pairs:
        if ":" in p:
            t, r = p.split(":")
            try:
                can_curve[float(t.strip())] = float(r.strip())/100.0
            except:
                pass
can_curve = dict(sorted(can_curve.items()))

# Show current curve summary
if can_curve:
    cols = st.columns(4)
    label = f"As of {can_date}" if can_date else "User-provided"
    cols[0].markdown(f"**Status:** {label}")
    key_tenors = [0.5, 2.0, 5.0, 10.0, 30.0]
    vals = {t: interp_linear(can_curve, t) for t in key_tenors}
    cols[1].metric("2Y", f"{vals[2.0]*100:.2f}%")
    cols[2].metric("10Y", f"{vals[10.0]*100:.2f}%")
    spread_2s10s_bp = (vals[10.0] - vals[2.0]) * 1e4
    cols[3].metric("2s10s Spread", f"{spread_2s10s_bp:.0f} bp")

# Plot Canada curve
if can_curve:
    fig = plt.figure()
    xs = sorted(can_curve.keys())
    ys = [can_curve[x]*100 for x in xs]
    plt.plot(xs, ys, marker="o", label="Canada Spot (current/proxy)")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Spot Rate (%)")
    plt.title("Current Canada Spot Curve")
    plt.legend()
    st.pyplot(fig)

st.divider()

# ======================
# Tabs
# ======================
tab1, tab2 = st.tabs(["Single-Yield Pricing (YTM)", "Curve-Based Pricing (Spots)"])

with tab1:
    st.subheader("Single-Yield Pricing (Closed-form duration/convexity)")
    ytm_pct = st.number_input("Yield to Maturity (annual %, bond-equivalent)", -50.0, 200.0, 4.80, step=0.10)
    ytm = ytm_pct / 100.0

    P, D_mac, D_mod, C = macaulay_mod_duration_convexity(face, coupon, ytm, maturity, freq)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"{P:,.2f}")
    c2.metric("Macaulay Duration", f"{D_mac:.4f} yrs")
    c3.metric("Modified Duration", f"{D_mod:.4f} yrs")
    c4.metric("Convexity", f"{C:.4f} yrs²")

    # Cashflow table
    t = cashflow_times(maturity, freq)
    per = (t*freq).astype(int)
    cpn = coupon * face / freq
    cfs = np.full_like(t, cpn); cfs[-1] += face
    y = ytm / freq
    disc = 1.0 / (1.0 + y) ** per
    pv = cfs * disc
    df = pd.DataFrame({"Time (yrs)": t, "CF": cfs, "DF": disc, "PV": pv})
    st.dataframe(df, use_container_width=True)

with tab2:
    st.subheader("Curve-Based Pricing (Fisher–Weil measures, no bumps)")
    base_choice = st.selectbox("Base Curve Shape (for comparison)", ["Upward (Normal)", "Flat", "Inverted"], index=0)

    def base_curve(shape: str) -> Dict[float,float]:
        keys = np.array([0.5,1,2,3,5,7,10,20,30], dtype=float)
        if shape == "Upward (Normal)":
            rates = np.array([0.035,0.037,0.040,0.042,0.045,0.047,0.048,0.049,0.050])
        elif shape == "Flat":
            rates = np.array([0.040]*len(keys))
        else:
            rates = np.array([0.050,0.049,0.047,0.045,0.043,0.041,0.040,0.038,0.037])
        return dict(zip(keys, rates))

    base = base_curve(base_choice)

    # Plot base vs current Canada
    fig2 = plt.figure()
    xs_b = sorted(base.keys()); ys_b = [base[x]*100 for x in xs_b]
    plt.plot(xs_b, ys_b, label=f"Base: {base_choice}")
    if can_curve:
        xs_c = sorted(can_curve.keys()); ys_c = [can_curve[x]*100 for x in xs_c]
        plt.plot(xs_c, ys_c, marker="o", label="Canada Spot (current/proxy)")
    plt.xlabel("Maturity (years)"); plt.ylabel("Spot Rate (%)")
    plt.title("Base vs Current Canada Curve")
    plt.legend()
    st.pyplot(fig2)

    if not can_curve:
        st.warning("Provide a Canada spot curve above to price off spots.")
    else:
        P_curve, table = price_from_spot_curve(face, coupon, maturity, freq, can_curve)
        P_fw, FW_D, FW_C = fisher_weil_measures(face, coupon, maturity, freq, can_curve)

        c1, c2, c3 = st.columns(3)
        c1.metric("Price (Curve)", f"{P_curve:,.2f}")
        c2.metric("Fisher–Weil Duration", f"{FW_D:.4f} yrs")
        c3.metric("FW Convexity", f"{FW_C:.4f} yrs²")

        # Show implied single-yield from curve price (nice talking point)
        ytm_impl = ytm_from_price(face, coupon, maturity, freq, P_curve) * 100
        st.metric("Implied YTM from Curve Price", f"{ytm_impl:.2f}%")

        st.dataframe(table, use_container_width=True)

st.caption("Notes: YTM tab uses closed-form Macaulay/Modified duration and discrete convexity. Curve tab uses spot discounts with Fisher–Weil measures. Canada curve uses BoC data when available; otherwise enter manually.")

