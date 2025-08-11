import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bond Valuation & Yield Curve Explorer", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
def cashflow_schedule(maturity_years: float, freq: int) -> np.ndarray:
    n = int(round(maturity_years * freq))
    # exact years to cashflows assuming equal periods
    return np.array([(i+1)/freq for i in range(n)], dtype=float)

def price_from_yield(face: float, coupon_rate: float, ytm: float, maturity_years: float, freq: int) -> float:
    c = coupon_rate * face / freq
    t = cashflow_schedule(maturity_years, freq)
    df = 1.0 / (1.0 + ytm / freq) ** (t * freq)
    pv_coupons = np.sum(c * df)
    pv_principal = face * df[-1]
    return float(pv_coupons + pv_principal)

def bump_yield_price(face, coupon_rate, ytm, maturity_years, freq, bump=1e-4):
    p0 = price_from_yield(face, coupon_rate, ytm, maturity_years, freq)
    p_up = price_from_yield(face, coupon_rate, ytm + bump, maturity_years, freq)
    p_dn = price_from_yield(face, coupon_rate, ytm - bump, maturity_years, freq)
    return p0, p_up, p_dn

def effective_measures_from_yield(face, coupon_rate, ytm, maturity_years, freq, bump=1e-4):
    p0, p_up, p_dn = bump_yield_price(face, coupon_rate, ytm, maturity_years, freq, bump=bump)
    dur = (p_dn - p_up) / (2 * p0 * bump)
    conv = (p_dn + p_up - 2 * p0) / (p0 * bump**2)
    return p0, dur, conv

# --- Curve tools (spot curve, continuous comp for stability) ---
def base_curve(curve_type: str) -> Dict[float, float]:
    # simple stylized spot curves in decimal (annual)
    # key tenors in years
    key = np.array([0.5,1,2,3,5,7,10,20,30], dtype=float)
    if curve_type == "Upward (Normal)":
        rates = np.array([0.035,0.037,0.040,0.042,0.045,0.047,0.048,0.049,0.050])
    elif curve_type == "Flat":
        rates = np.array([0.04]*len(key))
    elif curve_type == "Inverted":
        rates = np.array([0.050,0.049,0.047,0.045,0.043,0.041,0.040,0.038,0.037])
    else:
        rates = np.array([0.04]*len(key))
    return dict(zip(key, rates))

def apply_shocks(curve: Dict[float, float], parallel_bp: float, twist_bp: float) -> Dict[float, float]:
    # twist: -at short end, +at long end, 0 near 5Y, linear in between
    tenors = np.array(sorted(curve.keys()))
    rates = np.array([curve[k] for k in tenors])
    parallel = parallel_bp / 1e4

    # piecewise linear twist weight: -1 at 0.5Y → 0 at 5Y → +1 at 30Y
    def twist_weight(t):
        if t <= 5:
            return (t - 0.5) / (5 - 0.5) * (0 - (-1)) + (-1)  # -1 to 0
        else:
            return (t - 5) / (30 - 5) * (1 - 0) + 0           # 0 to +1

    weights = np.array([twist_weight(t) for t in tenors])
    twist = (twist_bp / 1e4) * weights
    shocked = rates + parallel + twist
    return dict(zip(tenors, shocked))

def interp_spot(curve: Dict[float, float], t: float) -> float:
    tenors = np.array(sorted(curve.keys()))
    spots = np.array([curve[k] for k in tenors])
    if t <= tenors[0]:  # flat extrapolate short end
        return float(spots[0])
    if t >= tenors[-1]: # flat extrapolate long end
        return float(spots[-1])
    i = np.searchsorted(tenors, t)
    t0, t1 = tenors[i-1], tenors[i]
    r0, r1 = spots[i-1], spots[i]
    w = (t - t0) / (t1 - t0)
    return float((1 - w) * r0 + w * r1)

def price_from_curve(face: float, coupon_rate: float, maturity_years: float, freq: int, curve: Dict[float,float]) -> Tuple[float, pd.DataFrame]:
    c = coupon_rate * face / freq
    t = cashflow_schedule(maturity_years, freq)
    cashflows = np.full_like(t, c)
    cashflows[-1] += face

    # continuous comp discounting: DF = exp(-s(t)*t)
    spots = np.array([interp_spot(curve, ti) for ti in t])
    dfs = np.exp(-spots * t)
    pv = cashflows * dfs
    price = float(np.sum(pv))

    df = pd.DataFrame({
        "Time (yrs)": t,
        "Cash Flow": cashflows,
        "Spot Rate": spots,
        "Discount Factor": dfs,
        "PV": pv
    })
    return price, df

def effective_measures_from_curve(face, coupon_rate, maturity_years, freq, curve, bump=1e-4):
    # bump the entire curve in parallel by +/- bump
    tenors = list(curve.keys())
    base_rates = np.array([curve[k] for k in tenors])
    def with_bump(delta):
        bumped = {k: r + delta for k, r in zip(tenors, base_rates)}
        p, _ = price_from_curve(face, coupon_rate, maturity_years, freq, bumped)
        return p
    p0, _ = price_from_curve(face, coupon_rate, maturity_years, freq, curve)
    p_up = with_bump(+bump)
    p_dn = with_bump(-bump)
    dur = (p_dn - p_up) / (2 * p0 * bump)
    conv = (p_dn + p_up - 2 * p0) / (p0 * bump**2)
    return p0, dur, conv

# ---------------------------
# UI
# ---------------------------
st.markdown("""
<h1 style="margin-bottom:0">💵 Bond Valuation & Yield Curve Explorer</h1>
<p style="color:#6b7280;margin-top:4px">CFA L2-style pricing, duration, convexity, and curve shocks — all interactive.</p>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Bond Inputs")
    face = st.number_input("Face Value", 0.0, 10_000_000.0, 100.0, step=100.0)
    coupon_rate = st.number_input("Coupon Rate (annual %)", 0.0, 100.0, 5.0, step=0.25) / 100.0
    maturity_years = st.number_input("Maturity (years)", 0.25, 50.0, 10.0, step=0.25)
    freq = st.selectbox("Coupon Frequency", [1,2,4], index=1)
    st.divider()
    st.caption("Effective measures use ±1 bp by default.")

tab1, tab2 = st.tabs(["Single-Yield Pricing", "Curve-Based Pricing"])

with tab1:
    st.subheader("Single-Yield (YTM) Pricing")
    ytm_pct = st.number_input("Yield to Maturity (annual %, bond-equivalent)", -50.0, 200.0, 4.8, step=0.1)
    bump_bp = st.slider("Bump (bp) for Duration/Convexity", 0.5, 10.0, 1.0, step=0.5)
    ytm = ytm_pct / 100.0
    bump = bump_bp / 1e4

    p0, dur, conv = effective_measures_from_yield(face, coupon_rate, ytm, maturity_years, freq, bump=bump)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"{p0:,.2f}")
    c2.metric("YTM", f"{ytm_pct:.2f}%")
    c3.metric("Eff. Duration", f"{dur:.4f}")
    c4.metric("Eff. Convexity", f"{conv:.2f}")

    # Cashflow table (using YTM discounting for display)
    t = cashflow_schedule(maturity_years, freq)
    c = coupon_rate * face / freq
    cashflows = np.full_like(t, c); cashflows[-1] += face
    dfs = 1.0 / (1.0 + ytm / freq) ** (t * freq)
    pv = cashflows * dfs
    df = pd.DataFrame({"Time (yrs)": t, "Cash Flow": cashflows, "DF (YTM)", dfs.name if hasattr(dfs, "name") else "Discount Factor": dfs, "PV": pv})
    st.dataframe(df, use_container_width=True)

with tab2:
    st.subheader("Curve-Based Pricing (Spot Discounts)")
    curve_type = st.selectbox("Base Curve", ["Upward (Normal)", "Flat", "Inverted"], index=0)
    parallel_bp = st.slider("Parallel Shift (bp)", -300, 300, 0, step=5)
    twist_bp = st.slider("Twist / Steepen-Flatten (bp)", -200, 200, 0, step=5,
                         help="Negative = steepen (lower short, higher long). Positive = flatten (higher short, lower long).")

    base = base_curve(curve_type)
    shocked = apply_shocks(base, parallel_bp, twist_bp)

    # Plot curve
    fig = plt.figure()
    ten = sorted(base.keys())
    plt.plot(ten, [base[k]*100 for k in ten], label="Base")
    plt.plot(ten, [shocked[k]*100 for k in ten], label="Shocked")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Spot Rate (%)")
    plt.title("Spot Curve")
    plt.legend()
    st.pyplot(fig)

    price_curve, table = price_from_curve(face, coupon_rate, maturity_years, freq, shocked)
    p0_c, dur_c, conv_c = effective_measures_from_curve(face, coupon_rate, maturity_years, freq, shocked, bump=1e-4)

    c1, c2, c3 = st.columns(3)
    c1.metric("Price (Curve)", f"{price_curve:,.2f}")
    c2.metric("Eff. Duration (Curve)", f"{dur_c:.4f}")
    c3.metric("Eff. Convexity (Curve)", f"{conv_c:.2f}")

    st.dataframe(table, use_container_width=True)

st.caption("Notes: YTM pricing uses bond-equivalent compounding. Curve pricing uses continuous-comp spot discounting for stability. Duration/convexity are effective (1 bp).")
