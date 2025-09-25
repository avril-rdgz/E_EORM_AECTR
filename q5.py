# --- Question 5 --- 
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

# Load data
DF = pd.read_csv("crsp_data.csv")
DF.columns = DF.columns.str.lower()
DF["date"] = pd.to_datetime(DF["date"])
DF = DF.sort_values(["ticker","date"])

# Residual Diagnostics Helper
def ljung_box(resid, lags=20):
    lb1 = acorr_ljungbox(resid, lags=lags, return_df=True)
    lb2 = acorr_ljungbox(resid**2, lags=lags, return_df=True)
    return (
        lb1["lb_stat"].iloc[-1], lb1["lb_pvalue"].iloc[-1],
        lb2["lb_stat"].iloc[-1], lb2["lb_pvalue"].iloc[-1]
    )

def arch_lm(resid, L=10):
    z2 = resid**2
    y = z2[L:]
    X = np.column_stack([z2[L-i-1:-i-1] for i in range(L)])
    X = np.column_stack([np.ones(len(X)), X])  # 常數
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_tot = ((y - y.mean())**2).sum()
    ss_res = ((y - yhat)**2).sum()
    R2 = 0.0 if ss_tot <= 0 else 1 - ss_res/ss_tot
    T = len(y)
    stat = T * R2
    p = 1 - stats.chi2.cdf(stat, df=L)
    return stat, p

def jarque_bera(resid):
    JB, p = stats.jarque_bera(resid)
    return JB, p

def persistence_half_life(alpha, beta):
    phi = alpha + beta
    hl = np.inf if (phi <= 0 or phi >= 1) else np.log(0.5)/np.log(phi)
    return phi, hl

# sigma_t^2 & Standardized Residuals ----------
def build_sigma2_series(x, sigma1_sq, model, params):
    """
    model: 'GARCH' | 'GARCH-M' | 'GARCH-M-L'
    params keys:
      mu, omega, alpha, beta, [lam], [delta], [gamma]
    """
    n = len(x)
    sigma2 = np.empty(n, dtype=float)
    sigma2[0] = max(sigma1_sq, 1e-10)

    mu    = float(params.get("mu", 0.0))
    lam   = float(params.get("lam", 0.0))
    omega = float(params["omega"])
    alpha = float(params["alpha"])
    beta  = float(params["beta"])
    delta = float(params.get("delta", 0.0))
    gamma = float(params.get("gamma", 0.0))

    for t in range(1, n):
        lever = 0.0
        if model == "GARCH-M-L":
            lever = delta * x[t-1] + gamma * (1.0 if x[t-1] >= 0 else -1.0) * x[t-1]
        sigma2[t] = omega + alpha*(x[t-1]**2) + beta*sigma2[t-1] + lever
        if sigma2[t] <= 0:
            sigma2[t] = 1e-10  

    sigma = np.sqrt(sigma2)
    z = (x - mu - lam*sigma2) / sigma
    return sigma2, z

# ---------- main ----------
def run_q5_for_ticker(df, ticker, n_obs=2500, lb_lags=20, lm_lags=10,
                      model="GARCH-M-L", params=None,
                      assume_params_in_percent=False): # False = percentage
    x = df[df["ticker"]==ticker]["ret"].dropna().values[:n_obs]

    # initial: first 50 (/n）
    init = x[:50]
    sigma1_sq = ((init - init.mean())**2).mean()

    if params is None:
        raise ValueError("Need params（according to question 4）")

    p = params.copy()
    if assume_params_in_percent:
        for key in ["mu","lam","delta","gamma"]:
            if key in p:
                p[key] = p[key] / 100.0

    sigma2_hat, z = build_sigma2_series(x, sigma1_sq, model, p)

    # test
    Qz, pz, Qz2, pz2 = ljung_box(z, lags=lb_lags)
    LM, pLM = arch_lm(z, L=lm_lags)
    JB, pJB = jarque_bera(z)
    phi, HL = persistence_half_life(p["alpha"], p["beta"])

    out = {
        "Ticker": ticker,
        "Model": model,
        "LB(z)_stat": Qz,   "LB(z)_p": pz,
        "LB(z2)_stat": Qz2, "LB(z2)_p": pz2,
        "ARCH-LM_stat": LM, "ARCH-LM_p": pLM,
        "JB_stat": JB,      "JB_p": pJB,
        "alpha+beta": phi,  "Half-life_days": HL
    }
    return out, sigma2_hat, z

# ---------- The optimal model parameters from Q4 ----------
BEST = {
    "PFE":  {"model":"GARCH-M-L", "params":{"mu":-0.045, "lam":0.122, "omega":0.000, "alpha":0.044, "beta":0.920, "delta":0.027, "gamma":17.124}},
    "JNJ":  {"model":"GARCH-M-L", "params":{"mu":0.033,  "lam":0.063, "omega":0.003, "alpha":0.026, "beta":0.916, "delta":0.017, "gamma":1.229}},
    "MRK":  {"model":"GARCH-M",   "params":{"mu":0.077,  "lam":0.065, "omega":0.015, "alpha":0.040, "beta":0.904}},
    "AAPL": {"model":"GARCH-M-L", "params":{"mu":0.096,  "lam":0.027, "omega":0.023, "alpha":0.069, "beta":0.911, "delta":0.061, "gamma":10.474}},
}

ASSUME_PCT = False

# ---------- Batch run for four stocks, export results to CSV ----------
rows = []
STORE = {}  # # Store each stock’s (sigma², z) for Q6 plotting
for tkr, spec in BEST.items():
    out, s2, z = run_q5_for_ticker(
        DF, tkr, n_obs=2500, lb_lags=20, lm_lags=10,
        model=spec["model"], params=spec["params"],
        assume_params_in_percent=ASSUME_PCT
    )
    rows.append(out)
    STORE[tkr] = {"sigma2": s2, "z": z}

diag_df = pd.DataFrame(rows)
diag_df_rounded = diag_df.copy()
num_cols = [c for c in diag_df.columns if c not in ["Ticker","Model"]]
diag_df_rounded[num_cols] = diag_df_rounded[num_cols].astype(float).round(4)
print(diag_df_rounded)

diag_df_rounded.to_csv("q5_diagnostics.csv", index=False)
print("Saved: q5_diagnostics.csv")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

RETURNS_CSV = "crsp_data.csv"

# 2) In-sample cutoff (after the first 2,500 obs in the assignment).
IN_SAMPLE_CUTOFF_DATE = "2021-01-04"

# 3) Q4 parameter estimates
#    Put None for irrelevant fields (e.g., lam for GARCH is 0; delta,gamma for models w/o leverage are 0).
#    Stock tickers MUST be 'PFE','JNJ','MRK' (AAPL not required in Q5).
PARAMS = {
    "PFE": {
        "GARCH":     dict(mu=0.059, lam=0.0,   omega=0.000, alpha=0.043, beta=0.915, delta=0.0,   gamma=0.0,   nu=4.659),
        "GARCH-M":   dict(mu=0.019, lam=0.060, omega=0.000, alpha=0.042, beta=0.917, delta=0.0,   gamma=0.0,   nu=4.662),
        "GARCH-M-L": dict(mu=-0.045,lam=0.122, omega=0.000, alpha=0.044, beta=0.920, delta=0.027, gamma=17.124,nu=4.903),
    },
    "JNJ": {
        "GARCH":     dict(mu=0.067, lam=0.0,   omega=0.002, alpha=0.054, beta=0.904, delta=0.0,   gamma=0.0,   nu=9.941),
        "GARCH-M":   dict(mu=0.058, lam=0.032, omega=0.000, alpha=0.025, beta=0.919, delta=0.0,   gamma=0.0,   nu=4.387),
        "GARCH-M-L": dict(mu=0.033, lam=0.063, omega=0.003, alpha=0.026, beta=0.916, delta=0.017, gamma=1.029, nu=4.557),
    },
    "MRK": {
        "GARCH":     dict(mu=0.078, lam=0.0,   omega=0.078, alpha=0.043, beta=0.832, delta=0.0,   gamma=0.0,   nu=4.763),
        "GARCH-M":   dict(mu=0.077, lam=-0.015,omega=0.015, alpha=0.040, beta=0.904, delta=0.0,   gamma=0.0,   nu=4.502),
        "GARCH-M-L": dict(mu=0.094, lam=0.065, omega=0.044, alpha=0.076, beta=0.867, delta=0.076, gamma=0.121, nu=9.977),
    },
}

# 4) X-range (in percent) for news-impact curves (assignment example uses ~[-5,5])
X_MIN, X_MAX, X_N = -5.0, 5.0, 501

# ========================= MODEL FUNCTIONS =========================

def news_impact_sigma2(x_grid, p):
    """
    News-impact curve for model (1), holding sigma_{t-1}^2 = 1.
    sigma_t^2 = omega + (alpha + delta * tanh(-gamma * x_{t-1}))
                * ((x_{t-1} - mu - lam * sigma_{t-1}^2) / sigma_{t-1})^2
                + beta * sigma_{t-1}^2
    """
    mu    = float(p.get("mu",    0.0) or 0.0)
    lam   = float(p.get("lam",   0.0) or 0.0)
    omega = float(p.get("omega", 0.0) or 0.0)
    alpha = float(p.get("alpha", 0.0) or 0.0)
    beta  = float(p.get("beta",  0.0) or 0.0)
    delta = float(p.get("delta", 0.0) or 0.0)
    gamma = float(p.get("gamma", 0.0) or 0.0)

    sig2_prev = 1.0
    sig_prev  = 1.0
    inner = (x_grid - mu - lam*sig2_prev) / sig_prev
    mult  = alpha + delta * np.tanh(-gamma * x_grid)
    sig2  = omega + mult * (inner**2) + beta*sig2_prev
    return sig2

def filter_sigma2(x, p):
    """
    Filter conditional variances sigma_t^2 over the full sample using model (1).
    Per assignment: sigma_1^2 = population variance of first 50 returns (divide by 50, not 49).
    x must be in percent units.
    """
    mu    = float(p.get("mu",    0.0) or 0.0)
    lam   = float(p.get("lam",   0.0) or 0.0)
    omega = float(p.get("omega", 0.0) or 0.0)
    alpha = float(p.get("alpha", 0.0) or 0.0)
    beta  = float(p.get("beta",  0.0) or 0.0)
    delta = float(p.get("delta", 0.0) or 0.0)
    gamma = float(p.get("gamma", 0.0) or 0.0)

    x = np.asarray(x, dtype=float)
    T = len(x)
    if T < 60:
        raise ValueError("Not enough observations to set sigma_1^2 from first 50 returns.")

    # sigma_1^2: population variance of first 50 returns
    x50 = x[:50]
    sig2 = np.empty(T)
    sig2[0] = np.mean((x50 - x50.mean())**2)
    sig = np.sqrt(max(sig2[0], 1e-12))

    for t in range(1, T):
        inner = (x[t-1] - mu - lam*sig2[t-1]) / (sig if sig > 0 else 1e-12)
        mult  = alpha + delta * np.tanh(-gamma * x[t-1])
        sig2[t] = omega + mult * (inner**2) + beta * sig2[t-1]
        sig = np.sqrt(max(sig2[t], 1e-12))
    return sig2

# Load data
df = pd.read_csv(RETURNS_CSV)
df.columns = [c.lower() for c in df.columns]
required = {"date", "ticker", "ret"}
if not required.issubset(df.columns):
    raise ValueError(f"Returns CSV must contain columns: {required}")
df["date"] = pd.to_datetime(df["date"])
df["ticker"] = df["ticker"].str.upper()

# Keep only the three stocks required by Q5 (rows in the panel)
stocks = ["PFE", "JNJ", "MRK"]

# Plot 3×2 panel
x_grid = np.linspace(X_MIN, X_MAX, X_N)
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10), constrained_layout=True)

for i, s in enumerate(stocks):
    sub = df[df["ticker"] == s].sort_values("date")
    if sub.empty:
        raise ValueError(f"No rows found for ticker {s} in {RETURNS_CSV}.")
    
    from matplotlib.ticker import AutoMinorLocator

# ---------- LEFT: news-impact curves ----------
    axL = axes[i, 0]
    
    styles = {
        "GARCH":     dict(color="#E68400", linestyle="-", lw=1.8),  # orange
        "GARCH-M":   dict(color="#2ca02c", linestyle="-", lw=1.8),  # green
        "GARCH-M-L": dict(color="#1f77b4", linestyle="-", lw=1.8),  # blue
    }
    
    for model in ["GARCH", "GARCH-M", "GARCH-M-L"]:
        p = PARAMS[s][model]
        y = news_impact_sigma2(x_grid, p)
        axL.plot(x_grid, y, label=model.replace("-", "–"), **styles[model])
    
    axL.set_title(f"News impact curve for {s}", fontweight="bold")
    axL.set_xlabel(r"$x_{t-1}$", fontweight="bold")
    axL.set_ylabel("News impact", rotation=90, labelpad=8, fontweight="bold")
    
    # Fix X & Free Y + padding(0.2)
    axL.set_xlim(-5.2, 5.2)
    axL.relim(); axL.autoscale()
    ymin, ymax = axL.get_ylim()
    axL.set_ylim(ymin - 0.2, ymax + 0.2)
    
    axL.set_xticks(np.linspace(-5, 5, 5))
    axL.xaxis.set_minor_locator(AutoMinorLocator(2))
    axL.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    axL.set_facecolor("#E5E5E5")
    axL.grid(True, which="major", axis="both", color="white", linewidth=1.2)
    axL.grid(True, which="minor", axis="both", color="white", linewidth=0.6, alpha=0.7)
    axL.tick_params(direction="out", length=5, width=1)
    
    axL.legend(frameon=True, loc="upper right")

# ---------- RIGHT: filtered volatilities (GARCH-M-L) ----------
    axR = axes[i, 1]
    p_ml = PARAMS[s]["GARCH-M-L"]
    
    dates    = sub["date"].values
    rets_raw = sub["ret"].values
    rets_pct = rets_raw * 100.0 if np.nanmedian(np.abs(rets_raw)) < 2.0 else rets_raw.copy()
    sig2     = filter_sigma2(rets_pct, p_ml)
    
    axR.set_facecolor("#E5E5E5")
    
    # Returns: grey
    axR.fill_between(dates, 0.0, rets_pct, color="#636363", alpha=0.6, label="Returns (%)", linewidth=0)
    axR.plot(dates, rets_pct, color="#636363", alpha=0.6, lw=0.8)
    
    # σ²: blue line
    axR.plot(dates, sig2, lw=1.8, alpha=0.95, color="#1f77b4", label=r"GARCH–M–L ($\sigma_t^2$)")
    
    # cutoff 
    cutoff = pd.to_datetime(IN_SAMPLE_CUTOFF_DATE)
    axR.axvline(cutoff, linestyle="--", linewidth=1.2, color="0.2")
    
    axR.set_title(f"Filtered volatilities for {s}", fontweight="bold")
    
    # ylim
    ylim = max(np.nanmax(np.abs(rets_pct)), np.nanmax(sig2)) * 1.10
    axR.set_ylim(-ylim, ylim)
    axR.yaxis.set_ticks_position("left")
    
    # year: every 5 years
    axR.xaxis.set_major_locator(mdates.YearLocator(base=5))
    axR.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axR.xaxis.set_minor_locator(mdates.YearLocator())
    
    axR.grid(True, which="major", color="white", linewidth=1.2)
    axR.grid(True, which="minor", color="white", linewidth=0.6, alpha=0.7)
    
    axR.legend(frameon=False, loc="upper right")


# fig.suptitle("Q5: News-impact curves and filtered volatilities", y=1.02)

# Save & show
fig.savefig("fig_q5_panel.pdf", bbox_inches="tight")
fig.savefig("fig_q5_panel.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved: fig_q5_panel.pdf and fig_q5_panel.png")
