import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# loda data
df = pd.read_csv("crsp_data.csv")
df.columns = df.columns.str.lower()

# keep cols in questions
keep_cols = ['date', 'ticker', 'ret']
df = df[keep_cols].copy()

# date to datetime
df['date'] = pd.to_datetime(df['date'])

# ret to numeric
df['ret'] = pd.to_numeric(df['ret'], errors='coerce')

# ret to percentage
df['ret'] = df['ret'] * 100

# summary statistics for AAPL
aapl = df.loc[df['ticker']=='AAPL', 'ret'].dropna()
print("AAPL mean:", aapl.mean())
print("AAPL std:", aapl.std(ddof=1))  # sample std
print("AAPL min:", aapl.min())
print("AAPL max:", aapl.max())

# question 2


# --- settings from the assignment ---
alpha = 0.4
deltas = [0.3, 0.1, 0.0, -0.3]
gammas = [0.01, 0.1, 1.0]

# shock grid - (start, end, number)
x = np.linspace(-5, 5, 1000)

def NIC(x, alpha, delta, gamma):
    return (alpha + delta * np.tanh(-gamma * x)) * (x**2)

# --- plot 2x2 grid ---
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
axes = axes.ravel()

for i, delta in enumerate(deltas):
    ax = axes[i]
    for gamma in gammas:
        y = NIC(x, alpha, delta, gamma)
        ax.plot(x, y, label=f"γ={gamma}")
    ax.set_title(f"δ = {delta}")
    ax.set_xlabel("Shock x")
    ax.set_ylabel("NIC(x)")
    ax.grid(True, alpha=0.3)  # alpha=0.3 是透明度，數字越小越淡
    ax.legend(frameon=False)  # frameon=False 代表不要顯示圖例的邊框

# fig.suptitle("News-Impact Curves: α=0.4, μ=0, λ=0, σ²_{t−1}=1", y=1.03)
plt.tight_layout()
fig.savefig("fig_q2_nic_2x2.png", dpi=300, bbox_inches="tight")
fig.savefig("fig_q2_nic_2x2.pdf", bbox_inches="tight")
print("Saved: fig_q2_nic_2x2.(png|pdf)")
