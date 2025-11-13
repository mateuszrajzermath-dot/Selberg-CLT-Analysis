import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro, kstest, anderson, skew, kurtosis, probplot
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib.backends.backend_pdf import PdfPages
import io
import logging

# --- Configuration ---
# Disable non-critical logs from matplotlib to keep the console clean
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# --- Core Functions ---

def theta(t):
    """
    Approximation of the Riemann-Siegel theta function.
    """
    return 0.5 * t * np.log(t / (2 * np.pi)) - 0.5 * t - np.pi / 8

def zeta_riemann_siegel(t):
    """
    Approximates |ζ(1/2+it)| using the Riemann-Siegel formula (main term).
    Uses an approximation for the number of terms N.
    """
    # N = number of terms in the approximation
    N = int(np.sqrt(t / (2 * np.pi)))
    if N <= 0: # Safety check for small t
        return 1.0
        
    n = np.arange(1, N + 1)
    
    # Main sum of the R-S formula
    s = np.sum(n**(-0.5) * np.cos(t * np.log(n) - theta(t)))
    
    # Return the modulus (absolute value)
    return abs(2 * s)

def sample_X_T(T, N_samples=1000):
    """
    Generates X_T(t) samples for a given T and N_samples.
    X_T(t) = log|ζ(1/2+it)| / sqrt(0.5 * log(log(T)))
    """
    # Random t points in the range [T, 2T]
    t_values = np.random.uniform(T, 2 * T, N_samples)
    
    # Calculate log|ζ| for each t value
    # Add 1e-12 (epsilon) to avoid log(0) in case of hitting a Riemann zero
    logabs_values = np.array([np.log(zeta_riemann_siegel(t) + 1e-12) for t in t_values])
    
    # Calculate the Selberg scaling factor
    scaling_factor = np.sqrt(0.5 * np.log(np.log(T)))
    
    X = logabs_values / scaling_factor
    return X, np.var(X) # Return samples AND variance

def kolmogorov_distance(data):
    """
    Calculates the maximum distance (supremum) between the ECDF and the N(0,1) CDF.
    Used for the Kolmogorov-Smirnov test.
    """
    ecdf = ECDF(data)
    x_space = np.linspace(min(data), max(data), 1000)
    # Calculate the absolute difference between the empirical CDF and the theoretical N(0,1) CDF
    return np.max(np.abs(ecdf(x_space) - norm.cdf(x_space)))

# --- Main Parameters ---
T_values = [1e3, 1e5, 1e8, 1e12] # T ranges
N = 2000 # Number of samples for each T

results = []
ks_distances = []
all_samples = {} # Storage for the generated samples

print("Starting Selberg CLT analysis...")

# --- Generating Plots (Figure 1: Histograms and QQ-plots) ---
plt.figure(figsize=(12, 10))

for i, T in enumerate(T_values, 1):
    print(f"Calculating for T = {T:.0e} ...")
    X, variance = sample_X_T(T, N) # Get samples and variance
    all_samples[T] = X # Store samples for PDF report

    # --- Statistical Tests ---
    shapiro_p = shapiro(X)[1]
    ks_p = kstest(X, 'norm')[1]
    anderson_stat = anderson(X, dist='norm').statistic
    
    # Moments: mean, skewness, kurtosis (non-fisher, i.e., 3 is normal)
    m2, m3, m4 = np.mean(X), skew(X), kurtosis(X, fisher=False)
    
    # KS distance
    sup_dist = kolmogorov_distance(X)

    # Append ALL results, including the calculated variance
    results.append((T, shapiro_p, ks_p, anderson_stat, m2, m3, m4, variance))
    ks_distances.append(sup_dist)

    # --- Plot 1: Histogram + KDE + Gaussian Curve ---
    plt.subplot(2, len(T_values), i)
    plt.hist(X, bins=30, density=True, alpha=0.6, color='skyblue', label='Empirical Data (X_T)')
    x_grid = np.linspace(-4, 4, 500)
    plt.plot(x_grid, norm.pdf(x_grid), 'r-', lw=2, label="N(0,1) Distribution")
    plt.title(f"Histogram for T = {T:.0e}")
    plt.xlabel(r"$X_T$")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)

    # --- Plot 2: QQ-plot ---
    plt.subplot(2, len(T_values), i + len(T_values))
    probplot(X, dist="norm", plot=plt)
    plt.title(f"QQ-plot for T={T:.0e}")

plt.suptitle("Empirical Analysis of Selberg's CLT for $\\log|\\zeta(1/2+it)|$", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# --- Moment and KS Distance Analysis (DataFrame) ---

# Creating DataFrame with results
df = pd.DataFrame(results, columns=["T", "p-Shapiro", "p-KS", "Anderson", "Mean", "Skewness", "Kurtosis", "Variance"])
df["KS-distance"] = ks_distances

print("\n--- Table of Statistical Test Results ---")
print(df)

# --- Generating Plots (Figure 2: Convergence) ---
plt.figure(figsize=(10, 5))

# X-axis for convergence plots
loglogT_axis = np.log(np.log(df["T"]))

# Plot 3: Moment Convergence
plt.subplot(1, 2, 1)
plt.plot(loglogT_axis, df["Mean"], 'o-', label='Mean (expected: 0)')
plt.axhline(0, color='gray', linestyle='--')
plt.plot(loglogT_axis, df["Variance"], 'o-', label='Variance (expected: 1)')
plt.axhline(1, color='gray', linestyle='--')
plt.plot(loglogT_axis, df["Skewness"], 'o-', label='Skewness (expected: 0)')
plt.axhline(0, color='gray', linestyle=':')
plt.plot(loglogT_axis, df["Kurtosis"], 'o-', label='Kurtosis (expected: 3)')
plt.axhline(3, color='gray', linestyle='-.')
plt.xlabel(r"$\log\log T$")
plt.legend()
plt.title("Moment Convergence")
plt.grid(alpha=0.3)

# Plot 4: KS distance convergence
plt.subplot(1, 2, 2)
plt.plot(loglogT_axis, df["KS-distance"], 'o-', color='red')
plt.xlabel(r"$\log\log T$")
plt.ylabel("Kolmogorov-Smirnov distance")
plt.title("Rate of convergence (KS)")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# --- Generating Plots (Figure 3: Scaling Comparison) ---
plt.figure(figsize=(10, 4))
T0 = 1e8
# Use the samples we already generated
scaled_data = all_samples[T0] 
# Re-calculate raw data just for this comparison plot
t_values_comp = np.random.uniform(T0, 2 * T0, N)
raw_data = np.array([np.log(zeta_riemann_siegel(t) + 1e-12) for t in t_values_comp])


plt.hist(raw_data, bins=30, alpha=0.6, label='Raw data (unscaled)', density=True)
plt.hist(scaled_data, bins=30, alpha=0.6, label='Scaled data (X_T)', density=True)
x_grid_comp = np.linspace(min(scaled_data), max(scaled_data), 500)
plt.plot(x_grid_comp, norm.pdf(x_grid_comp), 'r-', lw=2, label='N(0,1)')
plt.legend()
plt.title(f"Effect of Selberg scaling (T = {T0:.0e})")
plt.grid(alpha=0.3)
plt.show()

# --- Generating PDF Report ---
print(f"\nGenerating PDF report (selberg_report.pdf)...")

with PdfPages("selberg_report.pdf") as pdf:
    
    # Page 1-4: Histograms + QQ plots for each T
    # Use the samples stored in the 'all_samples' dictionary
    for T, X in all_samples.items(): 
        
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        
        # Histogram + Gauss
        ax[0].hist(X, bins=30, density=True, alpha=0.6, color='skyblue')
        x_grid_pdf = np.linspace(-4, 4, 500)
        ax[0].plot(x_grid_pdf, norm.pdf(x_grid_pdf), 'r-', lw=2)
        ax[0].set_title(f"Histogram of scaled log|ζ(1/2+it)| values")
        ax[0].set_xlabel("X_T(t)")
        ax[0].grid(alpha=0.3)

        # QQ-plot
        probplot(X, dist="norm", plot=ax[1])
        ax[1].set_title(f"QQ-plot (T = {T:.0e})")

        fig.suptitle(f"Empirical analysis of Selberg's CLT (T = {T:.0e})", fontsize=12, fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # Page 5: Convergence of moments and KS tests
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
    
    # moments
    ax2[0].plot(loglogT_axis, df["Mean"], 'o-', label='Mean')
    ax2[0].axhline(0, color='gray', linestyle='--')
    ax2[0].plot(loglogT_axis, df["Variance"], 'o-', label='Variance')
    ax2[0].axhline(1, color='gray', linestyle='--')
    ax2[0].plot(loglogT_axis, df["Skewness"], 'o-', label='Skewness')
    ax2[0].axhline(0, color='gray', linestyle=':')
    ax2[0].plot(loglogT_axis, df["Kurtosis"], 'o-', label='Kurtosis')
    ax2[0].axhline(3, color='gray', linestyle='-.')
    ax2[0].set_xlabel(r"$\log\log T$")
    ax2[0].set_title("Moment Convergence")
    ax2[0].legend()
    ax2[0].grid(alpha=0.3)

    # KS distance
    ax2[1].plot(loglogT_axis, df["KS-distance"], 'o-', color='red')
    ax2[1].set_xlabel(r"$\log\log T$")
    ax2[1].set_ylabel("KS distance")
    ax2[1].set_title("Rate of convergence (KS)")
    ax2[1].grid(alpha=0.3)

    fig2.suptitle("Convergence Analysis", fontsize=12, fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig2)
    plt.close(fig2)
    
    # Page 6: Results table (DataFrame)
    fig3, ax3 = plt.subplots(figsize=(10, 2 + 0.3 * len(df)))
    ax3.axis('off')
    
    # Formatting DataFrame for the report
    df_report = df.round(4)
    table_data = df_report.values
    columns = df_report.columns
    
    table = ax3.table(cellText=table_data, colLabels=columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    fig3.suptitle("Table of Experiment Results", fontsize=12, fontweight="bold")
    pdf.savefig(fig3)
    plt.close(fig3)

    # Page 7: Text summary
    buffer = io.StringIO()
    buffer.write("Summary of Selberg's CLT experiment\n\n")
    for i, row in df.iterrows():
        buffer.write(f"T = {row['T']:.0e}\n")
        buffer.write(f"  Mean: {row['Mean']:.4f} (Expected: 0)\n")
        buffer.write(f"  Variance: {row['Variance']:.4f} (Expected: 1)\n")
        buffer.write(f"  Skewness: {row['Skewness']:.4f} (Expected: 0)\n")
        buffer.write(f"  Kurtosis: {row['Kurtosis']:.4f} (Expected: 3)\n")
        buffer.write(f"  KS-distance: {row['KS-distance']:.4f}\n")
        buffer.write(f"  P-value (Shapiro-Wilk): {row['p-Shapiro']:.4f}\n")
        buffer.write(f"  P-value (KS-Test): {row['p-KS']:.4f}\n")
        buffer.write("-" * 50 + "\n")

    fig4, ax4 = plt.subplots(figsize=(8.5, 11))
    ax4.axis("off")
    ax4.text(0.05, 0.95, buffer.getvalue(), va='top', family='monospace')
    fig4.suptitle("Statistical Summary", fontsize=12, fontweight="bold")
    pdf.savefig(fig4)
    plt.close(fig4)

print(f"PDF report has been generated: selberg_report.pdf")