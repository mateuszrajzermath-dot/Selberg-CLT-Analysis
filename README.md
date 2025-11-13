# Empirical Analysis of Selberg's Central Limit Theorem

This repository contains the Python code used for the analysis presented in the talk **"Empirical Analysis of Selberg’s Central Limit Theorem for the Riemann Zeta Function"**.

This project provides a computational verification of the Selberg Central Limit Theorem, which describes the statistical distribution of the logarithm of the Riemann Zeta function $\log|\zeta(\frac{1}{2}+it)|$ on the critical line.

---

### Core Concept

Selberg's theorem (1946) states that as $T \to \infty$, the random variable $X_T(t)$—defined as:

$$
X_T(t) = \frac{\log|\zeta(\frac{1}{2}+it)|}{\sqrt{\frac{1}{2} \log\log T}}
$$

(where $t$ is chosen uniformly at random from $[T, 2T]$) converges in distribution to a standard normal (Gaussian) distribution $N(0, 1)$.

This script tests this convergence by:

1.  Sampling $\log|\zeta(\frac{1}{2}+it)|$ at high values of $T$ (up to $10^{12}$).
2.  Applying the Selberg scaling factor $\sqrt{\frac{1}{2} \log\log T}$.
3.  Performing statistical analysis (moments, KS-tests, Shapiro-Wilk) to check for normality.
4.  Visualizing the results (Histograms, QQ-plots) to show convergence to $N(0, 1)$.

---

### Code Structure

The main script `selberg_clt_analysis.py` performs the following actions:

1.  **`zeta_riemann_siegel(t)`**: Approximates $|\zeta(\frac{1}{2}+it)|$ using the main term of the Riemann-Siegel formula. This is a computationally efficient method for large $t$.
2.  **`sample_X_T(T, N_samples)`**: Generates $N$ scaled samples of the log-zeta function in the range $[T, 2T]$.
3.  **Statistical Analysis**: Calculates mean, skewness, kurtosis, and performs Shapiro-Wilk, Kolmogorov-Smirnov, and Anderson-Darling tests for normality.
4.  **Visualization**: Generates histograms and QQ-plots comparing the empirical data to the $N(0, 1)$ distribution for increasing $T$.
5.  **Report Generation**: Automatically generates a multi-page PDF report (`selberg_report.pdf`) summarizing all findings, plots, and statistical tables.

---
  ### Core Reference

The specific formulation and scaling factor used in this analysis are based on the modern treatment of the theorem as presented in:

M. Radziwiłł and K. Soundararajan. “Selberg’s central limit theorem for $\log |\zeta(\frac{1}{2} + it)|$”. In: L’Enseignement Mathematique 63.2 (2017), pp. 1–19.
### How to Run

The script is built with standard Python libraries.

### Dependencies

You must have the following libraries installed:
* `numpy`
* `matplotlib`
* `scipy`
* `pandas`
* `statsmodels`

You can install them via pip:
```bash
pip install numpy matplotlib scipy pandas statsmodels
```
###  Execution

Simply run the script from your terminal:
bash
python selberg_clt_analysis.py


The script will print the statistical results to the console, display the summary plots, and save a detailed report named selberg_report.pdf in the same directory.




### License
This project is licensed under the MIT License.
You are free to use, modify, and distribute this code, provided you credit the original author. This is intended as an educational and research tool.

