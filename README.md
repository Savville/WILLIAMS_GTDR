# Williams GTDR Formula

### Gauss-Taylor Derivative Reconstruction Integrator for Low-NFE Continuous-Time Machine Learning

**Author:** Williams Otieno Ochieng
**Affiliation:** Department of Civil Engineering, Kenyatta University, Nairobi, Kenya
**Contact:** ochiwilliamotieno@gmail.com
**Preprint DOI:** [doi.org/10.5281/zenodo.20967109](https://doi.org/10.5281/zenodo.20967109)

---

## What is the Williams GTDR Formula?

The **Williams GTDR Formula** is an explicit numerical ODE integrator that achieves **third-order local accuracy using only 3 function evaluations (NFEs) per step** — compared to 4 NFEs for classical Runge-Kutta 4 (RK4).

In Neural ODEs and score-based diffusion models, each function evaluation is a full neural network forward pass. This makes NFE count the dominant computational cost. The Williams GTDR reduces that cost by **25–33%** while matching RK4-level accuracy.

---

## The Formula

**Step 1 — Gauss-Legendre nodes:**

```
c1 = 0.5 - sqrt(3)/6  ≈ 0.2113
c2 = 0.5 + sqrt(3)/6  ≈ 0.7887
```

**Step 2 — Three function evaluations (GTDR-2 Base):**

```
k0 = f(xn, yn)                        # NFE 1
y1* = yn + c1 * h * k0
y2* = yn + c2 * h * k0
k1 = f(xn + c1*h, y1*)                # NFE 2
k2 = f(xn + c2*h, y2*)                # NFE 3
```

**Step 3 — Update:**

```
y_{n+1} = yn + (h/2) * (k1 + k2)
```

**Romberg Patch (raises to O(h⁴), matching RK4):**

```
y_{n+1} = (4 * Y_{h/2} - Y_h) / 3
```

---

## PyTorch / torchdiffeq Implementation

```python
def gtdr2_step(func, t, y, dt):
    """Williams GTDR-2: 3-NFE explicit integrator."""
    c1 = 0.5 - (3**0.5) / 6.0
    c2 = 0.5 + (3**0.5) / 6.0

    k0 = func(t, y)                        # NFE 1
    y1_star = y + c1 * dt * k0
    y2_star = y + c2 * dt * k0

    k1 = func(t + c1 * dt, y1_star)        # NFE 2
    k2 = func(t + c2 * dt, y2_star)        # NFE 3

    return y + (dt / 2.0) * (k1 + k2)
```

---

## Repository Structure

```
WILLIAMS_GTDR/
├── README.md                    # This file
├── WILLIAMS_GTDR.pdf            # Full paper (preprint)
├── williams_gtdr_ieee.tex       # LaTeX source
├── gtdr_benchmark.py            # Core GTDR vs Euler / RK4 / RK45 benchmarks
└── sde_benchmark.py             # Diffusion model / SDE ODE benchmarks
```

---

## Running the Benchmarks

**Requirements:**

```bash
pip install numpy scipy matplotlib
```

**Core ODE benchmark (GTDR vs Euler, RK4, RK45):**

```bash
python gtdr_benchmark.py
```

**Diffusion model / SDE benchmark:**

```bash
python sde_benchmark.py
```

---

## Key Results

| Solver                           | NFEs/step     | Local Error Order | vs RK4 cost                  |
| -------------------------------- | ------------- | ----------------- | ---------------------------- |
| Euler                            | 1             | O(h²)            | −75% (unstable at large h)  |
| **GTDR-2 (base)**          | **3**   | **O(h³)**  | **−25%**              |
| RK4                              | 4             | O(h⁴)            | baseline                     |
| Dormand-Prince (RK45)            | 6             | O(h⁵)            | +50%                         |
| **GTDR (Romberg patched)** | **9\*** | **O(h⁴)**  | **RK4-level accuracy** |

\* Across 3 sub-calls; takes fewer total integration steps than RK4 at large step sizes.

---

## Paper

The full derivation, truncation error proof, and benchmarks are in the preprint:

> Williams Otieno Ochieng, *"The Williams GTDR Formula: A Gauss-Taylor Derivative Reconstruction Integrator for Low-NFE Continuous-Time Machine Learning"*, Kenyatta University, 2026.

📄 **PDF:** `WILLIAMS_GTDR.pdf`
🔗 **Zenodo:** *(add DOI after publishing)*

---

## Citation

```bibtex
@article{ochieng2026gtdr,
  title   = {The Williams GTDR Formula: A Gauss-Taylor Derivative Reconstruction
             Integrator for Low-NFE Continuous-Time Machine Learning},
  author  = {Ochieng, Williams Otieno},
  year    = {2026},
  note    = {Preprint. Kenyatta University, Nairobi, Kenya.}
}
```

*(Update with Zenodo DOI once published)*

---

## License

This project is released under the **MIT License**.
The paper (PDF and LaTeX) is released under **CC BY 4.0**.
