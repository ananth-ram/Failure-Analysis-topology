# Failure Analysis Report: Topological Phase Prediction from Materials Descriptors

## 1. Objective

The goal of the project was to construct a machine learning model capable of predicting **topological or electronic phases of materials** (metal / insulator / semimetal or related proxies) using a dataset derived from a large materials database (~20k–300k crystal structures).

The workflow consisted of:

- Converting raw JSON structural data into an ML-ready dataset
- Extracting physics-informed descriptors (atomic numbers, electronegativity range, SOC proxy, symmetry indicators, etc.)
- Training a supervised classifier (XGBoost)
- Evaluating performance on phase classification tasks

---

## 2. Physical Background

### 2.1 What determines electronic/topological phases?

Electronic phases in crystalline materials are fundamentally governed by:

- **Band structure topology**
- **Spin–orbit coupling (SOC) strength**
- **Crystal symmetry constraints**
- **Orbital hybridization and band inversion**
- **Berry curvature distribution in momentum space**

In particular:

### Metal vs Insulator
- Controlled mainly by:
  - band gap
  - orbital overlap
  - electron filling

### Topological phases (e.g., topological insulators, semimetals)
- Require:
  - band inversion at high-symmetry points
  - non-trivial Berry phase
  - symmetry-protected degeneracies
  - strong SOC effects

---

## 3. Machine Learning Formulation

The problem was formulated as:

\[
X_{\text{structure descriptors}} \rightarrow y_{\text{phase label}}
\]

Where features included:
- Atomic number statistics (Z_mean, Z_max, Z_var)
- SOC proxy (∝ ⟨Z⁴⟩)
- Electronegativity range
- Density, volume, number of sites
- Space group number
- Magnetic moment

Classifier used:
- Gradient boosted decision trees  
  (:contentReference[oaicite:0]{index=0})

---

## 4. Observed Results

### Key observations across experiments:

- Overall accuracy: high (~0.95–0.99 in some runs)
- ROC-AUC: ranged from ~0.65 to ~0.97 depending on label definition
- Severe class imbalance:
  - Majority class (metal / trivial phase): ~90–99%
  - Minority class (semimetal / nontrivial phase): ~1–10%

### Example failure pattern:

- Precision for minority class: very low (0.04–0.12)
- Recall for minority class: extremely low (0.08–0.22)
- Confusion matrix dominated by false negatives

---

## 5. Core Failure Mechanisms

### 5.1 Class imbalance collapse

The dataset is heavily skewed:

- Model learns to minimize loss by predicting majority class
- Minority phase (often physically most interesting) is underrepresented

Even with `scale_pos_weight`, the imbalance is too extreme for reliable boundary learning.

---

### 5.2 Feature space is not topology-aware

The descriptors used are:

- statistical (Z_mean, EN_range)
- scalar (density, volume)
- symmetry index (space_group number)
- heuristic SOC proxy (Z⁴ scaling)

### Critical limitation:

These features **do not encode quantum band topology explicitly**.

Missing:

- band structure eigenvalues
- orbital-resolved information
- k-space band crossings
- Berry curvature
- symmetry eigenvalues at TRIM points

Thus the model is learning:

> “chemical similarity patterns”  
rather than  
> “topological invariants”

---

### 5.3 Representation mismatch (key physics issue)

Topological phases are not determined by local scalar descriptors alone.

They are defined in **reciprocal space (k-space)**, while the model operates in:

> real-space averaged feature space

This causes a fundamental mismatch:
- topology = global wavefunction property
- ML input = local statistical descriptors

---

### 5.4 Label noise / proxy limitation

In many cases:
- “is_metal” or “semimetal” labels are derived from band gap thresholds
- or database heuristics rather than rigorous topological classification

This introduces:
- ambiguous boundaries
- physically inconsistent labels
- noisy supervision signal

---

## 6. Why the model appears “good” sometimes (illusion of performance)

High accuracy (~0.99) is misleading because:

- dominated by majority class prediction
- ROC-AUC inflated by separability of trivial cases
- model learns dataset bias rather than physics

This is a classic **imbalanced classification illusion problem**.

---

## 7. Physical Interpretation of Failure

The failure indicates:

> The chosen descriptor space does not span the manifold of topological quantum phases.

In condensed matter terms:

- You are attempting to learn a **non-local quantum invariant**
- using **local classical descriptors**

This violates representational completeness.

---

## 8. Correct Direction Forward

### 8.1 Improve feature space (intermediate fix)

Add physics-informed descriptors:

- symmetry indicators (irreducible representations)
- band filling constraints
- orbital character proxies (s, p, d occupancy)
- more refined SOC estimators
- graph-based local environments

---

### 8.2 Move to graph representation (major upgrade)

Replace feature vectors with crystal graphs:

- nodes = atoms
- edges = bonds (cutoff-based)
- message passing captures local quantum environment

This is implemented using:

:contentReference[oaicite:1]{index=1}

---

### 8.3 Why graph models solve the issue

Graph models naturally encode:

- symmetry indirectly
- local bonding geometry
- periodic atomic connectivity
- non-linear orbital interactions

This makes them much closer to the **Hamiltonian representation of solids**.

---

## 9. Final Conclusion

The failure is not a model failure — it is a **representation mismatch problem**:

### Root cause hierarchy:

1. ❌ Incomplete physical feature space  
2. ❌ Strong class imbalance  
3. ❌ Proxy-based labeling of phases  
4. ❌ Lack of k-space/topological information  
5. ❌ Classical ML model limitation for quantum topology  

---

## 10. Key Insight

> Predicting topological phases requires learning the *structure of the Hamiltonian*, not just statistical properties of atomic composition.

This shifts the problem from:

- classical tabular ML  
to
- graph-based quantum representation learning

---

## 11. Recommended next step

Move from:

- descriptor-based XGBoost model  
→ to  
- graph neural network-based electronic structure learning pipeline
