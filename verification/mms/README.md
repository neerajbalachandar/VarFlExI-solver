# Method of Manufactured Solutions (MMS) Verification Framework

**VarFlExI: Variational Flexible Interface Aeroelastic Solver**  
*Verification framework for publication in Computer Methods in Applied Mechanics and Engineering (CMAME).*

---

## 1. Directory Structure

```text
verification/mms/
    mms_problem.py          # Single mathematical definition of the manufactured problem
    mms_driver.py           # Single Python execution and analysis driver
    mms_rvpm.jl             # Julia-side particle-level rVPM MMS integration
    mms_parameters.yaml     # Single source of all MMS-specific experimental parameters
    README.md               # This document
```

---

## 2. Mathematical Scope and Guarantees

### 2.1 Independence Guarantee

The fundamental MMS principle requires that the manufactured source $S^*$ is derived from the governing differential operator $\mathcal{L}$ applied analytically to an independently prescribed target solution $q^*$:

$$S^* = \mathcal{L}(q^*)$$

**Strict independence guarantees implemented here:**
- The structural manufactured source is derived symbolically via SymPy from the nonlinear Reissner–Mindlin equations in `mms_problem.py`. It is assembled into the FEniCS weak form as an independent right-hand side `inner(f_sym, v) * dx`. It **never** evaluates `model.mass_form()`, `model.damping_form()`, or `model.internal_work()` to construct the source.
- The particle manufactured source $b_p^* = \dot{y}_p^* - \mathcal{G}_p^{\mathrm{independent}}(y_p^*)$ uses an independent direct-sum $O(N^2)$ Biot–Savart velocity and a finite-difference Jacobian for stretching in `mms_rvpm.jl`. It **never** evaluates `vpm.update_particle_states` on a shadow particle field.
- The interface force transfer MMS uses exact Gauss–Legendre quadrature of the analytical traction against FEM test functions. It **never** uses the production CRM transfer matrix to generate the target load.

### 2.2 Discrete-Model vs. Continuum Verification

> [!WARNING]
> **Scope Distinction:**
> The fluid-side MMS in `mms_rvpm.jl` is explicitly a **particle-level discrete-model MMS**. It verifies that the production RK3 time integrator in FLOWVPM correctly advances particle positions $X_p$, circulations $\boldsymbol{\Gamma}_p$, and core radii $\sigma_p$ under a known source.
>
> It does **not** claim to verify the continuum Navier–Stokes equations, as that would require an independent continuum PDE source projection onto the discrete particle population.

---

## 3. Mathematical Formulation

### 3.1 Structural Manufactured Solution

The 5-field Reissner–Mindlin plate state is:

$$\mathbf{q}_s^*(x,y,t) = \big(u_0^*, v_0^*, w^*, \theta_x^*, \theta_y^*\big)^T$$

Spatial envelope with fourth-order boundary zeros:

$$B(x, y) = \sin^4\left(\frac{\pi x}{L_x}\right) \sin^4\left(\frac{\pi y}{L_y}\right)$$

Temporal modulation with compatible zero initial displacement and velocity:

$$g_j(t) = 1 - \cos(\Omega_j t), \quad j \in \{u, v, w, \theta_x, \theta_y\}$$

Target fields:

$$u_0^*(x,y,t) = A_u B(x,y) g_u(t)$$
$$v_0^*(x,y,t) = A_v B(x,y) g_v(t)$$
$$w^*(x,y,t) = A_w B(x,y) g_w(t)$$
$$\theta_x^*(x,y,t) = A_x B(x,y) g_x(t)$$
$$\theta_y^*(x,y,t) = A_y B(x,y) g_y(t)$$

Amplitudes and frequencies are defined in `mms_parameters.yaml`. Incommensurable frequencies ensure independent excitation of all five coupled equations.

### 3.2 Function Space Admissibility

Let $\Omega = [0, L_x] \times [0, L_y]$ be the mid-surface domain.

| Field | Space | Regularity Requirement in Weak Form |
| :--- | :--- | :--- |
| Mid-surface translations $(u_0, v_0)$ | $[H^1(\Omega)]^2$ | Membrane strain $\boldsymbol{\varepsilon} \in [L^2]^3$; nonlinear term $\nabla w \otimes \nabla w \in [L^2]^3$ |
| Transverse deflection $w$ | $H^1(\Omega)$ | Transverse shear strain $\boldsymbol{\gamma} = \nabla w - \boldsymbol{\theta} \in [L^2]^2$ |
| Rotations $(\theta_x, \theta_y)$ | $[H^1(\Omega)]^2$ | Curvature $\boldsymbol{\kappa} = \frac{1}{2}(\nabla\boldsymbol{\theta} + \nabla\boldsymbol{\theta}^T) \in [L^2]^3$ |
| Velocities $(\dot{u}_0, \dot{v}_0, \dot{w}, \dot{\theta}_x, \dot{\theta}_y)$ | $[H^1(\Omega)]^5$ | Required for stiffness-proportional damping $\eta_k \mathbf{K}\dot{\mathbf{q}}$ |
| Accelerations $(\ddot{u}_0, \ddot{v}_0, \ddot{w}, \ddot{\theta}_x, \ddot{\theta}_y)$ | $[L^2(\Omega)]^5$ | Inertia terms in weak form |
| Manufactured load $\mathbf{f}^*$ | $[L^2(\Omega)]^5$ | Right-hand side functional $L(v) = \int_\Omega \mathbf{f}^* \cdot \mathbf{v} \, d\Omega$ |

The manufactured fields are $C^\infty(\Omega) \cap C^\infty([0,T])$. Because $B(x,y)$ has 4th-order zeros at all boundaries:
- Dirichlet BCs $\mathbf{q}^* = \mathbf{0}$ are identically satisfied.
- Normal derivatives $\partial_n \mathbf{q}^* = \mathbf{0}$ and $\partial_{nn} \mathbf{q}^* = \mathbf{0}$ vanish identically at all edges, satisfying the homogeneous natural (free) boundary conditions for the Reissner–Mindlin plate.

### 3.3 Particle Trajectory Parametrization

For each particle $p \in \{1, \dots, N\}$:

$$\mathbf{X}_p^*(t) = \mathbf{X}_{0,p} + \mathbf{a}_p \sin(\omega_x t) + \mathbf{b}_p (1 - \cos(\omega_x t))$$
$$\boldsymbol{\Gamma}_p^*(t) = \boldsymbol{\Gamma}_{0,p} + \mathbf{c}_p \sin(\omega_\Gamma t) + \mathbf{d}_p (1 - \cos(\omega_\Gamma t))$$
$$\sigma_p^*(t) = \sigma_{0,p} + e_p (1 - \cos(\omega_\sigma t))$$

Spatial refinement scales $N = N_{\mathrm{base}} \times L$ where $L \in \{1, 2, 4, 8\}$ is the refinement level. The core-overlap ratio $\lambda_\sigma = \sigma_{0,p} / h_p$ is held strictly constant across refinement levels, where $h_p = L_y / N$ is the nominal inter-particle spacing.

### 3.4 Interface Traction and Work Duality

Prescribed smooth analytical traction:

$$\mathbf{t}^*(\eta_s, \eta_c, t) = \begin{pmatrix}
T_x \sin(2\pi\eta_s)\cos(\pi\eta_c) \\
T_y \cos(2\pi\eta_s)\sin(\pi\eta_c) \\
T_z \sin(\pi\eta_s)\sin(\pi\eta_c)
\end{pmatrix} \sin(\omega_t t)$$

Exact panel force is integrated via 2D Gauss–Legendre quadrature:

$$\mathbf{F}_{\mathrm{panel}, j}^* = \int_{\eta_{s,j}}^{\eta_{s,j+1}} \int_0^1 \mathbf{t}^*(\eta_s, \eta_c, t) \, d\eta_c \, d\eta_s$$

Virtual-work duality requires that the force transfer matrix $\mathbf{H}_F$ and displacement transfer matrix $\mathbf{H}_W$ satisfy:

$$\mathbf{H}_F = \mathbf{H}_W^T \implies \delta W_{\mathrm{ALM}} = \delta W_{\mathrm{FEM}}$$

This is verified across three independent virtual displacement modes: rigid translation, rigid rotation, and a sinusoidal bending mode.

---

## 4. Error Norms and Observed Orders

### 4.1 Structural Discretization Error

Relative $L^2(\Omega)$ norm via FEM quadrature:

$$E_{L^2}(q_j) = \left( \frac{\int_\Omega |q_{j,h}(x,y,T) - q_j^*(x,y,T)|^2 \, d\Omega}{\int_\Omega |q_j^*(x,y,T)|^2 \, d\Omega} \right)^{1/2}$$

Space-time norm: integrated over $[0, T]$ using trapezoidal quadrature.

### 4.2 Particle Error Norms

Discrete relative $L^2$ norms over the particle population:

$$e_x = \left( \frac{\sum_{p=1}^N \|\mathbf{X}_{p}^h - \mathbf{X}_p^*\|^2}{\sum_{p=1}^N \|\mathbf{X}_p^*\|^2} \right)^{1/2}, \quad
e_\Gamma = \left( \frac{\sum_{p=1}^N \|\boldsymbol{\Gamma}_p^h - \boldsymbol{\Gamma}_p^*\|^2}{\sum_{p=1}^N \|\boldsymbol{\Gamma}_p^*\|^2} \right)^{1/2}, \quad
e_\sigma = \left( \frac{\sum_{p=1}^N |\sigma_p^h - \sigma_p^*|^2}{\sum_{p=1}^N |\sigma_p^*|^2} \right)^{1/2}$$

### 4.3 Observed Order of Accuracy

For consecutive refinement levels with mesh sizes $h_{i-1}$ and $h_i$:

$$p_i = \frac{\ln(e_{i-1} / e_i)}{\ln(h_{i-1} / h_i)}$$

> [!IMPORTANT]
> **Reporting Rule:**
> Observed orders are reported only when:
> 1. At least three consecutive refinement pairs are available.
> 2. Errors are strictly decreasing ($e_i < e_{i-1}$).
> 3. Errors are above machine precision (no plateau).
> If a plateau occurs, the asymptotic range has not been reached or rounding error dominates; this is explicitly documented rather than masked.

---

## 5. Execution Guide

Run from the repository root:

```bash
# Activate the FEniCS conda environment
conda activate fenics-env

# 1. Standalone Structural MMS (Spatial + Temporal refinement, Jacobian check, scalar ODE check)
python verification/mms/mms_driver.py structural

# 2. Interface Transfer MMS (CRM & RBF force/moment conservation and work duality)
python verification/mms/mms_driver.py transfer

# 3. Particle-level rVPM MMS (standalone source property & stretching FD checks)
julia --project=../FLOWUnsteady verification/mms/mms_rvpm.jl \
      --manifest verification/mms/results/coupled/manifest_level1.json \
      --output verification/mms/results/particles \
      --study standalone

# 4. Coupled FSI MMS (Orchestration across refinement levels)
python verification/mms/mms_driver.py coupled --levels 1,2,4

# 5. Full MMS Verification Suite
python verification/mms/mms_driver.py all
```

---

## 6. Verification Hierarchy and Expected Rates

| Stage | Subsystem | Discretization | Theoretical Rate | Verified By |
| :--- | :--- | :--- | :--- | :--- |
| **0** | Mathematical Definitions | Analytical | Machine precision ($\approx 10^{-15}$) | `mms_problem.py` boundary checks |
| **1a** | Structural Spatial | CG2 Triangular FEM | $O(h^3)$ for displacements, $O(h^2)$ for rotations | `mms_driver.py structural --study spatial` |
| **1b** | Structural Temporal | Generalized-$\alpha$ ($\rho_\infty = 0.9$) | $O(\Delta t^2)$ | `mms_driver.py structural --study temporal` |
| **2** | Particle rVPM | Low-storage RK3 | $O(\Delta t^3)$ for isolated RK3; $O(\Delta t^2)$ with source | `mms_rvpm.jl --study standalone` |
| **3** | Interface Transfer | Common Refinement (CRM) | $O(h^2)$ force/moment; $\Delta W \approx 10^{-14}$ (duality) | `mms_driver.py transfer` |
| **4** | Coupled FSI | Partitioned Staggered | $O(\Delta t)$ (first-order splitting error) | `mms_driver.py coupled` |

---

## 7. Known Scope Limitations

1. **Particle MMS is discrete:** It tests the ODE integration of discrete particles, not the continuum Navier–Stokes equations.
2. **First-order splitting:** The production partitioned staggered FSI scheme introduces an $O(\Delta t)$ temporal splitting error. Second-order temporal convergence in the coupled FSI requires sub-iterative coupling (e.g., Picard or quasi-Newton iterations), which is not active in the explicit staggered mode.
3. **Inviscid reference configuration:** The fluid MMS is configured with `viscous_scheme: none` consistent with the production reference configuration; core-radius diffusion source terms are therefore zero.
