"""
mms_rvpm.jl — Julia-side MMS integration for the VarFlExI particle-level rVPM.

SCOPE
-----
This file verifies the discrete rVPM particle ODE at the PARTICLE LEVEL. It
does NOT claim to verify the continuum Navier–Stokes equations. Time marching
uses a stage-faithful MMS adapter that mirrors FLOWVPM's RK3 recurrence and
calls production state-update operations; it is not an unmodified call to the
native integrator.

INDEPENDENCE GUARANTEE
-----------------------
  Position source:    computed by independent direct-sum O(N²) Biot-Savart
                      (function `independent_biot_savart`), NOT via FLOWVPM's
                      UJ operators (vpm.UJ_freestream, vpm.UJ_direct, etc.).
  Stretching source:  computed by finite-difference Jacobian of the independent
                      Biot-Savart field (function `independent_stretching_rhs`),
                      NOT by calling vpm.update_particle_states on a shadow copy.
  Core source:        includes the stretching-induced term
                      b_sigma = sigma_dot* + sigma* Z. Viscous diffusion and
                      remapping are separate production operations; they are
                      not included in this independently derived source.

PRODUCTION HOOKS
----------------
This file provides `compute_particle_source(target, t, U_inf)` which returns
the source vector `b_p* = Ẏ_p* − G_p^independent(Y_p*, t)` for all particles.
The stage-faithful RK3 adapter `fluid/src/mms_rk3.jl` injects this source via
`MMSRK3StateSource`. The adapter mirrors the installed stage recurrence and
calls production state-update operations.

CLI
---
Run with:
    julia --project=<FLOWUnsteady_project> verification/mms_2/mms_rvpm.jl \\
          --manifest <path/to/manifest.json> \\
          --output   <results/particles> \\
          --study    standalone

Studies:
    standalone   Build particle population from manifest; compute sources
                 and report b_p* at selected times (no time marching).
    coupled      Load a coupled-run result JSON; compute errors vs. target.
    refinement   Particle-population source-scaling study (not solution convergence).
"""

using LinearAlgebra
using Statistics         # for mean()
using JSON               # JSON loader
using Printf

# Independent libm erf wrapper (zero external dependencies)
@inline _erf(x::Float64)::Float64 = ccall(:erf, Cdouble, (Cdouble,), x)

const _HAS_FLOWVPM = try
    @eval using FLOWVPM
    true
catch
    false
end
const vpm = _HAS_FLOWVPM ? FLOWVPM : nothing

if _HAS_FLOWVPM
    _rk3_path = joinpath(@__DIR__, "../../fluid/src/mms_rk3.jl")
    if isfile(_rk3_path)
        include(_rk3_path)
    end
end

# ---------------------------------------------------------------------------
# 0. JSON LOADING HELPER
# ---------------------------------------------------------------------------

function load_json(path::AbstractString)
    open(path, "r") do io
        return JSON.parse(io)
    end
end

function write_json(path::AbstractString, data)
    open(path, "w") do io
        JSON.print(io, data, 2)
    end
end

# ---------------------------------------------------------------------------
# 1. PARTICLE TARGET STATE
#
# Trajectory parametrization (must match mms_problem.py ParticlePopulation):
#   X_p*(t) = X0_p + a_p sin(ω_x,p t) + b_p (1 − cos(ω_x,p t))
#   Γ_p*(t) = Γ0_p + c_p sin(ω_Γ,p t) + d_p (1 − cos(ω_Γ,p t))
#   σ_p*(t) = σ0_p + e_p (1 − cos(ω_σ,p t))
#
# Initial conditions:
#   X_p*(0) = X0_p,  Ẋ_p*(0) = ω_x,p a_p
#   Γ_p*(0) = Γ0_p,  Γ̇_p*(0) = ω_Γ,p c_p
#   σ_p*(0) = σ0_p,  σ̇_p*(0) = 0
# ---------------------------------------------------------------------------

struct ParticleTarget
    n::Int
    ids::Vector{Int}
    x0::Matrix{Float64}       # (3, N)  initial positions
    gamma0::Matrix{Float64}   # (3, N)  initial circulations
    sigma0::Vector{Float64}   # (N,)
    a::Matrix{Float64}        # (3, N)  position amp 1
    b::Matrix{Float64}        # (3, N)  position amp 2
    c::Matrix{Float64}        # (3, N)  circulation amp 1
    d::Matrix{Float64}        # (3, N)  circulation amp 2
    e::Vector{Float64}        # (N,)    core modulation amp
    omega_x::Vector{Float64}  # (N,)
    omega_gamma::Vector{Float64}
    omega_sigma::Vector{Float64}
    lambda_sigma::Float64
end

"""Load particle target from manifest dict (read from JSON)."""
function load_particle_target(manifest::AbstractDict)::ParticleTarget
    p = manifest["particles"]
    # JSON arrays are row-major; reshape to (3, N)
    n = length(p["ids"])
    function to_mat(key)
        raw = p[key]
        if isempty(raw)
            return zeros(Float64, 3, 0)
        end
        if raw[1] isa AbstractVector
            M = Matrix{Float64}(undef, 3, length(raw))
            for i in eachindex(raw)
                M[:, i] .= Float64.(raw[i])
            end
            return M
        else
            return reshape(Float64.(raw), 3, n)
        end
    end
    to_vec(key) = Float64.(p[key])
    return ParticleTarget(
        n,
        Int.(p["ids"]),
        to_mat("x0"),
        to_mat("gamma0"),
        to_vec("sigma0"),
        to_mat("a"),
        to_mat("b"),
        to_mat("c"),
        to_mat("d"),
        to_vec("e"),
        to_vec("omega_x"),
        to_vec("omega_gamma"),
        to_vec("omega_sigma"),
        Float64(get(manifest, "lambda_sigma", 2.0)),
    )
end

"""Return (X*, Γ*, σ*) at time t.  X and Γ are (3, N) matrices."""
function particle_target_state(tgt::ParticleTarget, t::Float64)
    t = Float64(t)
    tx = tgt.omega_x' .* t            # (1, N) broadcast
    tg = tgt.omega_gamma' .* t
    ts = tgt.omega_sigma  .* t
    X     = tgt.x0     .+ tgt.a .* sin.(tx) .+ tgt.b .* (1.0 .- cos.(tx))
    Gamma  = tgt.gamma0 .+ tgt.c .* sin.(tg) .+ tgt.d .* (1.0 .- cos.(tg))
    sigma  = tgt.sigma0 .+ tgt.e .* (1.0 .- cos.(ts))
    return X, Gamma, sigma
end

"""Return (Ẋ*, Γ̇*, σ̇*) at time t.  Analytical time derivatives."""
function particle_derivatives(tgt::ParticleTarget, t::Float64)
    t = Float64(t)
    tx = tgt.omega_x' .* t
    tg = tgt.omega_gamma' .* t
    ts = tgt.omega_sigma  .* t
    Xdot    = (tgt.omega_x' .* (tgt.a .* cos.(tx) .+ tgt.b .* sin.(tx)))
    Gdot    = (tgt.omega_gamma' .* (tgt.c .* cos.(tg) .+ tgt.d .* sin.(tg)))
    sdot    = tgt.e .* tgt.omega_sigma .* sin.(ts)
    return Xdot, Gdot, sdot
end

# ---------------------------------------------------------------------------
# 2. INDEPENDENT BIOT-SAVART (Gaussian-erf kernel)
#
# u(x) = (1/4π) Σ_p  g(r_p/σ_p) / r_p³  ×  (Γ_p × (x − x_p))
#
# Gaussian-erf kernel (reproduced from FLOWVPM; independent implementation):
#   g(ρ) = erf(ρ/√2) − √(2/π) ρ exp(−ρ²/2)
#
# This function is computed WITHOUT calling any FLOWVPM UJ operator.
# ---------------------------------------------------------------------------

"""Gaussian-erf regularization kernel g(r/σ)."""
function rvpm_kernel_g(rho::Float64)::Float64
    return _erf(rho / sqrt(2.0)) - sqrt(2.0 / π) * rho * exp(-0.5 * rho^2)
end

"""
    independent_biot_savart(x_target, X, Gamma, sigma)

Compute the regularized Biot-Savart velocity at `x_target` (length-3 vector)
induced by particles with positions `X` (3×N), circulations `Gamma` (3×N),
and core radii `sigma` (length N).

This is an O(N²) direct sum using the Gaussian-erf kernel.
It does NOT call FLOWVPM's UJ_direct, UJ_fmm, or any other FLOWVPM operator.
"""
function independent_biot_savart(x_target::Vector{Float64},
                                  X::Matrix{Float64},
                                  Gamma::Matrix{Float64},
                                  sigma::Vector{Float64})::Vector{Float64}
    n = size(X, 2)
    u = zeros(Float64, 3)
    inv4pi = 1.0 / (4.0 * π)
    @inbounds for i in 1:n
        dx = x_target .- X[:, i]
        r2 = dot(dx, dx)
        r2 < 1.0e-28 && continue
        r = sqrt(r2)
        rho = r / sigma[i]
        g = rvpm_kernel_g(rho)
        # Γ_p × (x − x_p)
        cross_val = cross(Gamma[:, i], dx)
        u .+= inv4pi .* (g / (r2 * r)) .* cross_val
    end
    return u
end

"""
    independent_stretching_rhs(X, Gamma, sigma; f_rvpm, g_rvpm)

Compute the rVPM vortex stretching RHS for all particles:
  dΓ_p/dt = S_p − 3 Z_p Γ_p

where:
  S_p = Jᵀ Γ_p  (transposed stretching, production default)
  J   = ∂u_i/∂x_j  (velocity Jacobian at x_p, by FD of independent Biot-Savart)
  Z_p = [(f+g)/(1+3f)] S_p·Γ_p / |Γ_p|²

This is derived from Alvarez & Ning (2022), NOT by calling
vpm.update_particle_states on a shadow copy.

Arguments
---------
  X      : (3, N) particle positions
  Gamma  : (3, N) particle circulations
  sigma  : (N,) particle core radii
  f_rvpm : rVPM formulation constant f (default 1/2)
  g_rvpm : rVPM formulation constant g (default 5/4)

Returns
-------
  dGamma_dt : (3, N) independent stretching RHS
  dsigma_dt : (N,)   independent stretching-induced core evolution RHS: dσ/dt = -σ Z
"""
function independent_stretching_rhs(X::Matrix{Float64},
                                     Gamma::Matrix{Float64},
                                     sigma::Vector{Float64};
                                     f_rvpm::Float64 = 0.5,
                                     g_rvpm::Float64 = 1.25)
    n = size(X, 2)
    dGamma_dt = zeros(Float64, 3, n)
    dsigma_dt = zeros(Float64, n)
    h_fd = 1.0e-5   # finite-difference step for velocity Jacobian

    # Compute velocity at all particle positions (independent Biot-Savart)
    U = Matrix{Float64}(undef, 3, n)
    @inbounds for p in 1:n
        U[:, p] = independent_biot_savart(X[:, p], X, Gamma, sigma)
    end

    # Velocity Jacobian at each particle by central finite difference
    @inbounds for p in 1:n
        J = zeros(Float64, 3, 3)
        @inbounds for k in 1:3
            x_plus  = copy(X[:, p]); x_plus[k]  += h_fd
            x_minus = copy(X[:, p]); x_minus[k] -= h_fd
            u_plus  = independent_biot_savart(x_plus,  X, Gamma, sigma)
            u_minus = independent_biot_savart(x_minus, X, Gamma, sigma)
            J[:, k] = (u_plus .- u_minus) ./ (2.0 * h_fd)
        end
        # Transposed stretching: S = Jᵀ Γ_p
        S  = J' * Gamma[:, p]
        G2 = dot(Gamma[:, p], Gamma[:, p])
        Z  = 0.0
        if G2 > 0.0
            Z = (f_rvpm + g_rvpm) / (1.0 + 3.0 * f_rvpm) * dot(S, Gamma[:, p]) / G2
        end
        # rVPM: dΓ/dt = S − 3 Z Γ  (no SFS term for inviscid reference)
        dGamma_dt[:, p] = S .- 3.0 .* Z .* Gamma[:, p]
        # rVPM: stretching-induced core evolution rate dσ/dt = -σ Z
        dsigma_dt[p] = -sigma[p] * Z
    end
    return dGamma_dt, dsigma_dt
end

# ---------------------------------------------------------------------------
# 3. INDEPENDENT PARTICLE SOURCE
#
#   b_x,p  = Ẋ_p* − (U_BiS(X_p*, Γ_all, σ_all) + U_inf)
#   b_Γ,p  = Γ̇_p* − dΓ_dt^independent(X_p*, Γ_p*, σ_p*)
#   b_σ,p  = σ̇_p* − dσ_dt^independent(X_p*, Γ_p*, σ_p*)
#          = σ̇_p* − (−σ_p* Z_p*) = σ̇_p* + σ_p* Z_p*
#
# INDEPENDENCE NOTE: b_Γ and b_σ use independent_stretching_rhs, NOT
# vpm.update_particle_states applied to a shadow copy.
# Core evolution actively exercises the native stretching-induced core update.
# ---------------------------------------------------------------------------

"""
    compute_particle_source(tgt, t, U_inf; f_rvpm, g_rvpm)

Compute the manufactured source `b_p* = Ẏ_p* − G_p^independent(Y_p*, t)`
for all particles in the target.

Returns named tuple (bx, bGamma, bsigma):
  bx     : (3, N) position source  [m/s]
  bGamma : (3, N) circulation source  [m³/s²]
  bsigma : (N,)   core source  [m/s]  (exercising dσ/dt = -σ Z)
"""
function compute_particle_source(tgt::ParticleTarget, t::Float64,
                                  U_inf::Vector{Float64};
                                  f_rvpm::Float64 = 0.5,
                                  g_rvpm::Float64 = 1.25)
    # Target state and analytical derivatives
    X_star, Gamma_star, sigma_star = particle_target_state(tgt, t)
    Xdot_star, Gdot_star, sdot_star = particle_derivatives(tgt, t)

    # Independent position RHS: G_x = U_BiS(X*, Γ*, σ*) + U_inf
    n = tgt.n
    G_x = Matrix{Float64}(undef, 3, n)
    @inbounds for p in 1:n
        G_x[:, p] = independent_biot_savart(X_star[:, p], X_star, Gamma_star, sigma_star)
        G_x[:, p] .+= U_inf
    end

    # Independent circulation and core RHS from rVPM stretching
    G_Gamma, G_sigma = independent_stretching_rhs(X_star, Gamma_star, sigma_star;
                                                  f_rvpm=f_rvpm, g_rvpm=g_rvpm)

    # Sources: b = Ẏ* − G^independent(Y*)
    bx      = Xdot_star     .- G_x
    bGamma  = Gdot_star     .- G_Gamma
    bsigma  = sdot_star     .- G_sigma   # tests stretching-induced core evolution

    return (bx=bx, bGamma=bGamma, bsigma=bsigma)
end

# ---------------------------------------------------------------------------
# 4. ERROR NORMS  (particle-level, per-component)
#
#   e_x  = ( Σ_p ‖X_p^h − X_p*‖² / Σ_p ‖X_p*‖² )^{1/2}
#   e_Γ  = ( Σ_p ‖Γ_p^h − Γ_p*‖² / Σ_p ‖Γ_p*‖² )^{1/2}
#   e_σ  = ( Σ_p |σ_p^h − σ_p*|² / Σ_p |σ_p*|² )^{1/2}
#
# These are discrete L² norms on the particle population.
# ---------------------------------------------------------------------------

"""
    particle_l2_errors(X_num, Gamma_num, sigma_num, tgt, t)

Compute per-component relative L² errors between numerical particle state
and the analytical target at time t.

Arguments
---------
  X_num, Gamma_num : (3, N) matrices from the production solver
  sigma_num        : (N,) from the production solver
  tgt              : ParticleTarget
  t                : physical time

Returns
-------
  Named tuple (e_pos, e_circ, e_core).
"""
function particle_l2_errors(X_num::Matrix{Float64},
                              Gamma_num::Matrix{Float64},
                              sigma_num::Vector{Float64},
                              tgt::ParticleTarget,
                              t::Float64)
    isfinite(t) || error("Particle error evaluation time must be finite")
    X_star, Gamma_star, sigma_star = particle_target_state(tgt, t)
    size(X_num) == size(X_star) || error("Position arrays do not preserve fixed particle identity/ordering")
    size(Gamma_num) == size(Gamma_star) || error("Circulation arrays do not preserve fixed particle identity/ordering")
    length(sigma_num) == length(sigma_star) || error("Core arrays do not preserve fixed particle identity/ordering")
    all(isfinite, X_num) && all(isfinite, Gamma_num) && all(isfinite, sigma_num) ||
        error("Numerical particle state contains non-finite values")

    num2(A, B) = sum(norm(A[:, i] .- B[:, i])^2 for i in 1:size(A, 2))
    den2(A)    = sum(norm(A[:, i])^2 for i in 1:size(A, 2))
    num2s(a, b) = sum((a[i] - b[i])^2 for i in eachindex(a))
    den2s(a)    = sum(a[i]^2 for i in eachindex(a))

    scales = (den2(X_star), den2(Gamma_star), den2s(sigma_star))
    all(scale -> isfinite(scale) && scale > 0, scales) ||
        error("Particle target normalization scales must be finite and positive")
    e_pos   = sqrt(num2(X_num,     X_star)     / scales[1])
    e_circ  = sqrt(num2(Gamma_num, Gamma_star) / scales[2])
    e_core  = sqrt(num2s(sigma_num, sigma_star) / scales[3])

    return (e_pos=e_pos, e_circ=e_circ, e_core=e_core)
end

# ---------------------------------------------------------------------------
# 5. RECONSTRUCTED FIELD ERRORS  (probe-based)
#
# At fixed probe points, compare:
#   u_h(x_probe)  — velocity reconstructed by the production solver's particle field
#   u*(x_probe)   — velocity from the independent direct-sum Biot-Savart at target state
#
# The probe points are stored in the manifest so that the Python and Julia
# sides can compare on a common set.
# ---------------------------------------------------------------------------

"""
    reconstructed_field_errors(X_num, Gamma_num, sigma_num, tgt, t, probe_points)

Compare reconstructed velocity at probe points from:
  - Numerical particle field (X_num, Gamma_num, sigma_num)
  - Analytical target state (X_star, Gamma_star, sigma_star) via independent BiS

Returns vector of relative L² probe errors for velocity.
"""
function reconstructed_field_errors(X_num::Matrix{Float64},
                                     Gamma_num::Matrix{Float64},
                                     sigma_num::Vector{Float64},
                                     tgt::ParticleTarget,
                                     t::Float64,
                                     probe_points::Vector{Vector{Float64}})
    X_star, Gamma_star, sigma_star = particle_target_state(tgt, t)
    errors = Float64[]
    for x_probe in probe_points
        u_num  = independent_biot_savart(x_probe, X_num,   Gamma_num,   sigma_num)
        u_star = independent_biot_savart(x_probe, X_star,  Gamma_star,  sigma_star)
        num = norm(u_num .- u_star)
        den = norm(u_star)
        push!(errors, den > 1e-30 ? num / den : num)
    end
    return errors
end

# ---------------------------------------------------------------------------
# 6. STANDALONE SOURCE VERIFICATION  (no time marching)
#
# Computes b_p*(t) at several times and verifies:
#   (a) b_p*(0) ≠ 0 for all particles (nontrivial source)
#   (b) |b_x,p| / |U_inf| ≪ 1 at t=0 (physically small correction)
#   (c) FD consistency of Ẋ* ≈ G_x^independent + b_x  (tautological, by construction)
# ---------------------------------------------------------------------------

function run_standalone_verification(tgt::ParticleTarget, manifest::AbstractDict,
                                      output_dir::AbstractString)
    println("=" ^ 70)
    println("PARTICLE-LEVEL MMS  — Standalone Source Verification")
    println("Scope: discrete-model; NOT continuum Navier–Stokes verification")
    println("=" ^ 70)

    vinf = get(manifest, "freestream", Dict())
    vinf_val = get(vinf, "vinf", 15.0)
    aoa_deg  = get(vinf, "aoa_deg", 0.0)
    U_inf = [vinf_val * cosd(aoa_deg), 0.0, vinf_val * sind(aoa_deg)]

    t_checks = [0.0, 0.01, 0.05, 0.1]
    results = []

    for t_val in t_checks
        b = compute_particle_source(tgt, Float64(t_val), U_inf)
        bx_norm  = mean(norm(b.bx[:, p]) for p in 1:tgt.n)
        bg_norm  = mean(norm(b.bGamma[:, p]) for p in 1:tgt.n)
        bs_norm  = mean(abs(b.bsigma[p]) for p in 1:tgt.n)
        @printf("  t = %.3f s  |bx|_mean = %.4e  |bΓ|_mean = %.4e  |bσ|_mean = %.4e\n",
                t_val, bx_norm, bg_norm, bs_norm)
        push!(results, Dict(
            "t" => t_val,
            "bx_mean_norm"  => bx_norm,
            "bGamma_mean_norm" => bg_norm,
            "bsigma_mean_norm" => bs_norm,
        ))
    end

    # Check that source is non-trivial at t > 0
    b0 = compute_particle_source(tgt, 0.0, U_inf)
    b1 = compute_particle_source(tgt, 0.05, U_inf)
    if all(norm(b1.bx[:, p]) < 1e-30 for p in 1:tgt.n)
        @warn "Position source is zero at t=0.05 — check trajectory parametrization"
    else
        println("  Source nontriviality check: PASSED (source is nonzero for t > 0)")
    end

    # Physical plausibility: |bx| / |U_inf| should be ≪ 1 at t=0
    b_ratio = mean(norm(b0.bx[:, p]) for p in 1:tgt.n) / norm(U_inf)
    @printf("  |bx|(t=0) / |U_inf| = %.4e  (should be ≪ 1 for physical target)\n", b_ratio)
    if b_ratio > 0.5
        @warn "Position source at t=0 is large relative to U_inf — check amplitude scaling"
    end

    mkpath(output_dir)
    result_path = joinpath(output_dir, "particle_source_standalone.json")
    write_json(result_path, Dict("results" => results, "b0_ratio_vinf" => b_ratio))
    println("Results written: $result_path")
    return results
end

# ---------------------------------------------------------------------------
# 7. REFINEMENT STUDY  (particle spatial refinement)
#
# At each refinement level L:
#   N = n_base * L particles, h_p = Ly / N, σ₀ = λ_σ * h_p
#
# For each level, compute the source b_p*(t) at t = t_ref.
# Source magnitude versus N is a source-scaling study, not solution
# convergence. Reconstruction error requires fixed probes and an independent
# reference field.
#
# NOTE: Actual time-marching convergence (numerical vs. analytical) requires
# running the production solver, which is orchestrated by mms_driver.py.
# This function only computes standalone source properties.
# ---------------------------------------------------------------------------

function run_refinement_study(manifest::AbstractDict, output_dir::AbstractString)
    println("=" ^ 70)
    println("PARTICLE-LEVEL MMS — Population Source-Scaling Study")
    println("=" ^ 70)

    vinf = get(manifest, "freestream", Dict())
    vinf_val = get(vinf, "vinf", 15.0)
    aoa_deg  = get(vinf, "aoa_deg", 0.0)
    U_inf = [vinf_val * cosd(aoa_deg), 0.0, vinf_val * sind(aoa_deg)]

    Lx = Float64(manifest["Lx"])
    Ly = Float64(manifest["Ly"])
    lambda_sigma = Float64(get(manifest, "lambda_sigma", 2.0))

    # Base number of particles (from manifest or fallback)
    p_data = manifest["particles"]
    n_base = length(p_data["ids"])
    t_ref = 0.05

    study = get(get(manifest, "studies", Dict()), "particle_population", Dict())
    levels = Int.(get(study, "levels", Int[]))
    isempty(levels) && error("manifest is missing studies.particle_population.levels")
    t_ref = Float64(get(study, "evaluation_time", NaN))
    isfinite(t_ref) || error("manifest is missing a finite particle population evaluation_time")
    rows = []
    for level in levels
        n_particles = n_base * level
        h_p  = Ly / n_particles
        σ₀   = lambda_sigma * h_p

        # Build scaled target directly (reproducing Python ParticlePopulation.build)
        # Deterministic: same seed, scaled N
        tgt_level = build_scaled_target(n_particles, Lx, Ly, σ₀, lambda_sigma)

        b = compute_particle_source(tgt_level, Float64(t_ref), U_inf)
        bx_inf = maximum(norm(b.bx[:, p]) for p in 1:tgt_level.n)
        bg_inf = maximum(norm(b.bGamma[:, p]) for p in 1:tgt_level.n)

        @printf("  Level %2d  N=%4d  h_p=%.4e  σ₀=%.4e  |bx|_inf=%.4e  |bΓ|_inf=%.4e\n",
                level, n_particles, h_p, σ₀, bx_inf, bg_inf)
        push!(rows, Dict(
            "level" => level,
            "n_particles" => n_particles,
            "h_p" => h_p,
            "sigma0" => σ₀,
            "bx_inf_norm" => bx_inf,
            "bGamma_inf_norm" => bg_inf,
        ))
    end

    mkpath(output_dir)
    result_path = joinpath(output_dir, "particle_source_scaling.json")
    write_json(result_path, Dict("t_ref" => t_ref, "levels" => rows))
    println("Results written: $result_path")
    return rows
end

"""
Build a scaled particle target for refinement study (matches Python ParticlePopulation.build logic).
Uses deterministic layout: downstream wake, uniform spanwise distribution.
"""
function build_scaled_target(n::Int, Lx::Float64, Ly::Float64,
                              sigma0::Float64, lambda_sigma::Float64)::ParticleTarget
    # Deterministic layout (seed implicit via linspace)
    x0 = Matrix{Float64}(undef, 3, n)
    gamma0 = Matrix{Float64}(undef, 3, n)
    sigma0_arr = fill(sigma0, n)

    for i in 1:n
        eta = (i - 1) / max(n - 1, 1)
        x0[:, i] = [1.5 * Lx, 0.1 * Ly + 0.8 * Ly * eta, 0.0]
        gamma0[:, i] = [0.0, 0.0, 1.0e-4]
    end

    # Modulation amplitudes (small relative to Lx)
    a = fill(0.005 * Lx, 3, n)
    b = fill(0.0025 * Lx, 3, n)
    c = zeros(3, n)
    d = Matrix{Float64}(undef, 3, n)
    for i in 1:n; d[:, i] = [0.0, 0.0, 1.0e-5]; end
    e = fill(0.1 * sigma0, n)

    return ParticleTarget(
        n,
        collect(1:n),
        x0, gamma0, sigma0_arr,
        a, b, c, d, e,
        fill(2.0, n),   # omega_x
        fill(1.7, n),   # omega_gamma
        fill(1.3, n),   # omega_sigma
        lambda_sigma,
    )
end

# ---------------------------------------------------------------------------
# 7b. FIXED-POPULATION PARTICLE ODE TIME REFINEMENT (stage-faithful adapter)
# ---------------------------------------------------------------------------

"""
    run_particle_time_refinement(tgt, manifest, output_dir)

Execute fixed-population time-step refinement using the stage-faithful MMS
adapter around FLOWVPM production state-update operations.

Verifies:
  - Position evolution:     dX_p/dt = U_BiS(X_p) + U_inf + b_x,p*
  - Circulation evolution:  dΓ_p/dt = S_p - 3 Z_p Γ_p + b_Γ,p*
  - Core-radius evolution:  dσ_p/dt = -σ_p Z_p + b_σ,p*

All three particle-state equations are integrated simultaneously. At final
time T_end, compute matched particle-state errors and observed order; the
selected RK scheme does not guarantee a measured convergence rate.
"""
function run_particle_time_refinement(tgt::ParticleTarget, manifest::AbstractDict, output_dir::AbstractString)
    println("=" ^ 70)
    println("PARTICLE-LEVEL MMS — RK3 Adapter Temporal Study")
    println("Using MMSRK3StateSource around production state updates")
    println("=" ^ 70)

    if !_HAS_FLOWVPM
        error("FLOWVPM is required for particle ODE time refinement; no result was produced")
    end
    vpm = FLOWVPM

    vinf_data = get(manifest, "freestream", Dict())
    vinf_val  = Float64(get(vinf_data, "vinf", 15.0))
    aoa_deg   = Float64(get(vinf_data, "aoa_deg", 0.0))
    U_inf     = [vinf_val * cosd(aoa_deg), 0.0, vinf_val * sind(aoa_deg)]

    study = get(get(manifest, "studies", Dict()), "particle_temporal", Dict())
    T_end = Float64(get(study, "duration", NaN))
    dt_list = Float64.(get(study, "dt_s", Float64[]))
    (isfinite(T_end) && T_end > 0 && !isempty(dt_list) && all(dt -> dt > 0, dt_list)) ||
        error("manifest is missing valid studies.particle_temporal duration/dt_s")

    errors_x = Float64[]
    errors_g = Float64[]
    errors_s = Float64[]
    results = []

    for dt in dt_list
        n_steps = round(Int, T_end / dt)
        pfield = vpm.ParticleField(tgt.n; Uinf=(t)->U_inf)
        pfield.formulation = vpm.rVPM

        # Seed target initial condition at t = 0
        X0, G0, S0 = particle_target_state(tgt, 0.0)
        for i in 1:tgt.n
            vpm.add_particle(pfield, view(X0, :, i), view(G0, :, i), S0[i]; static=false)
        end

        cache = Dict{Float64, Any}()
        function source_callback(particle, stage_time, particle_slot, pf, a, b, step_dt)
            t_s = Float64(stage_time)
            if !haskey(cache, t_s)
                cache[t_s] = compute_particle_source(tgt, t_s, U_inf)
            end
            src = cache[t_s]
            p = Int(particle_slot)
            return (position=src.bx[:, p], circulation=src.bGamma[:, p], core=src.bsigma[p])
        end

        adapter = MMSRK3StateSource(source_callback, nothing)

        t_curr = 0.0
        for step in 1:n_steps
            pfield.t = t_curr
            adapter(pfield, dt)
            t_curr += dt
            pfield.t = t_curr
        end

        X_num = zeros(Float64, 3, tgt.n)
        Gamma_num = zeros(Float64, 3, tgt.n)
        sigma_num = zeros(Float64, tgt.n)
        for i in 1:tgt.n
            pt = vpm.get_particle(pfield, i)
            X_num[:, i] .= vpm.get_X(pt)
            Gamma_num[:, i] .= vpm.get_Gamma(pt)
            sigma_num[i] = vpm.get_sigma(pt)[]
        end

        errs = particle_l2_errors(X_num, Gamma_num, sigma_num, tgt, T_end)
        push!(errors_x, errs.e_pos)
        push!(errors_g, errs.e_circ)
        push!(errors_s, errs.e_core)
        @printf("  dt = %.4e s (%3d steps)  e_pos = %.4e  e_circ = %.4e  e_core = %.4e\n",
                dt, n_steps, errs.e_pos, errs.e_circ, errs.e_core)
        push!(results, Dict(
            "dt" => dt, "n_steps" => n_steps,
            "e_pos" => errs.e_pos, "e_circ" => errs.e_circ, "e_core" => errs.e_core
        ))
    end

    function calc_orders(errs, dts)
        ords = fill(NaN, length(errs))
        for i in 2:length(errs)
            if errs[i] > 0 && errs[i-1] > 0
                ords[i] = log(errs[i-1] / errs[i]) / log(dts[i-1] / dts[i])
            end
        end
        return ords
    end

    ord_x = calc_orders(errors_x, dt_list)
    ord_g = calc_orders(errors_g, dt_list)
    ord_s = calc_orders(errors_s, dt_list)

    println("\n" * "=" ^ 75)
    println("Particle RK3 Temporal Convergence Summary")
    println("=" ^ 75)
    @printf("%-12s %-12s %-12s %-12s %-8s %-8s %-8s\n",
            "dt (s)", "e_pos", "e_circ", "e_core", "ord_pos", "ord_circ", "ord_core")
    println("-" ^ 75)
    for i in 1:length(dt_list)
        @printf("%-12.4e %-12.4e %-12.4e %-12.4e %-8.2f %-8.2f %-8.2f\n",
                dt_list[i], errors_x[i], errors_g[i], errors_s[i],
                ord_x[i], ord_g[i], ord_s[i])
    end
    println("=" ^ 75)

    mkpath(output_dir)
    csv_path = joinpath(output_dir, "particle_temporal_convergence.csv")
    open(csv_path, "w") do io
        println(io, "dt,e_pos,e_circ,e_core,ord_pos,ord_circ,ord_core")
        for i in 1:length(dt_list)
            @printf(io, "%.6e,%.6e,%.6e,%.6e,%.2f,%.2f,%.2f\n",
                    dt_list[i], errors_x[i], errors_g[i], errors_s[i],
                    ord_x[i], ord_g[i], ord_s[i])
        end
    end
    println("CSV written: $csv_path")
    return results
end

# ---------------------------------------------------------------------------
# 8. COUPLED RUN ERROR LOADING
#
# When coupled MMS results are stored (by the production solver + mms_rk3.jl),
# this function loads the particle state at the final time and computes errors.
# ---------------------------------------------------------------------------

function run_coupled_errors(manifest::AbstractDict, result_path::AbstractString,
                             output_dir::AbstractString)
    println("=" ^ 70)
    println("PARTICLE-LEVEL MMS  — Coupled Run Errors")
    println("=" ^ 70)

    if !isfile(result_path)
        error("Coupled result file not found: $result_path")
    end

    result = load_json(result_path)
    tgt = load_particle_target(manifest)

    t_final = Float64(get(result, "t_final", 0.0))
    if haskey(result, "X_final") && haskey(result, "Gamma_final") && haskey(result, "sigma_final")
        X_num     = reshape(Float64.(result["X_final"]),     3, tgt.n)
        Gamma_num = reshape(Float64.(result["Gamma_final"]), 3, tgt.n)
        sigma_num = Float64.(result["sigma_final"])
        errs = particle_l2_errors(X_num, Gamma_num, sigma_num, tgt, t_final)
        @printf("  t_final = %.4f s\n", t_final)
        @printf("  e_pos   = %.4e\n", errs.e_pos)
        @printf("  e_circ  = %.4e\n", errs.e_circ)
        @printf("  e_core  = %.4e\n", errs.e_core)
        mkpath(output_dir)
        write_json(joinpath(output_dir, "coupled_particle_errors.json"), Dict(
            "t_final"  => t_final,
            "e_pos"    => errs.e_pos,
            "e_circ"   => errs.e_circ,
            "e_core"   => errs.e_core,
        ))
    else
        error("Result JSON is missing final particle state arrays; no particle error was computed")
    end
end

# ---------------------------------------------------------------------------
# 9. FD CONSISTENCY CHECK FOR INDEPENDENT STRETCHING RHS
#
# Verify: d/dt Γ_p*(t)|_{t=0} ≈ G_Γ^independent(X*(0), Γ*(0), σ*(0)) + b_Γ*(0)
# This is a tautological check (by construction), but confirms no sign/index errors.
# ---------------------------------------------------------------------------

function run_stretching_fd_check(tgt::ParticleTarget, U_inf::Vector{Float64},
                                  output_dir::AbstractString)
    println("--- Stretching source FD consistency check ---")
    h_fd = 1.0e-6
    t0 = 0.05

    _, Gdot_analytic, sdot_analytic = particle_derivatives(tgt, t0)
    b = compute_particle_source(tgt, t0, U_inf)
    G_Gamma, G_sigma = independent_stretching_rhs(
        particle_target_state(tgt, t0)[1],
        particle_target_state(tgt, t0)[2],
        particle_target_state(tgt, t0)[3]
    )

    # Check: Gdot* = G_Gamma + b_Gamma and sdot* = G_sigma + b_sigma
    residual_g = Gdot_analytic .- G_Gamma .- b.bGamma
    residual_s = sdot_analytic .- G_sigma .- b.bsigma
    max_res_g = maximum(norm(residual_g[:, p]) for p in 1:tgt.n)
    max_res_s = maximum(abs(residual_s[p]) for p in 1:tgt.n)
    @printf("  Stretching source definition residual: %.4e (circulation), %.4e (core)\n",
            max_res_g, max_res_s)
    if max_res_g > 1e-12 || max_res_s > 1e-12
        @warn "Source definition residual exceeds 1e-12 — check implementation"
    else
        println("  FD consistency check: PASSED")
    end

    # Finite-difference check on G_Gamma: compare to centered-FD of Γ* trajectory
    _, Gp, _ = particle_target_state(tgt, t0 + h_fd)
    _, Gm, _ = particle_target_state(tgt, t0 - h_fd)
    Gdot_fd  = (Gp .- Gm) ./ (2.0 * h_fd)
    err_fd = maximum(norm(Gdot_fd[:, p] .- Gdot_analytic[:, p]) for p in 1:tgt.n)
    @printf("  FD check on Ġ* vs analytical derivative: %.4e (should be < 1e-8)\n", err_fd)

    mkpath(output_dir)
    write_json(joinpath(output_dir, "stretching_fd_check.json"), Dict(
        "t0" => t0, "max_definition_residual" => max_res_g,
        "max_core_definition_residual" => max_res_s, "fd_derivative_error" => err_fd
    ))
end

# ---------------------------------------------------------------------------
# 10. MAIN ENTRYPOINT  (CLI dispatch)
# ---------------------------------------------------------------------------

function print_usage()
    println("""
Usage:
  julia --project=<FLOWUnsteady_project> verification/mms_2/mms_rvpm.jl \\
        --manifest <path/to/manifest.json> \\
        --output   <results/particles> \\
        --study    <standalone|coupled|refinement|all>

Options:
  --manifest PATH    JSON manifest file (required)
  --output   DIR     Output directory (default: verification/mms_2/results/particles)
  --study    STUDY   Study to run (default: standalone)
  --result   PATH    Coupled run result JSON (for --study coupled)
  --help             Print this message and exit
""")
end

function parse_cli(args::Vector{String})
    opts = Dict{String,String}(
        "manifest" => "",
        "output"   => "verification/mms_2/results/particles",
        "study"    => "standalone",
        "result"   => "",
    )
    i = 1
    while i <= length(args)
        key = args[i]
        if key == "--help" || key == "-h"
            print_usage()
            exit(0)
        end
        if startswith(key, "--") && i < length(args)
            k = key[3:end]
            opts[k] = args[i+1]
            i += 2
        else
            @warn "Unknown argument: $key"
            i += 1
        end
    end
    return opts
end

function main(args::Vector{String} = ARGS)
    opts = parse_cli(args)

    if isempty(opts["manifest"])
        println(stderr, "ERROR: --manifest is required")
        print_usage()
        exit(1)
    end

    manifest_path = opts["manifest"]
    if !isfile(manifest_path)
        println(stderr, "ERROR: manifest file not found: $manifest_path")
        exit(1)
    end

    println("Loading manifest: $manifest_path")
    manifest = load_json(manifest_path)
    tgt = load_particle_target(manifest)
    @printf("Particle target: N=%d  λ_σ=%.2f\n", tgt.n, tgt.lambda_sigma)

    output_dir = opts["output"]
    study      = opts["study"]

    vinf_data = get(manifest, "freestream", Dict())
    vinf_val  = Float64(get(vinf_data, "vinf", 15.0))
    aoa_deg   = Float64(get(vinf_data, "aoa_deg", 0.0))
    U_inf = [vinf_val * cosd(aoa_deg), 0.0, vinf_val * sind(aoa_deg)]

    if study == "standalone" || study == "all"
        run_standalone_verification(tgt, manifest, output_dir)
        run_stretching_fd_check(tgt, U_inf, output_dir)
    end

    if study == "refinement" || study == "all"
        run_refinement_study(manifest, output_dir)
        run_particle_time_refinement(tgt, manifest, output_dir)
    end

    if study == "coupled" || study == "all"
        result_path = opts["result"]
        if isempty(result_path)
            error("--result is required for coupled particle error extraction")
        else
            run_coupled_errors(manifest, result_path, output_dir)
        end
    end

    println("\nmms_rvpm.jl complete.")
end

# ---------------------------------------------------------------------------
# Top-level entry when run as script
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
