#!/usr/bin/env julia
"""Independent matrix/residual check for the installed FLOWVLM circulation solve.

The reference independently assembles the horseshoe influence matrix and
right-hand side, then compares the resulting circulation with FLOWVLM's solve
on the same prescribed geometry and freestream. It intentionally shares the
installed horseshoe velocity kernel; this checks matrix assembly and the
discrete no-penetration system, not the vortex kernel or coupled FSI path.
"""

using LinearAlgebra
using FLOWVLM

const vlm = FLOWVLM

# The production fluid entry point installs these robustness overrides before
# building or solving the aircraft VLM system. Mirror that kernel configuration
# in this standalone component process.
include(joinpath(@__DIR__, "..", "..", "fluid", "src", "vlm_patch.jl"))

function panel_normal(hs)
    _, A, B, _, CP, _, _, _ = hs
    n = cross(CP - A, B - A)
    nn = norm(n)
    nn > 0 || error("degenerate FLOWVLM panel normal")
    return n / nn
end

function reference_system(horseshoes, vinfs; t=0.0, vortexsheet=nothing,
                          extraVinf=nothing, extraVinfArgs...)
    n = length(horseshoes)
    length(vinfs) == n || error("horseshoe and freestream counts differ")
    A = zeros(Float64, n, n)
    rhs = zeros(Float64, n)
    normals = [panel_normal(hs) for hs in horseshoes]
    for i in 1:n
        cp = horseshoes[i][5]
        ni = normals[i]
        # Assemble one coefficient at a time, without calling the package's
        # matrix-building solver.
        for j in 1:n
            geom = vlm.VLMSolver.V(horseshoes[j], cp;
                                   ign_infvortex=(vortexsheet !== nothing))
            A[i, j] = dot(geom / (4pi), ni)
        end
        rhs[i] = -dot(vinfs[i], ni)
        if vortexsheet !== nothing
            rhs[i] -= dot(vortexsheet(cp, t), ni)
        end
        if extraVinf !== nothing
            rhs[i] -= dot(extraVinf(i, t; extraVinfArgs...), ni)
        end
    end
    return A, rhs
end

function main()
    nspan = 12
    wing = vlm.Wing(0.0, 0.0, 0.0, 1.0, 0.0)
    vlm.addchord(wing, 0.0, 4.0, 0.12, 0.72, 0.0, nspan; r=1.0)
    vlm.setVinf(wing, (x, t) -> [18.0, 0.0, 0.0])
    hs = vlm.getHorseshoes(wing)
    vinfs = vlm.getVinfs(wing)

    # Production package solve and independently assembled reference.
    gamma_prod = vlm.VLMSolver.solve(hs, vinfs)
    A, rhs = reference_system(hs, vinfs)
    gamma_ref = A \ rhs
    residual = A * gamma_prod - rhs
    circulation_rel = norm(gamma_ref - gamma_prod) / max(norm(gamma_ref), eps())
    residual_rel = norm(residual) / max(norm(rhs), eps())
    matrix_cond = cond(A)

    # Manufactured algebraic target: choose Gamma independently, construct a
    # compatible normal-velocity RHS, and recover it through the reference.
    gamma_hat = [0.07 * sin(0.63i) + 0.015 * cos(0.21i) for i in 1:nspan]
    rhs_hat = A * gamma_hat
    vinf_hat = [(-rhs_hat[i]) .* panel_normal(hs[i]) for i in 1:nspan]
    A_hat, rhs_rebuilt = reference_system(hs, vinf_hat)
    gamma_recovered = A_hat \ rhs_rebuilt
    manufactured_rel = norm(gamma_recovered - gamma_hat) / norm(gamma_hat)
    manufactured_residual = norm(A_hat * gamma_hat - rhs_rebuilt) /
                            max(norm(rhs_rebuilt), eps())

    metrics = (
        n_panels=nspan,
        circulation_relative_difference=circulation_rel,
        production_no_penetration_relative_residual=residual_rel,
        manufactured_circulation_relative_error=manufactured_rel,
        manufactured_no_penetration_relative_residual=manufactured_residual,
        influence_matrix_condition_number=matrix_cond,
        scope="FLOWVLM discrete circulation component; shared velocity kernel; static prescribed geometry; no wake or FSI",
    )
    println("status=", max(circulation_rel, residual_rel, manufactured_rel,
                         manufactured_residual) < 1e-10 ? "PASS" : "FAIL")
    for (key, value) in pairs(metrics)
        println(key, "=", value)
    end
    maximum((circulation_rel, residual_rel, manufactured_rel,
             manufactured_residual)) < 1e-10 || error("FLOWVLM circulation reference check failed")
end

main()
