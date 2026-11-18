using LinearAlgebra
using FLOWVPM
using Pkg

const vpm = FLOWVPM
const flowvpm_source_file = joinpath(dirname(pathof(FLOWVPM)), "FLOWVPM_timeintegration.jl")
isfile(flowvpm_source_file) || error("FLOWVPM integration source not found at runtime resolution")
const flowvpm_source = read(flowvpm_source_file, String)
for coefficient in ("(0.0, 1/3)", "(-5/9, 15/16)", "(-153/128, 8/15)")
    occursin(coefficient, flowvpm_source) || error(
        "Resolved FLOWVPM source does not contain expected RK3 coefficient $coefficient")
end
const flowvpm_version = Pkg.dependencies()[Base.PkgId(FLOWVPM).uuid].version

include(joinpath(@__DIR__, "../../fluid/src/mms_coupled_source.jl"))

function check_kernel_jacobian()
    X = [0.0 0.0 0.0; 0.17 -0.09 0.13; -0.21 0.16 -0.07]
    Gamma = [0.3 -0.2 0.1; -0.13 0.27 0.19; 0.08 0.11 -0.23]
    sigma = [0.08, 0.12, 0.10]
    x = [0.04, -0.03, 0.06]
    J = independent_biot_savart_jacobian(x, X, Gamma, sigma)
    all(isfinite, J) || error("analytic Jacobian must be finite")

    step_errors = Float64[]
    for relative_step in (2.0e-3, 1.0e-3, 5.0e-4)
        Jfd = zeros(3,3)
        for axis in 1:3
            h = relative_step * minimum(sigma)
            xp, xm = copy(x), copy(x)
            xp[axis] += h
            xm[axis] -= h
            Jfd[:,axis] .= (independent_biot_savart_velocity(xp, X, Gamma, sigma) .-
                            independent_biot_savart_velocity(xm, X, Gamma, sigma)) ./ (2h)
        end
        push!(step_errors, norm(J-Jfd) / max(norm(J), norm(Jfd), 1.0e-14))
    end
    minimum(step_errors) < 2.0e-7 || error(
        "analytic Gaussian-erf Jacobian failed centered directional oracle: $(step_errors)")

    gamma_probe = [0.21, -0.17, 0.09]
    h_orientation = 5.0e-5 * minimum(sigma)
    s_fd = zeros(3)
    for axis in 1:3
        xp, xm = copy(x), copy(x)
        xp[axis] += h_orientation
        xm[axis] -= h_orientation
        s_fd[axis] = (dot(independent_biot_savart_velocity(xp, X, Gamma, sigma), gamma_probe) -
                      dot(independent_biot_savart_velocity(xm, X, Gamma, sigma), gamma_probe)) /
                     (2h_orientation)
    end
    s_pred = J' * gamma_probe
    orientation_error = norm(s_pred-s_fd) / max(norm(s_pred), norm(s_fd), 1.0e-14)
    orientation_error < 2.0e-7 || error("stretching tensor orientation S=J'Gamma failed: $orientation_error")

    # The production direct kernel skips exact self interactions. Its velocity
    # and Jacobian contribution at a one-particle self evaluation are zero.
    self_X = reshape([0.1, -0.2, 0.3], 1, 3)
    self_Gamma = reshape([0.4, -0.1, 0.2], 1, 3)
    self_sigma = [0.09]
    self_velocity = independent_biot_savart_velocity(vec(self_X), self_X, self_Gamma, self_sigma)
    self_J = independent_biot_savart_jacobian(vec(self_X), self_X, self_Gamma, self_sigma)
    norm(self_velocity) == 0.0 && norm(self_J) == 0.0 || error(
        "FLOWVPM exact self-interaction exclusion convention changed")

    small_rho = 3.0e-3
    expected_limit = sqrt(2.0 / pi) / 3.0
    kernel_limit = independent_rvpm_kernel_g(small_rho) / small_rho^3
    isapprox(kernel_limit, expected_limit; rtol=2.0e-5, atol=0.0) || error(
        "Gaussian-erf radial kernel small-core limit is incorrect")
    return (step_errors=step_errors, kernel_limit=kernel_limit,
            expected_kernel_limit=expected_limit, self_convention="exact source exclusion")
end

function check_bound_circulation_target()
    omega = 2.7
    gamma0 = [0.41, 0.46, 0.52]
    amplitude_a = [0.03, -0.02, 0.01]
    amplitude_b = [0.04, 0.015, -0.025]
    data = Dict{String,Any}("bound_circulation" => Dict{String,Any}(
        "omega" => omega, "gamma0" => gamma0, "a" => amplitude_a, "b" => amplitude_b))
    target = CoupledMMSTarget(data, Int[], Dict{Int,Int}())
    state_error = 0.0
    derivative_error = 0.0
    for t in (0.0, 0.137, 2π/omega, 4π/omega, 5.13)
        expected = gamma0 .+ amplitude_a .* sin(omega*t) .+ amplitude_b .* cos(omega*t)
        actual = bound_circulation_state(target, t, length(gamma0))
        state_error = max(state_error, maximum(abs.(actual .- expected)))
        analytic = bound_circulation_derivative(target, t, length(gamma0))
        for h in (1.0e-4, 5.0e-5)
            finite_difference = (bound_circulation_state(target, t+h, length(gamma0)) .-
                                 bound_circulation_state(target, t-h, length(gamma0))) ./ (2h)
            derivative_error = max(derivative_error,
                maximum(abs.(finite_difference .- analytic)) /
                max(maximum(abs.(analytic)), 1.0e-12))
        end
    end
    state_error == 0.0 || error("Julia bound-circulation state does not match canonical formula")
    derivative_error < 2.0e-7 || error(
        "Julia bound-circulation derivative centered-FD check failed: $derivative_error")
    return (state_error=state_error, derivative_error=derivative_error,
            times=[0.0, 0.137, 2π/omega, 4π/omega, 5.13], threshold=2.0e-7)
end

result = check_kernel_jacobian()
circulation_result = check_bound_circulation_target()
println("COUPLED_KERNEL_AND_CIRCULATION_CHECK PASS FLOWVPM=", flowvpm_version,
        " source=", flowvpm_source_file, " jacobian=", result,
        " circulation=", circulation_result,
        " CIRCULATION_STATE_ABS_ERROR=", circulation_result.state_error,
        " CIRCULATION_DERIVATIVE_REL_ERROR=", circulation_result.derivative_error)
