using FLOWUnsteady
using LinearAlgebra: cross, norm
import FLOWVPM as vpm
import FLOWVLM as vlm

const OUT = joinpath(@__DIR__, "results")
mkpath(OUT)

function radial_integral(f, rmax=8.0, n=20001)
    dr = rmax / (n - 1)
    total = 0.0
    for i in 1:n
        r = (i - 1) * dr
        weight = (i == 1 || i == n) ? 0.5 : 1.0
        total += weight * 4pi * r^2 * f(r) * dr
    end
    return total
end

normalization = radial_integral(vpm.zeta_gauserf)
abs(normalization - 1.0) < 2e-7 || error("Gaussian-erf kernel normalization failed: $normalization")
abs(vpm.g_gauserf(0.0)) < 1e-14 || error("Gaussian-erf velocity kernel is not regular at the origin")
abs(vpm.g_gauserf(8.0) - 1.0) < 1e-12 || error("Gaussian-erf velocity kernel does not approach the Newtonian limit")

nu = 1.0e-3
dt = 2.5e-4
sigma0 = 1.0e-2
scheme = vpm.CoreSpreading(nu, sigma0, 1.0)
pfield = vpm.ParticleField(4, Float64;
    viscous=scheme, integration=vpm.rungekutta3, kernel=vpm.kernel_gaussianerf)
vpm.add_particle(pfield, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], sigma0)
for (a, b) in ((0.0, 1/3), (-5/9, 15/16), (-153/128, 8/15))
    vpm.viscousdiffusion(pfield, scheme, dt; aux1=a, aux2=b)
end
sigma = vpm.get_sigma(vpm.get_particle(pfield, 1))[]
sigma2_error = abs(sigma^2 - sigma0^2 - 2nu*dt)
sigma2_error < 1e-14 || error("RK3 core-spreading increment failed: $sigma2_error")

# Exercise FLOWVPM's actual ReformulatedVPM RK3 state update with a prescribed
# spatially uniform velocity/zero gradient. This isolates RK staging and the
# particle position/circulation updates from the N-body velocity solver.
u_test = [0.4, -0.2, 0.1]
dt_advect = 1.0e-3
n_advect = 12
advect_field = vpm.ParticleField(2, Float64;
    viscous=vpm.Inviscid(), integration=vpm.rungekutta3,
    kernel=vpm.kernel_gaussianerf, Uinf=(t -> u_test))
x0 = [0.1, -0.3, 0.2]
gamma0 = [0.2, -0.1, 0.4]
vpm.add_particle(advect_field, x0, gamma0, 0.01)
function prescribed_zero_gradient!(field; kwargs...)
    for i in 1:field.np
        vpm.get_U(field, i) .= 0.0
        vpm.get_J(field, i) .= 0.0
    end
end
for _ in 1:n_advect
    vpm.rungekutta3(advect_field, dt_advect; custom_UJ=prescribed_zero_gradient!)
    advect_field.t += dt_advect
end
x_expected = x0 .+ n_advect * dt_advect .* u_test
x_actual = collect(vpm.get_X(advect_field, 1))
position_error = norm(x_actual - x_expected)
gamma_error = norm(collect(vpm.get_Gamma(advect_field, 1)) - gamma0)
position_error < 1e-12 || error("RK3 prescribed-velocity trajectory failed: $position_error")
gamma_error < 1e-12 || error("Zero-gradient circulation preservation failed: $gamma_error")

# Check actual FLOWVPM direct U/J reconstruction at one target against the
# regularized Biot-Savart kernel: u=g(r/sigma)*(Gamma x r)/(4*pi*|r|^3).
velocity_field = vpm.ParticleField(3, Float64;
    viscous=vpm.Inviscid(), integration=vpm.rungekutta3,
    kernel=vpm.kernel_gaussianerf, UJ=vpm.UJ_direct)
x_source = [0.0, 0.0, 0.0]
gamma_source = [0.0, 0.0, 1.0]
sigma_source = 0.2
x_target = [1.0, 0.0, 0.0]
vpm.add_particle(velocity_field, x_source, gamma_source, sigma_source)
vpm.add_particle(velocity_field, x_target, zeros(3), sigma_source)
vpm.UJ_direct(velocity_field; reset=true)
r = x_target - x_source
u_reference = vpm.g_gauserf(norm(r)/sigma_source) .* cross(gamma_source, r) ./ (4pi*norm(r)^3)
u_actual = collect(vpm.get_U(velocity_field, 2))
velocity_error = norm(u_actual-u_reference)
velocity_error < 1e-12 || error("Direct UJ velocity reconstruction failed: $velocity_error")

Vinf(X, t) = [15.0, 0.0, 0.5]
wing = vlm.Wing(0.0, 0.0, 0.0, 1.0, 0.0)
vlm.addchord(wing, 0.0, 1.0, 0.0, 1.0, 0.0, 4; r=1.0)
system = vlm.WingSystem()
vlm.addwing(system, "Wing", wing)
vlm.setVinf(system, Vinf)
system.sol["Gamma"] = zeros(vlm.get_m(system))
horseshoe = vlm.getHorseshoe(system, 1; t=0.0)
length(horseshoe) == 8 || error("FLOWVLM registered-freestream horseshoe smoke test failed")

open(joinpath(OUT, "flowvpm_component_metrics.csv"), "w") do io
    println(io, "test,status,value,reference,absolute_error")
    println(io, "gaussianerf_radial_vorticity_normalization,PASS,$normalization,1.0,$(abs(normalization-1.0))")
    println(io, "gaussianerf_g_at_origin,PASS,$(vpm.g_gauserf(0.0)),0.0,$(abs(vpm.g_gauserf(0.0)))")
    println(io, "rk3_core_spreading_sigma_squared,PASS,$(sigma^2-sigma0^2),$(2nu*dt),$sigma2_error")
    println(io, "rk3_particle_position_constant_velocity,PASS,$(norm(x_actual-x0)),$(norm(x_expected-x0)),$position_error")
    println(io, "rk3_particle_circulation_zero_gradient,PASS,$(norm(collect(vpm.get_Gamma(advect_field,1))-gamma0)),0.0,$gamma_error")
    println(io, "direct_UJ_gaussianerf_velocity,PASS,$(norm(u_actual)),$(norm(u_reference)),$velocity_error")
    println(io, "flowvlm_registered_freestream,PASS,$(length(horseshoe)),8,0")
end
println("FLOWVPM kernel/core-spreading component checks passed")
println("  radial kernel normalization = $normalization")
println("  RK3 sigma^2 increment      = $(sigma^2-sigma0^2) (expected $(2nu*dt))")
println("  RK3 prescribed-velocity particle advance = PASS")
println("  FLOWVPM direct regularized velocity      = PASS")
println("  FLOWVLM stored freestream  = PASS")
