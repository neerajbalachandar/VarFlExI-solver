using FLOWUnsteady
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
    println(io, "flowvlm_registered_freestream,PASS,$(length(horseshoe)),8,0")
end
println("FLOWVPM kernel/core-spreading component checks passed")
println("  radial kernel normalization = $normalization")
println("  RK3 sigma^2 increment      = $(sigma^2-sigma0^2) (expected $(2nu*dt))")
println("  FLOWVLM stored freestream  = PASS")
