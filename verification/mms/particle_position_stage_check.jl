#!/usr/bin/env julia
"""Fixed-population component check for stage-aware rVPM position forcing."""

import FLOWVPM as vpm
using LinearAlgebra
include(joinpath(@__DIR__, "..", "..", "fluid", "src", "mms_rk3.jl"))

function make_field(integration=vpm.rungekutta3; nu=0.02, beta=100.0, gamma=[0.0, 0.0, 0.0])
    pfield = vpm.ParticleField(8; Uinf=t -> [1.0, 0.0, 0.0],
                                   formulation=vpm.rVPM,
                                   kernel=vpm.gaussianerf,
                                   UJ=vpm.UJ_direct,
                                   integration=integration,
                                   viscous=vpm.CoreSpreading(nu, 0.1, vpm.zeta_direct; beta=beta))
    vpm.add_particle(pfield, [0.0, 0.0, 0.0], gamma, 0.1)
    return pfield
end

function run_registered_adapter!(pfield, dt)
    pfield.integration(pfield, dt)
end

dt = 0.12
source_velocity = [0.0, 0.25, -0.125]
source_circulation = [0.03, -0.02, 0.04]
source_core = 0.05
trace = NamedTuple[]
source! = (_particle, _time) -> (
    position=source_velocity,
    circulation=source_circulation,
    core=source_core,
)
gamma0 = [0.2, -0.1, 0.05]
pfield = make_field(MMSRK3StateSource(source!, trace); nu=0.0, gamma=gamma0)
run_registered_adapter!(pfield, dt)
position = collect(vpm.get_X(vpm.get_particle(pfield, 1)))
expected = dt .* ([1.0, 0.0, 0.0] .+ source_velocity)
position_error = norm(position - expected)
gamma = collect(vpm.get_Gamma(vpm.get_particle(pfield, 1)))
circulation_error = norm(gamma - (gamma0 .+ dt .* source_circulation))
sigma = vpm.get_sigma(vpm.get_particle(pfield, 1))[]
core_error = abs(sigma - (0.1 + dt * source_core))
length(trace) == 3 || error("expected one source evaluation per RK3 stage; got $(length(trace))")
all(isfinite, position) || error("non-finite particle position")
position_error <= 1e-12 || error("constant-source RK3 position error is $position_error")
circulation_error <= 1e-12 || error("constant-source RK3 circulation error is $circulation_error")
core_error <= 1e-12 || error("constant-source RK3 core-width error is $core_error")

zero_source = (_particle, _time) -> zeros(3)
native = make_field(MMSRK3StateSource(zero_source, NamedTuple[]); beta=1.0, gamma=gamma0)
run_registered_adapter!(native, dt)
native_position = collect(vpm.get_X(vpm.get_particle(native, 1)))
reference = make_field(; beta=1.0, gamma=gamma0)
vpm.rungekutta3(reference, dt)
source_disabled_error = norm(native.particles[:, 1:vpm.get_np(native)] -
                             reference.particles[:, 1:vpm.get_np(reference)])
source_disabled_error <= 1e-14 || error("zero-source path changed native RK3 state: $source_disabled_error")
native_sigma = vpm.get_sigma(vpm.get_particle(native, 1))[]
reference_sigma = vpm.get_sigma(vpm.get_particle(reference, 1))[]
core_spreading_error = abs(native_sigma - reference_sigma)
core_spreading_error <= 1e-14 || error("zero-source path changed core spreading: $core_spreading_error")
core_reset_error = abs(reference.viscous.t_sgm)
core_reset_error <= 1e-14 || error("test did not exercise the native core reset: $core_reset_error")

println("status=PASS")
println("verification_level=fixed-population discrete particle-state RK3 component with native core spreading")
println("position_error=$position_error")
println("circulation_error=$circulation_error")
println("core_source_error=$core_error")
println("source_disabled_error=$source_disabled_error")
println("core_spreading_error=$core_spreading_error")
println("core_reset_error=$core_reset_error")
println("stage_times_s=$(join((row.time for row in trace), ','))")
