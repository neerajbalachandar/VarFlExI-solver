#!/usr/bin/env julia
"""Fixed-population component check for stage-aware rVPM position forcing."""

import FLOWVPM as vpm
using LinearAlgebra
include(joinpath(@__DIR__, "..", "..", "fluid", "src", "mms_rk3.jl"))

function make_field()
    pfield = vpm.ParticleField(8; Uinf=t -> [1.0, 0.0, 0.0],
                                   formulation=vpm.rVPM,
                                   kernel=vpm.gaussianerf,
                                   UJ=vpm.UJ_direct,
                                   integration=vpm.rungekutta3)
    vpm.add_particle(pfield, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.1)
    return pfield
end

pfield = make_field()
dt = 0.12
source_velocity = [0.0, 0.25, -0.125]
trace = NamedTuple[]
source! = (_particle, _time) -> source_velocity
rungekutta3_with_position_source(pfield, dt, source!; stage_trace=trace)
position = collect(vpm.get_X(vpm.get_particle(pfield, 1)))
expected = dt .* ([1.0, 0.0, 0.0] .+ source_velocity)
position_error = norm(position - expected)
length(trace) == 3 || error("expected one source evaluation per RK3 stage; got $(length(trace))")
all(isfinite, position) || error("non-finite particle position")
position_error <= 1e-12 || error("constant-source RK3 position error is $position_error")

native = make_field()
zero_source = (_particle, _time) -> zeros(3)
rungekutta3_with_position_source(native, dt, zero_source)
native_position = collect(vpm.get_X(vpm.get_particle(native, 1)))
reference = make_field()
vpm.rungekutta3(reference, dt)
source_disabled_error = norm(native.particles[:, 1:vpm.get_np(native)] -
                             reference.particles[:, 1:vpm.get_np(reference)])
source_disabled_error <= 1e-14 || error("zero-source path changed native RK3 state: $source_disabled_error")

println("status=PASS")
println("verification_level=fixed-population discrete particle-position component")
println("position_error=$position_error")
println("source_disabled_error=$source_disabled_error")
println("stage_times_s=$(join((row.time for row in trace), ','))")
