#!/usr/bin/env julia
# Exercise pinned FLOWVPM read! overwrite/append behavior on two snapshots.
using FLOWUnsteady
const vpm = FLOWUnsteady.vpm

length(ARGS) == 2 || error("Usage: test_vpm_read_append.jl dynamic_snapshot.h5 static_snapshot.h5")
dynamic_path, static_path = abspath.(ARGS)
all(isfile, (dynamic_path, static_path)) || error("Both fixture snapshots must exist")
read_meta(path) = vpm.HDF5.h5open(path, "r") do h5
    (np=Int(vpm.HDF5.read(h5["np"])), t=Float64(vpm.HDF5.read(h5["t"])))
end
dynamic = read_meta(dynamic_path)
static = read_meta(static_path)
isapprox(dynamic.t, static.t; rtol=0, atol=1e-12) || error("Fixture times differ")
field = vpm.ParticleField(dynamic.np + static.np + 1, Float64; useGPU=0)
vpm.read!(field, basename(dynamic_path); path=dirname(dynamic_path), overwrite=true)
vpm.get_np(field) == dynamic.np || error("Dynamic overwrite load count mismatch")
all(!vpm.is_static(vpm.get_particle(field, i)) for i in 1:dynamic.np) ||
    error("Dynamic fixture unexpectedly contains static particles")
vpm.read!(field, basename(static_path); path=dirname(static_path), overwrite=false)
vpm.get_np(field) == dynamic.np + static.np || error("Static read! did not append to dynamic population")
all(vpm.is_static(vpm.get_particle(field, i)) for i in dynamic.np+1:dynamic.np+static.np) ||
    error("Static fixture particle flags were not preserved")
println("PASS: read!(overwrite=true) loads dynamic population and overwrite=false appends static population; ",
        "counts=$(dynamic.np)+$(static.np), t=$(dynamic.t)")
