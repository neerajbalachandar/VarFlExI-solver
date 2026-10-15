# using Pkg

# Pkg.activate(joinpath(@__DIR__,"..", "..", ".."))
# Pkg.instantiate()

using Sockets
using JSON
using LinearAlgebra

import FLOWUnsteady as uns
import FLOWVLM as vlm
import FLOWVPM as vpm


include("src/FluidTypes.jl")
include("src/FluidConfig.jl")
include("src/cli.jl")

include("src/vlm_patch.jl")
include("src/math_utils.jl")
include("src/interpolation.jl")
include("src/eta_mapping.jl")
include("src/socket.jl")
include("src/payload_decoder.jl")
include("src/geom_update.jl")
include("src/geom_receiver.jl")
include("src/wing_geom.jl")
include("src/runtime_callback.jl")

args = parse_cli()

solver_cfg, wake_cfg, coupling_cfg = load_configs(
    args["fluid"],
    args["solid"],
    args["coupling"],
)

include("src/setup.jl")
include("src/execute.jl")
