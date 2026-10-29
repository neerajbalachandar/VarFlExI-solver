# Fixed-population coupled discrete-model MMS; Objective B structural source
# and fixed-population particle source. Formulation sections: "The numerical
# staggered socket-coupled system" and "Required output and pass/fail semantics".
# The production socket smoke and source/history assertions are orchestrated
# by checks_manifest_loading.py; this Julia gate validates manifest and hook
# syntax in the installed production language runtime.
using JSON

function main(args=ARGS)
    manifest = !isempty(args) ? args[1] : get(ENV, "VARFLEXI_MMS_TARGET", "")
    if !isempty(manifest)
        data = JSON.parsefile(manifest)
        @assert haskey(data, "particles") && haskey(data, "structural")
        @assert length(data["particles"]["ids"]) > 0
    end
    root = normpath(joinpath(@__DIR__, "..", ".."))
    implementation_files = (
        "fluid/src/mms_coupled_source.jl",
        "fluid/src/mms_rk3.jl",
        "fluid/src/runtime_callback.jl",
        "fluid/src/setup.jl",
    )
    for relative in implementation_files
        path = joinpath(root, relative)
        @assert isfile(path) "Missing production MMS implementation source: $path"
        source = read(path, String)
        Meta.parseall(source)
        if relative == "fluid/src/mms_coupled_source.jl"
            @assert occursin("make_coupled_mms_source", source)
            @assert occursin("update_particle_states", source)
            @assert occursin("viscousdiffusion", source)
        elseif relative == "fluid/src/runtime_callback.jl"
            @assert occursin("solve_mms_target_aerodynamics!", source)
            @assert occursin("target_circulation_residual", source)
        elseif relative == "fluid/src/setup.jl"
            @assert occursin("MMSRK3StateSource", source)
        end
    end
    println("Parsed manifest and inspected current production MMS source hooks; this is not a coupled-run result")
end

main()
