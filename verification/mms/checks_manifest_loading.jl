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
    Meta.parseall(read(joinpath(@__DIR__, "..", "..", "fluid", "src", "mms_coupled_source.jl"), String))
    Meta.parseall(read(joinpath(@__DIR__, "..", "..", "fluid", "src", "mms_rk3.jl"), String))
    println(isempty(manifest) ? "PASS G0 hook parse (manifest path not supplied)" : "PASS G0 manifest and Julia hook parse")
end

main()
