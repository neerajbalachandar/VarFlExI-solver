function parse_cli()
    fluid_dir = normpath(joinpath(@__DIR__, ".."))
    final_dir = normpath(joinpath(fluid_dir, ".."))
    defaults = Dict(
        "fluid" => joinpath(final_dir, "config", "fluid_params.yaml"),
        "solid" => joinpath(final_dir, "config", "solid_params.yaml"),
        "coupling" => joinpath(final_dir, "config", "coupling_params.yaml"),
    )

    # The MMS options are optional; use an Any-valued dictionary so `nothing`
    # remains a valid sentinel alongside the string paths above.
    args = Dict{String,Any}(defaults)
    args["mms_target"] = nothing
    args["mms_output"] = nothing
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg in ("--fluid", "--fluid-config")
            i += 1
            i <= length(ARGS) || error("$(arg) requires a path")
            args["fluid"] = ARGS[i]
        elseif arg in ("--solid", "--solid-config")
            i += 1
            i <= length(ARGS) || error("$(arg) requires a path")
            args["solid"] = ARGS[i]
        elseif arg in ("--coupling", "--coupling-config")
            i += 1
            i <= length(ARGS) || error("$(arg) requires a path")
            args["coupling"] = ARGS[i]
        elseif arg in ("--mms-target", "--mms-output")
            i += 1
            i <= length(ARGS) || error("$(arg) requires a path")
            args[arg == "--mms-target" ? "mms_target" : "mms_output"] = ARGS[i]
        elseif arg in ("-h", "--help")
            println("Usage: julia FINAL/fluid/fluid.jl [--fluid PATH] [--solid PATH] [--coupling PATH] [--mms-target PATH] [--mms-output PATH]")
            exit(0)
        else
            error("Unknown argument: $(arg)")
        end
        i += 1
    end

    return args
end
