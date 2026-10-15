function parse_scalar(raw::AbstractString)
    value = strip(split(raw, "#"; limit=2)[1])
    isempty(value) && return ""
    if value in ("null", "~")
        return nothing
    elseif lowercase(value) in ("true", "false")
        return lowercase(value) == "true"
    elseif startswith(value, "\"") && endswith(value, "\"")
        return value[2:end-1]
    elseif startswith(value, "'") && endswith(value, "'")
        return value[2:end-1]
    elseif startswith(value, "[") && endswith(value, "]")
        body = strip(value[2:end-1])
        isempty(body) && return Any[]
        return [parse_scalar(part) for part in split(body, ",")]
    end

    try
        return parse(Int, value)
    catch
    end
    try
        return parse(Float64, value)
    catch
    end
    return value
end

function load_simple_yaml(path::AbstractString)
    isfile(path) || error("Config file not found: $(path)")
    out = Dict{String, Any}()
    for line in eachline(path)
        stripped = strip(line)
        (isempty(stripped) || startswith(stripped, "#")) && continue
        parts = split(line, ":"; limit=2)
        length(parts) == 2 || continue
        key = strip(parts[1])
        isempty(key) && continue
        value = parse_scalar(parts[2])
        if haskey(out, key)
            if out[key] isa Vector
                push!(out[key], value)
            else
                out[key] = Any[out[key], value]
            end
        else
            out[key] = value
        end
    end
    return out
end

function cfg_get(cfg::Dict{String, Any}, key::String, default; duplicate::Symbol=:last)
    if !haskey(cfg, key) || cfg[key] === nothing
        return default
    end
    value = cfg[key]
    if value isa Vector && !(value isa AbstractString)
        return duplicate == :first ? first(value) : last(value)
    end
    return value
end

as_float(cfg, key, default; duplicate::Symbol=:last) = Float64(cfg_get(cfg, key, default; duplicate=duplicate))
as_int(cfg, key, default; duplicate::Symbol=:last) = Int(cfg_get(cfg, key, default; duplicate=duplicate))
as_bool(cfg, key, default; duplicate::Symbol=:last) = Bool(cfg_get(cfg, key, default; duplicate=duplicate))
as_string(cfg, key, default; duplicate::Symbol=:last) = String(cfg_get(cfg, key, default; duplicate=duplicate))

function choose_viscous_scheme(raw)
    value = lowercase(strip(String(raw)))
    if startswith(value, "corespreading")
        return "CoreSpreading"
    elseif value in ("none", "nothing", "false", "off")
        return "None"
    end
    @warn "Unknown viscous_scheme=$(raw); defaulting to CoreSpreading"
    return "CoreSpreading"
end

function load_configs(fluid_path::AbstractString, solid_path::AbstractString, coupling_path::AbstractString)
    fluid = load_simple_yaml(fluid_path)
    solid = load_simple_yaml(solid_path)
    coupling = load_simple_yaml(coupling_path)

    ttot = as_float(coupling, "total_time", as_float(fluid, "total_time", 5.0))
    nsteps = as_int(coupling, "n_steps", as_int(fluid, "nsteps", 1000))
    dt = ttot / nsteps

    vinf = as_float(fluid, "vinf", 8.0)
    p_per_step = as_int(fluid, "particles_per_step", 1.0)
    lambda_vpm = as_float(fluid, "lambda_vpm", 2.0)
    span = as_float(solid, "span", as_float(fluid, "span", 0.8))

    solver_cfg = SolverConfig(
        as_string(fluid, "run_name", "fluid"),
        as_float(fluid, "aoa_deg", 8.0),
        vinf,
        as_float(fluid, "rho", 1.0),
        as_float(fluid, "nu", 1.0e-6),
        ttot,
        nsteps,
        dt,
        p_per_step,
        lambda_vpm,
        as_float(fluid, "relaxation", 0.35),
        as_float(fluid, "sigma_solver", -1.0),
        as_float(fluid, "sigma_surface", 0.05),
        as_int(fluid, "n_span", as_int(coupling, "communication_span", 80)),
        as_int(fluid, "n_chord", 1),
        as_float(fluid, "eta_cp", 0.75),
        as_float(fluid, "eta_bv", 0.25),
        span,
        as_float(solid, "root_chord", as_float(fluid, "root_chord", 0.12)),
        as_float(solid, "tip_chord", as_float(fluid, "tip_chord", 0.12)),
        as_float(solid, "leading_edge_sweep", as_float(fluid, "leading_edge_sweep", 0.0)),
        as_float(fluid, "twist_root", 0.0),
        as_float(fluid, "twist_tip", 0.0),
        as_float(fluid, "dihedral", 0.0),
        as_float(fluid, "x", 1.0),
        as_float(fluid, "y", 1.0),
        as_float(fluid, "z", 1.0),
        as_string(fluid, "save_directory", "results/fluid"),
        as_bool(fluid, "create_savepath", as_bool(fluid, "create_directory", true)),
        as_bool(fluid, "prompt", false),
        as_int(fluid, "nsteps_save", as_int(fluid, "save_every", 1)),
        as_bool(fluid, "save_horseshoes", true),
        as_bool(fluid, "save_geometry_csv", true),
        as_bool(fluid, "use_ftot_force", true),
        lowercase(get(ENV, "COUPLING_DEBUG_IO", "0")) ∉ ("0", "false", "no"),
        as_bool(fluid, "regularize_vlm", true),
        as_int(fluid, "max_particles", 400000),
        as_float(fluid, "max_particles_safety_factor", 3.0),
    )

    wake_cfg = WakeConfig(
        as_int(fluid, "remove_every", 100),
        as_float(fluid, "sphere_factor", 200.0),
        as_float(fluid, "min_factor", 1.0e-8; duplicate=:first),
        as_float(fluid, "max_factor", 50.0; duplicate=:first),
        as_float(fluid, "min_factor", 1.0e-3; duplicate=:last),
        as_float(fluid, "max_factor", 50.0; duplicate=:last),
        as_bool(fluid, "start_with_wake", true),
        as_bool(fluid, "unsteady", true),
        as_float(fluid, "unsteady_criterion", 0.0),
        as_bool(fluid, "wake_coupled", true),
        as_int(fluid, "p", 4),
        as_int(fluid, "ncrit", 50),
        as_float(fluid, "theta", 0.4),
        as_bool(fluid, "shrink_recenter", true),
        as_float(fluid, "relative_tolerance", 1.0e-3),
        as_float(fluid, "absolute_tolerance", 1.0e-3),
        as_bool(fluid, "autotune_p", true),
        as_bool(fluid, "autotune_ncrit", true),
        as_bool(fluid, "autotune_regularization", false),
        as_float(fluid, "default_rho_over_sigma", 1.0),
        as_int(fluid, "minimum_ncrit", 3),
        choose_viscous_scheme(cfg_get(fluid, "viscous_scheme", "CoreSpreading")),
    )

    custom_stride = cfg_get(coupling, "custom_span_stride", nothing)
    coupling_cfg = CouplingConfig(
        as_string(coupling, "host", "127.0.0.1"),
        as_int(coupling, "port", 9000),
        solver_cfg.n_span,
        as_int(solid, "ny", 240),
        lowercase(strip(as_string(coupling, "span_sampling_mode", "node-stride"))),
        custom_stride === nothing ? nothing : Int(custom_stride),
        as_float(coupling, "geom_relax", 1.0),
        as_float(coupling, "force_relax", 1.0),
    )

    return solver_cfg, wake_cfg, coupling_cfg
end
