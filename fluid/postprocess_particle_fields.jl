#!/usr/bin/env julia

# Reconstruct velocity and vorticity fields from FLOWVPM particle snapshots.
# Run from the repository root with the same Julia environment as fluid/fluid.jl:
#   julia fluid/postprocess_particle_fields.jl --input results/.../fluid --steps 0:10:300
# The input is the *_pfield.N.h5 particle data (XMF is its ParaView companion;
# wing/wake VTK files are visualization outputs, not particle-state inputs).

using JSON
using LinearAlgebra

import FLOWUnsteady as uns
import FLOWUnsteady: gt, vpm

include("src/FluidConfig.jl")

function parse_options(args)
    opts = Dict{String, String}(
        "input" => "",
        "output" => "",
        "steps" => "",
        "metadata" => "",
        "fluid-config" => joinpath(@__DIR__, "..", "config", "fluid_params.yaml"),
        "solid-config" => joinpath(@__DIR__, "..", "config", "solid_params.yaml"),
        "coupling-config" => joinpath(@__DIR__, "..", "config", "coupling_params.yaml"),
        "nx" => "32", "ny" => "32", "nz" => "32",
        "bounds" => "",
        "bounds-file" => "",
        "add-j" => "false",
        "add-wapprox" => "false",
        "max-particles" => "0",
        "include-static" => "true",
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            println("""
            Usage: julia fluid/postprocess_particle_fields.jl --input DIR [options]
              --input DIR              Directory containing PREFIX_pfield.N.h5
              --output DIR             Output directory (default: DIR/fluid_domain)
              --steps SPEC             e.g. 0:10:300, 1,2,3; default: all saved steps
              --metadata FILE          Select a specific _run_metadata/.../run_metadata.json
              --fluid-config FILE      Legacy-run fallback (no run manifest)
              --solid-config FILE      Legacy-run fallback
              --coupling-config FILE   Legacy-run fallback
              --nx N --ny N --nz N     Grid cell counts (default: 32,32,32)
              --bounds CSV             Fixed x0,x1,y0,y1,z0,z1 bounds (meters); default uses body-based bounds
              --bounds-file JSON      Per-step bounds map written by the analysis notebook
              --add-j BOOL             Save velocity Jacobian fields for Q and consistency checks
              --add-wapprox BOOL       Also save FLOWVPM zeta_fmm vorticity approximation as Wapprox (default: false)
              --max-particles N        Override particle-field capacity
              --include-static BOOL    Include PREFIX_staticpfield.N.h5 (default: true)
            """)
            exit(0)
        end
        startswith(arg, "--") || error("Unexpected argument: $(arg)")
        key = arg[3:end]
        haskey(opts, key) || error("Unknown option: $(arg)")
        i += 1
        i <= length(args) || error("$(arg) requires a value")
        opts[key] = args[i]
        i += 1
    end
    isempty(opts["input"]) && error("--input is required")
    return opts
end

function parse_steps(spec, available)
    isempty(spec) && return available
    steps = Int[]
    for part in split(spec, ',')
        fields = parse.(Int, split(strip(part), ':'))
        if length(fields) == 1
            push!(steps, fields[1])
        elseif length(fields) == 2
            append!(steps, fields[1]:fields[2])
        elseif length(fields) == 3
            fields[2] > 0 || error("Step range increment must be positive")
            append!(steps, fields[1]:fields[2]:fields[3])
        else
            error("Invalid step selector: $(part)")
        end
    end
    return sort!(unique(steps))
end

function find_metadata(input_dir, requested)
    if !isempty(requested)
        path = isabspath(requested) ? requested : joinpath(input_dir, requested)
        isfile(path) || error("Run metadata not found: $(path)")
        return JSON.parsefile(path), path
    end
    root = joinpath(input_dir, "_run_metadata")
    if isdir(root)
        candidates = String[]
        for (dir, _, files) in walkdir(root)
            "run_metadata.json" in files && push!(candidates, joinpath(dir, "run_metadata.json"))
        end
        if !isempty(candidates)
            sort!(candidates; by=p -> basename(dirname(p)))
            path = last(candidates)
            return JSON.parsefile(path), path
        end
    end
    return nothing, ""
end

function setting(config, key, default)
    return haskey(config, key) && config[key] !== nothing ? config[key] : default
end

function read_legacy_configs(opts)
    fluid = load_simple_yaml(opts["fluid-config"])
    solid = load_simple_yaml(opts["solid-config"])
    coupling = load_simple_yaml(opts["coupling-config"])
    nsteps = Int(setting(coupling, "n_steps", setting(fluid, "nsteps", 1000)))
    dt = Float64(setting(coupling, "total_time", setting(fluid, "total_time", 5.0))) / nsteps
    particles_per_step = Int(setting(fluid, "particles_per_step", 1))
    span_stations = Int(setting(fluid, "n_span", setting(coupling, "communication_span", 80)))
    estimated_particles = ceil(Int, (nsteps + 1) * (span_stations * (particles_per_step + 1) + particles_per_step))
    estimated_capacity = ceil(Int, Float64(setting(fluid, "max_particles_safety_factor", 3.0)) * estimated_particles)
    max_particles = max(Int(setting(fluid, "max_particles", 400000)), estimated_capacity)
    fmm = Dict{String, Any}(
        "p" => setting(fluid, "p", 4), "ncrit" => setting(fluid, "ncrit", 50),
        "theta" => setting(fluid, "theta", 0.4),
        "shrink_recenter" => setting(fluid, "shrink_recenter", true),
        "relative_tolerance" => setting(fluid, "relative_tolerance", 1.0e-3),
        "absolute_tolerance" => setting(fluid, "absolute_tolerance", 1.0e-3),
        "autotune_p" => setting(fluid, "autotune_p", true),
        "autotune_ncrit" => setting(fluid, "autotune_ncrit", true),
        "autotune_regularization" => setting(fluid, "autotune_regularization", false),
        "default_rho_over_sigma" => setting(fluid, "default_rho_over_sigma", 1.0),
        "minimum_ncrit" => setting(fluid, "minimum_ncrit", 3),
    )
    return Dict{String, Any}(
        "run_name" => setting(fluid, "run_name", "fluid"),
        "effective" => Dict{String, Any}(
            "dt" => dt,
            "rho" => setting(fluid, "rho", 1.0), "nu" => setting(fluid, "nu", 1.0e-6),
            "vinf" => setting(fluid, "vinf", 8.0), "aoa_deg" => setting(fluid, "aoa_deg", 8.0),
            "span" => setting(solid, "span", setting(fluid, "span", 0.8)),
            "root_chord" => setting(solid, "root_chord", setting(fluid, "root_chord", 0.12)),
            "tip_chord" => setting(solid, "tip_chord", setting(fluid, "tip_chord", 0.12)),
            "leading_edge_sweep" => setting(solid, "leading_edge_sweep", setting(fluid, "leading_edge_sweep", 0.0)),
            "twist_root" => setting(fluid, "twist_root", 0.0),
            "twist_tip" => setting(fluid, "twist_tip", 0.0),
            "n_span" => span_stations, "p_per_step" => particles_per_step,
            "max_particles" => max_particles,
            "dynamic_particle_prefix" => string(setting(fluid, "run_name", "fluid"), "_pfield"),
            "static_particle_prefix" => string(setting(fluid, "run_name", "fluid"), "_staticpfield"),
            "fmm" => fmm,
            "freestream" => Dict{String, Any}(
                "gradient" => setting(fluid, "freestream_gradient", zeros(9)),
                "oscillation" => setting(fluid, "freestream_oscillation", zeros(3)),
                "wavenumber" => setting(fluid, "freestream_wavenumber", zeros(3)),
                "frequency_hz" => setting(fluid, "freestream_frequency_hz", 0.0),
                "phase_rad" => deg2rad(setting(fluid, "freestream_phase_deg", 0.0)),
                "reference" => setting(fluid, "freestream_reference", zeros(3)),
            ),
        ),
    )
end

function load_run_settings(opts, input_dir)
    metadata, metadata_path = find_metadata(input_dir, opts["metadata"])
    if metadata === nothing
        @warn "No run metadata found. Using the explicitly selected/current config files; these may not describe historical snapshots."
        metadata = read_legacy_configs(opts)
        return metadata, "config fallback (unverified against historical run)"
    end
    return metadata, metadata_path
end

function freestream_at(X, t, effective)
    fs = effective["freestream"]
    reference = Float64.(fs["reference"])
    gradient_values = Float64.(fs["gradient"])
    length(gradient_values) == 9 || error("Saved freestream gradient must have 9 entries")
    gradient = permutedims(reshape(gradient_values, 3, 3))
    amplitudes = Float64.(fs["oscillation"])
    wavenumbers = Float64.(fs["wavenumber"])
    length(amplitudes) == length(wavenumbers) == length(reference) == 3 ||
        error("Saved freestream vectors must have three components")
    x, y, z = X .- reference
    kx, ky, kz = wavenumbers
    phase = 2pi * Float64(fs["frequency_hz"]) * t + Float64(fs["phase_rad"])
    gust = amplitudes .* sin.([phase + ky * y, phase + kz * z, phase + kx * x])
    base_speed = Float64(effective["vinf"])
    aoa = deg2rad(Float64(effective["aoa_deg"]))
    return base_speed .* [cos(aoa), 0.0, sin(aoa)] + gradient * (X .- reference) + gust
end

function make_fmm(data)
    return vpm.FMM(
        p=Int(data["p"]), ncrit=Int(data["ncrit"]), theta=Float64(data["theta"]),
        shrink_recenter=Bool(data["shrink_recenter"]),
        relative_tolerance=Float64(data["relative_tolerance"]),
        absolute_tolerance=Float64(data["absolute_tolerance"]),
        autotune_p=Bool(data["autotune_p"]), autotune_ncrit=Bool(data["autotune_ncrit"]),
        autotune_reg_error=Bool(data["autotune_regularization"]),
        default_rho_over_sigma=Float64(data["default_rho_over_sigma"]),
        min_ncrit=Int(data["minimum_ncrit"]),
    )
end

function main()
    opts = parse_options(ARGS)
    input_dir = abspath(opts["input"])
    isdir(input_dir) || error("Input directory not found: $(input_dir)")
    metadata, provenance = load_run_settings(opts, input_dir)
    effective = metadata["effective"]
    run_name = String(metadata["run_name"])
    dynamic_prefix = String(get(effective, "dynamic_particle_prefix", run_name * "_pfield"))
    static_prefix = String(get(effective, "static_particle_prefix", run_name * "_staticpfield"))
    dynamic_files = [f for f in readdir(input_dir) if endswith(f, ".h5") && occursin("_pfield.", f) && !occursin("_staticpfield.", f)]
    if !isempty(dynamic_files) && !any(startswith(f, dynamic_prefix * ".") for f in dynamic_files)
        inferred_prefix = first(split(first(dynamic_files), r"\.\d+\.h5$"))
        if !haskey(metadata, "config_snapshots")
            inferred_name = replace(inferred_prefix, r"_pfield$" => "")
            inferred_name == run_name || @warn "Legacy run config name differs from stored particle prefix; using stored prefix but other config values may also be from another run." config_run=run_name data_run=inferred_name
            run_name = inferred_name
            dynamic_prefix = inferred_prefix
            static_prefix = inferred_name * "_staticpfield"
        end
    end
    step_pattern = Regex("^" * dynamic_prefix * raw"\.(\d+)\.h5$")
    available = sort!(parse.(Int, [m.captures[1] for f in readdir(input_dir)
        for m in eachmatch(step_pattern, f)]))
    isempty(available) && error("No $(dynamic_prefix).N.h5 particle files found in $(input_dir)")
    steps = parse_steps(opts["steps"], available)
    missing = setdiff(steps, available)
    isempty(missing) || error("Requested particle steps are absent: $(missing)")
    bounds_by_step = isempty(opts["bounds-file"]) ? nothing : JSON.parsefile(abspath(opts["bounds-file"]))
    if bounds_by_step !== nothing
        missing_bounds = [step for step in steps if !haskey(bounds_by_step, string(step))]
        isempty(missing_bounds) || error("Bounds file lacks requested steps: $(missing_bounds)")
    end
    # Preserve the physical time recorded in the XMF companion when present.
    # Older writers sometimes stored integer step numbers here; that case is
    # detected and corrected using the saved run dt below.
    snapshot_times = Dict{Int,Float64}()
    for step in steps
        xmf = joinpath(input_dir, "$(dynamic_prefix).$(step).xmf")
        if isfile(xmf)
            match_time = match(r"<Time\s+Value=\"([^\"]+)\"", read(xmf, String))
            match_time === nothing || (snapshot_times[step] = parse(Float64, match_time.captures[1]))
        end
    end
    if haskey(metadata, "created_unix")
        stale = [step for step in steps if stat(joinpath(input_dir, "$(dynamic_prefix).$(step).h5")).mtime < Float64(metadata["created_unix"])]
        isempty(stale) || @warn "Some selected particle files predate the chosen run manifest. They may belong to another run; select matching steps or pass --metadata explicitly." stale_steps=stale metadata=provenance
    end

    output_dir = isempty(opts["output"]) ? joinpath(input_dir, "fluid_domain") : abspath(opts["output"])
    mkpath(output_dir)
    span = Float64(effective["span"])
    chord = max(Float64(effective["root_chord"]), Float64(effective["tip_chord"]))
    if isempty(opts["bounds"])
        pmin = [-chord, -0.1span, -0.5span]
        pmax = [4span, 1.1span, 0.5span]
    else
        bounds = parse.(Float64, split(opts["bounds"], ','))
        length(bounds) == 6 || error("--bounds must contain x0,x1,y0,y1,z0,z1")
        pmin = [bounds[1], bounds[3], bounds[5]]
        pmax = [bounds[2], bounds[4], bounds[6]]
        all(pmax .> pmin) || error("Every --bounds maximum must exceed its minimum")
    end
    divisions = [parse(Int, opts["nx"]), parse(Int, opts["ny"]), parse(Int, opts["nz"])]
    all(>(0), divisions) || error("Grid divisions must all be positive")
    node_count = prod(divisions .+ 1)
    fmm = make_fmm(effective["fmm"])
    capacity_setting = parse(Int, opts["max-particles"])
    capacity = capacity_setting > 0 ? capacity_setting : Int(effective["max_particles"]) + node_count
    capacity > node_count || error("Particle capacity must exceed the grid node count")
    pfield = vpm.ParticleField(capacity; UJ=vpm.UJ_fmm, fmm=fmm)
    include_static = lowercase(opts["include-static"]) in ("true", "1", "yes")
    dt = Float64(effective["dt"])

    println("Run metadata: $(provenance)")
    println("Particle data: $(input_dir)/$(dynamic_prefix).N.h5")
    println("Steps: $(first(steps)) through $(last(steps)) ($(length(steps)) snapshots)")
    println("Grid nodes: $(node_count); cells: $(divisions); bounds mode: ", bounds_by_step === nothing ? "fixed" : "per-step cumulative")
    println("Static particles requested: $(include_static); output: $(output_dir)")

    for step in steps
        if bounds_by_step !== nothing
            bounds = Float64.(bounds_by_step[string(step)])
            length(bounds) == 6 || error("Bounds entry for step $(step) must contain x0,x1,y0,y1,z0,z1")
            pmin = [bounds[1], bounds[3], bounds[5]]
            pmax = [bounds[2], bounds[4], bounds[6]]
            all(pmax .> pmin) || error("Invalid per-step bounds at step $(step)")
        end
        grid = gt.Grid(pmin, pmax, divisions)
        println("Grid bounds at step $(step): $(pmin) to $(pmax)")
        flush(stdout)
        xmf_time = get(snapshot_times, step, NaN)
        # The repository's legacy FLOWUnsteady XMF writer stored integer step
        # numbers as times. visualization_metadata.jl documents this behavior.
        # Correct only that recognizable legacy case; otherwise preserve XMF.
        legacy_step_time = isfinite(xmf_time) && dt != 1.0 && isapprox(xmf_time, Float64(step); atol=1e-10, rtol=0)
        time = legacy_step_time ? step * dt : (isfinite(xmf_time) ? xmf_time : step * dt)
        if legacy_step_time
            @warn "XMF contains the legacy integer step timestamp; converting using run dt" step xmf_time step_dt=dt corrected_time=time
        elseif isfinite(xmf_time) && abs(time - step * dt) > max(1e-8, 1e-5 * max(abs(time), abs(step * dt)))
            @warn "XMF timestamp differs from step*dt; preserving XMF timestamp" step xmf_time step_dt=step*dt
        end
        println("Processing step $(step) at t=$(time) s")
        flush(stdout)
        vpm.read!(pfield, "$(dynamic_prefix).$(step).h5"; path=input_dir, overwrite=true)
        static_file = joinpath(input_dir, "$(static_prefix).$(step).h5")
        if include_static && isfile(static_file)
            vpm.read!(pfield, basename(static_file); path=input_dir, overwrite=false)
        elseif include_static
            @warn "Static particle snapshot missing at step $(step); processing dynamic particles only" file=static_file
        end

        uns.computefluiddomain(pfield, [grid];
            add_Uinf=false, add_J=lowercase(opts["add-j"]) in ("true", "1", "yes"),
            add_Wapprox=lowercase(opts["add-wapprox"]) in ("true", "1", "yes"),
            # The installed utilities_fluiddomain.jl references an undefined
            # timing variable in verbose messages. The outer loop logs progress.
            save_path=nothing, verbose=false,
        )
        velocity = grid.field["U"]["field_data"]
        induced_velocity = copy(velocity)
        gt.add_field(grid, "Uinduced", "vector", induced_velocity, "node")
        ambient_velocity = zeros(size(velocity))
        for node in 1:grid.nnodes
            ambient_velocity[:, node] .= freestream_at(view(grid.nodes, 1:3, node), time, effective)
        end
        gt.add_field(grid, "Uinf", "vector", ambient_velocity, "node")
        velocity .+= ambient_velocity
        gt.save(grid, "$(run_name)_fluiddomain"; path=output_dir, num=step, time=time)
    end
    println("Done. XMF/H5 grids containing U (total velocity) and W (particle-induced vorticity) are in $(output_dir).")
end

main()
