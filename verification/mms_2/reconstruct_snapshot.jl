#!/usr/bin/env julia

"""Reconstruct a complete saved FLOWVPM particle snapshot onto a Cartesian grid.

Manifest schema: `run_metadata_path`, `particle_directory`,
`required_population_names`, and an ordered `populations` array. Each entry
contains name, role, file_prefix, read_path, required, expected_count, and
saved_metadata_count. Dynamic must be first and unique; all other populations
must be explicitly declared static.
"""

using FLOWUnsteady
using JSON
const vpm = FLOWUnsteady.vpm
const gt = FLOWUnsteady.gt
include(joinpath(@__DIR__, "reconstruct_fields.jl"))

function arguments(args)
    out = Dict{String,String}()
    i = 1
    while i <= length(args)
        key = args[i]
        key in ("--manifest", "--snapshot", "--grid-config", "--output") ||
            error("Unknown argument $key")
        i < length(args) || error("Missing value for $key")
        out[key] = args[i+1]
        i += 2
    end
    all(haskey(out, k) for k in ("--manifest", "--snapshot", "--grid-config", "--output")) ||
        error("Usage: reconstruct_snapshot.jl --manifest FILE --snapshot N --grid-config YAML --output DIR")
    return out
end

readjson(path) = JSON.parsefile(path)

function run_manifest(path, snapshot)
    manifest = readjson(path)
    for key in ("snapshot", "physical_time_s", "run_metadata_path", "particle_directory",
                "required_population_names", "populations")
        haskey(manifest, key) || error("Manifest is missing required field '$key'")
    end
    Int(manifest["snapshot"]) == snapshot ||
        error("Manifest snapshot does not match --snapshot")
    metadata_path = abspath(String(manifest["run_metadata_path"]))
    isfile(metadata_path) || error("Production run metadata does not exist: $metadata_path")
    particle_dir = abspath(String(manifest["particle_directory"]))
    isdir(particle_dir) || error("Particle snapshot directory does not exist: $particle_dir")
    runmeta = readjson(metadata_path)
    eff = runmeta["effective"]
    dyn_prefix = String(eff["dynamic_particle_prefix"])
    stat_prefix = String(eff["static_particle_prefix"])
    required_names = String.(manifest["required_population_names"])
    entries = [Dict{String,Any}(String(k)=>v for (k,v) in e) for e in manifest["populations"]]
    isempty(entries) && error("Manifest has no required particle populations")
    names = String.(getindex.(entries, "name"))
    roles = String.(getindex.(entries, "role"))
    prefixes = String.(getindex.(entries, "file_prefix"))
    length(unique(names)) == length(names) || error("Duplicate population identifiers")
    length(unique(required_names)) == length(required_names) || error("Duplicate required population identifiers")
    Set(names) == Set(required_names) || error("Manifest population inventory is incomplete or contains extras")
    Set(required_names) == Set(("dynamic", "static_or_bound")) || error(
        "Production run metadata schema requires dynamic and static_or_bound populations")
    count(==("dynamic"), roles) == 1 && first(findall(==("dynamic"), roles)) == 1 ||
        error("Exactly one dynamic population must be listed first")
    Set([names[i] for i in eachindex(names) if roles[i] == "static"]) == Set(("static_or_bound",)) ||
        error("Every static/bound population must be explicitly named static_or_bound")
    for (entry, role, prefix) in zip(entries, roles, prefixes)
        role in ("dynamic", "static") || error("Unexpected particle role '$role'")
        Bool(entry["required"]) || error("Optional populations are not allowed in a complete-field manifest")
        expected_prefix = role == "dynamic" ? dyn_prefix : stat_prefix
        prefix == expected_prefix || error("Manifest prefix '$prefix' does not match production metadata '$expected_prefix'")
        Int(entry["expected_count"]) == Int(entry["saved_metadata_count"]) ||
            error("Expected and saved counts differ for $(entry["name"])")
        raw_path = String(entry["read_path"])
        resolved_path = isabspath(raw_path) ? raw_path : joinpath(particle_dir, raw_path)
        isdir(resolved_path) || error("Population read_path is not a directory: $resolved_path")
        entry["read_path"] = abspath(resolved_path)
    end
    files = [joinpath(String(e["read_path"]), String(e["file_prefix"]) * ".$(snapshot).h5") for e in entries]
    length(unique(normpath.(files))) == length(files) || error("Duplicate snapshot file in manifest")
    all(isfile, files) || error("A required population snapshot file is missing: " * join(filter(f -> !isfile(f), files), ", "))
    length(unique(realpath.(files))) == length(files) || error("Population entries resolve to duplicate snapshot files")
    return manifest, runmeta, entries, files
end

function make_freestream(eff)
    fs = eff["freestream"]
    grad = Float64.(fs["gradient"])
    amp = Float64.(fs["oscillation"])
    wave = Float64.(fs["wavenumber"])
    reference = Float64.(fs["reference"])
    length(grad) == 9 && length(amp) == 3 && length(wave) == 3 && length(reference) == 3 ||
        error("Invalid freestream metadata dimensions")
    any(!iszero, grad) && error("Spatially varying freestream gradient is unsupported")
    any(!iszero, wave) && error("Spatially varying freestream wavenumber is unsupported")
    speed, aoa = Float64(eff["vinf"]), deg2rad(Float64(eff["aoa_deg"]))
    freq, phase = Float64(fs["frequency_hz"]), Float64(fs["phase_rad"])
    return t -> [speed*cos(aoa), 0.0, speed*sin(aoa)] .+ amp .* sin.(2pi*freq*t + phase)
end

function make_fmm(d)
    vpm.FMM(p=Int(d["p"]), ncrit=Int(d["ncrit"]), theta=Float64(d["theta"]),
        shrink_recenter=Bool(d["shrink_recenter"]),
        relative_tolerance=Float64(d["relative_tolerance"]),
        absolute_tolerance=Float64(d["absolute_tolerance"]),
        autotune_p=Bool(d["autotune_p"]), autotune_ncrit=Bool(d["autotune_ncrit"]),
        autotune_reg_error=Bool(d["autotune_regularization"]),
        default_rho_over_sigma=Float64(d["default_rho_over_sigma"]),
        min_ncrit=Int(d["minimum_ncrit"]))
end

function new_field(capacity, runmeta)
    eff = runmeta["effective"]
    fmm = make_fmm(eff["fmm"])
    Uinf = make_freestream(eff)
    return vpm.ParticleField(capacity, Float64; viscous=vpm.Inviscid(), SFS=vpm.SFS_none,
        fmm=fmm, Uinf=Uinf, useGPU=0)
end

function load_into!(pfield, snapshot, entries, files)
    manifest = Dict{String,Any}(
        "snapshot"=>snapshot,
        "required_population_names"=>String.(getindex.(entries, "name")),
        "populations"=>entries)
    # Loader derives snapshot paths from read_path/prefix; ensure supplied paths
    # match the checked paths before allowing vpm.read! to access them.
    for (e, file) in zip(entries, files)
        expected = joinpath(String(e["read_path"]), String(e["file_prefix"]) * ".$(snapshot).h5")
        abspath(expected) == file || error("Snapshot path changed during manifest validation")
    end
    return load_complete_particle_snapshot!(pfield, snapshot, manifest)
end

function python_reference(bounds, cells, time_s, fluid_config)
    request = (bounds_m=[collect(bounds[d]) for d in 1:3], cell_counts=Int.(cells),
               time_s=Float64(time_s), fluid_config=abspath(fluid_config))
    tmp = tempname() * ".json"
    open(tmp, "w") do io; JSON.print(io, request); end
    script = joinpath(@__DIR__, "mms2_driver.py")
    python = get(ENV, "MMS2_PYTHON", "python3")
    output = read(`$python $script coordinate-reference --request $tmp`, String)
    rm(tmp; force=true)
    data = JSON.parse(output)
    return data["coordinate_reference"], Float64.(data["freestream_m_s"])
end

function write_field(path, grid, name, time_s)
    haskey(grid.field, name) || error("Reconstructed grid is missing field $name")
    vpm.HDF5.h5open(path, "w") do h5
        h5["nodes"] = Float64.(grid.nodes[1:3, :])
        h5["field_data"] = Float64.(grid.field[name]["field_data"])
        h5["NDIVS"] = Int.(grid.NDIVS)
        h5["time_s"] = Float64(time_s)
        h5["component_order"] = collect(codeunits("x,y,z"))
        h5["flattening"] = collect(codeunits("GeometricTools column-major; first coordinate fastest"))
    end
end

function save_json(path, object)
    open(path, "w") do io; JSON.print(io, object, 2); println(io); end
end

function main(args)
    opts = arguments(args)
    snapshot = parse(Int, opts["--snapshot"])
    snapshot >= 0 || error("Snapshot number must be nonnegative")
    config_path = abspath(opts["--grid-config"])
    isfile(config_path) || error("Missing grid configuration $config_path")
    py = get(ENV, "MMS2_PYTHON", "python3")
    config_json = read(`$py -c "import yaml,json,sys; print(json.dumps(yaml.safe_load(open(sys.argv[1]))))" $config_path`, String)
    config = JSON.parse(config_json)
    manifest, runmeta, entries, files = run_manifest(opts["--manifest"], snapshot)
    eff = runmeta["effective"]
    count_total = sum(Int(e["expected_count"]) for e in entries)
    count_total > 0 || error("Cannot reconstruct an empty population")
    pfield = new_field(count_total + 1, runmeta)
    audit = load_into!(pfield, snapshot, entries, files)
    info = particle_envelope(pfield)

    # The reconstruction box and target scale come from the MMS 2 grid config.
    bounds = config["reconstruction"]["bounds"]
    target_scale = Float64(config["target"]["smallest_target_length_scale"])
    core_intervals = Int(config["particles"]["core_grid_intervals_minimum"])
    target_intervals = Int(config["reconstruction"]["target_intervals_per_length_scale"])
    spacing_limit = min(info.sigma_min/core_intervals, target_scale/target_intervals)
    lengths = [Float64(bounds[d][2]-bounds[d][1]) for d in 1:3]
    cells = ceil.(Int, lengths ./ spacing_limit)
    all(cells .> 0) || error("Derived grid has invalid dimensions")
    fine_cells = cells .* Int(config["reconstruction"]["sensitivity_factor"])
    nnodes_fine_big = prod(BigInt.(fine_cells .+ 1))
    max_grid_points = BigInt(get(config["reconstruction"], "maximum_grid_points", 2_000_000))
    nnodes_fine_big <= max_grid_points || error(
        "Derived doubled grid requires $nnodes_fine_big nodes, above configured maximum_grid_points=$max_grid_points; " *
        "reduce the declared domain or explicitly raise the memory limit (grid spacing will not be coarsened)")
    nnodes_fine = Int(nnodes_fine_big)
    pfield = new_field(count_total + nnodes_fine + 1, runmeta)
    audit = load_into!(pfield, snapshot, entries, files)
    actual_time = Float64(audit["physical_time_s"])
    haskey(manifest, "physical_time_s") &&
        !isapprox(Float64(manifest["physical_time_s"]), actual_time; rtol=0, atol=1e-12) &&
        error("Manifest physical time differs from saved HDF5 time")
    lifecycle = get(eff, "mms_fixed_population_lifecycle", Dict("enabled"=>false))
    fs = make_freestream(eff)(actual_time)
    outdir = abspath(opts["--output"])
    recon = joinpath(outdir, "reconstruction")
    mkpath(recon)
    fluid_cfg_entry = runmeta["config_snapshots"]["fluid"]["path"]
    # Prefer an explicit fluid config path in the manifest; otherwise resolve
    # the run metadata's relative snapshot against its saved run directory.
    fluid_cfg = haskey(manifest, "fluid_config_path") ? String(manifest["fluid_config_path"]) :
        joinpath(dirname(dirname(dirname(abspath(String(manifest["run_metadata_path"]))))), String(fluid_cfg_entry))
    isfile(fluid_cfg) || error("Could not resolve saved fluid config: $fluid_cfg")

    function reconstruct(cells_this, suffix)
        grid = gt.Grid(Float64.([bounds[d][1] for d in 1:3]),
                       Float64.([bounds[d][2] for d in 1:3]), Int.(cells_this))
        reference, py_fs = python_reference(bounds, cells_this, actual_time, fluid_cfg)
        metadata = reconstruct_complete_field!(pfield, grid;
            core_support_radii=Float64(get(config["reconstruction"], "core_support_radii", 4.0)),
            minimum_clearance_core_radii=Float64(get(config["reconstruction"], "minimum_boundary_clearance_core_radii", 2.0)),
            target_length_scale_m=target_scale, core_intervals_minimum=core_intervals,
            target_intervals_minimum=target_intervals, population_audit=audit,
            expected_freestream=py_fs, coordinate_reference=reference, verbose=false)
        grid.field["U_induced"] = deepcopy(grid.field["U"])
        grid.field["U_induced"]["field_data"] .-= reshape(fs, 3, 1)
        write_field(joinpath(recon, "velocity_total$(suffix).h5"), grid, "U", actual_time)
        write_field(joinpath(recon, "velocity_induced$(suffix).h5"), grid, "U_induced", actual_time)
        write_field(joinpath(recon, "vorticity_Wapprox$(suffix).h5"), grid, "Wapprox", actual_time)
        return grid, metadata
    end
    coarse, coarse_metadata = reconstruct(cells, "")
    fine, fine_metadata = reconstruct(fine_cells, "_doubled")
    sensitivity = compare_nested_grid_fields(coarse, fine, Int(config["reconstruction"]["sensitivity_factor"]);
        freestream=fs,
        relative_tolerance=Float64(config["reconstruction"]["sensitivity_relative_tolerance"]),
        norm_floor=Float64(get(config["reconstruction"], "sensitivity_norm_floor", 1e-14)))
    validated_entries = [merge(copy(e), Dict("actual_loaded_count"=>audit["actual_loaded_counts"][String(e["name"])]))
                         for e in entries]
    save_json(joinpath(recon, "validated_particle_population_manifest.json"), Dict(
        "snapshot"=>snapshot, "physical_time_s"=>actual_time,
        "required_population_names"=>manifest["required_population_names"],
        "run_metadata_path"=>abspath(String(manifest["run_metadata_path"])),
        "populations"=>validated_entries))
    identity = Dict("status"=>"BLOCKED_SINGLE_SNAPSHOT", "stable_particle_ids_available"=>false,
        "creation_removal_role_change_events_available"=>false,
        "limitation"=>"Saved production snapshots contain particle states and static flags but no persistent particle identifiers or lifecycle event log; per-snapshot count/role checks cannot establish identity continuity.")
    gridmeta = Dict("nominal"=>coarse_metadata, "doubled"=>fine_metadata,
        "sensitivity"=>sensitivity, "nominal_cells"=>cells, "doubled_cells"=>fine_cells,
        "physical_time_s"=>actual_time, "coordinate_order"=>"x,y,z; column-major first coordinate fastest")
    save_json(joinpath(recon, "snapshot_metadata.json"), Dict("snapshot"=>snapshot,
        "physical_time_s"=>actual_time, "production_run_metadata"=>abspath(String(manifest["run_metadata_path"])),
        "population_manifest"=>abspath(opts["--manifest"]), "freestream_m_s"=>fs,
        "total_velocity_includes_freestream_once"=>true))
    save_json(joinpath(recon, "particle_population_audit.json"), Dict("snapshot_load"=>audit,
        "fixed_population_history"=>"NOT_AUDITED_SINGLE_SNAPSHOT", "identity_audit"=>identity,
        "runtime_lifecycle_controls"=>lifecycle,
        "probe_particles_removed"=>vpm.get_np(pfield) == count_total))
    save_json(joinpath(recon, "grid_metadata.json"), gridmeta)
    sensitivity["status"] == "PASS" || error(
        "Primary induced-velocity/vorticity grid sensitivity failed; see $(joinpath(recon, "grid_metadata.json"))")
    println(JSON.json(Dict("status"=>"PASS_SNAPSHOT_RECONSTRUCTION", "physical_time_s"=>actual_time,
        "particle_count"=>count_total, "nominal_cells"=>cells, "doubled_cells"=>fine_cells,
        "identity_audit"=>identity["status"], "output"=>recon)))
end

main(ARGS)
