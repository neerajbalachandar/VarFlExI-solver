"""FLOWUnsteady/FLOWVPM full-particle-field Eulerian reconstruction.

Call `reconstruct_complete_field!(pfield, grid; ...)` with the actual
production ParticleField at a saved MMS time. The input field must already
contain every production population exactly once (dynamic, static/bound, and
other contributing particles). No subset or target aerodynamic field is
accepted by this interface.
"""

using FLOWUnsteady
const vpm = FLOWUnsteady.vpm

"""Load one saved production snapshot from an explicit population manifest.

The manifest must contain `required_population_names` copied from the run
metadata and a `populations` vector. Every population entry must have `name`,
`role` (`dynamic` or `static`), `read_path`, `file_prefix`, `required=true`,
and both `expected_count` and `saved_metadata_count`. The names must exactly
match the independent metadata inventory; per-file counts are checked against
the HDF5 `np` dataset before loading.
"""
function load_complete_particle_snapshot!(pfield, snapshot::Int, manifest)
    haskey(manifest, "populations") && haskey(manifest, "required_population_names") ||
        error("Manifest must contain populations and an independent saved-metadata inventory")
    names = String[]
    files = String[]
    roles = String[]
    entries = collect(manifest["populations"])
    isempty(entries) && error("Particle population manifest is required")
    required_names = String.(manifest["required_population_names"])
    isempty(required_names) && error("Saved metadata lists no required particle populations")
    for entry in entries
        required = Bool(entry["required"])
        required || error("Optional populations are ambiguous in a full-field MMS manifest")
        name, role = String(entry["name"]), String(entry["role"])
        role in ("dynamic", "static") || error("Unsupported population role '$role'")
        count_expected = Int(entry["expected_count"])
        count_metadata = Int(entry["saved_metadata_count"])
        count_expected == count_metadata || error(
            "Manifest/saved metadata count mismatch for $name: $count_expected != $count_metadata")
        count_expected >= 0 || error("Negative expected particle count for $name")
        push!(names, name); push!(roles, role)
        push!(files, joinpath(String(entry["read_path"]), String(entry["file_prefix"]) * ".$(snapshot).h5"))
    end
    length(unique(names)) == length(names) || error("Duplicate population name in reconstruction manifest")
    length(unique(required_names)) == length(required_names) ||
        error("Saved particle metadata contains duplicate population names")
    Set(names) == Set(required_names) || error(
        "Manifest populations do not match the complete required population inventory")
    length(unique(files)) == length(files) || error("Same particle snapshot file was listed more than once")
    all(isfile, files) || error("A required particle population snapshot is missing")
    count(==("dynamic"), roles) == 1 || error("Manifest must identify exactly one dynamic particle population")
    first(findall(==("dynamic"), roles)) == 1 || error("Dynamic population must be loaded first")

    loaded = Dict{String,Int}()
    snapshot_times = Float64[]
    for (i, entry) in enumerate(entries)
        filepath = files[i]
        snapshot_metadata = vpm.HDF5.h5open(filepath, "r") do h5
            "np" in keys(h5) || error("Snapshot $filepath has no saved particle-count metadata")
            "t" in keys(h5) || error("Snapshot $filepath has no saved physical-time metadata")
            (count=Int(vpm.HDF5.read(h5["np"])), time=Float64(vpm.HDF5.read(h5["t"])))
        end
        hdf5_count = snapshot_metadata.count
        hdf5_count == Int(entry["saved_metadata_count"]) || error(
            "Snapshot metadata for $(entry["name"]) reports $hdf5_count particles, " *
            "manifest says $(entry["saved_metadata_count"])" )
        before = vpm.get_np(pfield)
        filename = String(entry["file_prefix"]) * ".$(snapshot).h5"
        vpm.read!(pfield, filename; path=String(entry["read_path"]), overwrite=(i == 1))
        after = vpm.get_np(pfield)
        actual_count = after - (i == 1 ? 0 : before)
        actual_count == Int(entry["expected_count"]) || error(
            "Loaded $actual_count particles for $(entry["name"]), metadata expects $(entry["expected_count"])" )
        slots = (after-actual_count+1):after
        expected_static = roles[i] == "static"
        all(j -> vpm.is_static(vpm.get_particle(pfield, j)) == expected_static, slots) || error(
            "Particle static flags disagree with role $(roles[i]) for $(entry["name"])")
        loaded[names[i]] = actual_count
        loaded[names[i]] == hdf5_count || error("Loaded and saved counts differ for $(names[i])")
        push!(snapshot_times, snapshot_metadata.time)
    end
    n_dynamic = sum(loaded[names[i]] for i in eachindex(names) if roles[i] == "dynamic")
    n_static = sum(loaded[names[i]] for i in eachindex(names) if roles[i] == "static")
    n_dynamic + n_static == vpm.get_np(pfield) || error("Population manifest does not account for every loaded particle")
    length(snapshot_times) == length(entries) || error("Not all population snapshots supplied a physical time")
    all(isapprox(snapshot_times[1], t; rtol=0, atol=1e-12) for t in snapshot_times) ||
        error("Dynamic and static snapshots have different physical times: $snapshot_times")
    if haskey(manifest, "physical_time_s")
        isapprox(Float64(manifest["physical_time_s"]), snapshot_times[1]; rtol=0, atol=1e-12) ||
            error("Manifest physical time disagrees with saved snapshot metadata")
    end
    if haskey(manifest, "snapshot")
        Int(manifest["snapshot"]) == snapshot || error("Manifest snapshot number disagrees with requested snapshot")
    end
    return Dict("snapshot"=>snapshot, "populations"=>loaded,
                "particle_counts"=>Dict("dynamic"=>n_dynamic,"static_or_bound"=>n_static,
                                        "total"=>n_dynamic+n_static),
                "population_names"=>names, "source_files"=>files,
                "physical_time_s"=>snapshot_times[1],
                "actual_loaded_counts"=>copy(loaded))
end

function assert_fixed_population_count(population_audit, observed_count::Integer, time_s::Real)
    expected = Int(population_audit["particle_counts"]["total"])
    observed_count == expected || error(
        "Fixed-population MMS particle count changed at t=$time_s: $observed_count != $expected")
    return Dict("time_s"=>Float64(time_s), "expected_count"=>expected,
                "observed_count"=>Int(observed_count), "status"=>"PASS")
end

"""Check every recorded step and return a serializable fixed-count audit."""
function audit_fixed_population_history(population_audit, observations)
    isempty(observations) && error("At least one time-step particle count is required")
    step_values = [Int(row["step"]) for row in observations]
    all(diff(step_values) .== 1) || length(step_values) == 1 ||
        error("Particle-count audit must include consecutive time steps in order")
    time_values = [Float64(row["time_s"]) for row in observations]
    all(isfinite, time_values) || error("Particle-count audit times must be finite")
    (length(time_values) == 1 || all(diff(time_values) .> 0)) ||
        error("Particle-count audit times must be strictly increasing")
    rows = [assert_fixed_population_count(population_audit,
             Int(row["particle_count"]), Float64(row["time_s"])) for row in observations]
    return Dict("status"=>"PASS", "steps_checked"=>length(rows), "steps"=>rows,
                "constant_count"=>true)
end

"""Fail-closed lifecycle/identity audit for caller-supplied per-step records.

Each row must provide `step`, `time_s`, `particles` (records with persistent
`id` and `role`), plus `creation_events`, `removal_events`, and
`role_change_events` arrays. The current production snapshot format does not
provide these records, so a single-snapshot reconstruction cannot call this
function successfully or claim identity preservation.
"""
function audit_population_identity_history(population_audit, observations)
    isempty(observations) && error("Particle identity history is required")
    first_ids = nothing
    first_roles = nothing
    results = Any[]
    previous_step = nothing
    previous_time = nothing
    expected_total = Int(population_audit["particle_counts"]["total"])
    for row in observations
        for key in ("step", "time_s", "particles", "creation_events", "removal_events", "role_change_events")
            haskey(row, key) || error("Identity audit row is missing '$key'")
        end
        step, time_s = Int(row["step"]), Float64(row["time_s"])
        isfinite(time_s) || error("Identity audit time must be finite")
        previous_step === nothing || step == previous_step + 1 || error("Identity audit steps must be consecutive")
        previous_time === nothing || time_s > previous_time || error("Identity audit times must increase")
        isempty(row["creation_events"]) || error("Unexpected particle creation at step $step")
        isempty(row["removal_events"]) || error("Unexpected particle removal at step $step")
        isempty(row["role_change_events"]) || error("Unexpected particle role change at step $step")
        particles = collect(row["particles"])
        length(particles) == expected_total || error("Identity record count disagrees with complete snapshot population")
        ids = [String(p["id"]) for p in particles]
        length(unique(ids)) == length(ids) || error("Duplicate persistent particle identifier at step $step")
        roles = Dict(String(p["id"])=>String(p["role"]) for p in particles)
        all(role in ("dynamic", "static") for role in values(roles)) || error("Unexpected particle role at step $step")
        if first_ids === nothing
            first_ids = Set(ids)
            first_roles = roles
        else
            Set(ids) == first_ids || error("Particle identity membership changed at step $step")
            roles == first_roles || error("Particle dynamic/static role changed at step $step")
        end
        push!(results, Dict("step"=>step, "time_s"=>time_s, "particle_count"=>length(ids), "status"=>"PASS"))
        previous_step, previous_time = step, time_s
    end
    return Dict("status"=>"PASS", "steps_checked"=>length(results), "steps"=>results,
                "unique_particle_ids"=>length(first_ids), "lifecycle_events_absent"=>true)
end

function particle_envelope(pfield)
    n = vpm.get_np(pfield)
    n > 0 || error("Cannot reconstruct an empty production ParticleField")
    xyz_min = fill(Inf, 3)
    xyz_max = fill(-Inf, 3)
    sigmas = Float64[]
    n_static = 0
    n_dynamic = 0
    for particle in vpm.iterate(pfield; include_static=true)
        x = Float64.(vpm.get_X(particle))
        sigma = Float64(vpm.get_sigma(particle)[])
        all(isfinite, x) && isfinite(sigma) && sigma > 0 ||
            error("Particle positions/core radii must be finite and core radii positive")
        xyz_min .= min.(xyz_min, x)
        xyz_max .= max.(xyz_max, x)
        push!(sigmas, sigma)
        if vpm.is_static(particle)
            n_static += 1
        else
            n_dynamic += 1
        end
    end
    n_static + n_dynamic == n || error("Particle iteration did not cover get_np(pfield) exactly once")
    return (minimum=xyz_min, maximum=xyz_max, sigma_min=minimum(sigmas),
            sigma_max=maximum(sigmas), sigma_mean=sum(sigmas)/length(sigmas),
            total=n, dynamic=n_dynamic, static=n_static)
end

"""
    reconstruct_complete_field!(pfield, grid; core_support_radii=3,
        minimum_clearance_core_radii=2, save_path=nothing, file_pref="mms2")

Use the installed FLOWUnsteady `computefluiddomain` API on the complete
ParticleField. The field's default kernel is preserved; all particle cores use
scale_sigma=1 and particle strengths are unmodified. Zero-strength grid probes
are used, so the sampling nodes add no artificial particle field. The actual
uniform freestream is added exactly once to velocity; vorticity is taken from
the installed FLOWVPM RBF `zeta_fmm` reconstruction (`Wapprox`).
"""
function reconstruct_complete_field!(pfield, grid;
        core_support_radii=4.0, minimum_clearance_core_radii=2.0,
        target_length_scale_m=0.32, core_intervals_minimum=8,
        target_intervals_minimum=20, population_audit,
        expected_freestream, coordinate_reference,
        save_path=nothing, file_pref="mms2", num=nothing, verbose=false)
    haskey(population_audit, "particle_counts") || error("Complete population audit is required")
    isfinite(core_support_radii) && core_support_radii >= 0 ||
        error("core_support_radii must be finite and nonnegative")
    isfinite(minimum_clearance_core_radii) && minimum_clearance_core_radii >= 0 ||
        error("minimum_clearance_core_radii must be finite and nonnegative")
    isfinite(target_length_scale_m) && target_length_scale_m > 0 ||
        error("target_length_scale_m must be finite and positive")
    core_intervals_minimum > 0 && target_intervals_minimum > 0 ||
        error("Grid interval criteria must be positive")
    info = particle_envelope(pfield)
    declared = population_audit["particle_counts"]
    (info.total == declared["total"] && info.dynamic == declared["dynamic"] &&
     info.static == declared["static_or_bound"]) || error(
        "Loaded ParticleField counts disagree with the saved population manifest")
    pfield.viscous isa vpm.Inviscid || error(
        "Field MMS requires disabled viscosity; effective scheme is $(typeof(pfield.viscous))")
    pfield.SFS isa vpm.NoSFS || error(
        "Field MMS requires disabled SFS; effective model is $(typeof(pfield.SFS))")
    size(grid.nodes, 1) >= 3 || error("Expected a three-dimensional Cartesian grid")
    all(isfinite, grid.nodes) || error("Reconstruction grid coordinates must be finite")
    grid_min = vec(minimum(grid.nodes[1:3, :]; dims=2))
    grid_max = vec(maximum(grid.nodes[1:3, :]; dims=2))
    grid_spacing = (Float64.(grid.P_max) .- Float64.(grid.P_min)) ./ Float64.(grid.NDIVS)
    all(isfinite, grid_spacing) && all(grid_spacing .> 0) ||
        error("Reconstruction grid spacings must be finite and positive")
    max_spacing = min(info.sigma_min/core_intervals_minimum,
                      target_length_scale_m/target_intervals_minimum)
    maximum(grid_spacing) <= max_spacing * (1 + 1.0e-12) || error(
        "Eulerian spacing $(grid_spacing) does not resolve smallest core " *
        "($(info.sigma_min) m; require $core_intervals_minimum intervals) and target scale " *
        "($(target_length_scale_m) m; require $target_intervals_minimum intervals); " *
        "maximum allowed spacing is $max_spacing m")
    # FLOWVPM's Gaussian-erf kernel has no finite support. This is an explicit
    # N-sigma truncation envelope used only for domain-clearance checks.
    support_min = info.minimum .- core_support_radii * info.sigma_max
    support_max = info.maximum .+ core_support_radii * info.sigma_max
    clearance = min.(support_min .- grid_min, grid_max .- support_max)
    minimum(clearance) >= minimum_clearance_core_radii * info.sigma_max || error(
        "Particle/core envelope is too close to or beyond reconstruction boundary; " *
        "clearance=$(clearance), required=$(minimum_clearance_core_radii * info.sigma_max) m")

    # `computefluiddomain` iterates over all free and static particles already
    # resident in pfield. It temporarily appends grid probes and removes them.
    expected_capacity = info.total + grid.nnodes
    pfield.maxparticles >= expected_capacity || error(
        "ParticleField capacity $(pfield.maxparticles) is below " *
        "source_count + grid_nodes = $expected_capacity; reserve probe capacity")
    freestream = Float64.(pfield.Uinf(pfield.t))
    length(freestream) == 3 && all(isfinite, freestream) ||
        error("ParticleField freestream must be a finite 3-vector")
    expected = Float64.(expected_freestream)
    length(expected) == 3 && all(isfinite, expected) ||
        error("Expected Python freestream must be a finite 3-vector")
    max_fs_error = maximum(abs.(freestream .- expected))
    max_fs_error <= 1.0e-12 || error(
        "Python/ParticleField freestream mismatch: max abs error=$max_fs_error m/s")
    coordinate_check = verify_python_coordinate_samples(grid, coordinate_reference)
    save_args = save_path === nothing ? (; save_path=nothing) :
        (; save_path=String(save_path), file_pref=file_pref, grid_names=["field"], num=num)
    FLOWUnsteady.computefluiddomain(pfield, [grid];
        add_Uinf=true, Uinf=freestream, add_Wapprox=true, zeta=vpm.zeta_fmm,
        scale_sigma=1.0, f_Gamma=0.0, remove_nodeparticles=true,
        verbose=verbose, save_args...)
    vpm.get_np(pfield) == info.total || error(
        "Reconstruction leaked probe particles into production ParticleField")
    return Dict(
        "particle_counts" => Dict("total"=>info.total, "dynamic"=>info.dynamic,
                                  "static_or_bound"=>info.static,
                                  "manifest_populations"=>population_audit["populations"]),
        "particle_core_m" => Dict("minimum"=>info.sigma_min, "maximum"=>info.sigma_max,
                                  "mean"=>info.sigma_mean,
                                  "sigma_definition"=>"FLOWVPM smoothing-radius parameter; for the default Gaussian-erf kernel it is the Gaussian radial scale (kernel uses r/sigma), not a finite physical diameter or hard support radius"),
        "sigma_min"=>info.sigma_min, "sigma_max"=>info.sigma_max,
        "sigma_mean"=>info.sigma_mean,
        "particle_position_min_m"=>info.minimum,
        "particle_position_max_m"=>info.maximum,
        "particle_core_truncation_envelope_bounds_m"=>Dict("minimum"=>support_min, "maximum"=>support_max),
        "core_truncation_envelope_sigma_factor"=>core_support_radii,
        "core_truncation_envelope_is_hard_support"=>false,
        "boundary_clearance_m"=>clearance,
        "grid_min_m"=>grid_min, "grid_max_m"=>grid_max,
        "grid_points"=>grid.nnodes,
        "grid_cells"=>Int.(grid.NDIVS), "grid_spacing_m"=>grid_spacing,
        "grid_spacing_x_m"=>grid_spacing[1], "grid_spacing_y_m"=>grid_spacing[2],
        "grid_spacing_z_m"=>grid_spacing[3],
        "intervals_per_minimum_sigma"=>info.sigma_min ./ grid_spacing,
        "intervals_per_minimum_core"=>info.sigma_min ./ grid_spacing,
        "intervals_per_target_scale"=>target_length_scale_m ./ grid_spacing,
        "core_resolution_convention"=>"intervals are measured per FLOWVPM sigma Gaussian radial scale for the default Gaussian-erf kernel; no hard core diameter is assumed",
        "core_intervals_minimum"=>core_intervals_minimum,
        "target_length_scale_m"=>target_length_scale_m,
        "target_intervals_minimum"=>target_intervals_minimum,
        "freestream_added_once_m_s"=>freestream,
        "freestream_expected_from_python_m_s"=>expected,
        "freestream_max_abs_difference_m_s"=>max_fs_error,
        "coordinate_reference_check"=>coordinate_check,
        "viscous_scheme_effective"=>string(typeof(pfield.viscous)),
        "sfs_effective"=>string(typeof(pfield.SFS)),
        "sfs_enabled"=>vpm.isSFSenabled(pfield.SFS),
        "kernel"=>string(typeof(pfield.kernel)),
        "core_scale_sigma"=>1.0,
        "probe_circulation_factor"=>0.0,
        "vorticity_field"=>"Wapprox (FLOWVPM zeta_fmm RBF)"
    )
end

"""Match saved Python known-point coordinates to actual GeometricTools nodes."""
function verify_python_coordinate_samples(grid, reference; atol=1e-12)
    haskey(reference, "samples") && haskey(reference, "bounds_m") &&
        haskey(reference, "cell_counts") || error("Python coordinate reference metadata is incomplete")
    expected_min = Float64.([reference["bounds_m"][d][1] for d in 1:3])
    expected_max = Float64.([reference["bounds_m"][d][2] for d in 1:3])
    expected_cells = Int.(reference["cell_counts"])
    length(expected_cells) == 3 && Int.(grid.NDIVS) == expected_cells ||
        error("Python/Julia reconstruction grid dimensions differ")
    isapprox(Float64.(grid.P_min), expected_min; rtol=0, atol=atol) &&
        isapprox(Float64.(grid.P_max), expected_max; rtol=0, atol=atol) ||
        error("Python/Julia reconstruction domains differ")

    samples = reference["samples"]
    checked = 0
    seen = Set{Int}()
    sampled_indices = Vector{NTuple{3,Int}}()
    for sample in samples
        flat0 = Int(sample["flat_index_geometrictools_columnmajor"])
        index = flat0 + 1
        1 <= index <= grid.nnodes || error("Python sample node index is outside Julia grid")
        flat0 in seen && error("Python coordinate samples contain duplicate flattened indices")
        push!(seen, flat0)
        expected = Float64.(sample["xyz_m"])
        actual = Float64.(grid.nodes[:, index])
        length(expected) == 3 && maximum(abs.(expected .- actual)) <= atol || error(
            "Python/Julia node mismatch at flattened index $(index-1): $expected != $actual")
        expected_probe = Float64.(sample["mapping_probe_vector"])
        x, y, z = actual
        actual_probe = [x + 2y + 3z, -2x + y - 4z, 5x - 3y + z]
        length(expected_probe) == 3 && all(isfinite, expected_probe) ||
            error("Python known-value mapping probe must be a finite 3-vector")
        maximum(abs.(expected_probe .- actual_probe)) <= atol || error(
            "Python/Julia analytical value or component-order mismatch at flattened index $(index-1)")
        dims = Int.(grid.NDIVS) .+ 1
        i0 = (flat0 % dims[1], (flat0 ÷ dims[1]) % dims[2], flat0 ÷ (dims[1]*dims[2]))
        push!(sampled_indices, i0)
        checked += 1
    end
    checked > 0 || error("No Python coordinate samples were supplied for cross-language check")
    if Bool(get(reference, "exact_all_node_check", false))
        checked == grid.nnodes && length(seen) == grid.nnodes ||
            error("Exact Python/Julia coordinate check did not cover every node")
    else
        dims = Int.(grid.NDIVS) .+ 1
        for axis in 1:3
            values = unique([index[axis] for index in sampled_indices])
            length(values) >= 5 && minimum(values) == 0 && maximum(values) == dims[axis]-1 ||
                error("Python/Julia samples do not span axis $axis and both grid boundaries")
        end
    end
    return Dict("status"=>"PASS", "points_compared"=>checked,
                "grid_points_total"=>grid.nnodes,
                "exact_all_node_check"=>Bool(get(reference, "exact_all_node_check", false)),
                "coordinate_order"=>"x,y,z; GeometricTools column-major, first coordinate fastest")
end

"""Compare same particle field on nested Cartesian grids at common nodes."""
function compare_nested_grid_fields(coarse, fine, factor::Int;
        freestream, relative_tolerance=0.01, norm_floor=1.0e-14)
    factor > 1 || error("Grid sensitivity factor must exceed one")
    isfinite(relative_tolerance) && relative_tolerance > 0 ||
        error("Grid sensitivity tolerance must be finite and positive")
    isfinite(norm_floor) && norm_floor > 0 ||
        error("Grid sensitivity norm floor must be finite and positive")
    coarse.P_min == fine.P_min && coarse.P_max == fine.P_max ||
        error("Grid-sensitivity grids must cover the identical physical box")
    Int.(fine.NDIVS) == factor .* Int.(coarse.NDIVS) ||
        error("Fine grid dimensions do not equal sensitivity_factor times coarse dimensions")
    coarse_dims = Tuple(Int.(coarse.NDIVS) .+ 1)
    fine_dims = Tuple(Int.(fine.NDIVS) .+ 1)
    fine_linear = LinearIndices(fine_dims)
    spacing = (Float64.(coarse.P_max) .- Float64.(coarse.P_min)) ./ Float64.(coarse.NDIVS)
    weights_sum = 0.0
    fields = ("U", "U_induced", "Wapprox")
    accum = Dict(name => Dict("difference2"=>0.0,"reference2"=>0.0,"max_abs"=>0.0)
                 for name in fields)
    all(name -> name in keys(coarse.field), ("U", "Wapprox")) ||
        error("Nominal grid is missing U or Wapprox")
    all(name -> name in keys(fine.field), ("U", "Wapprox")) ||
        error("Refined grid is missing U or Wapprox")
    common_fs = Float64.(freestream)
    length(common_fs) == 3 && all(isfinite, common_fs) ||
        error("Sensitivity comparison requires one finite common freestream vector")
    for I in CartesianIndices(coarse_dims)
        fine_I = ((I[1]-1)*factor+1, (I[2]-1)*factor+1, (I[3]-1)*factor+1)
        ic = LinearIndices(coarse_dims)[I]
        jf = fine_linear[fine_I...]
        # Explicit coordinate equality prevents a shape-compatible but shifted
        # or differently ordered comparison.
        isapprox(coarse.nodes[:,ic], fine.nodes[:,jf]; rtol=0, atol=1e-12) || error(
            "Nested grids disagree at coarse node $I")
        weight = prod(spacing) * (I[1] in (1,coarse_dims[1]) ? 0.5 : 1.0) *
                 (I[2] in (1,coarse_dims[2]) ? 0.5 : 1.0) *
                 (I[3] in (1,coarse_dims[3]) ? 0.5 : 1.0)
        weights_sum += weight
        for name in fields
            c = name == "U_induced" ? coarse.field["U"]["field_data"][:,ic] .- common_fs :
                                      coarse.field[name]["field_data"][:,ic]
            f = name == "U_induced" ? fine.field["U"]["field_data"][:,jf] .- common_fs :
                                      fine.field[name]["field_data"][:,jf]
            all(isfinite,c) && all(isfinite,f) || error("Non-finite $name in grid sensitivity comparison")
            d = f .- c
            accum[name]["difference2"] += weight * sum(abs2,d)
            accum[name]["reference2"] += weight * sum(abs2,f)
            accum[name]["max_abs"] = max(accum[name]["max_abs"], maximum(abs.(d)))
        end
    end
    metrics = Dict{String,Any}()
    for name in fields
        ref2 = accum[name]["reference2"]
        sqrt(ref2) > norm_floor || error(
            "Cannot normalize $name grid sensitivity: fine-field norm $(sqrt(ref2)) " *
            "does not exceed norm floor $norm_floor")
        relative = sqrt(accum[name]["difference2"]/ref2)
        metrics[name] = Dict("relative_l2"=>relative,
                             "absolute_l2"=>sqrt(accum[name]["difference2"]),
                             "reference_l2"=>sqrt(ref2), "max_absolute"=>accum[name]["max_abs"],
                             "normalization_field"=>"fine reconstruction",
                             "norm_floor"=>norm_floor,
                             "relative_tolerance"=>relative_tolerance,
                             "status"=>relative <= relative_tolerance ? "PASS" : "FAIL")
    end
    primary_fields = ("U_induced", "Wapprox")
    primary_pass = all(metrics[name]["status"] == "PASS" for name in primary_fields)
    return Dict("common_nodes"=>coarse.nnodes, "domain_min_m"=>coarse.P_min,
                "domain_max_m"=>coarse.P_max, "factor"=>factor,
                "coarse_cells"=>coarse.NDIVS, "fine_cells"=>fine.NDIVS,
                "coarse_spacing_m"=>spacing, "fine_spacing_m"=>spacing ./ factor,
                "common_domain_weight_sum_m3"=>weights_sum, "fields"=>metrics,
                "primary_sensitivity_fields"=>collect(primary_fields),
                "norm_floor"=>norm_floor,
                "status"=>primary_pass ? "PASS" : "FAIL")
end

function run_reconstruction_sensitivity!(pfield, coarse, fine;
        sensitivity_level::Int, sensitivity_factor::Int,
        sensitivity_relative_tolerance::Real, sensitivity_norm_floor::Real,
        coarse_coordinate_reference,
        fine_coordinate_reference, expected_freestream, kwargs...)
    sensitivity_level > 0 || error("sensitivity_level must be positive")
    coarse_metadata = reconstruct_complete_field!(pfield, coarse;
        coordinate_reference=coarse_coordinate_reference,
        expected_freestream=expected_freestream, kwargs...)
    fine_metadata = reconstruct_complete_field!(pfield, fine;
        coordinate_reference=fine_coordinate_reference,
        expected_freestream=expected_freestream, kwargs...)
    comparison = compare_nested_grid_fields(coarse, fine, sensitivity_factor;
        freestream=expected_freestream,
        relative_tolerance=sensitivity_relative_tolerance,
        norm_floor=sensitivity_norm_floor)
    return Dict("sensitivity_level"=>sensitivity_level,
                "sensitivity_factor"=>sensitivity_factor,
                "nominal_metadata"=>coarse_metadata, "refined_metadata"=>fine_metadata,
                "comparison"=>comparison)
end
