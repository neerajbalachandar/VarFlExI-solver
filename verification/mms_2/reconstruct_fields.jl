"""FLOWUnsteady/FLOWVPM full-particle-field Eulerian reconstruction.

Call `reconstruct_complete_field!(pfield, grid; ...)` with the actual
production ParticleField at a saved MMS time. The input field must already
contain every production population exactly once (dynamic, static/bound, and
other contributing particles). No subset or target aerodynamic field is
accepted by this interface.
"""

using FLOWUnsteady
const vpm = FLOWUnsteady.vpm

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
        target_wavelength_m=0.96, core_intervals_minimum=3,
        target_intervals_minimum=8,
        save_path=nothing, file_pref="mms2", num=nothing, verbose=false)
    info = particle_envelope(pfield)
    pfield.viscous isa vpm.Inviscid || error(
        "Field MMS requires disabled viscosity; effective scheme is $(typeof(pfield.viscous))")
    pfield.SFS isa vpm.NoSFS || error(
        "Field MMS requires disabled SFS; effective model is $(typeof(pfield.SFS))")
    size(grid.nodes, 1) >= 3 || error("Expected a three-dimensional Cartesian grid")
    grid_min = vec(minimum(grid.nodes[1:3, :]; dims=2))
    grid_max = vec(maximum(grid.nodes[1:3, :]; dims=2))
    grid_spacing = (Float64.(grid.P_max) .- Float64.(grid.P_min)) ./ Float64.(grid.NDIVS)
    max_spacing = min(info.sigma_min/core_intervals_minimum,
                      target_wavelength_m/target_intervals_minimum)
    maximum(grid_spacing) <= max_spacing * (1 + 1.0e-12) || error(
        "Eulerian spacing $(grid_spacing) does not resolve smallest core " *
        "($(info.sigma_min) m; require $core_intervals_minimum intervals) and target scale " *
        "($(target_wavelength_m) m; require $target_intervals_minimum intervals); " *
        "maximum allowed spacing is $max_spacing m")
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
                                  "static_or_bound"=>info.static),
        "particle_core_m" => Dict("minimum"=>info.sigma_min, "maximum"=>info.sigma_max,
                                  "mean"=>info.sigma_mean),
        "particle_position_min_m"=>info.minimum,
        "particle_position_max_m"=>info.maximum,
        "particle_core_support_bounds_m"=>Dict("minimum"=>support_min, "maximum"=>support_max),
        "boundary_clearance_m"=>clearance,
        "grid_min_m"=>grid_min, "grid_max_m"=>grid_max,
        "grid_points"=>grid.nnodes,
        "grid_cells"=>Int.(grid.NDIVS), "grid_spacing_m"=>grid_spacing,
        "core_intervals_minimum"=>core_intervals_minimum,
        "target_wavelength_m"=>target_wavelength_m,
        "target_intervals_minimum"=>target_intervals_minimum,
        "freestream_added_once_m_s"=>freestream,
        "viscous_scheme_effective"=>string(typeof(pfield.viscous)),
        "sfs_effective"=>string(typeof(pfield.SFS)),
        "sfs_enabled"=>vpm.isSFSenabled(pfield.SFS),
        "kernel"=>string(typeof(pfield.kernel)),
        "core_scale_sigma"=>1.0,
        "probe_circulation_factor"=>0.0,
        "vorticity_field"=>"Wapprox (FLOWVPM zeta_fmm RBF)"
    )
end
