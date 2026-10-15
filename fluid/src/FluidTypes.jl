# Only for defining structs

struct SolverConfig

    run_name::String

    aoa::Float64
    vinf::Float64
    rho::Float64
    nu::Float64

    ttot::Float64
    nsteps::Int
    dt::Float64

    p_per_step::Int

    lambda_vpm::Float64

    vlm_rlx::Float64

    sigma_vlm_solver::Float64
    sigma_vlm_surface::Float64

    n_span::Int
    n_chord::Int
    eta_cp::Float64
    eta_bv::Float64

    span::Float64
    root_chord::Float64
    tip_chord::Float64
    leading_edge_sweep::Float64
    twist_root::Float64
    twist_tip::Float64
    dihedral::Float64

    disp_scale_x::Float64
    disp_scale_y::Float64
    disp_scale_z::Float64

    save_path::String
    create_savepath::Bool
    prompt::Bool
    nsteps_save::Int
    save_horseshoes::Bool
    save_geometry_csv::Bool
    use_ftot_force::Bool
    debug_io::Bool
    regularize_vlm::Bool
    max_particles::Int
    max_particles_safety_factor::Float64

end

struct WakeConfig

    remove_every::Int

    sphere_factor::Float64

    wake_strength_min::Float64
    wake_strength_max::Float64

    wake_sigma_min::Float64
    wake_sigma_max::Float64

    shed_starting::Bool
    shed_unsteady::Bool

    unsteady_shedcrit::Float64
    wake_coupled::Bool

    fmm_p::Int
    fmm_ncrit::Int
    fmm_theta::Float64
    fmm_shrink_recenter::Bool
    fmm_relative_tolerance::Float64
    fmm_absolute_tolerance::Float64
    fmm_autotune_p::Bool
    fmm_autotune_ncrit::Bool
    fmm_autotune_regularization::Bool
    fmm_default_rho_over_sigma::Float64
    fmm_minimum_ncrit::Int

    viscous_scheme::String

end

struct CouplingConfig

    host::String

    port::Int

    comm_nspan::Int

    solid_ny::Int

    span_sampling::String

    custom_stride::Union{Nothing,Int}

    geom_relax::Float64

    force_relax::Float64

end

mutable struct RuntimeState

    sock

    u_prev_cp

    omega_prev_cp

    forces_prev

    step_hist

    force_res_hist

    geom_res_hist

    force_trace_io

    step_ref

    use_ftot_force

end
