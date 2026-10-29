AOA = solver_cfg.aoa
magVinf = solver_cfg.vinf
rho = solver_cfg.rho
nu = solver_cfg.nu
DEBUG_IO = solver_cfg.debug_io

span = solver_cfg.span
root_chord = solver_cfg.root_chord
tip_chord = solver_cfg.tip_chord
leading_edge_sweep = solver_cfg.leading_edge_sweep
b = span
twist_root = solver_cfg.twist_root
twist_tip = solver_cfg.twist_tip
n_span = solver_cfg.n_span
eta_cp = solver_cfg.eta_cp
eta_bv = solver_cfg.eta_bv

ttot = solver_cfg.ttot
nsteps = solver_cfg.nsteps
dt = solver_cfg.dt
p_per_step = solver_cfg.p_per_step
lambda_vpm = solver_cfg.lambda_vpm
sigma_vpm_overwrite = lambda_vpm * magVinf * dt / max(p_per_step, 1)
sigma_vlm_solver = solver_cfg.sigma_vlm_solver
sigma_vlm_surf = solver_cfg.sigma_vlm_surface * b
shed_starting = wake_cfg.shed_starting
use_unsteady_shedding = wake_cfg.shed_unsteady
unsteady_shedcrit = wake_cfg.unsteady_shedcrit
vlm_rlx = solver_cfg.vlm_rlx

geom_relax = coupling_cfg.geom_relax
force_relax = coupling_cfg.force_relax
disp_scale_x = solver_cfg.disp_scale_x
disp_scale_y = solver_cfg.disp_scale_y
disp_scale_z = solver_cfg.disp_scale_z
wake_remove_every = wake_cfg.remove_every
wake_sphere_factor = wake_cfg.sphere_factor
wake_strength_min_factor = wake_cfg.wake_strength_min
wake_strength_max_factor = wake_cfg.wake_strength_max
wake_sigma_min_factor = wake_cfg.wake_sigma_min
wake_sigma_max_factor = wake_cfg.wake_sigma_max
wake_coupled = wake_cfg.wake_coupled

span_sampling_mode = coupling_cfg.span_sampling
solid_ny_for_sampling = coupling_cfg.solid_ny
custom_span_stride = coupling_cfg.custom_stride

Vinf = configured_freestream(
    magVinf, AOA, solver_cfg.freestream_gradient, solver_cfg.freestream_oscillation,
    solver_cfg.freestream_wavenumber, solver_cfg.freestream_frequency,
    solver_cfg.freestream_phase, solver_cfg.freestream_reference,
)
freestream_gradient = solver_cfg.freestream_gradient
freestream_amplitude = solver_cfg.freestream_oscillation
polar_label = isempty(solver_cfg.section_polar) ? "disabled" : solver_cfg.section_polar

if any(x -> !iszero(x), freestream_gradient)
    @warn "Spatial freestream gradient is active for VLM force evaluation; this FLOWVPM version advects wake particles using Uinf(t) sampled at the global origin, so wake convection does not resolve spatial gradients."
end
if any(x -> !iszero(x), freestream_amplitude)
    @warn "Synthetic spatial-temporal gust enabled. FLOWVPM advects wake particles with the gust sampled at the global origin; wake convection does not resolve spatial gust variation. This is a prescribed gust, not resolved turbulence/LES."
end

if solver_cfg.regularize_vlm
    vlm.VLMSolver._regularize(true)
end

println(
    "Fluid config: AoA=$(AOA) deg, U=$(magVinf) m/s, rho=$(rho), nu=$(nu), span=$(span), " *
    "n_span=$(n_span), nsteps=$(nsteps), dt=$(dt), p_per_step=$(p_per_step), " *
    "polar=$(polar_label), vpm_sfs=$(solver_cfg.vpm_sfs), " *
    "shed_boundarylayer=$(solver_cfg.shed_boundarylayer)"
)

wing = make_cantilever_template(
    span, root_chord, tip_chord, leading_edge_sweep, 0.0, twist_root, twist_tip, n_span
)
wing_ref = deepcopy(wing)

# Optional fixed-population coupled MMS target. A target manifest takes
# precedence over the legacy constant-source variables; the legacy path is
# unchanged when no manifest is provided.
mms_target_path = strip(string(something(args["mms_target"], get(ENV, "VARFLEXI_MMS_TARGET", ""))))
mms_target = nothing
mms_target_enabled = !isempty(mms_target_path)
if mms_target_enabled
    if !isdefined(@__MODULE__, :CoupledMMSTarget)
        include(joinpath(@__DIR__, "mms_coupled_source.jl"))
    end
    mms_target = CoupledMMSTarget(mms_target_path)
end
mms_state_source_text = strip(get(ENV, "VARFLEXI_MMS_STATE_SOURCE", ""))
mms_position_source_text = strip(get(ENV, "VARFLEXI_MMS_POSITION_SOURCE", ""))
if mms_target_enabled && (!isempty(mms_state_source_text) || !isempty(mms_position_source_text))
    println(stderr, "WARNING: VARFLEXI_MMS_TARGET takes precedence over VARFLEXI_MMS_STATE_SOURCE and VARFLEXI_MMS_POSITION_SOURCE")
end
mms_source_text = isempty(mms_state_source_text) ? mms_position_source_text : mms_state_source_text
mms_state_source_enabled = mms_target_enabled || !isempty(mms_source_text)
mms_rk3_stage_trace = NamedTuple[]
mms_target_seeded = Ref(false)
mms_target_start = Ref(0)
if !mms_state_source_enabled
    vpm_integration = vpm.rungekutta3
elseif mms_target_enabled
    mms_source! = make_coupled_mms_source(mms_target, nothing)
    mms_initializer! = pf -> seed_coupled_mms_particles!(mms_target, pf)
    vpm_integration = MMSRK3StateSource(mms_source!, mms_rk3_stage_trace; initialize! = mms_initializer!)
    println("Target-driven fixed-population rVPM MMS source active: $(mms_target_path)")
else
    !isempty(mms_state_source_text) && !isempty(mms_position_source_text) && error(
        "Set only one of VARFLEXI_MMS_STATE_SOURCE and VARFLEXI_MMS_POSITION_SOURCE"
    )
    lowercase(strip(solver_cfg.vpm_sfs)) in ("none", "off", "false") || error(
        "The particle-position MMS component requires vpm_sfs=none so its position source cannot enter SFS strength evolution"
    )
    mms_source_values = parse.(Float64, split(mms_source_text, ','))
    if !isempty(mms_state_source_text)
        length(mms_source_values) == 7 || error(
            "VARFLEXI_MMS_STATE_SOURCE must contain 7 comma-separated values: " *
            "position source (3 m/s), circulation source (3 strength/s), core-width source (m/s)"
        )
        mms_source = (
            position=mms_source_values[1:3],
            circulation=mms_source_values[4:6],
            core=mms_source_values[7],
        )
    else
        length(mms_source_values) == 3 || error(
            "VARFLEXI_MMS_POSITION_SOURCE must contain three comma-separated values in m/s"
        )
        mms_source = (position=mms_source_values, circulation=zeros(3), core=0.0)
    end
    all(isfinite, mms_source_values) || error("MMS state source must be finite")
    mms_source! = (_particle, _stage_time) -> mms_source
    vpm_integration = MMSRK3StateSource(mms_source!, mms_rk3_stage_trace)
    println("Opt-in rVPM RK3 state source active: $(mms_source)")
end

polar_path = isempty(solver_cfg.section_polar_path) ? nothing : solver_cfg.section_polar_path
parasitic_args = polar_path === nothing ? () : (read_path=polar_path,)
aerodynamic_force_calculator = uns.generate_calc_aerodynamicforce(
    add_parasiticdrag=!isempty(solver_cfg.section_polar),
    airfoilpolar=solver_cfg.section_polar,
    parasiticdrag_args=parasitic_args,
)

system = vlm.WingSystem()
vlm.addwing(system, "Wing", wing)
vlm.setVinf(system, Vinf)
ensure_gamma!(system, vlm.get_m(system))
previous_force_system = Ref(deepcopy(system))
vehicle = uns.VLMVehicle(system; vlm_system=system, wake_system=system)

Vvehicle(t) = zeros(3)
anglevehicle(t) = zeros(3)
maneuver = uns.KinematicManeuver((), (), Vvehicle, anglevehicle)

simulation = uns.Simulation(
    vehicle, maneuver, 0.0, 0.0, ttot;
    Vinit=zeros(3), Winit=zeros(3)
)

fluid_dir = normpath(joinpath(@__DIR__, ".."))
final_dir = normpath(joinpath(fluid_dir, ".."))
save_path = isabspath(solver_cfg.save_path) ?
            solver_cfg.save_path :
            normpath(joinpath(final_dir, solver_cfg.save_path))
run_name = solver_cfg.run_name
configured_mms_output = strip(string(something(args["mms_output"], get(ENV, "VARFLEXI_MMS_OUTPUT", ""))))
if !isempty(configured_mms_output)
    save_path = abspath(configured_mms_output)
end
create_savepath = solver_cfg.create_savepath
prompt = solver_cfg.prompt
nsteps_save = solver_cfg.nsteps_save
save_horseshoes = solver_cfg.save_horseshoes
mkpath(save_path)

estimated_particles = Int(ceil((nsteps + 1) * (vlm.get_m(vehicle.vlm_system) * (p_per_step + 1) + p_per_step)))
max_particles = max(
    solver_cfg.max_particles,
    Int(ceil(solver_cfg.max_particles_safety_factor * estimated_particles))
)
omit_shedding_rows = Int[]

rmv_strength = 2 * 2 / max(p_per_step, 1) * dt / (1 / 12)
minmaxGamma = rmv_strength .* [wake_strength_min_factor, wake_strength_max_factor]
wake_treatment_strength = uns.remove_particles_strength(
    minmaxGamma[1]^2, minmaxGamma[2]^2; every_nsteps=max(wake_remove_every, 1)
)

minmaxsigma = sigma_vpm_overwrite .* [wake_sigma_min_factor, wake_sigma_max_factor]
wake_treatment_sigma = uns.remove_particles_sigma(
    minmaxsigma[1], minmaxsigma[2]; every_nsteps=max(wake_remove_every, 1)
)

wake_treatment_sphere = uns.remove_particles_sphere(
    (wake_sphere_factor * b)^2, 1; Xoff=[0.5 * b, 0.0, 0.0]
)

wake_treatment = uns.concatenate(
    wake_treatment_sphere,
    wake_treatment_strength,
    wake_treatment_sigma
)

vpm_fmm_settings = vpm.FMM(
    p=wake_cfg.fmm_p,
    ncrit=wake_cfg.fmm_ncrit,
    theta=wake_cfg.fmm_theta,
    shrink_recenter=wake_cfg.fmm_shrink_recenter,
    relative_tolerance=wake_cfg.fmm_relative_tolerance,
    absolute_tolerance=wake_cfg.fmm_absolute_tolerance,
    autotune_p=wake_cfg.fmm_autotune_p,
    autotune_ncrit=wake_cfg.fmm_autotune_ncrit,
    autotune_reg_error=wake_cfg.fmm_autotune_regularization,
    default_rho_over_sigma=wake_cfg.fmm_default_rho_over_sigma,
    min_ncrit=wake_cfg.fmm_minimum_ncrit
)

core_spreading_beta = parse(Float64, get(ENV, "VARFLEXI_CORE_SPREADING_BETA", "1.0"))
isfinite(core_spreading_beta) && core_spreading_beta > 0 || error(
    "VARFLEXI_CORE_SPREADING_BETA must be finite and positive"
)
vpm_viscous = wake_cfg.viscous_scheme == "CoreSpreading" ?
              vpm.CoreSpreading(nu, sigma_vpm_overwrite; beta=core_spreading_beta) :
              nothing
vpm_sfs = configured_vpm_sfs(solver_cfg.vpm_sfs)

# Preserve exact configs and effective solver settings for each execution.
run_id = replace(string(round(Int, time() * 1.0e6)), "." => "")
run_metadata_dir = joinpath(save_path, "_run_metadata", "$(run_name)_$(run_id)")
mkpath(run_metadata_dir)
config_snapshots = Dict{String, Any}()
for (label, source) in (("fluid", args["fluid"]), ("solid", args["solid"]), ("coupling", args["coupling"]))
    source_path = abspath(source)
    isfile(source_path) || error("Cannot snapshot missing $(label) config: $(source_path)")
    snapshot_path = joinpath(run_metadata_dir, "$(label)_params.yaml")
    cp(source_path, snapshot_path; force=false)
    config_snapshots[label] = Dict(
        "path" => relpath(snapshot_path, save_path),
        "sha256" => bytes2hex(sha256(read(source_path))),
    )
end
run_metadata = Dict{String, Any}(
    "schema_version" => 1,
    "run_id" => run_id,
    "created_utc" => string(Dates.now(Dates.UTC)),
    "created_unix" => time(),
    "run_name" => run_name,
    "config_snapshots" => config_snapshots,
    "effective" => Dict(
        "dt" => dt, "nsteps" => nsteps, "rho" => rho, "nu" => nu,
        "vinf" => magVinf, "aoa_deg" => AOA, "span" => span,
        "root_chord" => root_chord, "tip_chord" => tip_chord,
        "leading_edge_sweep" => leading_edge_sweep, "twist_root" => twist_root,
        "twist_tip" => twist_tip, "n_span" => n_span, "p_per_step" => p_per_step,
        "lambda_vpm" => lambda_vpm, "sigma_vpm_overwrite" => sigma_vpm_overwrite,
        "max_particles" => max_particles,
        "dynamic_particle_prefix" => run_name * "_pfield",
        "static_particle_prefix" => run_name * "_staticpfield",
        "viscous_scheme" => wake_cfg.viscous_scheme,
        "vpm_sfs" => solver_cfg.vpm_sfs,
        "fmm" => Dict(
            "p" => wake_cfg.fmm_p, "ncrit" => wake_cfg.fmm_ncrit,
            "theta" => wake_cfg.fmm_theta,
            "shrink_recenter" => wake_cfg.fmm_shrink_recenter,
            "relative_tolerance" => wake_cfg.fmm_relative_tolerance,
            "absolute_tolerance" => wake_cfg.fmm_absolute_tolerance,
            "autotune_p" => wake_cfg.fmm_autotune_p,
            "autotune_ncrit" => wake_cfg.fmm_autotune_ncrit,
            "autotune_regularization" => wake_cfg.fmm_autotune_regularization,
            "default_rho_over_sigma" => wake_cfg.fmm_default_rho_over_sigma,
            "minimum_ncrit" => wake_cfg.fmm_minimum_ncrit,
        ),
        "freestream" => Dict(
            "gradient" => solver_cfg.freestream_gradient,
            "oscillation" => solver_cfg.freestream_oscillation,
            "wavenumber" => solver_cfg.freestream_wavenumber,
            "frequency_hz" => solver_cfg.freestream_frequency,
            "phase_rad" => solver_cfg.freestream_phase,
            "reference" => solver_cfg.freestream_reference,
        ),
    ),
)
open(joinpath(run_metadata_dir, "run_metadata.json"), "w") do io
    JSON.print(io, run_metadata, 2)
    println(io)
end
println("Run provenance: $(joinpath(run_metadata_dir, "run_metadata.json"))")

println(
    "Shedding config: spanwise-only wing, n_span=$(n_span), " *
    "wake_sphere_factor=$(wake_sphere_factor), wake_remove_every=$(wake_remove_every), " *
    "wake_strength_factors=($(wake_strength_min_factor),$(wake_strength_max_factor)), " *
    "wake_sigma_factors=($(wake_sigma_min_factor),$(wake_sigma_max_factor)), " *
    "estimated_particles=$(estimated_particles), max_particles=$(max_particles)"
)

sock = connect_to_server(coupling_cfg.host, coupling_cfg.port)

m_span = vlm.get_m(wing)
if mms_target_enabled
    mms_target_wing = deepcopy(wing_ref)
    mms_target_system = vlm.WingSystem()
    vlm.addwing(mms_target_system, "Wing", mms_target_wing)
    vlm.setVinf(mms_target_system, Vinf)
    ensure_gamma!(mms_target_system, vlm.get_m(mms_target_system))
    previous_target_force_system = Ref(deepcopy(mms_target_system))
    mms_target_history_initialized = Ref(false)
    target_forces_prev = zeros(m_span, 3)
end
ys_ref = [wing_ref._ym[i] for i in 1:m_span]
eta_span_fluid = [clamp(ys_ref[i] / span, 0.0, 1.0) for i in 1:m_span]
eta_span_coupling = build_coupling_eta_span(
    m_span, solid_ny_for_sampling, span_sampling_mode; custom_stride=custom_span_stride
)
if span_sampling_mode == "custom-stride"
    assert_eta_close("Fluid VLM eta vs custom coupling eta", eta_span_fluid, eta_span_coupling; mode=span_sampling_mode)
end
eta_span_force_payload = span_sampling_mode == "custom-stride" ? eta_span_coupling : eta_span_fluid
eta_chord_force = [eta_cp]

u_prev_cp = zeros(Float64, m_span, 3)
u_prev_le = zeros(Float64, m_span, 3)
u_prev_te = zeros(Float64, m_span, 3)
omega_prev_cp = zeros(Float64, m_span, 3)
forces_prev = zeros(Float64, m_span, 3)

r_vortex_ref = zeros(Float64, m_span, 3)
r_le_ref = zeros(Float64, m_span, 3)
r_te_ref = zeros(Float64, m_span, 3)
for i in 1:m_span
    cp = [wing_ref._xm[i], wing_ref._ym[i], wing_ref._zm[i]]
    r_vortex_ref[i, :] .= [wing_ref._xn[i] - cp[1], wing_ref._yn[i] - cp[2], wing_ref._zn[i] - cp[3]]
    r_le_ref[i, :] .= [wing_ref._xlwingdcr[i] - cp[1], wing_ref._ywingdcr[i] - cp[2], wing_ref._zlwingdcr[i] - cp[3]]
    r_te_ref[i, :] .= [wing_ref._xtwingdcr[i] - cp[1], wing_ref._ywingdcr[i] - cp[2], wing_ref._ztwingdcr[i] - cp[3]]
end

if solver_cfg.save_geometry_csv
    fluid_geom_csv = joinpath(save_path, run_name * "_fluid_cp_le_te_coords.csv")
    open(fluid_geom_csv, "w") do io
        println(io, "index,i_span,x_cp,y_cp,z_cp,x_le,y_le,z_le,x_te,y_te,z_te")
        for i in 1:m_span
            println(
                io,
                string(
                    i, ",", i, ",",
                    wing_ref._xm[i], ",", wing_ref._ym[i], ",", wing_ref._zm[i], ",",
                    wing_ref._xlwingdcr[i], ",", wing_ref._ywingdcr[i], ",", wing_ref._zlwingdcr[i], ",",
                    wing_ref._xtwingdcr[i], ",", wing_ref._ywingdcr[i], ",", wing_ref._ztwingdcr[i]
                )
            )
        end
    end
end

if DEBUG_IO
    println("Coordinate system check: x=chord, y=span, z=normal")
    println("FLUID first CP/LE/TE = ", [wing_ref._xm[1], wing_ref._ym[1], wing_ref._zm[1]], " / ",
            [wing_ref._xlwingdcr[1], wing_ref._ywingdcr[1], wing_ref._zlwingdcr[1]], " / ",
            [wing_ref._xtwingdcr[1], wing_ref._ywingdcr[1], wing_ref._ztwingdcr[1]])
end

step_hist = Int[]
force_res_hist = Float64[]
force_ref_norm_hist = Float64[]
force_rel_error_hist = Float64[]
geom_res_hist = Float64[]
geom_ref_norm_hist = Float64[]
geom_rel_error_hist = Float64[]
lift_hist = Float64[]
drag_hist = Float64[]
cl_hist = Float64[]
cd_hist = Float64[]
cm_hist = Float64[]
fluid_step_time_hist = Float64[]
fluid_interstep_time_hist = Float64[]
previous_callback_end_ns = Ref{UInt64}(0)
force_trace_path = joinpath(save_path, run_name * "_force_payload_history.jsonl")
force_trace_io = open(force_trace_path, "w")
force_payload_history = Dict{String,Any}[]
wake_probe_history = Dict{String,Any}[]
diag_path = joinpath(save_path, run_name * "_coupling_diagnostics.csv")

msg0 = read_json_line(sock, "init")
if haskey(msg0, "dt")
    dt_solid = Float64(msg0["dt"])
    rel = abs(dt_solid - dt) / max(abs(dt), 1.0e-16)
    rel > 1.0e-10 && @warn "Solid/fluid dt mismatch at init: solid=$(dt_solid), fluid=$(dt)"
end
if haskey(msg0, "n_span")
    nspan_solid = Int(msg0["n_span"])
    if nspan_solid != m_span
        error("Spanwise mismatch at init: solid n_span=$(nspan_solid), fluid m_span=$(m_span). " *
              "Set fluid n_span or coupling communication span consistently.")
    end
end
if span_sampling_mode == "custom-stride" && haskey(msg0, "eta_span") && length(msg0["eta_span"]) == m_span
    eta_span_solid = Float64.(msg0["eta_span"])
    assert_eta_close("Solid eta_span vs fluid coupling eta", eta_span_solid, eta_span_coupling; mode=span_sampling_mode)
end
apply_from_message!(msg0; first_step=true, step=0)

step_ref = Ref(0)
runtime_pipeline = uns.concatenate(wake_treatment, coupling_runtime_function)
