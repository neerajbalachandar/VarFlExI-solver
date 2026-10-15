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

Vinf(X, t) = magVinf * [cosd(AOA), 0.0, sind(AOA)]

if solver_cfg.regularize_vlm
    vlm.VLMSolver._regularize(true)
end

println(
    "Fluid config: AoA=$(AOA) deg, U=$(magVinf) m/s, rho=$(rho), nu=$(nu), span=$(span), " *
    "n_span=$(n_span), nsteps=$(nsteps), dt=$(dt), p_per_step=$(p_per_step)"
)

wing = make_cantilever_template(
    span, root_chord, tip_chord, leading_edge_sweep, 0.0, twist_root, twist_tip, n_span
)
wing_ref = deepcopy(wing)

system = vlm.WingSystem()
vlm.addwing(system, "Wing", wing)
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

vpm_viscous = wake_cfg.viscous_scheme == "CoreSpreading" ?
              vpm.CoreSpreading(nu, sigma_vpm_overwrite, 1.0) :
              nothing

println(
    "Shedding config: spanwise-only wing, n_span=$(n_span), " *
    "wake_sphere_factor=$(wake_sphere_factor), wake_remove_every=$(wake_remove_every), " *
    "wake_strength_factors=($(wake_strength_min_factor),$(wake_strength_max_factor)), " *
    "wake_sigma_factors=($(wake_sigma_min_factor),$(wake_sigma_max_factor)), " *
    "estimated_particles=$(estimated_particles), max_particles=$(max_particles)"
)

sock = connect_to_server(coupling_cfg.host, coupling_cfg.port)

m_span = vlm.get_m(wing)
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
fluid_step_time_hist = Float64[]
force_trace_path = joinpath(save_path, run_name * "_force_payload_history.jsonl")
force_trace_io = open(force_trace_path, "w")
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
use_ftot_force = Ref(solver_cfg.use_ftot_force)
runtime_pipeline = uns.concatenate(wake_treatment, coupling_runtime_function)
