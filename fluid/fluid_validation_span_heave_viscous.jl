using Sockets
using JSON
using LinearAlgebra
import FLOWUnsteady as uns
import FLOWVLM as vlm
import FLOWVPM as vpm
using Dates
using Statistics

function wing_maneuver(;
    disp_plot=true,
    vehicle_velocity::Real=0.30,
    angle_of_attack::Real=0.0)

    chord = 0.1

    kG = 1.90

    freq = kG * vehicle_velocity / (π * chord)

    h_non_dim = 0.175

    a_root = h_non_dim * chord

    ω = 2π * freq

    println("Validation frequency = $freq Hz")
    println("Root amplitude       = $a_root m")

    vehicle_velocity_func(t) = begin

        t_phys = t * ttot

        vz = -a_root * ω * cos(ω * t_phys)

        [
            vehicle_velocity,
            0.0,
            0.0 # no need for defining twice
        ]
    end

    # No pitching rotation
    vehicle_angle_func(t) = [
        0.0,
        angle_of_attack,
        0.0
    ]

    wing_angles = ()

    rotor_rpms = ()

    maneuver = uns.KinematicManeuver(
        wing_angles,
        rotor_rpms,
        vehicle_velocity_func,
        vehicle_angle_func
    )

    if disp_plot
        uns.plot_maneuver(maneuver)
    end

    return maneuver
end


# Keep FLOWVLM robust against colinearity/typing edge cases.
vlm.VLMSolver._regularize(true)

function vlm.VLMSolver._V_AB(A::Vector{<:vlm.VLMSolver.FWrap}, B, C, gamma; ign_col::Bool=false)
    r0 = B - A
    r1 = C - A
    r2 = C - B
    crss = LinearAlgebra.cross(r1, r2)
    magsqr = LinearAlgebra.dot(crss, crss) + (vlm.VLMSolver.regularize ? vlm.VLMSolver.core_rad : 0)

    TF = gamma === nothing ? promote_type(eltype(A), eltype(B), eltype(C)) :
                             promote_type(eltype(A), eltype(B), eltype(C), typeof(gamma))

    if vlm.VLMSolver._check_collinear(magsqr / LinearAlgebra.norm(r0), vlm.VLMSolver.col_crit; ign_col=ign_col)
        return zeros(TF, 3)
    end

    F1 = crss / magsqr
    aux = r1 / sqrt(LinearAlgebra.dot(r1, r1)) - r2 / sqrt(LinearAlgebra.dot(r2, r2))
    F2 = LinearAlgebra.dot(r0, aux)

    if vlm.VLMSolver.blobify
        F1 *= vlm.VLMSolver.gw(LinearAlgebra.norm(crss) / LinearAlgebra.norm(r0), vlm.VLMSolver.smoothing_rad)
    end

    return gamma === nothing ? (F1 * F2) : ((gamma / 4 / pi) * F1 * F2)
end

function vlm.VLMSolver._V_Ainf_out(A::Vector{<:vlm.VLMSolver.FWrap},
                                   infD::Vector{<:vlm.VLMSolver.FWrap}, C, gamma;
                                   ign_col::Bool=false)
    AC = C - A
    unitinfD = infD / sqrt(LinearAlgebra.dot(infD, infD))
    AAp = LinearAlgebra.dot(unitinfD, AC) * unitinfD
    Ap = AAp + A

    boundAAp = vlm.VLMSolver._V_AB(A, Ap, C, gamma; ign_col=ign_col)

    ApC = C - Ap
    crss = LinearAlgebra.cross(infD, ApC)
    mag = sqrt(LinearAlgebra.dot(crss, crss) + (vlm.VLMSolver.regularize ? vlm.VLMSolver.core_rad : 0))

    TF = gamma === nothing ? promote_type(eltype(A), eltype(infD), eltype(C)) :
                             promote_type(eltype(A), eltype(infD), eltype(C), typeof(gamma))

    if vlm.VLMSolver._check_collinear(mag, vlm.VLMSolver.col_crit; ign_col=ign_col)
        return zeros(TF, 3)
    end

    h = mag / sqrt(LinearAlgebra.dot(infD, infD))
    n = crss / mag
    F = n / h

    if vlm.VLMSolver.blobify
        F *= vlm.VLMSolver.gw(h, vlm.VLMSolver.smoothing_rad)
    end

    return gamma === nothing ? (F + boundAAp) : ((gamma / 4 / pi) * F + boundAAp)
end


# ---------------------------------- CONFIG ----------------------------------
AOA = parse(Float64, get(ENV, "FLUID_AOA_DEG", "0.0"))
magVinf = parse(Float64, get(ENV, "FLUID_VINF", "0.30"))
rho = parse(Float64, get(ENV, "FLUID_RHO", "998"))
DEBUG_IO = lowercase(get(ENV, "COUPLING_DEBUG_IO", "0")) ∉ ("0", "false", "no")

nu = parse(Float64, get(ENV, "FLUID_NU", "1.0e-6"))

# Match solid coordinate system: x=chord, y=span, z=normal displacement.
span = parse(Float64, get(ENV, "WING_SPAN", "0.3"))
root_chord = parse(Float64, get(ENV, "WING_ROOT_CHORD", "0.1"))
tip_chord = parse(Float64, get(ENV, "WING_TIP_CHORD", "0.1"))
leading_edge_sweep = parse(Float64, get(ENV, "WING_X_TIP", "0.0"))

b = span
ar = span / tip_chord
tr = tip_chord / root_chord
twist_root = 0.0
twist_tip = 0.0
gamma = 0.0

# Keep fluid spanwise discretization aligned with coupling payload by default.
comm_n_span = parse(Int, get(ENV, "COUPLING_NSPAN_COMM", "80"))
n_span = parse(Int, get(ENV, "FLUID_N_SPAN", string(comm_n_span)))
span_sampling_mode = lowercase(strip(get(ENV, "COUPLING_SPAN_SAMPLING", "node-stride")))
solid_ny_for_sampling = parse(Int, get(ENV, "SOLID_NY", "240"))
custom_span_stride_raw = get(ENV, "COUPLING_SPAN_STRIDE", "")

ttot = parse(Float64, get(ENV, "COUPLING_TTOT", "15"))
nsteps = parse(Int, get(ENV, "COUPLING_NSTEPS", "3000"))
dt = ttot / nsteps

p_per_step = parse(Int, get(ENV, "FLUID_P_PER_STEP", "1"))
lambda_vpm = 2.0
sigma_vpm_overwrite = lambda_vpm * magVinf * dt / max(p_per_step, 1)
sigma_vlm_solver = -1
sigma_vlm_surf = 0.005 * b
shed_starting = true
use_unsteady_shedding = true
unsteady_shedcrit = 0.0
vlm_rlx = 0.35

# Make a balance for this

geom_relax = parse(Float64, get(ENV, "FLUID_GEOM_RELAX", "0.8"))
force_relax = parse(Float64, get(ENV, "FLUID_FORCE_RELAX", "0.8"))
disp_scale_x = parse(Float64, get(ENV, "FLUID_DISP_SCALE_X", "1.0"))
disp_scale_y = parse(Float64, get(ENV, "FLUID_DISP_SCALE_Y", "1.0"))
disp_scale_z = parse(Float64, get(ENV, "FLUID_DISP_SCALE_Z", "1.0"))
wake_remove_every = parse(Int, get(ENV, "FLUID_WAKE_REMOVE_EVERY", "20"))
wake_sphere_factor = parse(Float64, get(ENV, "FLUID_WAKE_SPHERE_FACTOR", "25.0"))
wake_strength_min_factor = parse(Float64, get(ENV, "FLUID_WAKE_STRENGTH_MIN_FACTOR", "1.0e-6"))
wake_strength_max_factor = parse(Float64, get(ENV, "FLUID_WAKE_STRENGTH_MAX_FACTOR", "30.0"))
wake_sigma_min_factor = parse(Float64, get(ENV, "FLUID_WAKE_SIGMA_MIN_FACTOR", "1.0e-3"))
wake_sigma_max_factor = parse(Float64, get(ENV, "FLUID_WAKE_SIGMA_MAX_FACTOR", "50.0"))

# Spanwise-only fluid discretization, CP/BV sampled internally from LE/TE.
eta_cp = 0.75
eta_bv = 0.25

function build_coupling_eta_span(nspan::Int, ny_solid::Int, mode::String)
    nspan < 1 && error("COUPLING_NSPAN_COMM/FLUID_N_SPAN must be >= 1")
    nspan == 1 && return [0.0]

    if mode == "midpoint"
        return [(i - 0.5) / nspan for i in 1:nspan]
    elseif mode == "node-stride"
        stride = max(1, ny_solid ÷ nspan)
        idx = [(i - 1) * stride for i in 1:nspan]
        if length(unique(idx)) < nspan || maximum(idx) > ny_solid
            idx = round.(Int, range(0, max(ny_solid - 1, 0); length=nspan))
        end
        return [idx_i / max(float(ny_solid), 1.0) for idx_i in idx]
    elseif mode == "custom-stride"
        isempty(custom_span_stride_raw) &&
            error("COUPLING_SPAN_SAMPLING=custom-stride requires COUPLING_SPAN_STRIDE")
        stride = parse(Int, custom_span_stride_raw)
        stride < 1 && error("COUPLING_SPAN_STRIDE must be >= 1")
        last_idx = (nspan - 1) * stride
        last_idx > ny_solid && error(
            "Unsafe custom span stride: " *
            "(COUPLING_NSPAN_COMM-1)*COUPLING_SPAN_STRIDE = $(last_idx) " *
            "exceeds SOLID_NY = $(ny_solid). Reduce stride/stations or increase SOLID_NY."
        )
        return [((i - 1) * stride) / max(float(ny_solid), 1.0) for i in 1:nspan]
    elseif mode == "linspace"
        return [(i - 1) / nspan for i in 1:nspan]
    end

    error("Unsupported COUPLING_SPAN_SAMPLING=$(mode)")
end

function assert_eta_close(name::String, a::Vector{Float64}, b::Vector{Float64}; tol=1.0e-10)
    length(a) == length(b) || error("$(name) length mismatch: $(length(a)) vs $(length(b))")
    err = maximum(abs.(a .- b))
    err <= tol || error(
        "$(name) mismatch for COUPLING_SPAN_SAMPLING=$(span_sampling_mode): " *
        "max |eta_a-eta_b| = $(err). The fluid VLM station locations must match " *
        "the coupling eta grid; otherwise forces are labeled at the wrong span stations."
    )
end

Vinf(X, t) = magVinf * [cosd(AOA), 0.0, sind(AOA)]

println(
    "Fluid v9 config: AoA=$(AOA) deg, U=$(magVinf) m/s, rho=$(rho), nu=$(nu), span=$(span), " *
    "n_span=$(n_span), nsteps=$(nsteps), dt=$(dt), p_per_step=$(p_per_step)"
)

# -------------------------------- GEOMETRY ----------------------------------
function chord_length_twist(xl, zl, xt, zt)
    dx = xt - xl
    dz = zt - zl
    c = sqrt(dx * dx + dz * dz)
    twist = atan(-dz, dx) * 180 / pi
    return c, twist
end

function make_cantilever_template(span, c_root, c_tip, x_tip, z_tip, twist_root, twist_tip, nspan)
    wing = vlm.Wing(0.0, 0.0, 0.0, c_root, twist_root)
    vlm.addchord(wing, x_tip, span, z_tip, c_tip, twist_tip, nspan; r=1.0)
    return wing
end

wing = make_cantilever_template(
    span, root_chord, tip_chord, leading_edge_sweep, 0.0, twist_root, twist_tip, n_span
)
wing_ref = deepcopy(wing)

system = vlm.WingSystem()
vlm.addwing(system, "Wing", wing)
vehicle = uns.VLMVehicle(system; vlm_system=system, wake_system=system)

# Vvehicle(t) = zeros(3)
# anglevehicle(t) = zeros(3)
# maneuver = uns.KinematicManeuver((), (), Vvehicle, anglevehicle)

maneuver = wing_maneuver(
    disp_plot=false,
    vehicle_velocity=magVinf,
    angle_of_attack=AOA
)

simulation = uns.Simulation(
    vehicle, maneuver, 0.0, 0.0, ttot;
    Vinit=zeros(3), Winit=zeros(3)
)

repo_root = normpath(joinpath(@__DIR__, ".."))
save_path = normpath(joinpath(repo_root, "results", "results_KG9_inflex", "fluid"))
run_name = "fluid_val_2"
mkpath(save_path)

# max_particles = Int((nsteps + 1) * (vlm.get_m(vehicle.vlm_system) * (p_per_step + 1) + p_per_step))
max_particles = 400000
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
    (wake_sphere_factor * root_chord)^2, 1; Xoff=[0.5 * b, 0.0, 0.0]
)

wake_treatment = uns.concatenate(
    wake_treatment_sphere,
    wake_treatment_strength,
    wake_treatment_sigma
)

vpm_fmm_settings = vpm.FMM(
    p=4,
    ncrit=50,
    theta=0.4,
    shrink_recenter=true,
    relative_tolerance=1e-3,
    absolute_tolerance=1e-3,
    autotune_p=true,
    autotune_ncrit=true,
    autotune_reg_error=false,
    default_rho_over_sigma=1.0,
    min_ncrit=3
)

# Viscous scheme ------------------------------------------------------
vpm_viscous = vpm.CoreSpreading(nu, sigma_vpm_overwrite, 1.0)

println(
    "Shedding config v9: spanwise-only wing, n_span=$(n_span), " *
    "omit_shedding rows=$(length(omit_shedding_rows)), wake_sphere_factor=$(wake_sphere_factor), " *
    "wake_remove_every=$(wake_remove_every), " *
    "wake_strength_factors=($(wake_strength_min_factor),$(wake_strength_max_factor)), " *
    "wake_sigma_factors=($(wake_sigma_min_factor),$(wake_sigma_max_factor))"
)


# ---------------------------- COUPLING HELPERS ------------------------------
function update_geometry_absolute_spanwise!(wing, wing_ref, u_cp, u_vortex, u_le, u_te)
    m = vlm.get_m(wing)
    @assert size(u_cp) == (m, 3)
    @assert size(u_vortex) == (m, 3)
    @assert size(u_le) == (m, 3)
    @assert size(u_te) == (m, 3)

    for i in 1:m
        wing._xm[i] = wing_ref._xm[i] + u_cp[i, 1]
        wing._ym[i] = wing_ref._ym[i] + u_cp[i, 2]
        wing._zm[i] = wing_ref._zm[i] + u_cp[i, 3]
    end

    for i in 1:(m + 1)
        idx = min(i, m)
        wing._xn[i] = wing_ref._xn[i] + u_vortex[idx, 1]
        wing._yn[i] = wing_ref._yn[i] + u_vortex[idx, 2]
        wing._zn[i] = wing_ref._zn[i] + u_vortex[idx, 3]
    end

    nch = length(wing._xlwingdcr)
    for i in 1:nch
        idx = min(i, m)
        wing._xlwingdcr[i] = wing_ref._xlwingdcr[i] + u_le[idx, 1]
        wing._zlwingdcr[i] = wing_ref._zlwingdcr[i] + u_le[idx, 3]
        wing._xtwingdcr[i] = wing_ref._xtwingdcr[i] + u_te[idx, 1]
        wing._ztwingdcr[i] = wing_ref._ztwingdcr[i] + u_te[idx, 3]
        wing._ywingdcr[i] = wing_ref._ywingdcr[i] + 0.5 * (u_le[idx, 2] + u_te[idx, 2])
    end

    wing._HSs = nothing
end

function read_json_line(sock::TCPSocket, tag::String)
    line = try
        readline(sock)
    catch err
        if err isa EOFError
            error("$tag: coupling socket closed")
        end
        rethrow(err)
    end
    s = String(line)
    isempty(strip(s)) && error("$tag: received empty line from coupling")
    return JSON.parse(s)
end

uniform_eta(n::Int) = n <= 1 ? [0.0] : collect(range(0.0, 1.0; length=n))

function interp_profile(eta_src::Vector{Float64}, vals::Matrix{Float64}, eta::Float64)
    n = length(eta_src)
    n == 1 && return copy(vals[1, :])

    e = clamp(eta, eta_src[1], eta_src[end])
    j = searchsortedlast(eta_src, e)
    if j <= 0
        return copy(vals[1, :])
    elseif j >= n
        return copy(vals[end, :])
    end

    e0 = eta_src[j]
    e1 = eta_src[j + 1]
    w = (e - e0) / max(e1 - e0, eps(Float64))
    return (1 - w) .* vals[j, :] .+ w .* vals[j + 1, :]
end

function decode_vector_payload(msg, key::String; allow_missing::Bool=false)
    if !haskey(msg, key)
        allow_missing && return zeros(Float64, 0, 0, 3), Float64[], Float64[]
        error("Payload missing key \"$key\"")
    end

    raw = msg[key]
    nraw = length(raw)
    if nraw == 0
        allow_missing && return zeros(Float64, 0, 0, 3), Float64[], Float64[]
        error("Received empty payload array for key \"$key\"")
    end

    vals = zeros(Float64, nraw, 3)
    for i in 1:nraw
        vals[i, 1] = Float64(raw[i][1])
        vals[i, 2] = Float64(raw[i][2])
        vals[i, 3] = Float64(raw[i][3])
    end

    n_span_in = haskey(msg, "n_span") ? Int(msg["n_span"]) : nraw
    n_chord_in = haskey(msg, "n_chord") ? Int(msg["n_chord"]) : 1
    if n_span_in < 1 || n_chord_in < 1 || n_span_in * n_chord_in != nraw
        eta_span = uniform_eta(nraw)
        eta_chord = [0.0]
        grid = reshape(copy(vals), nraw, 1, 3)
        return grid, eta_span, eta_chord
    end

    eta_span = haskey(msg, "eta_span") && length(msg["eta_span"]) == n_span_in ?
               Float64.(msg["eta_span"]) : uniform_eta(n_span_in)
    eta_chord = haskey(msg, "eta_chord") && length(msg["eta_chord"]) == n_chord_in ?
                Float64.(msg["eta_chord"]) : uniform_eta(n_chord_in)

    p_s = sortperm(eta_span)
    p_c = sortperm(eta_chord)
    eta_span = eta_span[p_s]
    eta_chord = eta_chord[p_c]

    indexing = haskey(msg, "indexing") ? String(msg["indexing"]) : "span-major"
    grid = zeros(Float64, n_span_in, n_chord_in, 3)

    idx = 1
    if indexing == "span-major"
        for i in 1:n_span_in
            for j in 1:n_chord_in
                grid[i, j, :] .= vals[idx, :]
                idx += 1
            end
        end
    elseif indexing == "chord-major"
        for j in 1:n_chord_in
            for i in 1:n_span_in
                grid[i, j, :] .= vals[idx, :]
                idx += 1
            end
        end
    else
        @warn "Unknown indexing='$indexing' for key '$key'; defaulting to span-major"
        for i in 1:n_span_in
            for j in 1:n_chord_in
                grid[i, j, :] .= vals[idx, :]
                idx += 1
            end
        end
    end

    sorted_grid = similar(grid)
    for i in 1:n_span_in, j in 1:n_chord_in
        sorted_grid[i, j, :] .= grid[p_s[i], p_c[j], :]
    end

    return sorted_grid, eta_span, eta_chord
end

function sample_span_chord(grid::Array{Float64, 3}, eta_span_src::Vector{Float64}, eta_chord_src::Vector{Float64},
                           eta_span_dst::Vector{Float64}, eta_chord_q::Float64)
    if size(grid, 1) == 0
        return zeros(Float64, length(eta_span_dst), 3)
    end

    ns = size(grid, 1)
    tmp = zeros(Float64, ns, 3)
    for i in 1:ns
        vals = reshape(grid[i, :, :], size(grid, 2), 3)
        tmp[i, :] .= interp_profile(eta_chord_src, vals, eta_chord_q)
    end

    out = zeros(Float64, length(eta_span_dst), 3)
    for i in eachindex(eta_span_dst)
        out[i, :] .= interp_profile(eta_span_src, tmp, eta_span_dst[i])
    end
    return out
end

function cross_rows2(a::Matrix{Float64}, b::Matrix{Float64})
    out = zeros(Float64, size(a, 1), 3)
    out[:, 1] .= a[:, 2] .* b[:, 3] .- a[:, 3] .* b[:, 2]
    out[:, 2] .= a[:, 3] .* b[:, 1] .- a[:, 1] .* b[:, 3]
    out[:, 3] .= a[:, 1] .* b[:, 2] .- a[:, 2] .* b[:, 1]
    return out
end

function ensure_gamma!(wing, m)
    if !haskey(wing.sol, "Gamma") || length(wing.sol["Gamma"]) != m
        wing.sol["Gamma"] = zeros(m)
    end
end

function safe_row_vec(sol::Dict{String, Any}, key::String, i::Int)
    if !haskey(sol, key)
        return nothing
    end
    arr = sol[key]
    if !(arr isa AbstractVector) || i > length(arr)
        return nothing
    end
    v = arr[i]
    if !(v isa AbstractVector) || length(v) < 3
        return nothing
    end
    vv = [Float64(v[1]), Float64(v[2]), Float64(v[3])]
    any(!isfinite, vv) && return nothing
    return vv
end

function fallback_relative_velocity(wing, i::Int, T::Float64)
    Xcp = [wing._xm[i], wing._ym[i], wing._zm[i]]
    Vrel = copy(Vinf(Xcp, T))
    Vind = safe_row_vec(wing.sol, "Vind", i)
    Vvpm = safe_row_vec(wing.sol, "Vvpm", i)
    Vkin = safe_row_vec(wing.sol, "Vkin", i)
    Vind !== nothing && (Vrel .+= Vind)
    Vvpm !== nothing && (Vrel .+= Vvpm)
    Vkin !== nothing && (Vrel .-= Vkin)
    return Vrel
end


# ------------------------------ SOCKET SETUP ---------------------------------
println("Connecting to coupling server...")
sock = connect(get(ENV, "COUPLING_HOST", "127.0.0.1"), parse(Int, get(ENV, "COUPLING_PORT", "9000")))
write(sock, JSON.json(Dict("role" => "fluid")) * "\n")
flush(sock)
println("Fluid connected.")

m_span = vlm.get_m(wing)
ys_ref = [wing_ref._ym[i] for i in 1:m_span]
eta_span_fluid = [clamp(ys_ref[i] / span, 0.0, 1.0) for i in 1:m_span]
eta_span_coupling = build_coupling_eta_span(m_span, solid_ny_for_sampling, span_sampling_mode)
if span_sampling_mode == "custom-stride"
    assert_eta_close("Fluid VLM eta vs custom coupling eta", eta_span_fluid, eta_span_coupling)
end
eta_span_force_payload = span_sampling_mode == "custom-stride" ? eta_span_coupling : eta_span_fluid

# For single-row spanwise wing, communication force channel is one chord location.
eta_chord_force = [eta_cp]

u_prev_cp = zeros(Float64, m_span, 3)
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

if DEBUG_IO
    println("Coordinate system check: x=chord, y=span, z=normal")
    println("FLUID first CP/LE/TE = ", [wing_ref._xm[1], wing_ref._ym[1], wing_ref._zm[1]], " / ",
            [wing_ref._xlwingdcr[1], wing_ref._ywingdcr[1], wing_ref._zlwingdcr[1]], " / ",
            [wing_ref._xtwingdcr[1], wing_ref._ywingdcr[1], wing_ref._ztwingdcr[1]])
    println("Saved fluid geometry coordinates: ", fluid_geom_csv)
end

step_hist = Int[]
Ct_history = Float64[]
thrust_history = Float64[]
force_res_hist = Float64[]
geom_res_hist = Float64[]
force_trace_path = joinpath(save_path, run_name * "_force_payload_history.jsonl")
force_trace_io = open(force_trace_path, "w")

# diag_path = joinpath(save_path, run_name * "_fluid_diagnostics.csv")

# diag_io = open(diag_path,"w")

# println(diag_io,
# "Step,Time,
# Particles,
# Lift,Drag,
# ForceNorm,
# Ct,
# GammaMax,GammaMean,
# CPUtime")



using DelimitedFiles

diag_file = joinpath(save_path, run_name * "_fluid_diagnostics.csv")

diag_io = open(diag_file, "w")

println(diag_io,
"step,time,nParticles,lift,drag,moment,
forceNorm,
gammaMax,gammaMean,
sigmaMax,sigmaMean,
cpuTime")

# ---------------- VALIDATION HISTORY ----------------

time_history = Float64[]
root_history = Float64[]
tip_history  = Float64[]

function extract_le_te_cp_bv(msg, eta_span_dst)
    # Preferred protocol: dedicated LE/TE payloads from solid.
    if haskey(msg, "geometry_le") && haskey(msg, "geometry_te")
        g_le, eta_s_le, eta_c_le = decode_vector_payload(msg, "geometry_le")
        g_te, eta_s_te, eta_c_te = decode_vector_payload(msg, "geometry_te")
        u_le = sample_span_chord(g_le, eta_s_le, eta_c_le, eta_span_dst, 0.0)
        u_te = sample_span_chord(g_te, eta_s_te, eta_c_te, eta_span_dst, 1.0)

        # Reconstruct CP/BV points from communicated edges.
        u_cp = (1 - eta_cp) .* u_le .+ eta_cp .* u_te
        u_bv = (1 - eta_bv) .* u_le .+ eta_bv .* u_te

        return u_le, u_te, u_cp, u_bv, true
    end

    # Backward-compatible protocol: use generic "geometry" payload and sample at needed chord etas.
    g, eta_s, eta_c = decode_vector_payload(msg, "geometry")
    u_le = sample_span_chord(g, eta_s, eta_c, eta_span_dst, 0.0)
    u_te = sample_span_chord(g, eta_s, eta_c, eta_span_dst, 1.0)
    u_cp = sample_span_chord(g, eta_s, eta_c, eta_span_dst, eta_cp)
    u_bv = sample_span_chord(g, eta_s, eta_c, eta_span_dst, eta_bv)
    return u_le, u_te, u_cp, u_bv, false
end

function extract_rotations(msg, eta_span_dst)
    if haskey(msg, "rotation_le") && haskey(msg, "rotation_te")
        r_le, eta_s_le, eta_c_le = decode_vector_payload(msg, "rotation_le")
        r_te, eta_s_te, eta_c_te = decode_vector_payload(msg, "rotation_te")
        omega_le = sample_span_chord(r_le, eta_s_le, eta_c_le, eta_span_dst, 0.0)
        omega_te = sample_span_chord(r_te, eta_s_te, eta_c_te, eta_span_dst, 1.0)
        omega_cp = (1 - eta_cp) .* omega_le .+ eta_cp .* omega_te
        omega_bv = (1 - eta_bv) .* omega_le .+ eta_bv .* omega_te
        return omega_le, omega_te, omega_cp, omega_bv
    end

    rot, eta_s, eta_c = decode_vector_payload(msg, "rotation"; allow_missing=true)
    if size(rot, 1) == 0
        z = zeros(Float64, length(eta_span_dst), 3)
        return z, z, z, z
    end
    omega_le = sample_span_chord(rot, eta_s, eta_c, eta_span_dst, 0.0)
    omega_te = sample_span_chord(rot, eta_s, eta_c, eta_span_dst, 1.0)
    omega_cp = sample_span_chord(rot, eta_s, eta_c, eta_span_dst, eta_cp)
    omega_bv = sample_span_chord(rot, eta_s, eta_c, eta_span_dst, eta_bv)
    return omega_le, omega_te, omega_cp, omega_bv
end

function apply_from_message!(msg; first_step::Bool=false, step::Int=0)
    u_le, u_te, u_cp, u_bv, used_edge_payload = extract_le_te_cp_bv(msg, eta_span_fluid)
    omega_le, omega_te, omega_cp, omega_bv = extract_rotations(msg, eta_span_fluid)

    # User-controlled scaling for debug/consistency sweeps.
    u_cp[:, 1] .*= disp_scale_x
    u_cp[:, 2] .*= disp_scale_y
    u_cp[:, 3] .*= disp_scale_z
    u_le[:, 1] .*= disp_scale_x
    u_le[:, 2] .*= disp_scale_y
    u_le[:, 3] .*= disp_scale_z
    u_te[:, 1] .*= disp_scale_x
    u_te[:, 2] .*= disp_scale_y
    u_te[:, 3] .*= disp_scale_z
    u_bv[:, 1] .*= disp_scale_x
    u_bv[:, 2] .*= disp_scale_y
    u_bv[:, 3] .*= disp_scale_z

    # Relax only CP displacement state; LE/TE/BV follow this step's message.
    u_cp .= geom_relax .* u_cp .+ (1 - geom_relax) .* u_prev_cp
    omega_cp .= geom_relax .* omega_cp .+ (1 - geom_relax) .* omega_prev_cp

    # Reconstruct LE/TE/BV from relaxed CP when edge payload is unavailable.
    if !used_edge_payload
        u_le .= u_cp
        u_te .= u_cp
        u_bv .= u_cp
    end

    # Rotational correction at each geometry point.
    # CP has no offset from itself; keep direct displacement at CP.
    u_bv .+= cross_rows2(omega_bv, r_vortex_ref)
    u_le .+= cross_rows2(omega_le, r_le_ref)
    u_te .+= cross_rows2(omega_te, r_te_ref)

    update_geometry_absolute_spanwise!(wing, wing_ref, u_cp, u_bv, u_le, u_te)
    ensure_gamma!(wing, m_span)

    u_prev_cp .= u_cp
    omega_prev_cp .= omega_cp

    if DEBUG_IO && (first_step || step == 1 || step % 20 == 0)
        println("RECV geometry mode = ", used_edge_payload ? "LE/TE direct" : "legacy geometry->sampled")
        println("RECV first CP/LE/TE = ", vec(u_cp[1, :]), " / ", vec(u_le[1, :]), " / ", vec(u_te[1, :]))
        println("RECV last  CP/LE/TE = ", vec(u_cp[end, :]), " / ", vec(u_le[end, :]), " / ", vec(u_te[end, :]))
    end
end

# Initial geometry
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
              "Set FLUID_N_SPAN or COUPLING_NSPAN_COMM consistently.")
    end
end
if span_sampling_mode == "custom-stride" && haskey(msg0, "eta_span") && length(msg0["eta_span"]) == m_span
    eta_span_solid = Float64.(msg0["eta_span"])
    assert_eta_close("Solid eta_span vs fluid coupling eta", eta_span_solid, eta_span_coupling)
end
apply_from_message!(msg0; first_step=true, step=0)

step_ref = Ref(0)
use_ftot_force = Ref(true)

function coupling_runtime_function(sim, PFIELD, T, DT; vprintln=(s)->nothing)

    step_cpu = time_ns()

    step_ref[] += 1
    step = step_ref[]

    ensure_gamma!(wing, m_span)
    gamma_w = wing.sol["Gamma"]

    gammaMax = maximum(abs.(gamma_w))
    gammaMean = mean(abs.(gamma_w))

    geom_res = 0.0
    
    force_out = Vector{Vector{Float64}}(undef, m_span)
    prev_snapshot = copy(forces_prev)

    frow = nothing
    if use_ftot_force[]
        try
            # FLOWVLM force postprocessing can fail for some wing layouts/types.
            # When it does, keep the coupling stable by permanently falling back
            # to Gamma-based force (KJ) rather than warning every step.
            vlm.calculate_field(wing, "Ftot"; rhoinf=rho, t=T)
            haskey(wing.sol, "Ftot") && (frow = wing.sol["Ftot"])
        catch err
            use_ftot_force[] = false
            @warn "Disabling Ftot-based force extraction; falling back to Gamma-based force. Root cause: $(sprint(showerror, err))"
        end
    end

    for i in 1:m_span
        γ = gamma_w[i]
        isfinite(γ) || (γ = 0.0)
        gamma_w[i] = γ

        fx_raw, fy_raw, fz_raw = 0.0, 0.0, 0.0
        if frow != nothing && i <= length(frow)
            fi = frow[i]
            if length(fi) == 3 && all(isfinite, fi)
                fx_raw, fy_raw, fz_raw = Float64(fi[1]), Float64(fi[2]), Float64(fi[3])
            else
                Vloc = fallback_relative_velocity(wing, i, T)
                lvec = [wing._xn[i + 1] - wing._xn[i], wing._yn[i + 1] - wing._yn[i], wing._zn[i + 1] - wing._zn[i]]
                Fkj = rho * γ * cross(Vloc, lvec)
                fx_raw, fy_raw, fz_raw = Fkj[1], Fkj[2], Fkj[3]
            end
        else
            Vloc = fallback_relative_velocity(wing, i, T)
            lvec = [wing._xn[i + 1] - wing._xn[i], wing._yn[i + 1] - wing._yn[i], wing._zn[i + 1] - wing._zn[i]]
            Fkj = rho * γ * cross(Vloc, lvec)
            fx_raw, fy_raw, fz_raw = Fkj[1], Fkj[2], Fkj[3]
        end

        fx = force_relax * fx_raw + (1 - force_relax) * forces_prev[i, 1]
        fy = force_relax * fy_raw + (1 - force_relax) * forces_prev[i, 2]
        fz = force_relax * fz_raw + (1 - force_relax) * forces_prev[i, 3]

        forces_prev[i, 1] = fx
        forces_prev[i, 2] = fy
        forces_prev[i, 3] = fz
        force_out[i] = [fx, fy, fz]
    end

    # -------------------------------------------------
        # TOTAL THRUST
        # -------------------------------------------------

        total_thrust = -sum(forces_prev[:,1])

        Sref = span * root_chord

        Ct = total_thrust / (0.5 * rho * magVinf^2 * Sref)

        push!(thrust_history, total_thrust)
        push!(Ct_history, Ct)


    # Shedding health diagnostic (helps catch end-of-run shedding errors).
    np = vpm.get_np(PFIELD)

    if step == 1 || step % 20 == 0 || step == nsteps
        println("Fluid step $step/$nsteps: Particles=$np")
    end

    force_mat = reduce(vcat, (reshape(force_out[k], 1, 3) for k in 1:length(force_out)))

    lift = sum(force_mat[:,3])

    drag = -sum(force_mat[:,1])
    
    forceNorm = norm(force_mat)

    force_res = norm(force_mat - prev_snapshot) / max(norm(force_mat), 1.0e-16)
    push!(step_hist, step)
    push!(force_res_hist, force_res)

    if step % 10 == 0
        println(
            force_trace_io,
            JSON.json(Dict(
                "step" => step,
                "n_span" => m_span,
                "n_chord" => 1,
                "indexing" => "span-major",
                "force" => force_out,
                "particles" => np,
            )),
        )
        flush(force_trace_io)

    end

    write(sock, JSON.json(Dict(
        "step" => step,
        "n_span" => m_span,
        "n_chord" => 1,
        "indexing" => "span-major",
        "dt" => dt,
        "ttot" => ttot,
        "eta_span" => eta_span_force_payload,
        "eta_chord" => eta_chord_force,
        "force" => force_out,
    )) * "\n")
    flush(sock)

    if step < nsteps
        msg = read_json_line(sock, "step $step")
        if haskey(msg, "dt")
            dt_solid = Float64(msg["dt"])
            rel = abs(dt_solid - dt) / max(abs(dt), 1.0e-16)
            if rel > 1.0e-10 && (step == 1 || step % 20 == 0)
                @warn "Solid/fluid dt mismatch at step $step: solid=$(dt_solid), fluid=$(dt)"
            end
        end
        u_prev_snapshot = copy(u_prev_cp)
        apply_from_message!(msg; first_step=false, step=step)

        # ---------------------------------------------------
        # SAVE ROOT/TIP DISPLACEMENT HISTORY
        # ---------------------------------------------------

        root_z = wing._zm[1]
        tip_z  = wing._zm[end]

        push!(time_history, T)
        push!(root_history, root_z)
        push!(tip_history, tip_z)

        geom_res = norm(u_prev_cp - u_prev_snapshot) / max(norm(u_prev_cp), 1.0e-16)
        push!(geom_res_hist, geom_res)
    end

    cpu_time = (time_ns()-step_cpu)/1e9

    println(diag_io,
        string(
            step,",",
            T,",",
            np,",",
            lift,",",
            drag,",",
            forceNorm,",",
            Ct,",",
            gammaMax,",",
            gammaMean,",",
            force_res,",",
            geom_res,",",
            cpu_time
        )
    )

    flush(diag_io)
    return step >= nsteps
end

runtime_pipeline = uns.concatenate(wake_treatment, coupling_runtime_function)

uns.run_simulation(simulation, nsteps;
    Vinf=Vinf,
    rho=rho,
    p_per_step=Int(p_per_step),
    max_particles=Int(max_particles),
    sigma_vlm_solver=sigma_vlm_solver,
    sigma_vlm_surf=sigma_vlm_surf,
    sigma_rotor_surf=sigma_vlm_surf,
    sigma_vpm_overwrite=sigma_vpm_overwrite,
    vpm_fmm=vpm_fmm_settings,
    vpm_viscous = vpm_viscous, # New VPM Viscous scheme
    shed_starting=shed_starting,
    shed_unsteady=use_unsteady_shedding,
    unsteady_shedcrit=unsteady_shedcrit,
    omit_shedding=omit_shedding_rows,
    wake_coupled=true,
    vlm_rlx=vlm_rlx,
    extra_runtime_function=runtime_pipeline,
    save_path=save_path,
    run_name=run_name,
    create_savepath=false,
    prompt=false,
    nsteps_save=5,
    save_horseshoes=true
)

# ---------------- SAVE VALIDATION CSV ----------------

using DelimitedFiles

disp_path = joinpath(save_path, "tip_displacement.csv")

data_out = hcat(
    time_history,
    root_history,
    tip_history
)

open(disp_path, "w") do io

    println(io, "time,root_z,tip_z")

    writedlm(io, data_out, ',')

end

println("Saved validation displacement history:")
println(disp_path)

diag_path = joinpath(save_path, run_name * "_coupling_diagnostics.csv")
open(diag_path, "w") do io
    println(io, "step,force_residual,geometry_residual")
    n = length(step_hist)
    for k in 1:n
        gres = k <= length(geom_res_hist) ? geom_res_hist[k] : NaN
        fres = k <= length(force_res_hist) ? force_res_hist[k] : NaN
        println(io, "$(step_hist[k]),$(fres),$(gres)")
    end
end

close(diag_io)
close(sock)
close(force_trace_io)
println("Fluid v9 finished.")
println("Outputs: $save_path")
println("Diagnostics: $diag_path")
println("Force trace: $force_trace_path")

using DelimitedFiles

ct_path = joinpath(save_path, "thrust_coefficient_history.csv")

writedlm(
    ct_path,
    hcat(time_history, thrust_history, Ct_history),
    ','
)

println("Saved Ct history: $ct_path")