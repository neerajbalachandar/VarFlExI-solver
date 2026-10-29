"""Fixed-population coupled discrete-model MMS. It exercises the installed
FLOWVLM/rVPM/Reissner–Mindlin operators and the production socket chronology.
It is not a continuum Navier–Stokes MMS, does not verify native particle
lifecycle, and does not treat CRM/RBF moment conservation as verified.

Implements formulation sections "Fixed-population target", "Actual particle
evolution operator", "RK-stage evaluation", and "Core spreading and remapping
are handled as actually implemented". The particle source is fixed-population;
the coupled structural source is Objective B (end-to-end transfer and FSI
accuracy). Target data are read-only and numerical particle states are never
assigned from the target.
"""

struct CoupledMMSTarget
    data::Dict{String,Any}
    ids::Vector{Int}
    id_to_slot::Dict{Int,Int}
end

mms_manifest_values(x::AbstractVector) = reduce(vcat, (mms_manifest_values(y) for y in x); init=Float64[])
mms_manifest_values(x) = [Float64(x)]
mms_manifest_matrix(rows) = reduce(vcat, (reshape(Float64.(row), 1, 3) for row in rows))

function CoupledMMSTarget(path::AbstractString)
    isfile(path) || error("MMS target manifest does not exist: $path")
    data = JSON.parsefile(path)
    haskey(data, "particles") || error("MMS manifest is missing particles")
    pdata = data["particles"]
    required = ("ids", "x0", "gamma0", "sigma0", "a", "b", "c", "d", "e",
                "omega_x", "omega_gamma", "omega_sigma")
    all(haskey(pdata, key) for key in required) || error("MMS manifest is missing particle target fields")
    ids = Int.(pdata["ids"])
    length(unique(ids)) == length(ids) || error("MMS particle IDs must be unique")
    n = length(ids)
    for key in ("x0", "gamma0", "a", "b", "c", "d")
        length(pdata[key]) == n && all(length(row) == 3 for row in pdata[key]) ||
            error("particle $key must be N by 3")
    end
    for key in ("sigma0", "e", "omega_x", "omega_gamma", "omega_sigma")
        length(pdata[key]) == n || error("particle $key must have N entries")
    end
    for key in required[2:end]
        all(isfinite, mms_manifest_values(pdata[key])) || error("particle target $key is non-finite")
    end
    all(Float64.(pdata["sigma0"]) .> 0) || error("target particle core widths must be positive")
    return CoupledMMSTarget(data, ids, Dict(id => i for (i, id) in enumerate(ids)))
end

function structural_state(target::CoupledMMSTarget, x, y, t)
    """Return q, q_t, q_tt in (u,v,w,theta_x,theta_y) order.

    Formulation section: "Target structural field".
    """
    Lx, Ly = Float64(target.data["Lx"]), Float64(target.data["Ly"])
    B = sin(pi*x/Lx)^4 * sin(pi*y/Ly)^4
    comps = target.data["structural"]["components"]
    q, qt, qtt = Float64[], Float64[], Float64[]
    for name in ("u", "v", "w", "theta_x", "theta_y")
        c = comps[name]
        A, w, mode = Float64(c["amplitude"]), Float64(c["omega"]), String(c["mode"])
        p = w * t
        if mode == "cos"
            g, gt, gtt = cos(p), -w*sin(p), -w^2*cos(p)
        elseif mode == "sin"
            g, gt, gtt = sin(p), w*cos(p), -w^2*sin(p)
        elseif mode == "ramp"
            g, gt, gtt = 1-cos(p), w*sin(p), w^2*cos(p)
        else
            error("unsupported structural temporal mode $mode")
        end
        push!(q, A*B*g); push!(qt, A*B*gt); push!(qtt, A*B*gtt)
    end
    return q, qt, qtt
end

function target_surface_point(target::CoupledMMSTarget, x, y, t, z=0.0)
    """Linearized surface map about x_bound, matching GeometryTarget.

    Formulation section: "Target aerodynamic geometry".
    """
    q, _, _ = structural_state(target, x, y, t)
    return [x + q[1], y + q[2], z + q[3] + q[4]*(x-Float64(target.data["x_bound"]))]
end

function particle_target_state(target::CoupledMMSTarget, t)
    """Evaluate every target particle without mutating the solver field.

    Formulation section: "Fixed-population target".
    """
    p = target.data["particles"]
    X0, G0 = mms_manifest_matrix(p["x0"]), mms_manifest_matrix(p["gamma0"])
    X, G = similar(X0), similar(G0)
    sigma = similar(Float64.(p["sigma0"]))
    for i in eachindex(target.ids)
        wx, wg, ws = Float64(p["omega_x"][i]), Float64(p["omega_gamma"][i]), Float64(p["omega_sigma"][i])
        for j in 1:3
            X[i,j] = X0[i,j] + p["a"][i][j]*sin(wx*t) + p["b"][i][j]*(1-cos(wx*t))
            G[i,j] = G0[i,j] + p["c"][i][j]*sin(wg*t) + p["d"][i][j]*(1-cos(wg*t))
        end
        sigma[i] = p["sigma0"][i] + p["e"][i]*(1-cos(ws*t))
    end
    return X, G, sigma
end

function particle_derivatives(target::CoupledMMSTarget, t)
    """Return analytic X_t, Gamma_t, sigma_t arrays.

    Formulation sections: "Fixed-population target" and "RK-stage evaluation".
    """
    p = target.data["particles"]
    n = length(target.ids)
    Xt, Gt, st = zeros(n,3), zeros(n,3), zeros(n)
    for i in 1:n
        wx, wg, ws = Float64(p["omega_x"][i]), Float64(p["omega_gamma"][i]), Float64(p["omega_sigma"][i])
        for j in 1:3
            Xt[i,j] = wx*(p["a"][i][j]*cos(wx*t)+p["b"][i][j]*sin(wx*t))
            Gt[i,j] = wg*(p["c"][i][j]*cos(wg*t)+p["d"][i][j]*sin(wg*t))
        end
        st[i] = p["e"][i]*ws*sin(ws*t)
    end
    return Xt, Gt, st
end

function seed_coupled_mms_particles!(target::CoupledMMSTarget, pfield)
    """Install only the prescribed startup initial data before RK stage one.

    Formulation sections: "Initial conditions and boundary conditions" and
    "Fixed-population target". This runs once, before numerical evolution;
    subsequent target values enter only through residual sources.
    """
    mms_target_seeded[] && return nothing
    X0, G0, S0 = particle_target_state(target, Float64(pfield.t))
    first_target = vpm.get_np(pfield) + 1
    for i in eachindex(target.ids)
        vpm.add_particle(pfield, view(X0,i,:), view(G0,i,:), S0[i]; static=false)
    end
    mms_target_start[] = first_target
    mms_target_seeded[] = true
    return nothing
end

function native_rhs_at_target(target::CoupledMMSTarget, X, Gamma, sigma, pfield, t)
    """Evaluate installed rVPM position and stretching RHS on a private field.

    Formulation sections: "Actual particle evolution operator" and "Core
    spreading and remapping are handled as actually implemented". FLOWVPM's
    update_particle_states uses U+Uinf for position and the installed
    transposed/classic Jacobian stretching plus rVPM correction for Gamma.
    Core spreading is a discrete diffusion/remap operation, so the ODE core
    source is zero. A deepcopy preserves production operator settings and
    ensures the caller's numerical PFIELD is not mutated.
    """
    tmp = deepcopy(pfield)
    tmp.t = t
    n = length(target.ids)
    total = vpm.get_np(tmp)
    total >= n || error("production ParticleField does not contain the fixed target population")
    first_target = Int(mms_target_start[])
    first_target > 0 && first_target + n - 1 <= total ||
        error("fixed-population MMS target slots are missing from the production field")
    for i in 1:n
        slot = first_target + i - 1
        vpm.set_X(tmp, slot, view(X, i, :))
        vpm.set_Gamma(tmp, slot, view(Gamma, i, :))
        vpm.set_sigma(tmp, slot, sigma[i])
        vpm.set_static(tmp, slot, 0.0)
    end
    tmp.UJ(tmp)
    Uinf = tmp.Uinf(t)
    f, g = tmp.formulation.f, tmp.formulation.g
    zeta0 = tmp.kernel.zeta(0.0)
    rhsX, rhsG = zeros(n,3), zeros(n,3)
    first_target = Int(mms_target_start[])
    for i in 1:n
        part = vpm.get_particle(tmp, first_target + i - 1)
        U, J, G = vpm.get_U(part), vpm.get_J(part), vpm.get_Gamma(part)
        rhsX[i,:] .= U .+ Uinf
        if tmp.transposed
            S = [J[1]*G[1]+J[2]*G[2]+J[3]*G[3],
                 J[4]*G[1]+J[5]*G[2]+J[6]*G[3],
                 J[7]*G[1]+J[8]*G[2]+J[9]*G[3]]
        else
            S = [J[1]*G[1]+J[4]*G[2]+J[7]*G[3],
                 J[2]*G[1]+J[5]*G[2]+J[8]*G[3],
                 J[3]*G[1]+J[6]*G[2]+J[9]*G[3]]
        end
        G2, C, SFS = sum(abs2, G), vpm.get_C(part)[1], vpm.get_SFS(part)
        Z = 0.0
        if G2 > 0
            Z = (f+g)/(1+3*f) * dot(S,G) / G2
            Z -= f/(1+3*f) * C * dot(SFS,G) * sigma[i]^3 / (zeta0*G2)
        end
        rhsG[i,:] .= S .- 3 .* Z .* G .- C .* SFS .* (sigma[i]^3) ./ zeta0
    end
    return rhsX, rhsG, zeros(n)
end

function make_coupled_mms_source(target::CoupledMMSTarget, pfield)
    """Build sources from a private copy of each actual native RK stage.

    The shadow copy starts from analytic target particle states and target
    low-storage memories. It executes the production U/J, particle update, and
    CoreSpreading/remap in their native order. The residual corrects that
    stage map to the analytic state at the next RK abscissa. The numerical
    PFIELD is never read as a target assignment destination.
    """
    n = length(target.ids)
    cache = Dict{Float64,Any}()
    shadow_memory = [zeros(8) for _ in 1:n]
    stage_c = (0.0, 1.0/3.0, 3.0/4.0)
    stage_next_c = (1.0/3.0, 3.0/4.0, 1.0)

    function evaluate_stage_residual!(active_field, stage_time, stage_a, stage_b, dt)
        stage_idx = findfirst(c -> isapprox(stage_time, active_field.t + c*dt; atol=1e-9, rtol=1e-8), stage_c)
        stage_idx === nothing && error("MMS source stage time does not match FLOWVPM RK3 abscissae")
        if stage_idx == 1
            for memory in shadow_memory
                fill!(memory, 0.0)
            end
        end
        t0 = Float64(active_field.t)
        target_time = Float64(stage_time)
        next_time = t0 + stage_next_c[stage_idx] * Float64(dt)
        tmp = deepcopy(active_field)
        tmp.t = t0
        Xh, Gh, Sh = particle_target_state(target, target_time)
        first_target = Int(mms_target_start[])
        for j in 1:n
            slot = first_target + j - 1
            vpm.set_X(tmp, slot, view(Xh, j, :))
            vpm.set_Gamma(tmp, slot, view(Gh, j, :))
            vpm.set_sigma(tmp, slot, Sh[j])
            vpm.set_static(tmp, slot, 0.0)
            vpm.get_M(vpm.get_particle(tmp, slot)) .= shadow_memory[j]
        end

        # Re-run the actual production stage on this target shadow field.
        tmp.SFS(tmp, vpm.BeforeUJ(); a=stage_a, b=stage_b)
        tmp.UJ(tmp)
        tmp.SFS(tmp, vpm.AfterUJ(); a=stage_a, b=stage_b)
        Uinf = tmp.Uinf(t0)
        f, g = tmp.formulation.f, tmp.formulation.g
        zeta0 = tmp.kernel.zeta(0.0)
        vpm.update_particle_states(tmp, stage_a, stage_b, dt, Uinf, f, g, zeta0)
        vpm.viscousdiffusion(tmp, dt; aux1=stage_a, aux2=stage_b)

        Xnext, Gnext, Snext = particle_target_state(target, next_time)
        bdt = Float64(stage_b) * Float64(dt)
        abs(bdt) > eps(Float64) || error("MMS RK3 source encountered a zero stage weight")
        sx, sg, ss = zeros(n,3), zeros(n,3), zeros(n)
        for j in 1:n
            slot = first_target + j - 1
            part = vpm.get_particle(tmp, slot)
            sx[j,:] .= (Xnext[j,:] .- vpm.get_X(part)) ./ bdt
            sg[j,:] .= (Gnext[j,:] .- vpm.get_Gamma(part)) ./ bdt
            ss[j] = (Snext[j] - vpm.get_sigma(part)[]) / bdt

            # Advance target low-storage memory by the same residual terms
            # that MMSRK3StateSource adds after the production stage.
            memory = copy(vpm.get_M(part))
            memory[1:3] .+= Float64(dt) .* sx[j,:]
            memory[4:6] .+= Float64(dt) .* sg[j,:]
            memory[8] += Float64(dt) * ss[j]
            shadow_memory[j] .= memory
        end
        return (sx, sg, ss)
    end

    function source!(particle, stage_time, particle_slot, runtime_pfield,
                     stage_a, stage_b, dt)
        active_field = runtime_pfield === nothing ? pfield : runtime_pfield
        active_field === nothing && error("MMS source callback did not receive its production ParticleField")
        # The first integration precedes FLOWUnsteady's first runtime
        # callback, where the prescribed startup population is seeded.
        if !mms_target_seeded[]
            return (position=zeros(3), circulation=zeros(3), core=0.0)
        end
        vpm.get_np(active_field) >= n || error("fixed-population MMS target particles are missing")
        first_target = Int(mms_target_start[])
        first_target > 0 || error("fixed-population MMS particle IDs have not been seeded")
        dynamic_count = count(i -> !vpm.is_static(vpm.get_particle(active_field, i)),
                              1:vpm.get_np(active_field))
        dynamic_count == n || error("fixed-population MMS particle count changed")
        slot = Int(particle_slot)
        target_slot = slot - first_target + 1
        1 <= target_slot <= n || error("no immutable MMS particle ID for slot $slot")
        particle_id = target.ids[target_slot]
        idx = get(target.id_to_slot, particle_id, 0)
        idx == target_slot || error("MMS immutable particle ID mapping is inconsistent")
        if !haskey(cache, Float64(stage_time))
            cache[Float64(stage_time)] = evaluate_stage_residual!(
                active_field, Float64(stage_time), Float64(stage_a), Float64(stage_b), Float64(dt))
        end
        sx, sg, ss = cache[Float64(stage_time)]
        return (position=copy(sx[idx,:]), circulation=copy(sg[idx,:]), core=ss[idx])
    end
    return source!
end
