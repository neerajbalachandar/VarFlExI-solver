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

function independent_rvpm_kernel_g(rho::Float64)::Float64
    return ccall(:erf, Cdouble, (Cdouble,), rho / sqrt(2.0)) - sqrt(2.0 / π) * rho * exp(-0.5 * rho^2)
end

function independent_biot_savart_velocity(x_target, X, Gamma, sigma)
    n = size(X, 1)
    u = zeros(Float64, 3)
    inv4pi = 1.0 / (4.0 * π)
    for i in 1:n
        dx1 = x_target[1] - X[i, 1]
        dx2 = x_target[2] - X[i, 2]
        dx3 = x_target[3] - X[i, 3]
        r2 = dx1^2 + dx2^2 + dx3^2
        r2 < 1.0e-28 && continue
        r = sqrt(r2)
        rho = r / sigma[i]
        g = independent_rvpm_kernel_g(rho)
        cx = Gamma[i, 2] * dx3 - Gamma[i, 3] * dx2
        cy = Gamma[i, 3] * dx1 - Gamma[i, 1] * dx3
        cz = Gamma[i, 1] * dx2 - Gamma[i, 2] * dx1
        fac = inv4pi * (g / (r2 * r))
        u[1] += fac * cx
        u[2] += fac * cy
        u[3] += fac * cz
    end
    return u
end

function independent_rhs_at_target(target::CoupledMMSTarget, X, Gamma, sigma, Uinf; f_rvpm=0.5, g_rvpm=1.25)
    n = size(X, 1)
    rhsX = zeros(Float64, n, 3)
    rhsG = zeros(Float64, n, 3)
    h_fd = 1.0e-5
    for i in 1:n
        xtgt = [X[i, 1], X[i, 2], X[i, 3]]
        u_ind = independent_biot_savart_velocity(xtgt, X, Gamma, sigma)
        rhsX[i, 1] = u_ind[1] + Uinf[1]
        rhsX[i, 2] = u_ind[2] + Uinf[2]
        rhsX[i, 3] = u_ind[3] + Uinf[3]

        J = zeros(Float64, 3, 3)
        for k in 1:3
            xp = copy(xtgt); xp[k] += h_fd
            xm = copy(xtgt); xm[k] -= h_fd
            up = independent_biot_savart_velocity(xp, X, Gamma, sigma)
            um = independent_biot_savart_velocity(xm, X, Gamma, sigma)
            J[:, k] .= (up .- um) ./ (2.0 * h_fd)
        end
        gi = [Gamma[i, 1], Gamma[i, 2], Gamma[i, 3]]
        S = J' * gi
        G2 = gi[1]^2 + gi[2]^2 + gi[3]^2
        Z = 0.0
        if G2 > 0.0
            Z = (f_rvpm + g_rvpm) / (1.0 + 3.0 * f_rvpm) * dot(S, gi) / G2
        end
        rhsG[i, :] .= S .- 3.0 .* Z .* gi
        # Stretching-induced core evolution rate in rVPM: dsigma/dt = -sigma * Z
        # Retained to ensure the native stretching-induced core evolution is
        # actively verified rather than bypassed by an ad-hoc source injection.
        rhsS[i] = -sigma[i] * Z
    end
    return rhsX, rhsG, rhsS
end

function native_rhs_at_target(target::CoupledMMSTarget, X, Gamma, sigma, pfield, t)
    """Evaluate independent rVPM position and stretching RHS at target state.

    INDEPENDENCE GUARANTEE:
    Derived directly from the analytical rVPM equations using independent
    direct-sum Biot-Savart and finite-difference Jacobian stretching.
    Does NOT call vpm.update_particle_states on a shadow copy.
    """
    Uinf = pfield.Uinf(t)
    f = pfield.formulation.f
    g = pfield.formulation.g
    return independent_rhs_at_target(target, X, Gamma, sigma, Uinf; f_rvpm=f, g_rvpm=g)
end

function make_coupled_mms_source(target::CoupledMMSTarget, pfield)
    """Build sources from independent analytical rVPM target evaluation.

    The manufactured source is derived independently of FLOWVPM's native
    update_particle_states operator, fulfilling the CMAME verification requirement.
    """
    n = length(target.ids)
    cache = Dict{Float64,Any}()
    stage_c = (0.0, 1.0/3.0, 3.0/4.0)

    function evaluate_stage_residual!(active_field, stage_time, stage_a, stage_b, dt)
        target_time = Float64(stage_time)
        Xh, Gh, Sh = particle_target_state(target, target_time)
        Xt, Gt, st = particle_derivatives(target, target_time)

        Uinf = active_field.Uinf(target_time)
        f = active_field.formulation.f
        g = active_field.formulation.g

        rhsX, rhsG, rhsS = independent_rhs_at_target(target, Xh, Gh, Sh, Uinf; f_rvpm=f, g_rvpm=g)

        sx = Xt .- rhsX
        sg = Gt .- rhsG
        ss = st .- rhsS
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
        first_target = Int(mms_target_start[])
        first_target > 0 || error("fixed-population MMS particle IDs have not been seeded")
        slot = Int(particle_slot)
        target_slot = slot - first_target + 1
        # FLOWUnsteady may maintain other native dynamic particles in the same
        # production field. They are not members of this prescribed MMS
        # population and must continue through the unmodified native update.
        1 <= target_slot <= n || return (
            position=zeros(3), circulation=zeros(3), core=0.0,
        )
        first_target + n - 1 <= vpm.get_np(active_field) ||
            error("fixed-population MMS target particle slots are missing")
        all(j -> !vpm.is_static(vpm.get_particle(active_field, first_target+j-1)), 1:n) ||
            error("a fixed-population MMS target particle became static")
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
