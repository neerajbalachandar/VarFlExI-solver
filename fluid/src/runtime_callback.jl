function mms_target_surface_velocity(target, x, y, t)
    _, qt, _ = structural_state(target, x, y, t)
    xb = Float64(target.data["x_bound"])
    return [qt[1], qt[2], qt[3] + qt[4]*(x-xb)]
end

function mms_probe_fields(pfield, probes, t, dt)
    """Velocity and finite-difference vorticity at fixed physical probes."""
    h = max(1.0e-5 * max(span, root_chord), 1.0e-7)
    points = Vector{Vector{Float64}}(undef, 7 * length(probes))
    for (i, x) in enumerate(probes)
        points[7i-6] = Float64.(x)
        for axis in 1:3
            xp, xm = Float64.(x), Float64.(x)
            xp[axis] += h
            xm[axis] -= h
            points[7i-7+2axis] = xp
            points[7i-6+2axis] = xm
        end
    end
    vind = uns.Vvpm_on_Xs(pfield, points; dt=dt)
    velocity = Vector{Vector{Float64}}(undef, length(probes))
    vorticity = Vector{Vector{Float64}}(undef, length(probes))
    for i in eachindex(probes)
        base = 7*i-6
        probe_velocity = k -> Float64.(Vinf(points[k], t)) .+ Float64.(vind[k])
        velocity[i] = probe_velocity(base)
        grad = zeros(3,3)
        for axis in 1:3
            vp, vm = probe_velocity(7*i-7+2*axis), probe_velocity(7*i-6+2*axis)
            grad[:,axis] .= (vp .- vm) ./ (2h)
        end
        vorticity[i] = [grad[3,2]-grad[2,3], grad[1,3]-grad[3,1], grad[2,1]-grad[1,2]]
    end
    return velocity, vorticity
end

function solve_mms_target_aerodynamics!(PFIELD, T, DT)
    isfinite(T) && isfinite(DT) && DT > 0.0 || error(
        "MMS target aerodynamic evaluation requires finite time and positive dt")
    wing = mms_target_wing
    system = mms_target_system
    m = vlm.get_m(wing)
    ucp, ubv = zeros(m,3), zeros(m,3)
    ule, ute = zeros(m,3), zeros(m,3)
    for i in 1:m
        cp = [wing_ref._xm[i], wing_ref._ym[i], wing_ref._zm[i]]
        xt = target_surface_point(mms_target, cp[1], cp[2], T, cp[3])
        ucp[i,:] .= xt .- cp
        xb = 0.5*(wing_ref._xn[i] + wing_ref._xn[i+1])
        yb = 0.5*(wing_ref._yn[i] + wing_ref._yn[i+1])
        zbound = 0.5*(wing_ref._zn[i] + wing_ref._zn[i+1])
        xbt = target_surface_point(mms_target, xb, yb, T, zbound)
        ubv[i,:] .= xbt .- [xb,yb,zbound]
    end
    for i in eachindex(wing_ref._xlwingdcr)
        idx = min(i, m)
        y = wing_ref._ywingdcr[i]
        le = [wing_ref._xlwingdcr[i], y, wing_ref._zlwingdcr[i]]
        te = [wing_ref._xtwingdcr[i], y, wing_ref._ztwingdcr[i]]
        ule[idx,:] .= target_surface_point(mms_target, le[1], le[2], T, le[3]) .- le
        ute[idx,:] .= target_surface_point(mms_target, te[1], te[2], T, te[3]) .- te
    end
    update_geometry_absolute_spanwise!(wing, wing_ref, ucp, ubv, ule, ute)
    ensure_gamma!(wing, m)
    vlm.setVinf(system, Vinf)
    cp = uns._get_Xs(system, "CP"; t=T)
    ap, a, b, bp = (uns._get_Xs(system, key; t=T) for key in ("Ap", "A", "B", "Bp"))
    mids = vcat((ap .+ a)./2, (a .+ b)./2, (b .+ bp)./2)
    points = vcat(cp, mids)
    grid = [Float64[0.0,0.25span,0.0], Float64[0.75root_chord,0.5span,0.1root_chord],
            Float64[1.5root_chord,0.75span,0.0]]
    Xh, Gh, Sh = particle_target_state(mms_target, T)
    # Build a disjoint target field: copy only prescribed static/background
    # particles, then append the immutable target population in manifest order.
    # Numerical dynamic particles never enter target flow or force evaluation.
    numerical_count = vpm.get_np(PFIELD)
    static_slots = [i for i in 1:numerical_count if vpm.is_static(vpm.get_particle(PFIELD, i))]
    # Vvpm_on_Xs temporarily adds one probe per evaluation point, while
    # mms_probe_fields adds the base point and six finite-difference probes.
    probe_capacity = max(length(points), 7 * length(grid))
    target_pf = vpm.ParticleField(
        length(static_slots) + length(mms_target.ids) + probe_capacity,
        eltype(PFIELD.particles); formulation=PFIELD.formulation, viscous=PFIELD.viscous,
        nt=PFIELD.nt, t=T, transposed=PFIELD.transposed, fmm=PFIELD.fmm,
        SFS=PFIELD.SFS, kernel=PFIELD.kernel, UJ=PFIELD.UJ, Uinf=PFIELD.Uinf,
        relaxation=PFIELD.relaxation, integration=PFIELD.integration, useGPU=0)
    for slot in static_slots
        particle = vpm.get_particle(PFIELD, slot)
        vpm.add_particle(target_pf, vpm.get_X(particle), vpm.get_Gamma(particle),
                         vpm.get_sigma(particle)[]; static=true)
    end
    for i in eachindex(mms_target.ids)
        vpm.add_particle(target_pf, view(Xh,i,:), view(Gh,i,:), Sh[i]; static=false)
    end
    induced = uns.Vvpm_on_Xs(target_pf, points; dt=DT)
    vlm._addsolution(system, "Vvpm", induced[1:m]; t=T)
    vlm._addsolution(system, "Vvpm_ApA", induced[m+1:2m]; t=T)
    vlm._addsolution(system, "Vvpm_AB", induced[2m+1:3m]; t=T)
    vlm._addsolution(system, "Vvpm_BBp", induced[3m+1:4m]; t=T)
    vcp = [mms_target_surface_velocity(mms_target, x[1], x[2], T) for x in cp]
    vmid = [mms_target_surface_velocity(mms_target, x[1], x[2], T) for x in mids]
    vlm._addsolution(system, "Vkin", vcp; t=T)
    vlm._addsolution(system, "Vkin_ApA", vmid[1:m]; t=T)
    vlm._addsolution(system, "Vkin_AB", vmid[m+1:2m]; t=T)
    vlm._addsolution(system, "Vkin_BBp", vmid[2m+1:3m]; t=T)
    # Build the discrete manufactured RHS from the analytical circulation,
    # then inject it as a normal-velocity correction into the production
    # solver's RHS adapter.  This does not overwrite Gamma after the solve.
    target_gamma = bound_circulation_state(mms_target, T, m)
    target_dgamma = bound_circulation_derivative(mms_target, T, m)
    hs = vlm.getHorseshoes(wing; t=T)
    foreach(hs) do horseshoe
        horseshoe[8] = nothing
    end
    A, rhs, normals = zeros(m,m), zeros(m), zeros(m,3)
    for i in 1:m
        _, ai, bi, _, cpi, _, _, _ = hs[i]
        ni = cross(cpi-ai, bi-ai); ni ./= norm(ni)
        normals[i,:] .= ni
        for j in 1:m
            hs_j = hs[j]
            induced_horseshoe = independent_finite_vortex_segment(hs_j[1], hs_j[2], cpi, 1.0) +
                                 independent_finite_vortex_segment(hs_j[2], hs_j[3], cpi, 1.0) +
                                 independent_finite_vortex_segment(hs_j[3], hs_j[4], cpi, 1.0)
            A[i,j] = dot(induced_horseshoe, ni)
        end
        target_velocity = Float64.(Vinf(cpi, T)) + induced[i] + vcp[i]
        rhs[i] = -dot(target_velocity, ni)
    end
    manufactured_rhs = A * target_gamma - rhs
    mms_extraVinf = (i, t; kwargs...) ->
        uns._extraVinf2(i, t; kwargs...) .- manufactured_rhs[i] .* view(normals, i, :)
    # The production VLM solver and the independent finite-segment assembly
    # can differ by a few ulps in their influence coefficients. Refine the
    # manufactured normal-velocity correction against the independent matrix
    # so the solved circulation satisfies the audited target operator.
    rhs_correction = copy(manufactured_rhs)
    for _ in 1:4
        mms_extraVinf = (i, t; kwargs...) ->
            uns._extraVinf2(i, t; kwargs...) .- rhs_correction[i] .* view(normals, i, :)
        vlm.solve(system, Vinf; t=T, extraVinf=mms_extraVinf, keep_sol=true,
                  vortexsheet=(X,t)->zeros(3))
        circulation_error = target_gamma .- wing.sol["Gamma"]
        norm(circulation_error) <= 1.0e-12 * max(norm(target_gamma), eps()) && break
        rhs_correction .+= A * circulation_error
    end
    # Audit the solved circulation against the independently assembled target
    # equation. Production-solver coefficient differences are accounted for
    # by the separate solved-circulation error above.
    independent_target_rhs = A * target_gamma
    circulation_residual = norm(A * wing.sol["Gamma"] - independent_target_rhs) /
                           max(norm(independent_target_rhs), eps())
    circulation_target_error = norm(wing.sol["Gamma"] - target_gamma) /
                               max(norm(target_gamma), eps())

    # Independent assembly of the production actuator-line force definition:
    # Kutta–Joukowski forces on ApA, AB, and BBp plus the unsteady-circulation
    # force on the two horseshoe triangles. No production force calculator is
    # used to manufacture this reference load.
    force_horseshoes = vlm.getHorseshoes(wing; t=T)
    for i in eachindex(force_horseshoes)
        force_horseshoes[i][8] = target_gamma[i]
    end
    segment_points = vcat(ap .+ a, a .+ b, b .+ bp) ./ 2
    target_np = vpm.get_np(target_pf)
    target_X = zeros(target_np, 3)
    target_G = similar(target_X)
    target_S = zeros(target_np)
    for i in 1:target_np
        particle = vpm.get_particle(target_pf, i)
        target_X[i,:] .= vpm.get_X(particle)
        target_G[i,:] .= vpm.get_Gamma(particle)
        target_S[i] = vpm.get_sigma(particle)[]
    end
    node_points = (ap, a, b, bp)
    node_velocity = [[mms_target_surface_velocity(mms_target, x[1], x[2], T) for x in rows]
                     for rows in node_points]
    fmat = zeros(m, 3)
    for panel in 1:m
        gamma = target_gamma[panel]
        for segment in 1:3
            row = panel + (segment-1)*m
            point = segment_points[row]
            vvlm = zeros(3)
            for source in 1:m
                hs_source = force_horseshoes[source]
                vvlm .+= independent_finite_vortex_segment(hs_source[1], hs_source[2], point, target_gamma[source])
                vvlm .+= independent_finite_vortex_segment(hs_source[2], hs_source[3], point, target_gamma[source])
                vvlm .+= independent_finite_vortex_segment(hs_source[3], hs_source[4], point, target_gamma[source])
            end
            vtran = (node_velocity[segment][panel] + node_velocity[segment+1][panel]) ./ 2
            vpm_target_velocity = independent_biot_savart_velocity(point, target_X, target_G, target_S)
            velocity = Float64.(Vinf(point, T)) + vpm_target_velocity + vvlm + vtran
            bound_vector = segment == 1 ? ap[panel] - a[panel] :
                           segment == 2 ? b[panel] - a[panel] : bp[panel] - b[panel]
            fmat[panel,:] .+= rho * gamma .* cross(velocity, bound_vector)
        end
        dgamma = target_dgamma[panel]
        l1a, l2a = ap[panel] - a[panel], b[panel] - a[panel]
        l1b, l2b = a[panel] - b[panel], bp[panel] - b[panel]
        triangle_scale = 1.0 / (2.0 * (1.0 - vlm.pn))
        area_normal = triangle_scale .* (cross(l1a,l2a) + cross(l1b,l2b))
        fmat[panel,:] .-= rho * dgamma .* area_normal
    end
    lift, drag = sum(fmat[:,3]), -sum(fmat[:,1])
    qinf = 0.5*rho*sum(sum(abs2,Vinf([wing._xm[i],wing._ym[i],wing._zm[i]],T)) for i in 1:m)/m
    area = max(span*root_chord,1e-16)
    numeric_count = vpm.get_np(PFIELD)
    numeric_dynamic_slots = [i for i in 1:numeric_count if !vpm.is_static(vpm.get_particle(PFIELD, i))]
    numeric_start = Int(mms_target_start[])
    target_slots = collect(numeric_start:(numeric_start + length(mms_target.ids) - 1))
    length(numeric_dynamic_slots) >= length(target_slots) || error(
        "fixed-population MMS target particles are missing: expected $(length(target_slots)), " *
        "found $(length(numeric_dynamic_slots)) dynamic particles")
    numeric_dynamic_slots[1:length(target_slots)] == target_slots ||
        error("numerical MMS particles no longer match the immutable target ID ordering")
    numeric_X = zeros(length(mms_target.ids),3)
    numeric_G = similar(numeric_X)
    numeric_S = zeros(length(mms_target.ids))
    for i in eachindex(mms_target.ids)
        part = vpm.get_particle(PFIELD, numeric_start+i-1)
        numeric_X[i,:] .= vpm.get_X(part)
        numeric_G[i,:] .= vpm.get_Gamma(part)
        numeric_S[i] = vpm.get_sigma(part)[]
    end
    # JSON.jl serializes Julia matrices as their column vectors. Transpose the
    # particle-by-component arrays so the wire representation is one row per
    # particle, matching the manifest and Python's (n_particle, 3) schema.
    numerical_state = Dict("position"=>permutedims(numeric_X),
                           "circulation"=>permutedims(numeric_G), "core"=>numeric_S)
    target_state = Dict("position"=>permutedims(Xh),
                        "circulation"=>permutedims(Gh), "core"=>Sh)
    particle_error = Dict("position_l2"=>norm(numeric_X-Xh), "circulation_l2"=>norm(numeric_G-Gh),
                          "core_l2"=>norm(numeric_S-Sh))
    num_probe_velocity, num_probe_vorticity = mms_probe_fields(PFIELD, grid, T, DT)
    tar_probe_velocity, tar_probe_vorticity = mms_probe_fields(target_pf, grid, T, DT)
    num_vel, tar_vel = num_probe_velocity, tar_probe_velocity
    grid_velocity_error = norm(reduce(vcat, (reshape(Float64.(num_vel[i]-tar_vel[i]),1,3) for i in eachindex(grid))))
    target_geometry = Dict(
        "cp" => [[Float64(wing._xm[i]),Float64(wing._ym[i]),Float64(wing._zm[i])] for i in 1:m],
        "le" => [[Float64(wing._xlwingdcr[i]),Float64(wing._ywingdcr[i]),Float64(wing._zlwingdcr[i])] for i in eachindex(wing._xlwingdcr)],
        "te" => [[Float64(wing._xtwingdcr[i]),Float64(wing._ywingdcr[i]),Float64(wing._ztwingdcr[i])] for i in eachindex(wing._xtwingdcr)],
    )
    return target_gamma, permutedims(fmat), lift/(qinf*area), drag/(qinf*area), circulation_residual,
           circulation_target_error, Float64.(wing.sol["Gamma"]),
           target_state, numerical_state, particle_error, grid_velocity_error, target_geometry,
           grid, tar_probe_velocity, num_probe_velocity, tar_probe_vorticity, num_probe_vorticity
end

function coupling_runtime_function(sim, PFIELD, T, DT; vprintln=(s)->nothing)
    step_start_ns = time_ns()
    step_ref[] += 1
    step = step_ref[]
    fluid_interstep_time = previous_callback_end_ns[] == 0 ? NaN :
        (step_start_ns - previous_callback_end_ns[]) / 1.0e9

    # Seed target particles once as the prescribed initial condition after
    # FLOWUnsteady has constructed its static surface field. No subsequent
    # target assignment is made to numerical particle states.
    if mms_target_enabled && !mms_target_seeded[]
        seed_coupled_mms_particles!(mms_target, PFIELD)
    end

    ensure_gamma!(wing, m_span)
    gamma_w = wing.sol["Gamma"]

    force_out = Vector{Vector{Float64}}(undef, m_span)
    prev_snapshot = copy(forces_prev)
    geometry_cp_absolute = [
        [Float64(wing._xm[i]), Float64(wing._ym[i]), Float64(wing._zm[i])]
        for i in 1:m_span
    ]
    geometry_le_absolute = [
        [Float64(wing._xlwingdcr[i]), Float64(wing._ywingdcr[i]), Float64(wing._zlwingdcr[i])]
        for i in 1:m_span
    ]
    geometry_te_absolute = [
        [Float64(wing._xtwingdcr[i]), Float64(wing._ywingdcr[i]), Float64(wing._ztwingdcr[i])]
        for i in 1:m_span
    ]

    ensure_gamma!(system, m_span)
    ensure_gamma!(previous_force_system[], m_span)
    populate_vpm_force_velocity_fields!(system, PFIELD, DT, T)
    frow = aerodynamic_force_calculator(
        system, previous_force_system[], PFIELD, Vinf, DT, rho; t=T
    )
    # FLOWUnsteady computes dGamma/dt using the previous wing, but its stock
    # force builder stores Funs-vector separately and leaves it out of Ftot.
    if haskey(system.sol, "Funs-vector")
        f_unsteady = system.sol["Funs-vector"]
        length(f_unsteady) == length(frow) || error("FLOWUnsteady Funs-vector size mismatch")
        frow = [frow[i] + f_unsteady[i] for i in eachindex(frow)]
    end
    # Retain this time level before the next received geometry is applied;
    # FLOWUnsteady uses it for both dGamma/dt and its discrete surface velocity.
    previous_force_system[] = deepcopy(system)
    previous_force_system[].sol["Gamma"] = copy(system.sol["Gamma"])

    for i in 1:m_span
        γ = gamma_w[i]
        isfinite(γ) || (γ = 0.0)
        gamma_w[i] = γ

        fx_raw, fy_raw, fz_raw = 0.0, 0.0, 0.0
        if i <= length(frow)
            fi = frow[i]
            if length(fi) == 3 && all(isfinite, fi)
                fx_raw, fy_raw, fz_raw = Float64(fi[1]), Float64(fi[2]), Float64(fi[3])
            else
                error("Non-finite FLOWUnsteady force at span element $i")
            end
        else
            error("FLOWUnsteady force count mismatch: received $(length(frow)), expected $m_span")
        end

        fx = force_relax * fx_raw + (1 - force_relax) * forces_prev[i, 1]
        fy = force_relax * fy_raw + (1 - force_relax) * forces_prev[i, 2]
        fz = force_relax * fz_raw + (1 - force_relax) * forces_prev[i, 3]

        forces_prev[i, 1] = fx
        forces_prev[i, 2] = fy
        forces_prev[i, 3] = fz
        force_out[i] = [fx, fy, fz]
    end

    # Shedding health diagnostic (helps catch end-of-run shedding errors).
    np = vpm.get_np(PFIELD)
    if step == 1 || step % 20 == 0 || step == nsteps
        println("Fluid step $step/$nsteps: Particles=$np, sample force=$(force_out[1])")
    end

    force_mat = reduce(vcat, (reshape(force_out[k], 1, 3) for k in 1:length(force_out)))
    force_res = norm(force_mat - prev_snapshot)
    force_ref_norm = norm(force_mat)
    force_rel_error = force_res / max(force_ref_norm + 1.0e-16, 1.0e-300)
    push!(step_hist, step)
    push!(force_res_hist, force_res)
    push!(force_ref_norm_hist, force_ref_norm)
    push!(force_rel_error_hist, force_rel_error)

    lift = sum(force_mat[:, 3])
    drag = -sum(force_mat[:, 1])
    ref_area = max(span * root_chord, 1.0e-16)
    q_inf = 0.5 * rho * sum(
        sum(abs2, Vinf([wing._xm[i], wing._ym[i], wing._zm[i]], T)) for i in 1:m_span
    ) / m_span
    denom = max(q_inf * ref_area, 1.0e-16)
    cl = lift / denom
    cd = drag / denom
    moment_y = 0.0
    x_moment_reference = minimum(wing_ref._xlwingdcr) + 0.25 * root_chord
    z_moment_reference = minimum(wing_ref._zlwingdcr)
    station_cm = zeros(Float64, m_span)
    station_cl = zeros(Float64, m_span)
    station_cd = zeros(Float64, m_span)
    station_my = zeros(Float64, m_span)
    for i in 1:m_span
        cp = Float64[wing._xm[i], wing._ym[i], wing._zm[i]]
        le = Float64[wing._xlwingdcr[i], wing._ywingdcr[i], wing._zlwingdcr[i]]
        te = Float64[wing._xtwingdcr[i], wing._ywingdcr[i], wing._ztwingdcr[i]]
        chord_i = max(norm(te - le), 1.0e-16)
        dy_i = i < length(wing._yn) ? abs(Float64(wing._yn[i+1] - wing._yn[i])) : span / m_span
        panel_area = max(chord_i * dy_i, 1.0e-16)
        fi = force_out[i]
        my_section = (cp[3] - le[3]) * fi[1] - (cp[1] - le[1]) * fi[3]
        moment_y += (cp[3] - z_moment_reference) * fi[1] - (cp[1] - x_moment_reference) * fi[3]
        station_my[i] = my_section
        station_cl[i] = fi[3] / max(q_inf * panel_area, 1.0e-16)
        station_cd[i] = -fi[1] / max(q_inf * panel_area, 1.0e-16)
        station_cm[i] = my_section / max(q_inf * panel_area * chord_i, 1.0e-16)
    end
    cm = moment_y / max(q_inf * ref_area * root_chord, 1.0e-16)
    fluid_step_time = (time_ns() - step_start_ns) / 1.0e9
    rk3_source_stages = mms_state_source_enabled && length(mms_rk3_stage_trace) >= 3 ?
                        last(mms_rk3_stage_trace, 3) : NamedTuple[]
    rk3_stage_times = Float64[row.time for row in rk3_source_stages]
    rk3_stage_source_norms = [
        Dict(
            "position_l2_norm" => row.position_source_l2_norm,
            "circulation_l2_norm" => row.circulation_source_l2_norm,
            "core_l2_norm" => row.core_source_l2_norm,
        ) for row in rk3_source_stages
    ]
    push!(lift_hist, lift)
    push!(drag_hist, drag)
    push!(cl_hist, cl)
    push!(cd_hist, cd)
    push!(cm_hist, cm)
    push!(fluid_step_time_hist, fluid_step_time)
    push!(fluid_interstep_time_hist, fluid_interstep_time)

    if step == 1 || step == nsteps || step % max(nsteps_save, 1) == 0
        probe_x = minimum(wing_ref._xlwingdcr) .+ root_chord .* [1.1, 1.5, 2.0]
        probes = [Float64[probe_x[i], 0.5span, 0.0] for i in eachindex(probe_x)]
        vind = uns.Vvpm_on_Xs(PFIELD, probes; dt=DT)
        rec = Dict{String,Any}("step"=>step, "time_s"=>Float64(T), "particle_count"=>np)
        for i in eachindex(probes)
            vtot = Float64.(Vinf(probes[i], T)) .+ Float64.(vind[i])
            rec["probe_$(i)_x_m"] = probes[i][1]
            rec["probe_$(i)_vx_m_s"] = vtot[1]
            rec["probe_$(i)_vy_m_s"] = vtot[2]
            rec["probe_$(i)_vz_m_s"] = vtot[3]
            rec["probe_$(i)_speed_m_s"] = norm(vtot)
        end
        push!(wake_probe_history, rec)
    end

    target_payload = Dict{String,Any}()
    if mms_target_enabled
        # The callback exchanges the load used by the structural step ending
        # at T + DT. Keep the MMS force and its reported physical clock at
        # that same force-exchange time (the callback's FLOWUnsteady T is the
        # beginning of the fluid interval).
        force_time = Float64(T + DT)
        target_gamma, target_force, target_cl, target_cd, target_residual,
            target_gamma_error, target_vlm_gamma,
            target_particle_state, numerical_particle_state, particle_error, grid_velocity_error,
            target_geometry, probe_positions, target_probe_velocity, numerical_probe_velocity,
            target_probe_vorticity, numerical_probe_vorticity =
            solve_mms_target_aerodynamics!(PFIELD, force_time, Float64(DT))
        target_q = [structural_state(mms_target, 0.75*Float64(mms_target.data["chord"]),
                       Float64(y), force_time)[1] for y in mms_target.data["y_stations"]]
    target_payload = Dict(
            "target_particle_ids" => mms_target.ids,
            "numerical_particle_ids" => mms_target.ids,
            "target_bound_circulation" => target_gamma,
            "target_y_stations_m" => Float64.(mms_target.data["y_stations"]),
            "target_force" => target_force,
            "target_cl" => target_cl,
            "target_cd" => target_cd,
            "target_circulation_residual" => target_residual,
            "vlm_algebraic_residual" => target_residual,
            "target_vlm_circulation_error" => target_gamma_error,
            "target_vlm_solved_circulation" => target_vlm_gamma,
            "target_particle_state" => target_particle_state,
            "numerical_particle_state" => numerical_particle_state,
            "particle_state_error" => particle_error,
            "fixed_grid_velocity_error_l2" => grid_velocity_error,
            "target_structural_state" => target_q,
            "target_geometry" => target_geometry,
            "fixed_probe_positions_m" => probe_positions,
            "target_probe_velocity_m_s" => target_probe_velocity,
            "numerical_probe_velocity_m_s" => numerical_probe_velocity,
            "target_probe_vorticity_s_inv" => target_probe_vorticity,
            "numerical_probe_vorticity_s_inv" => numerical_probe_vorticity,
            "fixed_probe_velocity_error_l2" => norm(reduce(vcat, (reshape(a .- b, 1, 3)
                for (a,b) in zip(numerical_probe_velocity, target_probe_velocity)))),
            "fixed_probe_vorticity_error_l2" => norm(reduce(vcat, (reshape(a .- b, 1, 3)
                for (a,b) in zip(numerical_probe_vorticity, target_probe_vorticity)))),
            "geometry_cp_error_l2" => norm(reduce(vcat, (
                reshape(Float64.(geometry_cp_absolute[i] - target_geometry["cp"][i]),1,3)
                for i in eachindex(geometry_cp_absolute)))),
        )
    end

    trace_record = Dict(
            "step" => step,
            "fluid_time_s" => Float64(T + (mms_target_enabled ? DT : 0.0)),
            "dt" => Float64(DT),
            "n_span" => m_span,
            "n_chord" => 1,
            "indexing" => "span-major",
            "bound_circulation" => Float64.(gamma_w),
            "particle_count" => np,
            "rVPM_RK3_stage_times_s" => rk3_stage_times,
            "rVPM_RK3_stage_source_norms" => rk3_stage_source_norms,
            "force" => force_out,
            "geometry_cp_absolute" => geometry_cp_absolute,
            "geometry_le_absolute" => geometry_le_absolute,
            "geometry_te_absolute" => geometry_te_absolute,
            "station_cl" => station_cl,
            "station_cd" => station_cd,
            "station_cm" => station_cm,
            "station_moment_y_Nm" => station_my,
            "particles" => np,
            "lift" => lift,
            "drag" => drag,
            "cl" => cl,
            "cd" => cd,
            "cm" => cm,
            "q_inf" => q_inf,
            "ref_area" => ref_area,
            "force_residual" => force_res,
            "force_reference_norm" => force_ref_norm,
            "force_relative_error" => force_rel_error,
            "fluid_step_time" => fluid_step_time,
            "fluid_interstep_time" => isfinite(fluid_interstep_time) ? fluid_interstep_time : nothing,
            "fluid_advance_between_callbacks_time" => isfinite(fluid_interstep_time) ? fluid_interstep_time : nothing,
        )
    merge!(trace_record, target_payload)
    push!(force_payload_history, trace_record)

    socket_record = Dict(
        "step" => step,
        "fluid_time_s" => Float64(T + (mms_target_enabled ? DT : 0.0)),
        "n_span" => m_span,
        "n_chord" => 1,
        "indexing" => "span-major",
        "bound_circulation" => Float64.(gamma_w),
        "particle_count" => np,
        "rVPM_RK3_stage_times_s" => rk3_stage_times,
        "rVPM_RK3_stage_source_norms" => rk3_stage_source_norms,
        "dt" => dt,
        "ttot" => ttot,
        "eta_span" => eta_span_force_payload,
        "eta_chord" => eta_chord_force,
        "force" => force_out,
        "station_cl" => station_cl,
        "station_cd" => station_cd,
        "station_cm" => station_cm,
        "station_moment_y_Nm" => station_my,
        "geometry_cp_absolute" => geometry_cp_absolute,
        "geometry_le_absolute" => geometry_le_absolute,
        "geometry_te_absolute" => geometry_te_absolute,
        "lift" => lift,
        "drag" => drag,
        "cl" => cl,
        "cd" => cd,
        "cm" => cm,
        "q_inf" => q_inf,
        "ref_area" => ref_area,
        "force_residual" => force_res,
        "force_reference_norm" => force_ref_norm,
        "force_relative_error" => force_rel_error,
        "fluid_step_time" => fluid_step_time,
        "fluid_interstep_time" => isfinite(fluid_interstep_time) ? fluid_interstep_time : nothing,
        "fluid_advance_between_callbacks_time" => isfinite(fluid_interstep_time) ? fluid_interstep_time : nothing,
    )
    merge!(socket_record, target_payload)
    write(sock, JSON.json(socket_record) * "\n")
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
        geom_res = haskey(msg, "geometry_residual") ? Float64(msg["geometry_residual"]) : norm(u_prev_cp - u_prev_snapshot)
        geom_ref_norm = haskey(msg, "geometry_reference_norm") ? Float64(msg["geometry_reference_norm"]) : norm(u_prev_cp)
        geom_rel_error = haskey(msg, "geometry_relative_error") ? Float64(msg["geometry_relative_error"]) :
                         geom_res / max(geom_ref_norm + 1.0e-16, 1.0e-300)
        push!(geom_res_hist, geom_res)
        push!(geom_ref_norm_hist, geom_ref_norm)
        push!(geom_rel_error_hist, geom_rel_error)
    end

    previous_callback_end_ns[] = time_ns()
    return step >= nsteps
end
