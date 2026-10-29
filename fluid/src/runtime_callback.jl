function coupling_runtime_function(sim, PFIELD, T, DT; vprintln=(s)->nothing)
    step_start_ns = time_ns()
    step_ref[] += 1
    step = step_ref[]
    fluid_interstep_time = previous_callback_end_ns[] == 0 ? NaN :
        (step_start_ns - previous_callback_end_ns[]) / 1.0e9

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
    push!(fluid_step_time_hist, fluid_step_time)
    push!(fluid_interstep_time_hist, fluid_interstep_time)

    println(
        force_trace_io,
        JSON.json(Dict(
            "step" => step,
            "fluid_time_s" => Float64(T),
            "n_span" => m_span,
            "n_chord" => 1,
            "indexing" => "span-major",
            "bound_circulation" => Float64.(gamma_w),
            "particle_count" => np,
            "rVPM_RK3_stage_times_s" => rk3_stage_times,
            "rVPM_RK3_stage_source_norms" => rk3_stage_source_norms,
            "force" => force_out,
            "particles" => np,
            "lift" => lift,
            "drag" => drag,
            "cl" => cl,
            "cd" => cd,
            "q_inf" => q_inf,
            "ref_area" => ref_area,
            "force_residual" => force_res,
            "force_reference_norm" => force_ref_norm,
            "force_relative_error" => force_rel_error,
            "fluid_step_time" => fluid_step_time,
            "fluid_interstep_time" => isfinite(fluid_interstep_time) ? fluid_interstep_time : nothing,
        )),
    )
    flush(force_trace_io)

    write(sock, JSON.json(Dict(
        "step" => step,
        "fluid_time_s" => Float64(T),
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
        "geometry_cp_absolute" => geometry_cp_absolute,
        "geometry_le_absolute" => geometry_le_absolute,
        "geometry_te_absolute" => geometry_te_absolute,
        "lift" => lift,
        "drag" => drag,
        "cl" => cl,
        "cd" => cd,
        "q_inf" => q_inf,
        "ref_area" => ref_area,
        "force_residual" => force_res,
        "force_reference_norm" => force_ref_norm,
        "force_relative_error" => force_rel_error,
        "fluid_step_time" => fluid_step_time,
        "fluid_interstep_time" => isfinite(fluid_interstep_time) ? fluid_interstep_time : nothing,
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
