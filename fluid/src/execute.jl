uns.run_simulation(simulation, nsteps;
    Vinf=Vinf,
    rho=rho,
    mu=rho * nu,
    p_per_step=Int(p_per_step),
    max_particles=Int(max_particles),
    sigma_vlm_solver=sigma_vlm_solver,
    sigma_vlm_surf=sigma_vlm_surf,
    sigma_rotor_surf=sigma_vlm_surf,
    sigma_vpm_overwrite=sigma_vpm_overwrite,
    vpm_fmm=vpm_fmm_settings,
    vpm_viscous = vpm_viscous,
    vpm_formulation=vpm.rVPM,
    vpm_kernel=vpm.gaussianerf,
    vpm_UJ=vpm.UJ_fmm,
    vpm_SFS=vpm_sfs,
    vpm_integration=vpm_integration,
    vpm_transposed=true,
    vpm_relaxation=vpm.pedrizzetti,
    vpm_surface=true,
    shed_starting=mms_target_enabled ? false : shed_starting,
    shed_unsteady=mms_target_enabled ? false : use_unsteady_shedding,
    shed_boundarylayer=solver_cfg.shed_boundarylayer,
    boundarylayer_prescribedCd=solver_cfg.boundarylayer_prescribed_cd,
    boundarylayer_d=solver_cfg.boundarylayer_dipole_width,
    unsteady_shedcrit=unsteady_shedcrit,
    omit_shedding=omit_shedding_rows,
    wake_coupled=wake_coupled,
    vlm_rlx=vlm_rlx,
    extra_runtime_function=runtime_pipeline,
    save_path=save_path,
    run_name=run_name,
    create_savepath=create_savepath,
    prompt=prompt,
    nsteps_save=nsteps_save,
    save_horseshoes=save_horseshoes
)

if mms_state_source_enabled
    mms_stage_path = joinpath(save_path, run_name * "_mms_rk3_stage_history.csv")
    open(mms_stage_path, "w") do io
        println(io, "stage,time_s,particle_count,position_source_l2_norm_m_per_s,circulation_source_l2_norm_strength_per_s,core_source_l2_norm_m_per_s,source_values_json")
        for row in mms_rk3_stage_trace
            println(io, "$(row.stage),$(row.time),$(row.particle_count),$(row.position_source_l2_norm),$(row.circulation_source_l2_norm),$(row.core_source_l2_norm),$(JSON.json(row.source_values))")
        end
    end
    println("MMS RK3 stage history: $mms_stage_path")
end

# FLOWUnsteady may recreate/clean its output directory after setup has opened
# trace files. Persist the in-memory coupling history after run_simulation so
# the source-enabled target/numerical records remain available on disk.
close(force_trace_io)
open(force_trace_path, "w") do io
    for row in force_payload_history
        println(io, JSON.json(row))
    end
end

    open(diag_path, "w") do io
    println(
        io,
        "step,time,force_residual,force_reference_norm,force_relative_error," *
        "geometry_residual,geometry_reference_norm,geometry_relative_error," *
        "lift,drag,cl,cd,cm,fluid_step_time,fluid_advance_between_callbacks_time"
    )
    n = length(step_hist)
    for k in 1:n
        gres = k <= length(geom_res_hist) ? geom_res_hist[k] : NaN
        gref = k <= length(geom_ref_norm_hist) ? geom_ref_norm_hist[k] : NaN
        grel = k <= length(geom_rel_error_hist) ? geom_rel_error_hist[k] : NaN
        fres = k <= length(force_res_hist) ? force_res_hist[k] : NaN
        fref = k <= length(force_ref_norm_hist) ? force_ref_norm_hist[k] : NaN
        frel = k <= length(force_rel_error_hist) ? force_rel_error_hist[k] : NaN
        lift = k <= length(lift_hist) ? lift_hist[k] : NaN
        drag = k <= length(drag_hist) ? drag_hist[k] : NaN
        cl = k <= length(cl_hist) ? cl_hist[k] : NaN
        cd = k <= length(cd_hist) ? cd_hist[k] : NaN
        cm = k <= length(cm_hist) ? cm_hist[k] : NaN
        step_time = k <= length(fluid_step_time_hist) ? fluid_step_time_hist[k] : NaN
        interstep_time = k <= length(fluid_interstep_time_hist) ? fluid_interstep_time_hist[k] : NaN
        time_value = step_hist[k] * dt
        println(io, "$(step_hist[k]),$(time_value),$(fres),$(fref),$(frel),$(gres),$(gref),$(grel),$(lift),$(drag),$(cl),$(cd),$(cm),$(step_time),$(interstep_time)")
    end
end

station_path = joinpath(save_path, run_name * "_aerodynamic_station_history.csv")
open(station_path, "w") do io
    println(io, "step,time_s,station,x_cp_m,y_cp_m,z_cp_m,bound_circulation_m2_s,fx_N,fy_N,fz_N,lift_N,drag_N,moment_y_Nm,cl_section,cd_section,cm_section")
    for rec in force_payload_history
        forces = rec["force"]
        points = rec["geometry_cp_absolute"]
        gamma = rec["bound_circulation"]
        my = rec["station_moment_y_Nm"]
        cls, cds, cms = rec["station_cl"], rec["station_cd"], rec["station_cm"]
        for i in eachindex(forces)
            f, p = forces[i], points[i]
            vals = (rec["step"], rec["fluid_time_s"], i, p[1], p[2], p[3], gamma[i],
                    f[1], f[2], f[3], f[3], -f[1], my[i], cls[i], cds[i], cms[i])
            println(io, join(vals, ","))
        end
    end
end

probe_path = joinpath(save_path, run_name * "_wake_probe_history.csv")
if !isempty(wake_probe_history)
    probe_columns = sort!(collect(keys(first(wake_probe_history))))
    open(probe_path, "w") do io
        println(io, join(probe_columns, ","))
        for row in wake_probe_history
            println(io, join((get(row, key, NaN) for key in probe_columns), ","))
        end
    end
end

function disconnect_server(sock)

    close(sock)

    println("Fluid socket closed.")

end


disconnect_server(sock)
println("Fluid v9 finished.")
println("Outputs: $save_path")
println("Diagnostics: $diag_path")
println("Aerodynamic station history: $station_path")
!isempty(wake_probe_history) && println("Wake probe history: $probe_path")
println("Force trace: $force_trace_path")
