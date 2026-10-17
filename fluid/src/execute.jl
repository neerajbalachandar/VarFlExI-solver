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
    vpm_viscous = vpm_viscous,
    shed_starting=shed_starting,
    shed_unsteady=use_unsteady_shedding,
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

open(diag_path, "w") do io
    println(
        io,
        "step,time,force_residual,force_reference_norm,force_relative_error," *
        "geometry_residual,geometry_reference_norm,geometry_relative_error," *
        "lift,drag,cl,cd,fluid_step_time,fluid_interstep_time"
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
        step_time = k <= length(fluid_step_time_hist) ? fluid_step_time_hist[k] : NaN
        interstep_time = k <= length(fluid_interstep_time_hist) ? fluid_interstep_time_hist[k] : NaN
        time_value = step_hist[k] * dt
        println(io, "$(step_hist[k]),$(time_value),$(fres),$(fref),$(frel),$(gres),$(gref),$(grel),$(lift),$(drag),$(cl),$(cd),$(step_time),$(interstep_time)")
    end
end

function disconnect_server(sock)

    close(sock)

    println("Fluid socket closed.")

end


disconnect_server(sock)
close(force_trace_io)
println("Fluid v9 finished.")
println("Outputs: $save_path")
println("Diagnostics: $diag_path")
println("Force trace: $force_trace_path")
