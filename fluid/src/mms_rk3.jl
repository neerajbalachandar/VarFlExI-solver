"""Opt-in stage-aware position-source adapter for FLOWVPM's native rVPM RK3.

This adapter manufactures only the particle position equation. It delegates all
three stages, native induced-velocity/Jacobian evaluation, stretching,
core-spreading, circulation updates, and relaxation to vpm.rungekutta3.
The source is added to the evaluated particle velocity exactly once per RK
stage, before the native low-storage position update.
"""

function rungekutta3_with_position_source(pfield, dt, source!;
                                           relax::Bool=false,
                                           custom_UJ=nothing,
                                           stage_trace=nothing)
    stage_times = (0.0, 1.0 / 3.0, 3.0 / 4.0)
    stage = Ref(0)
    t0 = Float64(pfield.t)

    function uj_with_mms_source!(pf; kwargs...)
        uj = isnothing(custom_UJ) ? pf.UJ : custom_UJ
        uj(pf; kwargs...)
        stage[] += 1
        stage[] <= length(stage_times) || error("FLOWVPM RK3 evaluated more stages than expected")
        stage_time = t0 + stage_times[stage[]] * Float64(dt)

        source_norm_sq = 0.0
        for i in 1:vpm.get_np(pf)
            particle = vpm.get_particle(pf, i)
            vpm.is_static(particle) && continue
            source = source!(particle, stage_time)
            length(source) == 3 || error("MMS position source must return a 3-vector")
            all(isfinite, source) || error("MMS position source returned non-finite values")
            source_norm_sq += sum(abs2, source)
            vpm.get_U(particle) .+= source
        end
        stage_trace === nothing || push!(stage_trace, (
            stage=stage[], time=stage_time, particle_count=vpm.get_np(pf),
            source_l2_norm=sqrt(source_norm_sq),
        ))
        return nothing
    end

    vpm.rungekutta3(pfield, dt; relax=relax, custom_UJ=uj_with_mms_source!)
    stage[] == length(stage_times) || error(
        "FLOWVPM RK3 evaluated $(stage[]) stages; expected $(length(stage_times))"
    )
    return nothing
end
