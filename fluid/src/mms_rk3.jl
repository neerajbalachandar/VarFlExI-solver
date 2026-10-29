"""Opt-in stage-aware state-source adapter for FLOWVPM's native rVPM RK3.

It retains FLOWVPM's native U/J, stretching update, particle-state update,
viscous/core-spreading update, and RK3 coefficients. Position sources enter
the stage velocity; circulation and core-width residuals enter the matching
low-storage RK accumulators after the native stage map.
"""

struct MMSRK3StateSource{S,T,I}
    source!::S
    stage_trace::T
    initialize!::I
end

MMSRK3StateSource(source!, trace; initialize! = (_pf)->nothing) =
    MMSRK3StateSource{typeof(source!),typeof(trace),typeof(initialize!)}(source!, trace, initialize!)

(adapter::MMSRK3StateSource)(pfield, dt; relax=false, custom_UJ=nothing) =
    (adapter.initialize!(pfield); rungekutta3_with_state_source(
        pfield, dt, adapter.source!; relax=relax, custom_UJ=custom_UJ,
        stage_trace=adapter.stage_trace,
    ))

function mms_source_components(source!, particle, stage_time, particle_id, pfield=nothing,
                               stage_a=nothing, stage_b=nothing, dt=nothing)
    value = if applicable(source!, particle, stage_time, particle_id, pfield,
                          stage_a, stage_b, dt)
        source!(particle, stage_time, particle_id, pfield, stage_a, stage_b, dt)
    elseif applicable(source!, particle, stage_time, particle_id, pfield)
        source!(particle, stage_time, particle_id, pfield)
    elseif applicable(source!, particle, stage_time, particle_id)
        source!(particle, stage_time, particle_id)
    else
        source!(particle, stage_time)
    end
    if value isa AbstractVector
        return (position=value, circulation=zeros(3), core=0.0)
    end
    return (
        position=get(value, :position, zeros(3)),
        circulation=get(value, :circulation, zeros(3)),
        core=get(value, :core, 0.0),
    )
end

function rungekutta3_with_state_source(pfield, dt, source!;
                                       relax::Bool=false,
                                       custom_UJ=nothing,
                                       stage_trace=nothing)
    stage_times = (0.0, 1.0 / 3.0, 3.0 / 4.0)
    stage = Ref(0)
    t0 = Float64(pfield.t)
    stage_sources = Vector{Any}(undef, vpm.get_np(pfield))

    function uj_with_mms_source!(pf; a, b, kwargs...)
        uj = isnothing(custom_UJ) ? pf.UJ : custom_UJ
        uj(pf; kwargs...)
        stage[] += 1
        stage[] <= length(stage_times) || error("FLOWVPM RK3 evaluated more stages than expected")
        stage_time = t0 + stage_times[stage[]] * Float64(dt)

        source_norm_sq = 0.0
        circulation_source_norm_sq = 0.0
        core_source_norm_sq = 0.0
        source_values = Any[]
        for i in 1:vpm.get_np(pf)
            particle = vpm.get_particle(pf, i)
            if vpm.is_static(particle)
                stage_sources[i] = nothing
                continue
            end
            source = mms_source_components(source!, particle, stage_time, i, pf, a, b, dt)
            length(source.position) == 3 || error("MMS position source must return a 3-vector")
            length(source.circulation) == 3 || error("MMS circulation source must return a 3-vector")
            all(isfinite, source.position) || error("MMS position source returned non-finite values")
            all(isfinite, source.circulation) || error("MMS circulation source returned non-finite values")
            isfinite(source.core) || error("MMS core source returned a non-finite value")
            stage_sources[i] = source
            push!(source_values, Dict("particle_slot"=>i,
                "position"=>Float64.(source.position),
                "circulation"=>Float64.(source.circulation),
                "core"=>Float64(source.core)))
            source_norm_sq += sum(abs2, source.position)
            circulation_source_norm_sq += sum(abs2, source.circulation)
            core_source_norm_sq += abs2(source.core)
            vpm.get_U(particle) .+= source.position
        end
        stage_trace === nothing || push!(stage_trace, (
            stage=stage[], time=stage_time, particle_count=vpm.get_np(pf),
            position_source_l2_norm=sqrt(source_norm_sq),
            circulation_source_l2_norm=sqrt(circulation_source_norm_sq),
            core_source_l2_norm=sqrt(core_source_norm_sq),
            source_values=source_values,
        ))
        return nothing
    end

    # Match the installed RK3 storage reset and stage order. The production
    # update and viscous dispatch perform the native RHS, stretching, and core
    # remap; manufactured circulation/core residuals enter the same RK memory
    # after the native stage map so they remain additive on later stages.
    for i in 1:vpm.get_np(pfield)
        particle = vpm.get_particle(pfield, i)
        vpm.is_static(particle) && continue
        vpm.get_M(particle) .= 0
    end
    Uinf = pfield.Uinf(pfield.t)
    f, g = pfield.formulation.f, pfield.formulation.g
    zeta0 = pfield.kernel.zeta(0.0)
    for (a, b) in ((0.0, 1/3), (-5/9, 15/16), (-153/128, 8/15))
        pfield.SFS(pfield, vpm.BeforeUJ(); a=a, b=b)
        uj_with_mms_source!(pfield; a=a, b=b,
                           reset_sfs=vpm.isSFSenabled(pfield.SFS), reset=true,
                           sfs=vpm.isSFSenabled(pfield.SFS))
        pfield.SFS(pfield, vpm.AfterUJ(); a=a, b=b)
        vpm.update_particle_states(pfield, a, b, dt, Uinf, f, g, zeta0)
        vpm.viscousdiffusion(pfield, dt; aux1=a, aux2=b)

        for i in 1:vpm.get_np(pfield)
            source = stage_sources[i]
            source === nothing && continue
            particle = vpm.get_particle(pfield, i)
            M = vpm.get_M(particle)
            Gamma = vpm.get_Gamma(particle)
            sigma = vpm.get_sigma(particle)
            for k in 1:3
                M[4+k-1] += dt * source.circulation[k]
                Gamma[k] += b * dt * source.circulation[k]
            end
            M[8] += dt * source.core
            sigma[] += b * dt * source.core
        end
    end

    if relax
        vpm._reset_particles(pfield)
        pfield.UJ(pfield)
        for i in 1:vpm.get_np(pfield)
            particle = vpm.get_particle(pfield, i)
            vpm.is_static(particle) || pfield.relaxation(pfield, i)
        end
    end
    stage[] == length(stage_times) || error(
        "FLOWVPM RK3 evaluated $(stage[]) stages; expected $(length(stage_times))"
    )
    return nothing
end

# FLOWVPM's CoreSpreading method recognizes RK3 by exact equality against the
# integration function stored in the ParticleField. MMSRK3StateSource is a
# callable adapter and cannot satisfy that identity check. This narrow method
# reproduces the installed RK3 core update (including its scheduled reset and
# strength remap) for fields registered with this adapter.
function production_zeta_fmm_compat!(pfield)
    """Evaluate FLOWVPM's installed zeta FMM with its required source flag.

    Formulation section: "Core spreading and remapping are handled as
    actually implemented". FLOWVPM 4.x calls the FastMultipole tree
    constructor without the required `target=false` positional flag; this
    adapter supplies that flag and preserves the installed direct-list zeta
    accumulation and downstream RBF remap.
    """
    # FLOWVPM's stock zeta_fmm calls Tree((pfield,); ...), but the installed
    # FastMultipole version requires the explicit `target` positional flag.
    # Empty fields have no remap contributions and need no tree.
    vpm.get_np(pfield) == 0 && return nothing
    options = pfield.fmm
    leaf_size, shrink_recenter = options.ncrit, options.shrink_recenter
    acceptance = options.theta
    zeta = pfield.kernel.zeta
    tree = vpm.FastMultipole.Tree((pfield,), false; leaf_size, shrink_recenter)
    _, direct_list = vpm.FastMultipole.build_interaction_lists(
        tree.branches, tree.branches, leaf_size, acceptance, false, true, true
    )
    sort_index = tree.sort_index_list[1]
    for (i_target, i_source) in direct_list
        target_branch, source_branch = tree.branches[i_target], tree.branches[i_source]
        for i_source_particle in source_branch.bodies_index[1]
            source_particle = vpm.get_particle(pfield, sort_index[i_source_particle])
            source_x = vpm.get_X(source_particle)
            for i_target_particle in target_branch.bodies_index[1]
                target_particle = vpm.get_particle(pfield, sort_index[i_target_particle])
                target_x = vpm.get_X(target_particle)
                dx = source_x[1]-target_x[1]
                dy = source_x[2]-target_x[2]
                dz = source_x[3]-target_x[3]
                r = sqrt(dx*dx + dy*dy + dz*dz)
                target_sigma = vpm.get_sigma(target_particle)[]
                zeta_sigma = zeta(r/target_sigma) / target_sigma^3
                target_gamma = vpm.get_Gamma(target_particle)
                J = vpm.get_J(source_particle)
                J[1] += target_gamma[1]*zeta_sigma
                J[2] += target_gamma[2]*zeta_sigma
                J[3] += target_gamma[3]*zeta_sigma
            end
        end
    end
    return nothing
end

function vpm.viscousdiffusion(
    pfield::vpm.ParticleField{R,F,V,U,S,K,J,I,L,N},
    scheme::vpm.CoreSpreading,
    dt;
    aux1=0,
    aux2=0,
) where {R,F,V,U,S,K,J,I<:MMSRK3StateSource,L,N}
    proceed = false
    for p in vpm.iterator(pfield)
        vpm.get_M(p)[7] = aux1 * vpm.get_M(p)[7] + dt * 2 * scheme.nu
        vpm.get_sigma(p)[] = sqrt(vpm.get_sigma(p)[]^2 + aux2 * vpm.get_M(p)[7])
    end
    if isapprox(aux2, 8 / 15, atol=1e-7)
        proceed = true
    end
    # The fixed-population MMS may deliberately start from an empty field
    # when it is used only to exercise coupled bound-flow feedback.  There is
    # no core-remap work to do in that case, and FLOWVPM's FMM zeta evaluator
    # does not accept an empty ParticleField.
    if proceed && vpm.get_np(pfield) > 0
        scheme.t_sgm += dt
        beta_cur = sqrt(2 * scheme.nu * scheme.t_sgm / scheme.sgm0^2 + 1)
        if beta_cur >= scheme.beta
            if scheme.zeta === vpm.zeta_fmm
                production_zeta_fmm_compat!(pfield)
            else
                scheme.zeta(pfield)
            end
            for p in vpm.iterator(pfield)
                for i in 1:3
                    vpm.get_M(p)[6+i] = vpm.get_J(p)[i]
                end
                vpm.get_sigma(p)[] = scheme.sgm0
            end
            scheme.rbf(pfield, scheme)
            scheme.t_sgm = 0
        end
    end
    return nothing
end
