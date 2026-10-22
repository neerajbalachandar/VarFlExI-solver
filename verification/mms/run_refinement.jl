#!/usr/bin/env julia

# Standalone FLOWUnsteady fluid-resolution studies. This file deliberately
# does not include fluid/src/setup.jl: that production file opens the FSI socket.
using FLOWUnsteady
using FLOWVLM
using FLOWVPM
using JSON
using LinearAlgebra
using Dates
using SHA

const HERE = @__DIR__
const ROOT = normpath(joinpath(HERE, "..", ".."))
const PROJECT = normpath(joinpath(ROOT, ".."))
const uns = FLOWUnsteady
const vlm = FLOWVLM
const vpm = FLOWVPM
include(joinpath(ROOT, "fluid", "src", "force_utils.jl"))
include(joinpath(ROOT, "fluid", "src", "math_utils.jl"))
include(joinpath(ROOT, "fluid", "src", "freestream.jl"))
include(joinpath(ROOT, "fluid", "src", "vpm_config.jl"))

function parse_args(args)
    opts = Dict{String,String}()
    flags = Set{String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--") && occursin("=", arg)
            k, v = split(arg[3:end], "="; limit=2); opts[k] = v
        elseif startswith(arg, "--") && i < length(args) && !startswith(args[i+1], "--")
            opts[arg[3:end]] = args[i+1]; i += 1
        elseif startswith(arg, "--")
            push!(flags, arg[3:end])
        else
            error("Unexpected argument: $arg")
        end
        i += 1
    end
    return opts, flags
end

function simple_yaml(path)
    out = Dict{String,Any}()
    for line in eachline(path)
        s = strip(first(split(line, "#"; limit=2)))
        (isempty(s) || !occursin(":", s)) && continue
        k, v = strip.(split(s, ":"; limit=2))
        if startswith(v, "[") && endswith(v, "]")
            out[k] = [parse(Float64, strip(x)) for x in split(v[2:end-1], ",") if !isempty(strip(x))]
        elseif lowercase(v) in ("true", "false")
            out[k] = lowercase(v) == "true"
        elseif startswith(v, "\"") && endswith(v, "\"")
            out[k] = v[2:end-1]
        else
            n = tryparse(Float64, v)
            out[k] = n === nothing ? v : n
        end
    end
    return out
end

function write_json(path, value)
    open(path, "w") do io
        JSON.print(io, value, 2); println(io)
    end
end

function make_cases(study, cfg, duration, dt)
    f = cfg["fluid"]
    pps = Int(get(f, "particles_per_step", 4))
    lam = Float64(get(f, "lambda_vpm", 2.0))
    nspan = Int(get(f, "n_span", 80))
    p = Int(get(f, "p", 4)); ncrit = Int(get(f, "ncrit", 50))
    theta = Float64(get(f, "theta", 0.4))
    base = Dict("dt"=>dt, "duration"=>duration, "particles_per_step"=>pps,
        "lambda_vpm"=>lam, "n_span"=>nspan, "fmm_p"=>p,
        "fmm_ncrit"=>ncrit, "fmm_theta"=>theta,
        "fmm_relative_tolerance"=>Float64(get(f,"relative_tolerance",1e-3)),
        "fmm_absolute_tolerance"=>Float64(get(f,"absolute_tolerance",1e-3)),
        "fmm_autotune_p"=>Bool(get(f,"autotune_p",true)),
        "fmm_autotune_ncrit"=>Bool(get(f,"autotune_ncrit",true)),
        "fmm_autotune_regularization"=>Bool(get(f,"autotune_regularization",false)),
        "remove_every"=>Int(get(f,"remove_every",100)),
        "sphere_factor"=>Float64(get(f,"sphere_factor",200.0)),
        "strength_min_factor"=>Float64(get(f,"min_factor",1e-3)),
        "strength_max_factor"=>Float64(get(f,"max_factor",50.0)),
        "sigma_min_factor"=>Float64(get(f,"min_factor",1e-3)),
        "sigma_max_factor"=>Float64(get(f,"max_factor",50.0)))
    levels = if study == "particles"
        [("pps_$n", merge(copy(base), Dict("particles_per_step"=>n))) for n in (1,2,4,8)]
    elseif study == "time"
        [("dt_div_$(2^i)", merge(copy(base), Dict("dt"=>dt/2.0^i,
            "particles_per_step"=>(4,2,1)[i+1]))) for i in 0:2]
    elseif study == "core"
        [("lambda_$x", merge(copy(base), Dict("lambda_vpm"=>x))) for x in (1.0,1.5,2.0,3.0)]
    elseif study == "panels"
        [("nspan_$n", merge(copy(base), Dict("n_span"=>n))) for n in (40,80,160)]
    elseif study == "fmm"
        [("tol_$(replace(string(x), "."=>"e"))", merge(copy(base),
            Dict("fmm_relative_tolerance"=>x,"fmm_absolute_tolerance"=>x))) for x in (1e-2,1e-3,1e-4)]
    elseif study == "wake"
        conservative = merge(copy(base), Dict("remove_every"=>typemax(Int), "sphere_factor"=>1e6,
            "strength_min_factor"=>0.0, "strength_max_factor"=>1e12,
            "sigma_min_factor"=>0.0, "sigma_max_factor"=>1e12))
        [("baseline_removal", base), ("retention", conservative),
         ("sphere_only", merge(copy(base),Dict("strength_min_factor"=>0.0,"strength_max_factor"=>1e12,
            "sigma_min_factor"=>0.0,"sigma_max_factor"=>1e12))),
         ("strength_only", merge(copy(base),Dict("sphere_factor"=>1e6,
            "sigma_min_factor"=>0.0,"sigma_max_factor"=>1e12))),
         ("sigma_only", merge(copy(base),Dict("sphere_factor"=>1e6,
            "strength_min_factor"=>0.0,"strength_max_factor"=>1e12)))]
    else
        error("Unknown study '$study'")
    end
    return [(name, merge(values, Dict("study"=>study))) for (name,values) in levels]
end

function csv_write(path, header, rows)
    open(path,"w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join((x isa AbstractFloat ? repr(x) : string(x) for x in row), ","))
        end
    end
end

function run_case(casepath)
    case = JSON.parsefile(casepath)
    cfg = simple_yaml(case["fluid_config_path"])
    f = cfg
    dt = Float64(case["dt"]); duration = Float64(case["duration"])
    nsteps = round(Int, duration/dt); duration = nsteps*dt
    nspan = Int(case["n_span"]); pps = Int(case["particles_per_step"])
    U = Float64(f["vinf"]); aoa = Float64(f["aoa_deg"])
    rho = Float64(f["rho"]); nu = Float64(f["nu"])
    span = Float64(case["span"]); cr = Float64(case["root_chord"]); ct = Float64(case["tip_chord"])
    sweep = Float64(case["leading_edge_sweep"])
    sigma0 = Float64(case["lambda_vpm"])*U*dt/pps
    Vinf = configured_freestream(U, aoa,
        Float64.(get(f, "freestream_gradient", zeros(9))),
        Float64.(get(f, "freestream_oscillation", zeros(3))),
        Float64.(get(f, "freestream_wavenumber", zeros(3))),
        Float64(get(f, "freestream_frequency_hz", 0.0)),
        deg2rad(Float64(get(f, "freestream_phase_deg", 0.0))),
        Float64.(get(f, "freestream_reference", zeros(3))))
    wing = FLOWVLM.Wing(0.0,0.0,0.0,cr,Float64(get(f,"twist_root",0.0)))
    FLOWVLM.addchord(wing,sweep,span,0.0,ct,Float64(get(f,"twist_tip",0.0)),nspan;r=1.0)
    system = FLOWVLM.WingSystem(); FLOWVLM.addwing(system,"Wing",wing)
    FLOWVLM.setVinf(system,Vinf)
    ensure_gamma!(system, FLOWVLM.get_m(system))
    vehicle = FLOWUnsteady.VLMVehicle(system;vlm_system=system,wake_system=system)
    Vvehicle(t)=zeros(3); anglevehicle(t)=zeros(3)
    maneuver=FLOWUnsteady.KinematicManeuver((),(),Vvehicle,anglevehicle)
    simulation=FLOWUnsteady.Simulation(vehicle,maneuver,0.0,0.0,duration;Vinit=zeros(3),Winit=zeros(3))
    fmm=FLOWVPM.FMM(p=Int(case["fmm_p"]),ncrit=Int(case["fmm_ncrit"]),
        theta=Float64(case["fmm_theta"]),shrink_recenter=Bool(get(f,"shrink_recenter",true)),
        relative_tolerance=Float64(case["fmm_relative_tolerance"]),
        absolute_tolerance=Float64(case["fmm_absolute_tolerance"]),
        autotune_p=Bool(case["fmm_autotune_p"]),autotune_ncrit=Bool(case["fmm_autotune_ncrit"]),
        autotune_reg_error=Bool(case["fmm_autotune_regularization"]),
        default_rho_over_sigma=Float64(get(f,"default_rho_over_sigma",1.0)),
        min_ncrit=Int(get(f,"minimum_ncrit",3)))
    viscous_name=lowercase(string(get(f,"viscous_scheme","CoreSpreading")))
    viscous=startswith(viscous_name,"corespreading") ?
        FLOWVPM.CoreSpreading(nu,sigma0,1.0) : nothing
    refarea=span*cr
    qinf_reference=0.5*rho*U^2
    polar=strip(string(get(f,"section_polar","")))
    polarpath=strip(string(get(f,"section_polar_path","")))
    shed_boundarylayer=Bool(get(f,"shed_boundarylayer",false))
    boundarylayer_width=Float64(get(f,"boundarylayer_dipole_width",0.0))
    shed_boundarylayer && boundarylayer_width <= 0 &&
        error("shed_boundarylayer requires boundarylayer_dipole_width > 0, matching production validation")
    parasitic_args=isempty(polarpath) ? NamedTuple() : (;read_path=polarpath)
    forcefun=FLOWUnsteady.generate_calc_aerodynamicforce(add_parasiticdrag=!isempty(polar),
        airfoilpolar=polar,parasitic_args...)
    previous=Ref(deepcopy(system))
    eta=collect(range(0.0,1.0;length=nspan+1))[1:end-1] .+ 0.5/nspan
    probes=[ [sweep+1.25cr, span*y, -0.25span] for y in (0.25,0.5,0.75)]
    headers=vcat(["step","time_s","particle_count","runtime_step_s","lift_N","drag_N","CL","CD"],
        ["Gamma_$i" for i in 1:nspan],
        ["u_probe$(j)_$(k)" for j in 1:3 for k in ("x","y","z")],
        ["omega_probe$(j)_$(k)" for j in 1:3 for k in ("x","y","z")])
    rows=Vector{Vector{Any}}(); step=Ref(0); started=time()
    fmm_error=Ref(NaN); removed=Ref(0); inserted=Ref(0); lastcount=Ref(0)
    effective_fmm=Ref(fmm)
    function collect_history(sim, PFIELD, T, DT; optargs...)
        t0=time(); step[]+=1; np=FLOWVPM.get_np(PFIELD)
        effective_fmm[]=deepcopy(PFIELD.fmm)
        gam=Float64.(wing.sol["Gamma"])
        force=zeros(3)
        ensure_gamma!(system, FLOWVLM.get_m(system))
        ensure_gamma!(previous[], FLOWVLM.get_m(previous[]))
        if !isempty(wing.sol["Gamma"])
            populate_vpm_force_velocity_fields!(system,PFIELD,DT,T)
            frow=forcefun(system,previous[],PFIELD,Vinf,DT,rho;t=T)
            if haskey(system.sol,"Funs-vector")
                f_unsteady=system.sol["Funs-vector"]
                frow=[frow[i]+f_unsteady[i] for i in eachindex(frow)]
            end
            force=reduce(+,frow;init=zeros(3))
        end
        previous[]=deepcopy(system)
        lift=force[3]; drag=-force[1]
        m=FLOWVLM.get_m(system)
        qinf=0.5*rho*sum(sum(abs2,Vinf([wing._xm[i],wing._ym[i],wing._zm[i]],T)) for i in 1:m)/m
        denom=max(qinf*refarea,1e-16)
        CL=lift/denom; CD=drag/denom
        induced=isempty(probes) || np==0 ? [zeros(3) for _ in probes] : FLOWUnsteady.Vvpm_on_Xs(PFIELD,probes;dt=DT)
        omega=[zeros(3) for _ in probes]
        if np>0
            for (j,X) in enumerate(probes), i in 1:np
                dx=X-FLOWVPM.get_X(PFIELD,i); sig=FLOWVPM.get_sigma(PFIELD,i)[]
                omega[j] .+= FLOWVPM.get_Gamma(PFIELD,i) .* (FLOWVPM.zeta_gauserf(norm(dx)/sig)/sig^3)
            end
        end
        vals=Any[step[],T,np,time()-t0,lift,drag,CL,CD]
        append!(vals,gam)
        for u in induced; append!(vals,u); end
        for w in omega; append!(vals,w); end
        push!(rows,vals)
        if case["study"]=="fmm" && step[]==nsteps && np<=2000
            direct=deepcopy(PFIELD); approx=deepcopy(PFIELD)
            FLOWVPM.UJ_direct(direct;reset=true)
            FLOWVPM.UJ_fmm(approx;reset=true,autotune=false)
            ud=hcat([collect(FLOWVPM.get_U(direct,i)) for i in 1:np]...)
            uf=hcat([collect(FLOWVPM.get_U(approx,i)) for i in 1:np]...)
            fmm_error[]=norm(uf-ud)/max(norm(ud),1e-14)
        end
        return false
    end
    # Wake-removal functions are the same FLOWUnsteady operators used by the
    # coupled setup; wrap them to count reliably observed removals.
    rmv=2*2/max(pps,1)*dt/(1/12)
    lo=rmv*Float64(case["strength_min_factor"]); hi=rmv*Float64(case["strength_max_factor"])
    smin=sigma0*Float64(case["sigma_min_factor"]); smax=sigma0*Float64(case["sigma_max_factor"])
    treatments=Function[]
    if Int(case["remove_every"]) < typemax(Int)
        push!(treatments,FLOWUnsteady.remove_particles_sphere((Float64(case["sphere_factor"])*span)^2,1;Xoff=[0.5span,0.0,0.0]))
        push!(treatments,FLOWUnsteady.remove_particles_strength(lo^2,hi^2;every_nsteps=max(Int(case["remove_every"]),1)))
        push!(treatments,FLOWUnsteady.remove_particles_sigma(smin,smax;every_nsteps=max(Int(case["remove_every"]),1)))
    end
    function pipeline(sim,PFIELD,T,DT;optargs...)
        before=FLOWVPM.get_np(PFIELD)
        inserted[] += max(0,before-lastcount[])
        for treatment in treatments; treatment(sim,PFIELD,T,DT;optargs...); end
        after=FLOWVPM.get_np(PFIELD)
        removed[] += max(0,before-after)
        lastcount[]=after
        return collect_history(sim,PFIELD,T,DT;optargs...)
    end
    sigma_surface=Float64(get(f,"sigma_surface",0.025))*span
    estimated_particles=Int(ceil((nsteps+1)*(FLOWVLM.get_m(system)*(pps+1)+pps)))
    maxparticles=max(Int(case["max_particles"]),
        Int(ceil(Float64(get(f,"max_particles_safety_factor",3.0))*estimated_particles)))
    FLOWVLM.VLMSolver._regularize(Bool(get(f,"regularize_vlm",true)))
    FLOWUnsteady.run_simulation(simulation,nsteps;Vinf=Vinf,rho,mu=rho*nu,
        p_per_step=pps,max_particles=maxparticles,vpm_formulation=FLOWVPM.rVPM,
        vpm_kernel=FLOWVPM.gaussianerf,vpm_UJ=FLOWVPM.UJ_fmm,
        vpm_SFS=configured_vpm_sfs(get(f,"vpm_sfs","none")),vpm_integration=FLOWVPM.rungekutta3,
        vpm_transposed=true,vpm_relaxation=FLOWVPM.pedrizzetti,vpm_surface=true,
        vpm_viscous=viscous,vpm_fmm=fmm,
        sigma_vpm_overwrite=sigma0,sigma_vlm_solver=Float64(get(f,"sigma_solver",-1.0)),
        sigma_vlm_surf=sigma_surface,sigma_rotor_surf=sigma_surface,
        shed_starting=Bool(get(f,"start_with_wake",true)),shed_unsteady=Bool(get(f,"unsteady",true)),
        unsteady_shedcrit=Float64(get(f,"unsteady_criterion",0.0)),
        shed_boundarylayer=shed_boundarylayer,
        boundarylayer_prescribedCd=Float64(get(f,"boundarylayer_prescribed_cd",0.1)),
        boundarylayer_d=boundarylayer_width,
        wake_coupled=Bool(get(f,"wake_coupled",true)),vlm_rlx=Float64(get(f,"relaxation",0.35)),
        extra_runtime_function=pipeline,save_path=nothing,run_name=case["case_id"],
        create_savepath=false,prompt=false,verbose=true,verbose_nsteps=max(1,round(Int,1/dt)),
        nsteps_save=max(nsteps+1,1),save_horseshoes=false,save_static_particles=false)
    csv_write(joinpath(case["case_output"],"history.csv"),headers,rows)
    actual_fmm=effective_fmm[]
    return Dict("status"=>"COMPLETE","runtime_s"=>time()-started,"nsteps_completed"=>length(rows),
        "final_particle_count"=>(isempty(rows) ? 0 : rows[end][3]),"particles_inserted_observed"=>inserted[],
        "particles_removed_observed"=>removed[],"fmm_direct_relative_velocity_error"=>(isfinite(fmm_error[]) ? fmm_error[] : nothing),
        "geometry"=>Dict("span_m"=>span,"root_chord_m"=>cr,"tip_chord_m"=>ct,
            "leading_edge_sweep_m"=>sweep,"aoa_deg"=>aoa,
            "twist_root_deg"=>Float64(get(f,"twist_root",0.0)),
            "twist_tip_deg"=>Float64(get(f,"twist_tip",0.0)),"n_span"=>nspan),
        "physical"=>Dict("freestream_speed_m_s"=>U,"aoa_deg"=>aoa,"rho_kg_m3"=>rho,
            "nu_m2_s"=>nu,"dynamic_pressure_reference_Pa"=>qinf_reference,"reference_area_m2"=>refarea,
            "section_polar"=>polar,"parasitic_drag_enabled"=>!isempty(polar),
            "boundary_layer_shedding_enabled"=>shed_boundarylayer),
        "effective"=>Dict("dt_s"=>dt,"duration_s"=>duration,"particles_per_step"=>pps,
            "lambda_vpm"=>case["lambda_vpm"],"sigma_vpm_initial_m"=>sigma0,
            "release_spacing_nominal_m"=>U*dt/pps,"nsteps"=>nsteps,"fmm_p"=>actual_fmm.p,
            "fmm_ncrit"=>actual_fmm.ncrit,"fmm_theta"=>actual_fmm.theta,
            "fmm_relative_tolerance"=>actual_fmm.relative_tolerance,
            "fmm_absolute_tolerance"=>actual_fmm.absolute_tolerance,
            "n_span"=>nspan,"core_spreading_beta"=>(viscous isa FLOWVPM.CoreSpreading ? viscous.beta : nothing),
            "fmm_autotune_p"=>case["fmm_autotune_p"],"fmm_autotune_ncrit"=>case["fmm_autotune_ncrit"],
            "fmm_autotune_regularization"=>case["fmm_autotune_regularization"],
            "fmm_default_rho_over_sigma"=>actual_fmm.default_rho_over_sigma,
            "fmm_min_ncrit"=>actual_fmm.min_ncrit,"fmm_shrink_recenter"=>actual_fmm.shrink_recenter,
            "vpm_sfs"=>string(get(f,"vpm_sfs","none")),
            "remove_every"=>case["remove_every"],"sphere_factor"=>case["sphere_factor"],
            "max_particles"=>maxparticles,"estimated_particles"=>estimated_particles,
            "strength_min_factor"=>case["strength_min_factor"],
            "strength_max_factor"=>case["strength_max_factor"],
            "sigma_min_factor"=>case["sigma_min_factor"],"sigma_max_factor"=>case["sigma_max_factor"],
            "viscous_scheme"=>string(typeof(viscous)),"integrator"=>"FLOWVPM.rungekutta3",
            "formulation"=>"FLOWVPM.rVPM","kernel"=>"FLOWVPM.gaussianerf"),
        "history_path"=>joinpath(case["case_output"],"history.csv"))
end

function main(args)
    opts,flags=parse_args(args)
    if haskey(opts,"internal-case")
        result=run_case(opts["internal-case"])
        write_json(joinpath(JSON.parsefile(opts["internal-case"])["case_output"],"result.json"),result)
        return
    end
    configdir=abspath(get(opts,"config-dir",joinpath(ROOT,"config")))
    fluidpath=joinpath(configdir,"fluid_params.yaml")
    solid=simple_yaml(joinpath(configdir,"solid_params.yaml"))
    coupling=simple_yaml(joinpath(configdir,"coupling_params.yaml"))
    fluid=simple_yaml(fluidpath)
    duration=Float64(get(coupling,"total_time",5.0)); dt0=duration/Int(get(coupling,"n_steps",1000))
    if haskey(opts,"smoke-steps")
        steps=parse(Int,opts["smoke-steps"]); dt0=parse(Float64,get(opts,"dt",string(dt0))); duration=steps*dt0
        fluid["max_particles"] = 10000
    end
    studies=split(get(opts,"study","particles"),",")
    "all" in studies && (studies=["particles","time","core","panels","fmm","wake"])
    out=abspath(get(opts,"output",joinpath(HERE,"results")))
    mkpath(out)
    if "execute" in flags
        for study in studies
            study_dir=joinpath(out,study); mkpath(study_dir)
            case_defs=make_cases(study,Dict("fluid"=>fluid),duration,dt0)
            if haskey(opts,"case-only")
                filter!(pair -> pair[1] == opts["case-only"], case_defs)
                isempty(case_defs) && error("--case-only $(opts["case-only"]) is not a level in study '$study'")
            end
            manifest=Dict{String,Any}("study"=>study,"generated_utc"=>string(now()),
                "executable"=>string(Base.julia_cmd()),"julia_version"=>string(VERSION),
                "package_versions"=>Dict("FLOWUnsteady"=>string(pkgversion(FLOWUnsteady)),
                    "FLOWVLM"=>string(pkgversion(FLOWVLM)),"FLOWVPM"=>string(pkgversion(FLOWVPM))),
                "active_project"=>Base.active_project(),"project_root"=>PROJECT,
                "project_sha256"=>(isfile(joinpath(PROJECT,"Project.toml")) ? bytes2hex(sha256(read(joinpath(PROJECT,"Project.toml")))) : "missing"),
                "manifest_sha256"=>(isfile(joinpath(PROJECT,"Manifest.toml")) ? bytes2hex(sha256(read(joinpath(PROJECT,"Manifest.toml")))) : "missing"),
                "source_config_paths"=>Dict(name=>joinpath(configdir,name) for name in
                    ("fluid_params.yaml","solid_params.yaml","coupling_params.yaml")),
                "source_config_sha256"=>Dict(name=>bytes2hex(sha256(read(joinpath(configdir,name)))) for name in
                    ("fluid_params.yaml","solid_params.yaml","coupling_params.yaml")),
                "base_fluid_config_path"=>fluidpath,"base_fluid_config_sha256"=>bytes2hex(sha256(read(fluidpath))),
                "baseline_parameters"=>Dict("lambda_vpm"=>get(fluid,"lambda_vpm",2.0),
                    "particles_per_step"=>get(fluid,"particles_per_step",4),"n_span"=>get(fluid,"n_span",80),
                    "vpm_sfs"=>get(fluid,"vpm_sfs","none")),
                "flow_formulation"=>Dict("vpm_formulation"=>"FLOWVPM.rVPM",
                    "kernel"=>"FLOWVPM.gaussianerf","velocity_solver"=>"FLOWVPM.UJ_fmm",
                    "integrator"=>"FLOWVPM.rungekutta3","sfs"=>get(fluid,"vpm_sfs","none"),
                    "viscous_scheme"=>get(fluid,"viscous_scheme","CoreSpreading")),
        "freestream_override"=>"none; uses shared configured V(x,t) function", "cases"=>Any[])
            for (name,values) in case_defs
                id="$(study)_$(name)"; cdir=joinpath(study_dir,id); mkpath(cdir)
                case=merge(values,Dict("case_id"=>id,"case_output"=>cdir,
                    "fluid_config_path"=>fluidpath,"span"=>Float64(solid["span"]),
                    "root_chord"=>Float64(solid["root_chord"]),"tip_chord"=>Float64(solid["tip_chord"]),
                    "max_particles"=>Int(get(fluid,"max_particles",400000)),
                    "leading_edge_sweep"=>Float64(get(solid,"leading_edge_sweep",0.0))))
                for config_name in ("fluid_params.yaml","solid_params.yaml","coupling_params.yaml")
                    cp(joinpath(configdir,config_name),joinpath(cdir,"source_"*config_name);force=true)
                end
                casefile=joinpath(cdir,"case_config.json"); write_json(casefile,case)
                cmd=`$(Base.julia_cmd()) --project=$PROJECT $(@__FILE__) --internal-case $casefile`
                stdout=open(joinpath(cdir,"stdout.log"),"w"); stderr=open(joinpath(cdir,"stderr.log"),"w")
                started=time(); code=0
                try
                    proc=run(pipeline(cmd,stdout=stdout,stderr=stderr);wait=false)
                    wait(proc); code=proc.exitcode
                catch err
                    code=1; write(stderr,"\nDriver failure: $(sprint(showerror,err))\n")
                finally
                    close(stdout);close(stderr)
                end
                entry=Dict("case_id"=>id,"requested"=>case,"config_path"=>casefile,
                    "config_sha256"=>bytes2hex(sha256(read(casefile))),"return_code"=>code,
                    "status"=>(code==0 && isfile(joinpath(cdir,"result.json")) ? "COMPLETE" : "FAILED"),
                    "runtime_s"=>time()-started,"stdout_path"=>joinpath(cdir,"stdout.log"),
                    "stderr_path"=>joinpath(cdir,"stderr.log"))
                isfile(joinpath(cdir,"result.json")) && (entry["result"]=JSON.parsefile(joinpath(cdir,"result.json")))
                push!(manifest["cases"],entry); write_json(joinpath(study_dir,"manifest.json"),manifest)
                println("[$study] $(entry["status"]) $id ($(round(entry["runtime_s"];digits=1)) s)")
            end
        end
    else
        for study in studies
            study in ("particles","time","core","panels","fmm","wake") || error("Unknown study $study")
            plans=make_cases(study,Dict("fluid"=>fluid),duration,dt0)
            write_json(joinpath(out,"$(study)_plan.json"),Dict("study"=>study,"execute"=>false,
                "freestream_override"=>"none; uses shared configured V(x,t) function","cases"=>[Dict("case_id"=>n,"parameters"=>v) for (n,v) in plans]))
            println("Wrote plan $(joinpath(out,"$(study)_plan.json")); no simulations launched.")
        end
    end
end

main(ARGS)
