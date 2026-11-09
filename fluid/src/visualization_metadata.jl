"""Write physical-time ParaView collections for FLOWUnsteady's snapshots.

FLOWUnsteady names particle snapshots with the integer solver step and, in
the installed legacy version, writes that integer into the XDMF `<Time>` tag.
That tag is seconds to ParaView, so normalize it to `step * dt` and provide
explicit temporal collections for particles and VLM panels.
"""
function _snapshot_step(name::String, prefix::String, extension::String)
    startswith(name, prefix) || return nothing
    endswith(name, extension) || return nothing
    stem_end = lastindex(name) - length(extension)
    stem_end < firstindex(name) && return nothing
    stem = name[firstindex(name):stem_end]
    suffix = stem[length(prefix)+1:end]
    return tryparse(Int, suffix)
end

function _xml_escape(value::String)
    return replace(value, "&" => "&amp;", "\"" => "&quot;", "<" => "&lt;", ">" => "&gt;")
end

function _write_xdmf_collection(path::String, entries)
    isempty(entries) && return
    open(path, "w") do io
        println(io, "<?xml version=\"1.0\"?>")
        println(io, "<Xdmf Version=\"3.0\" xmlns:xi=\"http://www.w3.org/2001/XInclude\">")
        println(io, "  <Domain>")
        println(io, "    <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">")
        for (_step, name) in entries
            println(io, "      <xi:include href=\"$(_xml_escape(name))\" xpointer=\"xpointer(/Xdmf/Domain/Grid)\" />")
        end
        println(io, "    </Grid>")
        println(io, "  </Domain>")
        println(io, "</Xdmf>")
    end
end

function _write_vtk_collection(path::String, entries, dt::Float64)
    isempty(entries) && return
    open(path, "w") do io
        println(io, "<?xml version=\"1.0\"?>")
        println(io, "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">")
        println(io, "  <Collection>")
        for (step, name) in entries
            println(io, "    <DataSet timestep=\"$(step * dt)\" part=\"0\" file=\"$(_xml_escape(name))\" />")
        end
        println(io, "  </Collection>")
        println(io, "</VTKFile>")
    end
end

function write_visualization_time_metadata(save_path::String, run_name::String, dt::Float64)
    files = readdir(save_path)
    particle_series = Tuple{Int,String}[]
    static_particle_series = Tuple{Int,String}[]
    panel_series = Tuple{Int,String}[]

    for name in files
        step = _snapshot_step(name, "$(run_name)_pfield.", ".xmf")
        if step !== nothing
            xmf_path = joinpath(save_path, name)
            content = read(xmf_path, String)
            physical_time = string(step * dt)
            content = replace(content, r"<Time Value=\"[^\"]+\"\s*/>" => "<Time Value=\"$physical_time\" />")
            write(xmf_path, content)
            push!(particle_series, (step, name))
            continue
        end

        step = _snapshot_step(name, "$(run_name)_staticpfield.", ".xmf")
        if step !== nothing
            xmf_path = joinpath(save_path, name)
            content = read(xmf_path, String)
            physical_time = string(step * dt)
            content = replace(content, r"<Time Value=\"[^\"]+\"\s*/>" => "<Time Value=\"$physical_time\" />")
            write(xmf_path, content)
            push!(static_particle_series, (step, name))
            continue
        end

        step = _snapshot_step(name, "$(run_name)_Wing_vlm.", ".vtk")
        step === nothing || push!(panel_series, (step, name))
    end

    sort!(particle_series; by=first)
    sort!(static_particle_series; by=first)
    sort!(panel_series; by=first)
    _write_xdmf_collection(joinpath(save_path, "$(run_name)_pfield_series.xmf"), particle_series)
    _write_xdmf_collection(joinpath(save_path, "$(run_name)_staticpfield_series.xmf"), static_particle_series)
    _write_vtk_collection(joinpath(save_path, "$(run_name)_Wing_vlm.pvd"), panel_series, dt)

    println("ParaView time metadata: particle, static-particle, and panel collections use physical seconds in $save_path")
end
