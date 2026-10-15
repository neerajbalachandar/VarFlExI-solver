function ensure_kj_vpm_velocity_fields!(wing, pfield, dt; fsgm=1)
    field_names = ("Vvpm_ApA", "Vvpm_AB", "Vvpm_BBp")
    all(name -> haskey(wing.sol, name), field_names) && return nothing

    ap = uns._get_Xs(wing, "Ap")
    a = uns._get_Xs(wing, "A")
    b = uns._get_Xs(wing, "B")
    bp = uns._get_Xs(wing, "Bp")
    midpoints = vcat((ap .+ a) ./ 2, (a .+ b) ./ 2, (b .+ bp) ./ 2)
    velocities = uns.Vvpm_on_Xs(pfield, midpoints; dt=dt, fsgm=fsgm)

    m = vlm.get_m(wing)
    for (i, name) in enumerate(field_names)
        if !haskey(wing.sol, name)
            wing.sol[name] = velocities[((i - 1) * m + 1):(i * m)]
        end
    end
    return nothing
end
