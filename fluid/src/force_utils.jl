function populate_vpm_force_velocity_fields!(vlm_system, pfield, dt, t)
    m = vlm.get_m(vlm_system)
    ap = uns._get_Xs(vlm_system, "Ap"; t=t)
    a = uns._get_Xs(vlm_system, "A"; t=t)
    b = uns._get_Xs(vlm_system, "B"; t=t)
    bp = uns._get_Xs(vlm_system, "Bp"; t=t)
    x_ap_a = (ap .+ a) ./ 2
    x_a_b = (a .+ b) ./ 2
    x_b_bp = (b .+ bp) ./ 2
    induced = uns.Vvpm_on_Xs(pfield, vcat(x_ap_a, x_a_b, x_b_bp); dt=dt)
    length(induced) == 3m || error("FLOWVPM returned an unexpected bound-vortex velocity count")
    vlm._addsolution(vlm_system, "Vvpm_ApA", induced[1:m]; t=t)
    vlm._addsolution(vlm_system, "Vvpm_AB", induced[m+1:2m]; t=t)
    vlm._addsolution(vlm_system, "Vvpm_BBp", induced[2m+1:3m]; t=t)
    return nothing
end
