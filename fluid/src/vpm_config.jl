function configured_vpm_sfs(name)
    model = lowercase(strip(String(name)))
    model in ("none", "off", "false") && return vpm.SFS_none
    model in ("dynamic_two_level", "cd_twolevel") && return vpm.SFS_Cd_twolevel_nobackscatter
    model in ("dynamic_three_level", "cd_threelevel") && return vpm.SFS_Cd_threelevel_nobackscatter
    error("Unsupported vpm_sfs='$name'; choose none, dynamic_two_level, or dynamic_three_level")
end
