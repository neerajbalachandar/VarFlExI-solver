"""Build the configured V(x,t) field shared by production and verification."""
function configured_freestream(speed, aoa_deg, gradient_values, amplitudes,
                               wavenumbers, frequency_hz, phase_rad, reference)
    length(gradient_values) == 9 || error("freestream_gradient must have 9 row-major values")
    length(amplitudes) == 3 || error("freestream_oscillation must have 3 components")
    length(wavenumbers) == 3 || error("freestream_wavenumber must have 3 components")
    length(reference) == 3 || error("freestream_reference must have 3 components")

    base = speed .* [cosd(aoa_deg), 0.0, sind(aoa_deg)]
    gradient = permutedims(reshape(gradient_values, 3, 3))
    return function (X, t)
        x, y, z = X .- reference
        kx, ky, kz = wavenumbers
        phase = 2pi * frequency_hz * t + phase_rad
        gust = amplitudes .* sin.(phase .+ [ky * y, kz * z, kx * x])
        return base + gradient * (X .- reference) + gust
    end
end
