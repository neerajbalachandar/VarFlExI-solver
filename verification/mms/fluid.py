"""Taylor-Green manufactured incompressible Navier-Stokes/vorticity fields.

This verifies the analytic reference and source identities only. It is not a
manufactured-source test of FLOWVPM's filtered particle equations.
"""

import sympy as sp


def taylor_green_source(rho=1.225, nu=1.5e-5, speed=1.0, wave_number=1.0):
    
    x, y, z, t = sp.symbols("x y z t", real=True)
    rho, nu, speed, k = map(sp.sympify, (rho, nu, speed, wave_number))
    
    decay = sp.exp(-2 * nu * k**2 * t)
    
    u = sp.Matrix([
        speed * sp.sin(k * x) * sp.cos(k * y) * decay,
        -speed * sp.cos(k * x) * sp.sin(k * y) * decay,
        0,
    ])
    
    p = rho * speed**2 / 4 * (
        sp.cos(2 * k * x) + sp.cos(2 * k * y)
    ) * sp.exp(-4 * nu * k**2 * t)
    
    coords = (x, y, z)
    
    grad_u = u.jacobian(coords)
    
    conv = grad_u * u
    
    lap_u = sp.Matrix([sum(sp.diff(u[i], q, 2) for q in coords) for i in range(3)])
    
    grad_p = sp.Matrix([sp.diff(p, q) / rho for q in coords])
    
    momentum_source = u.diff(t) + conv + grad_p - nu * lap_u
    
    vort = sp.Matrix([
        sp.diff(u[2], y) - sp.diff(u[1], z),
        sp.diff(u[0], z) - sp.diff(u[2], x),
        sp.diff(u[1], x) - sp.diff(u[0], y),
    ])
    expected_vort = sp.Matrix([
        0,
        0,
        2 * speed * k * sp.sin(k * x) * sp.sin(k * y) * decay,
    ])
    
    grad_vort = vort.jacobian(coords)
    
    lap_vort = sp.Matrix([sum(sp.diff(vort[i], q, 2) for q in coords) for i in range(3)])
    
    vort_source = vort.diff(t) + grad_vort * u - grad_u * vort - nu * lap_vort
    
    divergence = sum(sp.diff(u[i], coords[i]) for i in range(3))
    
    return {
        "coords": (x, y, z, t),
        "velocity": u,
        "pressure": p,
        "vorticity": vort,
        "expected_vorticity": expected_vort,
        "curl_residual": vort - expected_vort,
        "momentum_source": momentum_source,
        "vorticity_source": vort_source,
        "divergence": divergence,
    }
