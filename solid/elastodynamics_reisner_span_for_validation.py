from dolfin import *
import json
import os
import socket
import csv

import numpy as np
from scipy.spatial import cKDTree
from time import perf_counter

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True


T = float(os.getenv("COUPLING_TTOT", "15"))
Nsteps = int(os.getenv("COUPLING_NSTEPS", "3000"))
dt_value = T / Nsteps
dt = Constant(dt_value)

span = float(os.getenv("SOLID_SPAN", "0.3"))
root_chord = float(os.getenv("SOLID_ROOT_CHORD", "0.1"))
tip_chord = float(os.getenv("SOLID_TIP_CHORD", "0.1"))
thickness_ratio = float(os.getenv("SOLID_THICKNESS_RATIO", "0.01"))
leading_edge_sweep = float(os.getenv("SOLID_LE_SWEEP", "0.0"))

# For 2D Reissner-Mindlin plate: only chord and span directions (no thickness discretization)
nx = int(os.getenv("SOLID_NX", "20"))
ny = int(os.getenv("SOLID_NY", "240"))

# Communication stations for the fluid v10 spanwise panels.
n_span_comm = int(os.getenv("COUPLING_NSPAN_COMM", "80"))
n_chord_comm = 1
m_panels_comm = n_span_comm * n_chord_comm
span_sampling_mode = os.getenv("COUPLING_SPAN_SAMPLING", "node-stride").strip().lower()
span_custom_stride = os.getenv("COUPLING_SPAN_STRIDE")


def build_eta_span_comm(n_span_vals, ny_vals, mode):
    if n_span_vals < 1:
        raise ValueError("COUPLING_NSPAN_COMM must be >= 1")
    if n_span_vals == 1:
        return np.array([0.0], dtype=float), np.array([0], dtype=int)

    if mode == "midpoint":
        eta = (np.arange(n_span_vals, dtype=float) + 0.5) / n_span_vals
        idx = np.round(eta * ny_vals).astype(int)
        return eta, idx

    if mode == "node-stride":
        # Example: ny=200 and n_span=4 -> indices [0, 50, 100, 150].
        stride = max(1, ny_vals // n_span_vals)
        idx = np.arange(n_span_vals, dtype=int) * stride
        idx = np.clip(idx, 0, max(ny_vals, 1))
        if np.unique(idx).size < n_span_vals:
            idx = np.linspace(0, max(ny_vals - 1, 0), n_span_vals).round().astype(int)
        eta = idx.astype(float) / max(float(ny_vals), 1.0)
        return eta, idx

    if mode == "custom-stride":
        if span_custom_stride is None:
            raise ValueError(
                "COUPLING_SPAN_SAMPLING=custom-stride requires COUPLING_SPAN_STRIDE"
            )
        stride = int(span_custom_stride)
        if stride < 1:
            raise ValueError("COUPLING_SPAN_STRIDE must be >= 1")
        last_idx = (n_span_vals - 1) * stride
        if last_idx > ny_vals:
            raise ValueError(
                "Unsafe custom span stride: "
                f"(COUPLING_NSPAN_COMM-1)*COUPLING_SPAN_STRIDE = {last_idx} "
                f"exceeds SOLID_NY = {ny_vals}. Reduce stride/stations or increase SOLID_NY."
            )
        idx = np.arange(n_span_vals, dtype=int) * stride
        eta = idx.astype(float) / max(float(ny_vals), 1.0)
        return eta, idx

    if mode == "linspace":
        eta = np.linspace(0.0, 1.0, n_span_vals, endpoint=False)
        idx = np.round(eta * ny_vals).astype(int)
        return eta, idx

    raise ValueError(f"Unsupported COUPLING_SPAN_SAMPLING='{mode}'")


eta_span_comm, eta_span_comm_indices = build_eta_span_comm(n_span_comm, ny, span_sampling_mode)
eta_cp = float(os.getenv("COUPLING_ETA_CP", "0.75"))
eta_cp_comm = np.array([eta_cp], dtype=float)

work_conservative_mode = True
rbf_radius = float(os.getenv("COUPLING_RBF_RADIUS", os.getenv("COUPLING_RBF_EPS", "0.08")))
rbf_neighbors = int(os.getenv("COUPLING_RBF_NEIGHBORS", "24"))
max_abs_force_component = float(os.getenv("COUPLING_MAX_FORCE_COMPONENT", "5.0e3"))

DEBUG_IO = os.getenv("COUPLING_DEBUG_IO", "0").strip().lower() not in ("0", "false", "no")
edge_eval_xi_eps = float(os.getenv("COUPLING_EDGE_EVAL_XI_EPS", "1.0e-6"))

enforce_chord_projection = (
    os.getenv("COUPLING_ENFORCE_CHORD", "0").lower()
    not in ("0", "false", "no")
)

enforce_span_projection = (
    os.getenv("COUPLING_ENFORCE_SPAN", "0").lower()
    not in ("0", "false", "no")
)




#just use for now
enforce_chord_projection = False
enforce_span_projection = False




def chord_at(y_val):
    eta = min(max(y_val / span, 0.0), 1.0)
    return root_chord + (tip_chord - root_chord) * eta


def x_leading_edge_at(y_val):
    eta = min(max(y_val / span, 0.0), 1.0)
    return leading_edge_sweep * eta


def naca_half_thickness(xi):
    xi_clip = min(max(xi, 0.0), 1.0)
    return 5.0 * thickness_ratio * (
        0.2969 * np.sqrt(xi_clip)
        - 0.1260 * xi_clip
        - 0.3516 * xi_clip ** 2
        + 0.2843 * xi_clip ** 3
        - 0.1015 * xi_clip ** 4
    )


# 2d midsurface mesh z=0
mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, span), nx, ny)

# Map reference square [0,1]x[0,span] to physical tapered/swept wing planform in chord-span plane
coords = mesh.coordinates()
for i in range(coords.shape[0]):
    xi = coords[i, 0]
    y_val = coords[i, 1]
    chord = chord_at(y_val)
    x_le = x_leading_edge_at(y_val)
    coords[i, 0] = x_le + xi * chord
    # y-coordinate stays unchanged


def left(x, on_boundary):
    return near(x[1], 0.0) and on_boundary


# For 2D plate, mark the whole top surface as coupling interface
facet_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
facet_markers.set_all(0)
DomainBoundary().mark(facet_markers, 1)  # Mark all exterior facets
ds_aero = Measure("ds", domain=mesh, subdomain_data=facet_markers)

# Plate thickness
plate_thickness = root_chord * thickness_ratio
h = Constant(plate_thickness)

# Reissner-Mindlin mixed element space:
#   u_mem: membrane displacement (in-plane) [ux, uy] -> CG2
#   w: transverse displacement -> CG2
#   theta: plate rotations [theta_x, theta_y] -> CG2kG
U_el = VectorElement("CG", mesh.ufl_cell(), 2, dim=2)
W_el = FiniteElement("CG", mesh.ufl_cell(), 2)
T_el = VectorElement("CG", mesh.ufl_cell(), 2, dim=2)
mixed_element = MixedElement([U_el, W_el, T_el])
V = FunctionSpace(mesh, mixed_element)

# For output: expand 3D displacement (add zero z-component) for aero transfer
Vt = VectorFunctionSpace(mesh, "CG", 1, dim=3)
Vsig = TensorFunctionSpace(mesh, "DG", 0, shape=(2, 2))

t_aero = Function(Vt, name="AerodynamicTraction")

E = float(os.getenv("SOLID_E", "2.6e10")) # 5 to 4 to 2.5e9, 3.4e9; 2.6e10
nu = float(os.getenv("SOLID_NU", "0.35"))
rho_s = float(os.getenv("SOLID_RHO", "1100.0"))
rho = Constant(rho_s)
eta_m = Constant(float(os.getenv("SOLID_ETA_M", "0.02")))
eta_k = Constant(float(os.getenv("SOLID_ETA_K", "1.0e-6")))
kappa_shear = Constant(5.0 / 6.0)  # Shear correction factor for Reissner-Mindlin

alpha_m = Constant(0.10)
alpha_f = Constant(0.20)
gamma = Constant(0.5 + alpha_f - alpha_m)
beta = Constant((gamma + 0.5) ** 2 / 4.0)

print(
    f"Reissner-Mindlin plate v18: span={span} m, c_root={root_chord} m, c_tip={tip_chord} m, "
    f"h={plate_thickness:.4e} m, E={E:.3e} Pa, rho={rho_s} kg/m^3, comm_stations={n_span_comm}, "
    f"sampling={span_sampling_mode}"
)
print(f"Time setup: T={T} s, Nsteps={Nsteps}, dt={dt_value}")
print(
    f"Force transfer: compact Wendland C2 RBF, radius={rbf_radius:.4e} m, "
    f"neighbors={rbf_neighbors}"
)
print(
    f"Plate shear correction kappa={float(kappa_shear)}"
)
if DEBUG_IO:
    print(f"Span comm node indices = {eta_span_comm_indices.tolist()}")
    print(f"Span comm etas         = {eta_span_comm.tolist()}")

dq_trial = TrialFunction(V)
q_test = TestFunction(V)
q = Function(V, name="PlateState")
q_old = Function(V)
v_old = Function(V)
a_old = Function(V)

Uinf = 0.30
chord = 0.1

kG = 1.90

freq = kG * Uinf / (np.pi * chord)

omega = 2.0 * np.pi * freq

a_root = 0.175 * chord

# zero= Constant((0.0, 0.0, 0.0))

# bc = DirichletBC(V,zero, left)

u_zero_2d = Constant((0.0, 0.0))

# heave_expr = Expression(
#     "A*cos(omega*t)",
#     # A=a_root,
#     A = 0,
#     omega=omega,
#     t=0.0,
#     degree=2
# )

Tramp = 3.0 / freq     # one oscillation period

heave_expr = Expression(
    """
    (t < Tramp) ?
    A*pow(sin(0.5*pi*t/Tramp),2)*cos(omega*t)
    :
    A*cos(omega*t)
    """,
    A=a_root,
    omega=omega,
    Tramp=Tramp,
    pi=np.pi,
    t=0.0,
    degree=4
)

bc_u = DirichletBC(V.sub(0), u_zero_2d, left)

bc_w = DirichletBC(
    V.sub(1),
    heave_expr,
    left
)

bc_theta = DirichletBC(
    V.sub(2),
    u_zero_2d,
    left
)

bcs = [bc_u, bc_w, bc_theta]

# Homogenous Boundary Conditions
bc_w_hom = DirichletBC(V.sub(1), Constant(0.0), left)
bcs_hom = [bc_u, bc_w_hom, bc_theta]


I2 = Identity(2)



def split_state(q_fun):
    if isinstance(q_fun, tuple):
        return q_fun
    return split(q_fun)

def membrane_strain(u_mem):
    """2D membrane strain tensor: e(u) = sym(grad(u))"""
    return sym(grad(u_mem))


def curvature(theta):
    """Curvature tensor from plate rotations: k(theta) = sym(grad(theta))"""
    return sym(grad(theta))


def shear_strain(theta, w):
    """Transverse shear strain: gamma = grad(w) - theta"""
    return grad(w) - theta


def membrane_stress(u_mem):
    """Membrane stress resultant N = C_m : e(u)"""
    eps = membrane_strain(u_mem)
    coeff = E * h / (1.0 - nu ** 2)
    return coeff * ((1.0 - nu) * eps + nu * tr(eps) * I2)


def bending_moment(theta):
    """Bending moment M = D : k(theta)"""
    kap = curvature(theta)
    coeff = E * h ** 3 / (12.0 * (1.0 - nu ** 2))
    return coeff * ((1.0 - nu) * kap + nu * tr(kap) * I2)


def displacement_3d(q_fun):
    """Reconstruct 3D displacement from plate unknowns for aerodynamic coupling"""
    u_mem, w, _theta = split_state(q_fun)
    # Return 3D vector: [ux, uy, w] where w is out-of-plane (z-direction)
    return as_vector((u_mem[0], u_mem[1], w))


def m_form(q_trial, q_test):

    if isinstance(q_trial, tuple):
        u_t, w_t, theta_t = q_trial
    else:
        u_t, w_t, theta_t = split(q_trial)

    if isinstance(q_test, tuple):
        u_x, w_x, theta_x = q_test
    else:
        u_x, w_x, theta_x = split(q_test)

    inertia_rot = rho*h**3/12.0

    return (
        rho*h*(inner(u_t,u_x)+w_t*w_x)*dx
        + inertia_rot*inner(theta_t,theta_x)*dx
    )

#get from the fenics demo 
def k_form(q_trial, q_test):

    if isinstance(q_trial, tuple):
        u_t, w_t, theta_t = q_trial
    else:
        u_t, w_t, theta_t = split(q_trial)

    if isinstance(q_test, tuple):
        u_x, w_x, theta_x = q_test
    else:
        u_x, w_x, theta_x = split(q_test)

    eps_x = membrane_strain(u_x)
    kap_x = curvature(theta_x)
    gam_t = shear_strain(theta_t, w_t)
    gam_x = shear_strain(theta_x, w_x)

    N_t = membrane_stress(u_t)
    M_t = bending_moment(theta_t)
    G_shear = Constant(E / (2.0 * (1.0 + nu)))
    K_shear = kappa_shear * G_shear * h

    membrane_term = inner(N_t, eps_x) * dx
    bending_term = inner(M_t, kap_x) * dx
    shear_term = K_shear * inner(gam_t, gam_x) * dx
    return membrane_term + bending_term + shear_term


def c_form(q_trial, q_test):
    """Damping form: proportional damping"""
    return eta_m * m_form(q_trial, q_test) + eta_k * k_form(q_trial, q_test)


def Wext(q_test):
    """External work from aerodynamic loading"""
    v_disp = displacement_3d(q_test)
    return dot(v_disp, t_aero) * ds_aero(1)


def update_a(q_new, q_prev, v_prev, a_prev, ufl=True):
    if ufl:
        dt_ = dt
        beta_ = beta
    else:
        dt_ = float(dt)
        beta_ = float(beta)
    return (q_new - q_prev - dt_ * v_prev) / beta_ / dt_ ** 2 - (
        1.0 - 2.0 * beta_
    ) / (2.0 * beta_) * a_prev


def update_v(a_new, q_prev, v_prev, a_prev, ufl=True):
    if ufl:
        dt_ = dt
        gamma_ = gamma
    else:
        dt_ = float(dt)
        gamma_ = float(gamma)
    return v_prev + dt_ * ((1.0 - gamma_) * a_prev + gamma_ * a_new)


def update_fields(q_fun, q_prev, v_prev, a_prev):
    q_vec = q_fun.vector()
    q0_vec = q_prev.vector()
    v0_vec = v_prev.vector()
    a0_vec = a_prev.vector()

    a_vec = update_a(q_vec, q0_vec, v0_vec, a0_vec, ufl=False)
    v_vec = update_v(a_vec, q0_vec, v0_vec, a0_vec, ufl=False)

    v_prev.vector()[:] = v_vec
    a_prev.vector()[:] = a_vec
    q_prev.vector()[:] = q_vec

def avg(x_old, x_new, alpha):
    return alpha*x_old + (1.0-alpha)*x_new


# Split states FIRST
q_u, q_w, q_th = split(q)

qo_u, qo_w, qo_th = split(q_old)

vo_u, vo_w, vo_th = split(v_old)

ao_u, ao_w, ao_th = split(a_old)


# Accelerations componentwise
a_u_new = update_a(q_u, qo_u, vo_u, ao_u, ufl=True)
a_w_new = update_a(q_w, qo_w, vo_w, ao_w, ufl=True)
a_th_new = update_a(q_th, qo_th, vo_th, ao_th, ufl=True)

# Velocities componentwise
v_u_new = update_v(a_u_new, qo_u, vo_u, ao_u, ufl=True)
v_w_new = update_v(a_w_new, qo_w, vo_w, ao_w, ufl=True)
v_th_new = update_v(a_th_new, qo_th, vo_th, ao_th, ufl=True)


# Generalized-alpha fields
a_alpha = (
    avg(ao_u, a_u_new, alpha_m),
    avg(ao_w, a_w_new, alpha_m),
    avg(ao_th, a_th_new, alpha_m)
)

v_alpha = (
    avg(vo_u, v_u_new, alpha_f),
    avg(vo_w, v_w_new, alpha_f),
    avg(vo_th, v_th_new, alpha_f)
)

q_alpha = (
    avg(qo_u, q_u, alpha_f),
    avg(qo_w, q_w, alpha_f),
    avg(qo_th, q_th, alpha_f)
)


res = (
    m_form(a_alpha, q_test)
    + c_form(v_alpha, q_test)
    + k_form(q_alpha, q_test)
    - Wext(q_test)
)

jac = derivative(res, q, dq_trial)

# Coupled step-1 is commonly the hardest: the solid starts from rest/zero
# displacement and suddenly receives aerodynamic loads. The original defaults
# (1e-8/1e-7) are often unrealistically strict for this transient and can cause
# an early abort that then cascades into the fluid side as a socket disconnect.

newton_atol = float(os.getenv("SOLID_NEWTON_ATOL", "1.0e-6"))
newton_rtol = float(os.getenv("SOLID_NEWTON_RTOL", "1.0e-5"))
newton_maxit = int(os.getenv("SOLID_NEWTON_MAXIT", "120"))
dq_newton = Function(V)
linear_solver = LUSolver("mumps")

# Optional ramp on the applied coupling forces during the Newton iterations.
force_ramp_iters = int(os.getenv("SOLID_FORCE_RAMP_ITERS", "20"))


def local_project(v_expr, Vout, u_out=None):
    dv = TrialFunction(Vout)
    v_test = TestFunction(Vout)
    a_proj = inner(dv, v_test) * dx
    b_proj = inner(v_expr, v_test) * dx
    solver_local = LocalSolver(a_proj, b_proj)
    solver_local.factorize()
    if u_out is None:
        u_out = Function(Vout)
        solver_local.solve_local_rhs(u_out)
        return u_out
    solver_local.solve_local_rhs(u_out)
    return u_out


def as_eta_array(values, n):
    if values is None:
        return np.linspace(0.0, 1.0, n)
    arr = np.asarray(values, dtype=float).reshape(-1)
    if len(arr) != n:
        return np.linspace(0.0, 1.0, n)
    if n > 1:
        arr = np.clip(arr, 0.0, 1.0)
        arr = np.maximum.accumulate(arr)
        if arr[-1] > 0.0:
            arr = arr / arr[-1]
    return arr


def interp_profile(x_src, vals_src, x_dst):
    x_src = np.asarray(x_src, dtype=float).reshape(-1)
    vals_src = np.asarray(vals_src, dtype=float)
    x_dst = np.asarray(x_dst, dtype=float).reshape(-1)
    if vals_src.ndim == 1:
        vals_src = vals_src.reshape(-1, 1)
    if len(x_src) == 0 or vals_src.shape[0] == 0:
        return np.zeros((len(x_dst), vals_src.shape[1]), dtype=float)
    if len(x_src) == 1:
        return np.repeat(vals_src[:1, :], len(x_dst), axis=0)
    out = np.zeros((len(x_dst), vals_src.shape[1]), dtype=float)
    for c_idx in range(vals_src.shape[1]):
        out[:, c_idx] = np.interp(x_dst, x_src, vals_src[:, c_idx])
    return out


def resample_forces_to_shape(
    forces,
    n_span_out,
    n_chord_out,
    eta_span_in=None,
    eta_chord_in=None,
    eta_span_out=None,
    eta_chord_out=None,
):
    forces = np.asarray(forces, dtype=float).reshape(-1, 3)
    n_out = n_span_out * n_chord_out
    if len(forces) == 0:
        return np.zeros((n_out, 3), dtype=float)

    eta_span_out = as_eta_array(eta_span_out, n_span_out)
    eta_chord_out = as_eta_array(eta_chord_out, n_chord_out)

    n_in = len(forces)
    if eta_span_in is None or eta_chord_in is None:
        if n_in == n_out:
            return forces
        if n_in == 1:
            return np.repeat(forces[:1], n_out, axis=0)
        s_in = np.linspace(0.0, 1.0, n_in)
        s_out = np.linspace(0.0, 1.0, n_out)
        return interp_profile(s_in, forces, s_out)

    eta_span_in = np.asarray(eta_span_in, dtype=float).reshape(-1)
    eta_chord_in = np.asarray(eta_chord_in, dtype=float).reshape(-1)
    n_span_in = len(eta_span_in)
    n_chord_in = len(eta_chord_in)
    if n_span_in == 0 or n_chord_in == 0:
        return np.zeros((n_out, 3), dtype=float)
    if n_in != n_span_in * n_chord_in:
        return resample_forces_to_shape(forces, n_span_out, n_chord_out)

    grid_in = forces.reshape((n_span_in, n_chord_in, 3))
    eta_span_in = as_eta_array(eta_span_in, n_span_in)
    eta_chord_in = as_eta_array(eta_chord_in, n_chord_in)

    grid_span = np.zeros((n_span_out, n_chord_in, 3), dtype=float)
    for j_idx in range(n_chord_in):
        grid_span[:, j_idx, :] = interp_profile(
            eta_span_in, grid_in[:, j_idx, :], eta_span_out
        )

    grid_out = np.zeros((n_span_out, n_chord_out, 3), dtype=float)
    for i_idx in range(n_span_out):
        grid_out[i_idx, :, :] = interp_profile(
            eta_chord_in, grid_span[i_idx, :, :], eta_chord_out
        )

    return grid_out.reshape((n_out, 3))


def parse_force_payload(data, n_span_out, n_chord_out, eta_span_out, eta_chord_out):
    eta_span_out = as_eta_array(eta_span_out, n_span_out)
    eta_chord_out = as_eta_array(eta_chord_out, n_chord_out)

    if "n_span" in data and "n_chord" in data and "force" in data:
        n_span_in = int(data.get("n_span", 0))
        n_chord_in = int(data.get("n_chord", 0))
        force_raw = np.asarray(data.get("force", []), dtype=float).reshape(-1, 3)
        indexing = str(data.get("indexing", "span-major"))
        if n_span_in > 0 and n_chord_in > 0 and len(force_raw) == n_span_in * n_chord_in:
            force_grid = np.zeros((n_span_in, n_chord_in, 3), dtype=float)
            k_idx = 0
            if indexing == "span-major":
                for i_idx in range(n_span_in):
                    for j_idx in range(n_chord_in):
                        force_grid[i_idx, j_idx, :] = force_raw[k_idx, :]
                        k_idx += 1
            elif indexing == "chord-major":
                for j_idx in range(n_chord_in):
                    for i_idx in range(n_span_in):
                        force_grid[i_idx, j_idx, :] = force_raw[k_idx, :]
                        k_idx += 1
            else:
                raise RuntimeError(f"Unsupported force indexing '{indexing}'")
            force_raw = force_grid.reshape((-1, 3))
        eta_span_in = (
            as_eta_array(data.get("eta_span"), n_span_in) if n_span_in > 0 else None
        )
        eta_chord_in = (
            as_eta_array(data.get("eta_chord"), n_chord_in) if n_chord_in > 0 else None
        )
        if n_span_in > 0 and n_chord_in > 0:
            forces = resample_forces_to_shape(
                force_raw,
                n_span_out,
                n_chord_out,
                eta_span_in=eta_span_in,
                eta_chord_in=eta_chord_in,
                eta_span_out=eta_span_out,
                eta_chord_out=eta_chord_out,
            )
            forces = np.clip(forces, -max_abs_force_component, max_abs_force_component)
            return forces, True

    forces_legacy = np.asarray(data.get("force", []), dtype=float).reshape(-1, 3)
    forces = resample_forces_to_shape(
        forces_legacy,
        n_span_out,
        n_chord_out,
        eta_span_out=eta_span_out,
        eta_chord_out=eta_chord_out,
    )
    forces = np.clip(forces, -max_abs_force_component, max_abs_force_component)
    return forces, False


# def get_aero_surface_node_ids():
#     """Get node IDs of all nodes on the 2D plate surface"""
#     # For 2D plate, use nodes from the w (transverse) component
#     v_w = V.sub(1).collapse()
#     coords_v = v_w.tabulate_dof_coordinates().reshape((-1, 2))
#     ids = []
#     for i_node, X in enumerate(coords_v):
#         y_val = X[1]
#         if y_val >= 0.0 and y_val <= span:
#             ids.append(i_node)
#     # Map back to mixed space indices
#     w_dofs = np.asarray(V.sub(1).dofmap().dofs(), dtype=np.int64)
#     mixed_ids = w_dofs[np.asarray(sorted(set(ids)), dtype=np.int64)]
#     return mixed_ids, coords_v

def get_aero_surface_node_ids():

    v_w = V.sub(1).collapse()
    coords_v = v_w.tabulate_dof_coordinates().reshape((-1, 2))

    ids = []

    for i_node, X in enumerate(coords_v):
        y_val = X[1]

        if 0.0 <= y_val <= span:
            ids.append(i_node)

    ids = np.asarray(ids, dtype=np.int64)

    return ids, coords_v

def build_spanwise_targets(eta_span_vals, xi_val, xi_eps=0.0):
    pts = np.zeros((len(eta_span_vals), 3), dtype=float)
    xi_eff = float(np.clip(xi_val, 0.0, 1.0))
    if xi_eff <= 0.0:
        xi_eff = min(1.0, xi_eff + xi_eps)
    elif xi_eff >= 1.0:
        xi_eff = max(0.0, xi_eff - xi_eps)
    for i_idx, eta_s in enumerate(eta_span_vals):
        y_val = eta_s * span
        chord = chord_at(y_val)
        x_le = x_leading_edge_at(y_val)
        pts[i_idx, 0] = x_le + xi_eff * chord
        pts[i_idx, 1] = y_val
        pts[i_idx, 2] = 0.0
    return pts


def sample_vector_field_at_targets(u_fun, targets_xyz, fallback_tree=None, fallback_vals=None):
    out = np.zeros((targets_xyz.shape[0], 3), dtype=float)
    for k_idx in range(targets_xyz.shape[0]):
        pt = Point(
            float(targets_xyz[k_idx, 0]),
            float(targets_xyz[k_idx, 1]),
            float(targets_xyz[k_idx, 2]),
        )
        try:
            val = u_fun(pt)
            out[k_idx, 0] = float(val[0])
            out[k_idx, 1] = float(val[1])
            out[k_idx, 2] = float(val[2])
        except RuntimeError:
            if fallback_tree is None or fallback_vals is None:
                continue
            # _, idx = fallback_tree.query(targets_xyz[k_idx, :], k=1)
            _, idx = fallback_tree.query(targets_xyz[k_idx,:2], k=1)
            out[k_idx, :] = fallback_vals[int(idx), :]
    return out


def project_le_te_inextensible(u_le_arr, u_te_arr, le_ref, te_ref):
    # Enforce reference chord length at each span station in communicated geometry.
    Xle = le_ref + u_le_arr
    Xte = te_ref + u_te_arr
    mid = 0.5 * (Xle + Xte)

    ref_vec = te_ref - le_ref
    ref_len = np.linalg.norm(ref_vec, axis=1)

    cur_vec = Xte - Xle
    cur_len = np.linalg.norm(cur_vec, axis=1)
    eps = 1.0e-14

    ref_dir = np.zeros_like(ref_vec)
    ok_ref = ref_len > eps
    ref_dir[ok_ref, :] = ref_vec[ok_ref, :] / ref_len[ok_ref, None]
    if np.any(~ok_ref):
        ref_dir[~ok_ref, :] = np.array([1.0, 0.0, 0.0], dtype=float)

    cur_dir = np.zeros_like(cur_vec)
    ok_cur = cur_len > eps
    cur_dir[ok_cur, :] = cur_vec[ok_cur, :] / cur_len[ok_cur, None]
    if np.any(~ok_cur):
        cur_dir[~ok_cur, :] = ref_dir[~ok_cur, :]

    Xle_p = mid - 0.5 * ref_len[:, None] * cur_dir
    Xte_p = mid + 0.5 * ref_len[:, None] * cur_dir
    u_le_p = Xle_p - le_ref
    u_te_p = Xte_p - te_ref
    return u_le_p, u_te_p, cur_len, ref_len


def project_spanwise_inextensible_line(u_line, ref_line):
    # Enforce reference segment lengths along a spanwise polyline (LE or TE).
    X = ref_line + u_line
    n_pts = X.shape[0]
    if n_pts <= 1:
        return u_line, np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)

    seg_ref_vec = ref_line[1:, :] - ref_line[:-1, :]
    seg_ref_len = np.linalg.norm(seg_ref_vec, axis=1)
    seg_cur_vec = X[1:, :] - X[:-1, :]
    seg_cur_len = np.linalg.norm(seg_cur_vec, axis=1)

    Xp = np.zeros_like(X)
    Xp[0, :] = X[0, :]
    eps = 1.0e-14
    for i in range(1, n_pts):
        v = X[i, :] - X[i - 1, :]
        lv = np.linalg.norm(v)
        if lv > eps:
            d = v / lv
        else:
            rv = ref_line[i, :] - ref_line[i - 1, :]
            lrv = np.linalg.norm(rv)
            if lrv > eps:
                d = rv / lrv
            else:
                d = np.array([0.0, 1.0, 0.0], dtype=float)
        Xp[i, :] = Xp[i - 1, :] + seg_ref_len[i - 1] * d

    u_proj = Xp - ref_line
    return u_proj, seg_cur_len, seg_ref_len


def build_local_rbf_map(fluid_points, solid_points, radius, n_neighbors=24):
    fluid_points = np.asarray(fluid_points, dtype=float)
    solid_points = np.asarray(solid_points, dtype=float)
    support_radius = float(radius)
    if support_radius <= 0.0:
        raise ValueError("COUPLING_RBF_RADIUS must be positive for compact RBF transfer")
    n_s = solid_points.shape[0]
    k = int(max(1, min(n_neighbors, n_s)))

    tree = cKDTree(solid_points)
    dists, idx = tree.query(fluid_points, k=k)
    if k == 1:
        dists = dists.reshape(-1, 1)
        idx = idx.reshape(-1, 1)

    nbr_ids = idx.astype(np.int64)

    # Compact-support Wendland C2 kernel:
    #   phi(r) = (1 - q)^4 (4q + 1), q = r / R, for q < 1
    #   phi(r) = 0,                         for q >= 1
    # This keeps force transfer local and shuts off exactly outside radius R.
    q = dists / support_radius
    nbr_w = np.zeros_like(dists, dtype=float)
    inside = q < 1.0
    one_minus_q = 1.0 - q[inside]
    nbr_w[inside] = (one_minus_q ** 4) * (4.0 * q[inside] + 1.0)

    row_sum = np.sum(nbr_w, axis=1, keepdims=True)
    bad = np.where(row_sum[:, 0] <= 1.0e-16)[0]
    for bi in bad:
        # If the radius is too small for this CP, keep the map usable by assigning
        # the full CP force to the nearest solid node.
        nbr_w[bi, :] = 0.0
        nbr_w[bi, 0] = 1.0
    row_sum = np.sum(nbr_w, axis=1, keepdims=True)
    nbr_w /= np.maximum(row_sum, 1.0e-16)
    return nbr_ids, nbr_w


def map_forces_to_solid(f_fluid, n_solid_nodes, nbr_ids, nbr_w):
    n_f, k = nbr_ids.shape
    out = np.zeros((n_solid_nodes, 3), dtype=float)
    for q_idx in range(k):
        contrib = nbr_w[:, q_idx : q_idx + 1] * f_fluid
        np.add.at(out, nbr_ids[:, q_idx], contrib)
    return out


def compute_S_lumped(n_solid_nodes, nbr_ids, nbr_w, A_diag):
    _n_f, k = nbr_ids.shape
    S = np.zeros((n_solid_nodes,), dtype=float)
    for q_idx in range(k):
        np.add.at(S, nbr_ids[:, q_idx], nbr_w[:, q_idx] * A_diag)
    return np.maximum(S, 1.0e-14)


def apply_Tf_operator(Fa, n_solid_nodes, nbr_ids, nbr_w, A_diag, S_lumped):
    FaA = Fa * A_diag[:, None]
    rhs = map_forces_to_solid(FaA, n_solid_nodes, nbr_ids, nbr_w)
    Fs_coeff = rhs / S_lumped[:, None]
    return Fs_coeff, rhs


def add_nodal_forces_to_rhs_plate(rhs_vec, nodal_forces, node_ids, dofs_u_x, dofs_u_y, dofs_w):
    """Add 3D nodal forces to plate model RHS: map [fx, fy, fz] to [u_x, u_y, w] DOFs"""
    arr = rhs_vec.get_local()
    arr[dofs_u_x[node_ids]] += nodal_forces[:, 0]
    arr[dofs_u_y[node_ids]] += nodal_forces[:, 1]
    arr[dofs_w[node_ids]] += nodal_forces[:, 2]
    rhs_vec.set_local(arr)
    rhs_vec.apply("insert")


def get_nodal_displacements_plate(q_fun, node_ids, dofs_u_x, dofs_u_y, dofs_w):
    """Extract 3D displacement from plate state: [u_x, u_y, w]"""
    q_arr = q_fun.vector().get_local()
    out = np.zeros((len(node_ids), 3), dtype=float)
    out[:, 0] = q_arr[dofs_u_x[node_ids]]
    out[:, 1] = q_arr[dofs_u_y[node_ids]]
    out[:, 2] = q_arr[dofs_w[node_ids]]
    return out


sig = Function(Vsig, name="MembraneStress")

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
out_dir = os.path.join(repo_root, "results", "results_KG9_inflex", "solid")
os.makedirs(out_dir, exist_ok=True)

# =====================================================================
#                    DIAGNOSTICS AND RESTART SETUP
# =====================================================================

# -----------------------
# Restart settings
# -----------------------

restart_interval = int(os.getenv("SOLID_RESTART_INTERVAL", "100"))

restart_dir = os.path.join(out_dir, "restart")
os.makedirs(restart_dir, exist_ok=True)

# -----------------------
# Diagnostics settings
# -----------------------

diagnostics_dir = os.path.join(out_dir, "diagnostics")
os.makedirs(diagnostics_dir, exist_ok=True)

diagnostics_file = os.path.join(
    diagnostics_dir,
    "solid_diagnostics.csv"
)

# Flush every timestep
flush_every = 1

# =====================================================================
# Open diagnostics file BEFORE simulation starts
# =====================================================================

diag_fp = open(diagnostics_file, "w", newline="")

diag_writer = csv.writer(diag_fp)

diag_writer.writerow([
    "Step",
    "Time",

    # -----------------------------
    # Displacements
    # -----------------------------
    "Root_Uz",
    "Mid_Uz",
    "Tip_Uz",

    # -----------------------------
    # Velocities
    # -----------------------------
    "Root_Vz",
    "Mid_Vz",
    "Tip_Vz",

    # -----------------------------
    # Accelerations
    # -----------------------------
    "Root_Az",
    "Mid_Az",
    "Tip_Az",

    # -----------------------------
    # Energies
    # -----------------------------
    "ElasticEnergy",
    "KineticEnergy",
    "DampingEnergy",
    "TotalEnergy",

    "DeltaE",
    "DeltaE_over_E0",

    # -----------------------------
    # Work
    # -----------------------------
    "FluidWork",
    "StructuralWork",

    "CumFluidWork",
    "CumStructuralWork",

    "WorkError",

    # -----------------------------
    # Newton
    # -----------------------------
    "NewtonIterations",
    "NewtonResidual",

    # -----------------------------
    # Forces
    # -----------------------------
    "ForceNorm",

    # -----------------------------
    # CPU
    # -----------------------------
    "WallTime"
])

diag_fp.flush()
os.fsync(diag_fp.fileno())


xdmf_path = os.path.join(out_dir, "elastodynamics-results.xdmf")
xdmf_file = XDMFFile(xdmf_path)
xdmf_file.parameters["flush_output"] = True
xdmf_file.parameters["functions_share_mesh"] = True
xdmf_file.parameters["rewrite_function_mesh"] = False

mesh_pvd_path = os.path.join(out_dir, "solid_mesh.pvd")
q_pvd_path = os.path.join(out_dir, "plate_state.pvd")
sig_pvd_path = os.path.join(out_dir, "membrane_stress.pvd")
mesh_pvd = File(mesh_pvd_path)
q_pvd = File(q_pvd_path)
sig_pvd = File(sig_pvd_path)
mesh_pvd << mesh

print("Connecting solid to coupling server...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(
    (
        os.getenv("COUPLING_HOST", "127.0.0.1"),
        int(os.getenv("COUPLING_PORT", "9000")),
    )
)

sock_file = sock.makefile("r")
sock.sendall((json.dumps({"role": "solid"}) + "\n").encode())
print("Solid connected.")

print("Building interface sets and communication targets...")

# aero_node_ids, aero_coords = get_aero_surface_node_ids()
# interface_node_ids = aero_node_ids
# interface_coords = aero_coords[interface_node_ids, :]



aero_node_ids, aero_coords = get_aero_surface_node_ids()
print("len(aero_coords) =", len(aero_coords))


interface_node_ids = aero_node_ids
print("max(interface_node_ids) =", np.max(interface_node_ids))
interface_coords = aero_coords[interface_node_ids]

interface_tree = cKDTree(interface_coords)

# Rebuild target points for 2D plate (z=0 on midsurface)
cp_targets = build_spanwise_targets(eta_span_comm, eta_cp, xi_eps=0.0)
le_targets = build_spanwise_targets(eta_span_comm, 0.0, xi_eps=edge_eval_xi_eps)
te_targets = build_spanwise_targets(eta_span_comm, 1.0, xi_eps=edge_eval_xi_eps)

np.savetxt(
    os.path.join(out_dir, "coupling_cp_targets.csv"),
    cp_targets,
    delimiter=",",
    header="x,y,z",
    comments="",
)
np.savetxt(
    os.path.join(out_dir, "coupling_le_targets.csv"),
    le_targets,
    delimiter=",",
    header="x,y,z",
    comments="",
)
np.savetxt(
    os.path.join(out_dir, "coupling_te_targets.csv"),
    te_targets,
    delimiter=",",
    header="x,y,z",
    comments="",
)

cp_targets_2d = cp_targets[:, :2]

nbr_ids, nbr_w = build_local_rbf_map(
    cp_targets_2d,
    interface_coords,
    rbf_radius,
    n_neighbors=rbf_neighbors
)

# nbr_ids, nbr_w = build_local_rbf_map(
#     cp_targets, interface_coords, rbf_radius, n_neighbors=rbf_neighbors
# )

A_diag = np.ones((cp_targets.shape[0],), dtype=float)
S_lumped = compute_S_lumped(len(interface_node_ids), nbr_ids, nbr_w, A_diag)

print(
    f"Interface ready: surface_nodes={len(interface_node_ids)}, "
    f"cp_stations={len(cp_targets)}, RBF neighbors={nbr_ids.shape[1]}"
)

# Extract DOFs from mixed space for force/displacement mapping
# Components: [u_mem (2D), w, theta (2D)]
dofs_u_x = np.asarray(V.sub(0).sub(0).dofmap().dofs(), dtype=np.int64)  # ux component
dofs_u_y = np.asarray(V.sub(0).sub(1).dofmap().dofs(), dtype=np.int64)  # uy component
dofs_w = np.asarray(V.sub(1).dofmap().dofs(), dtype=np.int64)           # w component
print("len(dofs_w) =", len(dofs_w))
# Note: theta components (rotations) don't directly couple to point forces

ext_force_vec_template = q.vector().copy()
ext_force_vec_template.zero()

time = np.linspace(0.0, T, Nsteps + 1)

u_root = np.zeros(Nsteps+1)
u_mid  = np.zeros(Nsteps+1)
u_tip  = np.zeros(Nsteps+1)

v_root = np.zeros(Nsteps+1)
v_mid  = np.zeros(Nsteps+1)
v_tip  = np.zeros(Nsteps+1)

a_root = np.zeros(Nsteps+1)
a_mid  = np.zeros(Nsteps+1)
a_tip  = np.zeros(Nsteps+1)

# energies = np.zeros((Nsteps + 1, 4), dtype=float)

elastic_energy = np.zeros(Nsteps+1)
kinetic_energy = np.zeros(Nsteps+1)
damping_energy = np.zeros(Nsteps+1)
total_energy   = np.zeros(Nsteps+1)

delta_energy = np.zeros(Nsteps+1)
delta_energy_ratio = np.zeros(Nsteps+1)

E_damp_acc = 0.0
force_relax = 1.0
forces_prev = None
work_rel_errors = np.full((Nsteps,), np.nan, dtype=float)
work_Wf = np.full((Nsteps,), np.nan, dtype=float)
work_Ws = np.full((Nsteps,), np.nan, dtype=float)

fluid_work = np.zeros(Nsteps+1)
structural_work = np.zeros(Nsteps+1)

cum_fluid_work_array = np.zeros(Nsteps+1)
cum_structural_work_array = np.zeros(Nsteps+1)

work_error = np.zeros(Nsteps+1)

newton_iterations = np.zeros(Nsteps+1, dtype=int)

newton_residual = np.zeros(Nsteps+1)
force_norm = np.zeros(Nsteps+1)

# tip_x = x_leading_edge_at(span) + eta_cp * chord_at(span)
# tip_y = span - 1.0e-8
tip_x = 0.075      # 75% chord
tip_y = 0.30       # span tip



zero_payload = [[0.0, 0.0, 0.0] for _ in range(m_panels_comm)]
init_msg = {
    "step": 0,
    "dt": dt_value,
    "ttot": T,
    "nsteps": Nsteps,
    "n_span": n_span_comm,
    "n_chord": n_chord_comm,
    "indexing": "span-major",
    "eta_span": eta_span_comm.tolist(),
    "eta_chord": eta_cp_comm.tolist(),
    "geometry": zero_payload,
    "geometry_le": zero_payload,
    "geometry_te": zero_payload,
    "rotation": zero_payload,
    "rotation_le": zero_payload,
    "rotation_te": zero_payload,
}
sock.sendall((json.dumps(init_msg) + "\n").encode())
print("Initial zero geometry sent.")

# xdmf_file.write(q, 0.0)
# # Write membrane stress at t=0
# u_mem_f, w_f, theta_f = q.split(deepcopy=True)
# xdmf_file.write(q, 0.0)
# # q_pvd << (q, 0.0)
# q_pvd << (...)

# xdmf_file.write(q, 0.0)

# u_mem_f, w_f, theta_f = q.split(deepcopy=True)

# xdmf_file.write(q, 0.0)

# q_pvd << (...)


def save_restart(step, t, q_old, v_old, a_old):

    q_old.vector().apply("insert")
    v_old.vector().apply("insert")
    a_old.vector().apply("insert")

    File(os.path.join(restart_dir, "q_old.xml")) << q_old
    File(os.path.join(restart_dir, "v_old.xml")) << v_old
    File(os.path.join(restart_dir, "a_old.xml")) << a_old

    info = {
        "step": int(step),
        "time": float(t)
    }

    with open(os.path.join(restart_dir, "restart.json"), "w") as fp:
        json.dump(info, fp, indent=4)

# =====================================================================
#               CUMULATIVE DIAGNOSTICS
# =====================================================================

cum_fluid_work = 0.0
cum_structural_work = 0.0

E0 = None

simulation_walltime = 0.0


def save_restart(step,
                 current_time,
                 q_old,
                 v_old,
                 a_old):

    File(os.path.join(restart_dir, "q_old.xml")) << q_old
    File(os.path.join(restart_dir, "v_old.xml")) << v_old
    File(os.path.join(restart_dir, "a_old.xml")) << a_old

    restart_info = {
        "step": int(step),
        "time": float(current_time)
    }

    with open(os.path.join(restart_dir,
                           "restart.json"), "w") as fp:

        json.dump(restart_info,
                  fp,
                  indent=4)
        

for i_step in range(Nsteps):
    step_start = perf_counter()
    print(f"Solid step {i_step + 1}/{Nsteps}: waiting for force...")
    
    current_time = time[i_step]
    heave_expr.t = current_time

    for bc in bcs:
        bc.apply(q.vector())

    line = sock_file.readline()
    if line == "":
        raise RuntimeError("Coupling server disconnected while sending force data")

    data = json.loads(line)
    forces, used_structured_force = parse_force_payload(
        data, n_span_comm, n_chord_comm, eta_span_comm, eta_cp_comm
    )
    if not np.isfinite(forces).all():
        raise RuntimeError(f"Non-finite force data at solid step {i_step + 1}")

    if i_step == 0:
        if used_structured_force:
            print(
                f"Solid: received structured force payload ({n_span_comm}x{n_chord_comm})"
            )
        else:
            print("Solid: received legacy force payload and resampled to coupling stations")

    if forces_prev is None:
        forces_eff = forces.copy()
    else:
        forces_eff = force_relax * forces + (1.0 - force_relax) * forces_prev
    forces_prev = forces_eff.copy()

    force_norm[i_step+1] = np.linalg.norm(forces_eff)

    nodal_forces = None
    Fs_coeff = None
    if work_conservative_mode:
        Fs_coeff, nodal_forces = apply_Tf_operator(
            forces_eff, len(interface_node_ids), nbr_ids, nbr_w, A_diag, S_lumped
        )
        if not np.isfinite(nodal_forces).all():
            raise RuntimeError(f"Non-finite mapped nodal forces at solid step {i_step + 1}")


    ext_force_vec = ext_force_vec_template.copy()
    ext_force_vec.zero()
    if nodal_forces is not None:
        add_nodal_forces_to_rhs_plate(
            ext_force_vec, nodal_forces, interface_node_ids, dofs_u_x, dofs_u_y, dofs_w
        )

    converged = False
    r0 = None
    r_rel = np.inf


    for newton_it in range(newton_maxit):
        A = assemble(jac)
        R = assemble(res)
        b = R.copy()
        b *= -1.0
        
        if force_ramp_iters > 0:
            ramp = min(1.0, float(newton_it + 1) / float(force_ramp_iters))
        else:
            ramp = 1.0
        b.axpy(ramp, ext_force_vec)
        # for bc in bcs:
        #     bc.apply(A, b)

        for bc in bcs_hom:
            bc.apply(A, b)

        r_norm = b.norm("l2")
        if r0 is None:
            r0 = max(r_norm, 1.0e-16)
        r_rel = r_norm / r0
        if r_norm <= newton_atol or r_rel <= newton_rtol:
            converged = True
            break
        linear_solver.solve(A, dq_newton.vector(), b)

        print(
            "dq norm =",
            dq_newton.vector().norm("l2")
        )
        
        q.vector().axpy(1.0, dq_newton.vector())

        print(
            "q norm =",
            q.vector().norm("l2")
        )
        
        q.vector().apply("insert")

        for bc in bcs:
            bc.apply(q.vector())

        print(
            f"Step {i_step+1}, "
            f"Newton {newton_it}, "
            f"Residual = {r_norm:.6e}"
        )


    if not converged:
        raise RuntimeError(
            f"Newton failed at solid step {i_step + 1}: "
            f"residual={r_norm:.6e}, relative={r_rel:.6e}, "
            f"atol={newton_atol:.2e}, rtol={newton_rtol:.2e}, "
            f"maxit={newton_maxit}, ramp_iters={force_ramp_iters}"
        )
    # ---------------------------------------
    # Newton diagnostics
    # ---------------------------------------

    newton_iterations[i_step+1] = newton_it + 1
    newton_residual[i_step+1] = r_norm
    

    if work_conservative_mode and nodal_forces is not None and Fs_coeff is not None:
        interface_disp_prev = get_nodal_displacements_plate(
            q_old, interface_node_ids, dofs_u_x, dofs_u_y, dofs_w
        )
        u_cp_prev = sample_vector_field_at_targets(
            q_old,
            cp_targets,
            fallback_tree=interface_tree,
            fallback_vals=interface_disp_prev,
        )
        Wf = float(np.sum(u_cp_prev * (forces_eff * A_diag[:, None])))
        Ws = float(np.sum(interface_disp_prev * (Fs_coeff * S_lumped[:, None])))
        rel_work_err = abs(Wf - Ws) / max(abs(Wf), abs(Ws), 1.0e-16)
        work_rel_errors[i_step] = rel_work_err
        work_Wf[i_step] = Wf
        work_Ws[i_step] = Ws

        cum_fluid_work += Wf
        cum_structural_work += Ws

        cum_fluid_work_array[i_step+1] = cum_fluid_work
        cum_structural_work_array[i_step+1] = cum_structural_work

        fluid_work[i_step+1] = Wf
        structural_work[i_step+1] = Ws

        work_error[i_step+1] = rel_work_err

        if i_step == 0 or (i_step + 1) % 20 == 0:
            print(
                f"Work audit step {i_step + 1}: "
                f"Wf={Wf:.6e}, Ws={Ws:.6e}, rel_err={rel_work_err:.3e}"
            )

    update_fields(q, q_old, v_old, a_old)
    if ((i_step+1)%restart_interval)==0:

        save_restart(
            i_step+1,
            t,
            q_old,
            v_old,
            a_old
        )

        diag_fp.flush()
        os.fsync(diag_fp.fileno())


    t = time[i_step + 1]

    xdmf_file.write(q, t)
    # q_pvd << (q, float(t))

    # Compute strain energy from plate kinematics
    u_mem_t, w_t, theta_t = split(q_old)
    E_elas = assemble(
        0.5 * inner(membrane_stress(u_mem_t), membrane_strain(u_mem_t)) * dx
        + 0.5 * inner(bending_moment(theta_t), curvature(theta_t)) * dx
        + 0.5 * kappa_shear * (E / (2.0 * (1.0 + nu))) * h
          * inner(shear_strain(theta_t, w_t), shear_strain(theta_t, w_t)) * dx
    )
    E_kin = 0.5 * assemble(m_form(v_old, v_old))
    E_damp_acc += dt_value * assemble(c_form(v_old, v_old))
    E_tot = E_elas + E_kin + E_damp_acc
    # energies[i_step + 1, :] = np.array([E_elas, E_kin, E_damp_acc, E_tot])
    elastic_energy[i_step+1] = E_elas
    kinetic_energy[i_step+1] = E_kin
    damping_energy[i_step+1] = E_damp_acc
    total_energy[i_step+1] = E_tot

    if E0 is None:
        E0 = max(E_tot,1e-16)

    delta_energy[i_step+1] = E_tot-E0
    delta_energy_ratio[i_step+1] = (E_tot-E0)/E0

    # Get tip transverse displacement
    try:
        vals = q(Point(tip_x, tip_y))

        ux      = vals[0]
        uy      = vals[1]
        w_eval  = vals[2]
        theta_x = vals[3]
        theta_y = vals[4]

        u_tip[i_step + 1] = float(w_eval)
        try:
        
            vals_v = v_old(Point(tip_x, tip_y))

            v_tip[i_step+1] = float(vals_v[2])

        except RuntimeError:
        
            v_tip[i_step+1] = 0.0

        try:

            vals_a = a_old(Point(tip_x, tip_y))

            a_tip[i_step+1] = float(vals_a[2])

        except RuntimeError:
        
            a_tip[i_step+1] = 0.0
    
    except RuntimeError:
        u_tip[i_step + 1] = 0.0


    root_x = 0.075
    root_y = 0.0

    mid_x = 0.075
    mid_y = 0.15

    try:

        vals_mid = q(Point(mid_x, mid_y))
        u_mid[i_step+1] = float(vals_mid[2])

        vals_mid_v = v_old(Point(mid_x, mid_y))
        v_mid[i_step+1] = float(vals_mid_v[2])

        vals_mid_a = a_old(Point(mid_x, mid_y))
        a_mid[i_step+1] = float(vals_mid_a[2])

    except RuntimeError:

        u_mid[i_step+1] = 0.0
        v_mid[i_step+1] = 0.0
        a_mid[i_step+1] = 0.0

    # root_z = 0.0

    # try:
    #     u_root[i_step + 1] = vals = q(Point(root_x,root_y))
    # except RuntimeError:
    #     u_root[i_step+1] = float(vals[2])

    # try:
    #     vals_root = q(Point(root_x, root_y))

    #     u_root[i_step + 1] = float(vals_root[2])

    # except RuntimeError:
    #     u_root[i_step + 1] = 0.0

    try:

        vals_root = q(Point(root_x, root_y))
        u_root[i_step+1] = float(vals_root[2])
    
        vals_root_v = v_old(Point(root_x, root_y))
        v_root[i_step+1] = float(vals_root_v[2])
    
        vals_root_a = a_old(Point(root_x, root_y))
        a_root[i_step+1] = float(vals_root_a[2])
    
    except RuntimeError:
    
        u_root[i_step+1] = 0.0
        v_root[i_step+1] = 0.0
        a_root[i_step+1] = 0.0
    
    diag_writer.writerow([

        i_step+1,
        t,

        u_root[i_step+1],
        u_mid[i_step+1],
        u_tip[i_step+1],

        v_root[i_step+1],
        v_mid[i_step+1],
        v_tip[i_step+1],

        a_root[i_step+1],
        a_mid[i_step+1],
        a_tip[i_step+1],

        elastic_energy[i_step+1],
        kinetic_energy[i_step+1],
        damping_energy[i_step+1],
        total_energy[i_step+1],

        delta_energy[i_step+1],
        delta_energy_ratio[i_step+1],

        fluid_work[i_step+1],
        structural_work[i_step+1],

        cum_fluid_work_array[i_step+1],
        cum_structural_work_array[i_step+1],

        work_error[i_step+1],

        newton_iterations[i_step+1],
        newton_residual[i_step+1],

        force_norm[i_step+1],

        simulation_walltime

    ])

    if ((i_step+1)%flush_every)==0:

        diag_fp.flush()

        os.fsync(diag_fp.fileno())

    if i_step < Nsteps - 1:
        interface_disp_cur = get_nodal_displacements_plate(
            q, interface_node_ids, dofs_u_x, dofs_u_y, dofs_w
        )
        u_le_arr = sample_vector_field_at_targets(
            q,
            le_targets,
            fallback_tree=interface_tree,
            fallback_vals=interface_disp_cur,
        )
        u_te_arr = sample_vector_field_at_targets(
            q,
            te_targets,
            fallback_tree=interface_tree,
            fallback_vals=interface_disp_cur,
        )
        if enforce_chord_projection:
            u_le_arr, u_te_arr, chord_len_cur, chord_len_ref = project_le_te_inextensible(
                u_le_arr, u_te_arr, le_targets, te_targets
            )
            if DEBUG_IO and (i_step == 0 or (i_step + 1) % 20 == 0):
                rel_ch = np.abs(chord_len_cur - chord_len_ref) / np.maximum(chord_len_ref, 1.0e-14)
                print(
                    f"Chord projection step {i_step+1}: "
                    f"pre-proj rel chord err max/mean = {np.max(rel_ch):.3e}/{np.mean(rel_ch):.3e}"
                )
        if enforce_span_projection:
            u_le_arr, span_len_cur_le, span_len_ref_le = project_spanwise_inextensible_line(
                u_le_arr, le_targets
            )
            u_te_arr, span_len_cur_te, span_len_ref_te = project_spanwise_inextensible_line(
                u_te_arr, te_targets
            )
            if DEBUG_IO and (i_step == 0 or (i_step + 1) % 20 == 0):
                if span_len_ref_le.size > 0:
                    rel_sp_le = np.abs(span_len_cur_le - span_len_ref_le) / np.maximum(
                        span_len_ref_le, 1.0e-14
                    )
                    rel_sp_te = np.abs(span_len_cur_te - span_len_ref_te) / np.maximum(
                        span_len_ref_te, 1.0e-14
                    )
                    print(
                        f"Span projection step {i_step+1}: "
                        f"LE pre-proj rel seg err max/mean = {np.max(rel_sp_le):.3e}/{np.mean(rel_sp_le):.3e}, "
                        f"TE pre-proj rel seg err max/mean = {np.max(rel_sp_te):.3e}/{np.mean(rel_sp_te):.3e}"
                    )
        u_cp_arr = (1.0 - eta_cp) * u_le_arr + eta_cp * u_te_arr
        zero_rot = np.zeros_like(u_cp_arr)

        if DEBUG_IO and (i_step == 0 or (i_step + 1) % 20 == 0):
            print(f"SEND step {i_step + 1} first LE/TE = {u_le_arr[0, :].tolist()} / {u_te_arr[0, :].tolist()}")
            print(f"SEND step {i_step + 1} last  LE/TE = {u_le_arr[-1, :].tolist()} / {u_te_arr[-1, :].tolist()}")
        
        simulation_walltime = perf_counter()-step_start
        
        msg_geo = json.dumps(
            {
                "step": i_step + 1,
                "dt": dt_value,
                "ttot": T,
                "nsteps": Nsteps,
                "n_span": n_span_comm,
                "n_chord": n_chord_comm,
                "indexing": "span-major",
                "eta_span": eta_span_comm.tolist(),
                "eta_chord": eta_cp_comm.tolist(),
                "geometry": u_cp_arr.tolist(),
                "geometry_le": u_le_arr.tolist(),
                "geometry_te": u_te_arr.tolist(),
                "rotation": zero_rot.tolist(),
                "rotation_le": zero_rot.tolist(),
                "rotation_te": zero_rot.tolist(),
            }
        )
        sock.sendall((msg_geo + "\n").encode())
        print(f"Solid step {i_step + 1}/{Nsteps}: geometry sent.")

sock_file.close()
sock.close()
print("Solid solver finished.")
print(f"Solid field outputs: {xdmf_path}")
print(f"Solid VTK outputs: {q_pvd_path}, {sig_pvd_path}, {mesh_pvd_path}")

# diag_csv = os.path.join(out_dir, "solid_v18_diagnostics.csv")

# with open(diag_csv, "w") as fp:
#     # fp.write(
#     #     "step,time,u_tip,E_elas,E_kin,E_damp,E_tot,"
#     #     "work_Wf,work_Ws,work_rel_error\n"
#     # )
#     fp.write(
#         "step,time,u_tip,v_tip,a_tip,"
#         "E_elas,E_kin,E_damp,E_tot,"
#         "work_Wf,work_Ws,work_rel_error\n"
#     )

#     for k_idx in range(Nsteps):
#         fp.write(
#             f"{k_idx+1},"
#             f"{time[k_idx+1]:.12e},"
#             f"{u_tip[k_idx+1]:.12e},"
#             f"{v_tip[k_idx+1]:.12e},"
#             f"{a_tip[k_idx+1]:.12e},"
#             f"{energies[k_idx+1,0]:.12e},"
#             f"{energies[k_idx+1,1]:.12e},"
#             f"{energies[k_idx+1,2]:.12e},"
#             f"{energies[k_idx+1,3]:.12e},"
#             f"{work_Wf[k_idx]:.12e},"
#             f"{work_Ws[k_idx]:.12e},"
#             f"{work_rel_errors[k_idx]:.12e}\n"
#         )

diag_fp.close()