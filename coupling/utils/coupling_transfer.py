from dolfin import *
import numpy as np
from time import perf_counter
from scipy.spatial import cKDTree

from .common_refinement_mesh import CommonRefinementMesh
from .rbf_transfer import RBFTransfer


class CouplingTransfer:
    """Coupling-side geometry, sampling, and transfer operators."""

    def __init__(
        self,
        domain,
        model,
        eta_span_comm,
        eta_cp,
        force_transfer_mode="rbf",
        rbf_radius=0.08,
        rbf_neighbors=24,
        max_abs_force_component=5.0e3,
        edge_eval_xi_eps=1.0e-6,
        work_conservative_mode=True,
        enforce_chord_projection=False,
        enforce_span_projection=False,
        debug_io=False,
    ):
        self.domain = domain
        self.model = model
        self.force_transfer_mode = str(force_transfer_mode).strip().lower()
        if self.force_transfer_mode not in ("crm", "rbf"):
            raise ValueError(
                f"Unsupported force_transfer_mode '{force_transfer_mode}'. "
                "Expected 'crm' or 'rbf'."
            )
        self.eta_span_comm = np.asarray(eta_span_comm, dtype=float).reshape(-1)
        self.eta_cp = float(eta_cp)
        self.eta_cp_comm = np.array([self.eta_cp], dtype=float)
        self.rbf_radius = float(rbf_radius)
        self.rbf_neighbors = int(rbf_neighbors)
        self.max_abs_force_component = (
            None if max_abs_force_component is None else float(max_abs_force_component)
        )
        if self.max_abs_force_component is not None and (
            not np.isfinite(self.max_abs_force_component)
            or self.max_abs_force_component <= 0.0
        ):
            raise ValueError("max_abs_force_component must be positive or null to disable clipping")
        self.edge_eval_xi_eps = float(edge_eval_xi_eps)
        self.work_conservative_mode = bool(work_conservative_mode)
        self.enforce_chord_projection = bool(enforce_chord_projection)
        self.enforce_span_projection = bool(enforce_span_projection)
        self.debug_io = bool(debug_io)

        self.aero_node_ids, self.aero_coords = self.get_aero_surface_node_ids()
        self.interface_node_ids, self.interface_coords, self.interface_tree = (
            self._build_interface_sets()
        )
        self.cp_targets = self.domain.build_spanwise_targets(self.eta_span_comm, self.eta_cp, xi_eps=0.0)
        self.le_targets = self.domain.build_spanwise_targets(self.eta_span_comm, 0.0, xi_eps=0.0)
        self.te_targets = self.domain.build_spanwise_targets(self.eta_span_comm, 1.0, xi_eps=0.0)

        # Geometry transfer is intentionally separate from force transfer:
        # structural field evaluation near the physical LE/TE forms the common
        # displacement space, then an RBF map targets the aerodynamic LE/TE
        # geometry space used by FLOWUnsteady.
        self.geometry_common_le_targets = self.domain.build_spanwise_targets(
            self.eta_span_comm, 0.0, xi_eps=self.edge_eval_xi_eps
        )
        self.geometry_common_te_targets = self.domain.build_spanwise_targets(
            self.eta_span_comm, 1.0, xi_eps=self.edge_eval_xi_eps
        )
        self.geometry_common_targets = np.vstack(
            (self.geometry_common_le_targets, self.geometry_common_te_targets)
        )
        self.geometry_aero_targets = np.vstack((self.le_targets, self.te_targets))
        t_geometry_setup = perf_counter()
        self.geometry_nbr_ids, self.geometry_nbr_w = RBFTransfer.build_local_rbf_map(
            self.geometry_aero_targets[:, :2],
            self.geometry_common_targets[:, :2],
            self.rbf_radius,
            n_neighbors=self.rbf_neighbors,
        )
        self.geometry_rbf_setup_time = perf_counter() - t_geometry_setup

        self.crm_backend = CommonRefinementMesh(self.domain, self.eta_span_comm)
        self.rbf_backend = None
        self.crm_transfer_matrix = None
        self.crm_panel_areas = None
        self.crm_panel_polys = None
        self.nbr_ids = None
        self.nbr_w = None
        self.A_diag = np.ones((self.cp_targets.shape[0],), dtype=float)
        self.S_lumped = None
        self.crm_setup_time = 0.0
        self.rbf_setup_time = 0.0

        if self.force_transfer_mode == "crm":
            t_setup = perf_counter()
            self.crm_transfer_matrix, self.crm_panel_areas, self.crm_panel_polys = (
                self.crm_backend.build_common_refinement_operator()
            )
            self.crm_setup_time = perf_counter() - t_setup
        else:
            t_setup = perf_counter()
            cp_targets_2d = self.cp_targets[:, :2]
            self.rbf_backend = RBFTransfer(
                cp_targets_2d,
                self.interface_coords,
                self.rbf_radius,
                n_neighbors=self.rbf_neighbors,
                A_diag=self.A_diag,
            )
            self.nbr_ids = self.rbf_backend.nbr_ids
            self.nbr_w = self.rbf_backend.nbr_w
            self.S_lumped = self.rbf_backend.S_lumped
            self.rbf_setup_time = perf_counter() - t_setup


    def get_aero_surface_node_ids(self):
        v_w = self.model.V.sub(1).collapse()
        coords_v = v_w.tabulate_dof_coordinates().reshape((-1, 2))
        ids = []
        for i_node, X in enumerate(coords_v):
            y_val = X[1]
            if 0.0 <= y_val <= self.domain.span:
                ids.append(i_node)
        return np.asarray(ids, dtype=np.int64), coords_v

    def _build_interface_sets(self):
        crm_space = FunctionSpace(self.domain.mesh, "CG", 1)
        crm_coords = crm_space.tabulate_dof_coordinates().reshape((-1, 2))
        crm_tree = cKDTree(self.aero_coords)
        crm_dist, interface_node_ids = crm_tree.query(crm_coords, k=1)
        interface_node_ids = np.asarray(interface_node_ids, dtype=np.int64)
        if np.max(crm_dist) > 1.0e-10:
            raise RuntimeError(
                f"CRM vertex-to-solid dof mapping failed: max mismatch={np.max(crm_dist):.3e}"
            )
        interface_coords = self.aero_coords[interface_node_ids]
        interface_tree = cKDTree(interface_coords)
        return interface_node_ids, interface_coords, interface_tree

    @staticmethod
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

    @staticmethod
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
        self,
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

        eta_span_out = self.as_eta_array(eta_span_out, n_span_out)
        eta_chord_out = self.as_eta_array(eta_chord_out, n_chord_out)

        n_in = len(forces)
        if eta_span_in is None or eta_chord_in is None:
            if n_in == n_out:
                return forces
            if n_in == 1:
                return np.repeat(forces[:1], n_out, axis=0)
            s_in = np.linspace(0.0, 1.0, n_in)
            s_out = np.linspace(0.0, 1.0, n_out)
            return self.interp_profile(s_in, forces, s_out)

        eta_span_in = np.asarray(eta_span_in, dtype=float).reshape(-1)
        eta_chord_in = np.asarray(eta_chord_in, dtype=float).reshape(-1)
        n_span_in = len(eta_span_in)
        n_chord_in = len(eta_chord_in)
        if n_span_in == 0 or n_chord_in == 0:
            return np.zeros((n_out, 3), dtype=float)
        if n_in != n_span_in * n_chord_in:
            return self.resample_forces_to_shape(forces, n_span_out, n_chord_out)

        grid_in = forces.reshape((n_span_in, n_chord_in, 3))
        eta_span_in = self.as_eta_array(eta_span_in, n_span_in)
        eta_chord_in = self.as_eta_array(eta_chord_in, n_chord_in)

        grid_span = np.zeros((n_span_out, n_chord_in, 3), dtype=float)
        for j_idx in range(n_chord_in):
            grid_span[:, j_idx, :] = self.interp_profile(
                eta_span_in, grid_in[:, j_idx, :], eta_span_out
            )

        grid_out = np.zeros((n_span_out, n_chord_out, 3), dtype=float)
        for i_idx in range(n_span_out):
            grid_out[i_idx, :, :] = self.interp_profile(
                eta_chord_in, grid_span[i_idx, :, :], eta_chord_out
            )

        return grid_out.reshape((n_out, 3))

    def limit_force_components(self, forces):
        forces = np.asarray(forces, dtype=float)
        if not np.all(np.isfinite(forces)):
            raise RuntimeError("Non-finite aerodynamic force received during force transfer")
        if self.max_abs_force_component is None:
            return forces
        return np.clip(
            forces,
            -self.max_abs_force_component,
            self.max_abs_force_component,
        )

    def parse_force_payload(self, data, n_span_out, n_chord_out, eta_span_out, eta_chord_out):
        eta_span_out = self.as_eta_array(eta_span_out, n_span_out)
        eta_chord_out = self.as_eta_array(eta_chord_out, n_chord_out)

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
                self.as_eta_array(data.get("eta_span"), n_span_in) if n_span_in > 0 else None
            )
            eta_chord_in = (
                self.as_eta_array(data.get("eta_chord"), n_chord_in) if n_chord_in > 0 else None
            )
            if n_span_in > 0 and n_chord_in > 0:
                forces = self.resample_forces_to_shape(
                    force_raw,
                    n_span_out,
                    n_chord_out,
                    eta_span_in=eta_span_in,
                    eta_chord_in=eta_chord_in,
                    eta_span_out=eta_span_out,
                    eta_chord_out=eta_chord_out,
                )
                forces = self.limit_force_components(forces)
                return forces, True

        forces_legacy = np.asarray(data.get("force", []), dtype=float).reshape(-1, 3)
        forces = self.resample_forces_to_shape(
            forces_legacy,
            n_span_out,
            n_chord_out,
            eta_span_out=eta_span_out,
            eta_chord_out=eta_chord_out,
        )
        forces = self.limit_force_components(forces)
        return forces, False

    def build_spanwise_targets(self, eta_span_vals, xi_val, xi_eps=0.0):
        return self.domain.build_spanwise_targets(eta_span_vals, xi_val, xi_eps=xi_eps)

    def sample_vector_field_at_targets(self, u_fun, targets_xyz, fallback_tree=None, fallback_vals=None):
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
                _, idx = fallback_tree.query(targets_xyz[k_idx, :2], k=1)
                out[k_idx, :] = fallback_vals[int(idx), :]
        return out

    def apply_force_transfer(self, forces):
        if self.force_transfer_mode == "crm":
            nodal_forces = self.crm_backend.panel_loads_to_solid_nodal_forces(
                forces, self.crm_transfer_matrix, self.crm_panel_areas
            )
            return None, nodal_forces
        if self.rbf_backend is None:
            raise RuntimeError("RBF backend is not initialized")
        return self.rbf_backend.apply(forces)

    @staticmethod
    def apply_rbf_interpolation(values, nbr_ids, nbr_w):
        values = np.asarray(values, dtype=float).reshape(-1, 3)
        if values.shape[0] == 0:
            return np.zeros((nbr_ids.shape[0], 3), dtype=float)
        return np.sum(nbr_w[:, :, None] * values[nbr_ids, :], axis=1)

    def apply_geometry_transfer(self, u_fun, fallback_tree=None, fallback_vals=None):
        """Evaluate u_s on the common interface and RBF-map it to aero LE/TE."""
        u_common = self.sample_vector_field_at_targets(
            u_fun,
            self.geometry_common_targets,
            fallback_tree=fallback_tree,
            fallback_vals=fallback_vals,
        )
        u_aero_geometry = self.apply_rbf_interpolation(
            u_common, self.geometry_nbr_ids, self.geometry_nbr_w
        )
        n_span = len(self.eta_span_comm)
        u_le = u_aero_geometry[:n_span, :]
        u_te = u_aero_geometry[n_span:, :]
        return u_le, u_te, u_common, u_aero_geometry

    # Diagnostic only: this probes the existing force-transfer path without
    # redefining it as the transpose of the geometry-transfer operator.
    def force_transfer_to_aero_space(self, nodal_forces, Fs_coeff=None):
        if self.force_transfer_mode == "crm":
            return self.nodal_displacements_to_panel_average(
                nodal_forces, self.crm_transfer_matrix, self.crm_panel_areas
            )

        if Fs_coeff is None:
            raise RuntimeError("RBF force-transfer diagnostic requires Fs_coeff")
        Fs_coeff = np.asarray(Fs_coeff, dtype=float).reshape(-1, 3)
        structural_at_aero = np.sum(
            self.nbr_w[:, :, None] * Fs_coeff[self.nbr_ids, :], axis=1
        )
        return structural_at_aero * self.A_diag[:, None]

    def force_transfer_metrics(self, aero_forces, nodal_forces, Fs_coeff=None, epsilon_reg=1.0e-16):
        aero_forces = np.asarray(aero_forces, dtype=float).reshape(-1, 3)
        structural_common = self.force_transfer_to_aero_space(nodal_forces, Fs_coeff=Fs_coeff)
        if structural_common.shape != aero_forces.shape:
            raise RuntimeError(
                "Force-transfer diagnostic shape mismatch: "
                f"structural_common={structural_common.shape}, aero_forces={aero_forces.shape}"
            )
        residual = float(np.linalg.norm((structural_common - aero_forces).reshape(-1)))
        reference_norm = float(np.linalg.norm(aero_forces.reshape(-1)))
        relative_error = residual / max(reference_norm + float(epsilon_reg), 1.0e-300)
        return residual, relative_error, structural_common

    def displacement_transfer_metrics(self, nodal_displacements, epsilon_reg=1.0e-16):
        nodal_displacements = np.asarray(nodal_displacements, dtype=float).reshape(-1, 3)
        if self.force_transfer_mode == "crm":
            return self.crm_backend.round_trip_displacement_metrics(
                nodal_displacements,
                self.crm_transfer_matrix,
                self.crm_panel_areas,
                epsilon_reg=epsilon_reg,
            )
        return self.rbf_backend.round_trip_displacement_metrics(
            nodal_displacements, epsilon_reg=epsilon_reg
        )

    def span_edges_from_stations(self, eta_span_vals):
        return self.crm_backend.span_edges_from_stations(eta_span_vals)

    def panel_polygon_for_span_interval(self, y0, y1):
        return self.crm_backend.panel_polygon_for_span_interval(y0, y1)

    @staticmethod
    def polygon_area(poly):
        return CommonRefinementMesh.polygon_area(poly)

    @staticmethod
    def polygon_centroid(poly):
        return CommonRefinementMesh.polygon_centroid(poly)

    @staticmethod
    def is_inside_halfplane(p, a, b, eps=1.0e-14):
        return CommonRefinementMesh.is_inside_halfplane(p, a, b, eps=eps)

    @staticmethod
    def line_intersection(p1, p2, a, b):
        return CommonRefinementMesh.line_intersection(p1, p2, a, b)

    def clip_polygon_to_convex(self, poly, clip_poly):
        return self.crm_backend.clip_polygon_to_convex(poly, clip_poly)

    @staticmethod
    def barycentric_coords_triangle(p, tri):
        return CommonRefinementMesh.barycentric_coords_triangle(p, tri)

    @staticmethod
    def cg1_triangle_basis(bary):
        return CommonRefinementMesh.cg1_triangle_basis(bary)

    @staticmethod
    def triangle_quadrature_rule():
        return CommonRefinementMesh.triangle_quadrature_rule()

    @staticmethod
    def triangulate_polygon_fan(poly):
        return CommonRefinementMesh.triangulate_polygon_fan(poly)

    def build_common_refinement_operator(self):
        return self.crm_backend.build_common_refinement_operator()

    @staticmethod
    def panel_loads_to_solid_nodal_forces(panel_forces, C, panel_areas):
        return CommonRefinementMesh.panel_loads_to_solid_nodal_forces(panel_forces, C, panel_areas)

    @staticmethod
    def nodal_displacements_to_panel_average(nodal_disp, C, panel_areas):
        return CommonRefinementMesh.nodal_displacements_to_panel_average(nodal_disp, C, panel_areas)

    @staticmethod
    def project_le_te_inextensible(u_le_arr, u_te_arr, le_ref, te_ref):
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

    @staticmethod
    def project_spanwise_inextensible_line(u_line, ref_line):
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

    def build_local_rbf_map(self, fluid_points, solid_points, radius, n_neighbors=24):
        return RBFTransfer.build_local_rbf_map(fluid_points, solid_points, radius, n_neighbors=n_neighbors)

    @staticmethod
    def map_forces_to_solid(f_fluid, n_solid_nodes, nbr_ids, nbr_w):
        return RBFTransfer.map_forces_to_solid(f_fluid, n_solid_nodes, nbr_ids, nbr_w)

    @staticmethod
    def compute_S_lumped(n_solid_nodes, nbr_ids, nbr_w, A_diag):
        return RBFTransfer.compute_S_lumped(n_solid_nodes, nbr_ids, nbr_w, A_diag)

    @classmethod
    def apply_Tf_operator(cls, Fa, n_solid_nodes, nbr_ids, nbr_w, A_diag, S_lumped):
        return RBFTransfer.apply_Tf_operator(Fa, n_solid_nodes, nbr_ids, nbr_w, A_diag, S_lumped)

    @staticmethod
    def add_nodal_forces_to_rhs_plate(rhs_vec, nodal_forces, node_ids, dofs_u_x, dofs_u_y, dofs_w):
        arr = rhs_vec.get_local()
        arr[dofs_u_x[node_ids]] += nodal_forces[:, 0]
        arr[dofs_u_y[node_ids]] += nodal_forces[:, 1]
        arr[dofs_w[node_ids]] += nodal_forces[:, 2]
        rhs_vec.set_local(arr)
        rhs_vec.apply("insert")

    @staticmethod
    def get_nodal_displacements_plate(q_fun, node_ids, dofs_u_x, dofs_u_y, dofs_w):
        q_arr = q_fun.vector().get_local()
        out = np.zeros((len(node_ids), 3), dtype=float)
        out[:, 0] = q_arr[dofs_u_x[node_ids]]
        out[:, 1] = q_arr[dofs_u_y[node_ids]]
        out[:, 2] = q_arr[dofs_w[node_ids]]
        return out


__all__ = ["CouplingTransfer"]
