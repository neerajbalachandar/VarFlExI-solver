from dolfin import *
import numpy as np
from scipy.optimize import minimize


class CommonRefinementMesh:
    def __init__(self, domain, eta_span_comm):
        self.domain = domain
        self.eta_span_comm = np.asarray(eta_span_comm, dtype=float).reshape(-1)
        # The CRM force vector is ordered by this CG1 space's dof map. Keep
        # the matching coordinates beside it for force and moment audits;
        # the structural solver's P2 interface coordinates have a different
        # size and ordering.
        self.force_space = FunctionSpace(domain.mesh, "CG", 1)
        self.force_dof_coords = self.force_space.tabulate_dof_coordinates().reshape((-1, 2)).copy()

    @staticmethod
    def polygon_area(poly):
        poly = np.asarray(poly, dtype=float)
        if len(poly) < 3:
            return 0.0
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @staticmethod
    def polygon_centroid(poly):
        poly = np.asarray(poly, dtype=float)
        if len(poly) == 0:
            return np.zeros((2,), dtype=float)
        return np.mean(poly, axis=0)

    @staticmethod
    def is_inside_halfplane(p, a, b, eps=1.0e-14):
        return ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= -eps

    @staticmethod
    def line_intersection(p1, p2, a, b):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = a
        x4, y4 = b
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1.0e-16:
            return p2.copy()
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return np.array([px, py], dtype=float)

    def clip_polygon_to_convex(self, poly, clip_poly):
        poly = [np.asarray(p, dtype=float) for p in np.asarray(poly, dtype=float)]
        clip_poly = np.asarray(clip_poly, dtype=float)
        if len(poly) == 0:
            return np.zeros((0, 2), dtype=float)
        for i in range(len(clip_poly)):
            a = clip_poly[i]
            b = clip_poly[(i + 1) % len(clip_poly)]
            if len(poly) == 0:
                break
            out = []
            prev = poly[-1]
            prev_inside = self.is_inside_halfplane(prev, a, b)
            for cur in poly:
                cur_inside = self.is_inside_halfplane(cur, a, b)
                if cur_inside:
                    if not prev_inside:
                        out.append(self.line_intersection(prev, cur, a, b))
                    out.append(cur)
                elif prev_inside:
                    out.append(self.line_intersection(prev, cur, a, b))
                prev = cur
                prev_inside = cur_inside
            poly = out
        if len(poly) == 0:
            return np.zeros((0, 2), dtype=float)
        return np.asarray(poly, dtype=float)

    @staticmethod
    def barycentric_coords_triangle(p, tri):
        a = tri[0]
        b = tri[1]
        c = tri[2]
        v0 = b - a
        v1 = c - a
        v2 = p - a
        denom = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(denom) < 1.0e-16:
            return None
        l2 = (v2[0] * v1[1] - v1[0] * v2[1]) / denom
        l3 = (v0[0] * v2[1] - v2[0] * v0[1]) / denom
        l1 = 1.0 - l2 - l3
        return np.array([l1, l2, l3], dtype=float)

    @staticmethod
    def cg1_triangle_basis(bary):
        return np.asarray(bary, dtype=float)

    @staticmethod
    def triangle_quadrature_rule():
        bary = np.array(
            [[1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0], [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0]],
            dtype=float,
        )
        weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
        return bary, weights

    @staticmethod
    def triangulate_polygon_fan(poly):
        poly = np.asarray(poly, dtype=float)
        if len(poly) < 3:
            return []
        base = poly[0]
        tris = []
        for i in range(1, len(poly) - 1):
            tris.append(np.array([base, poly[i], poly[i + 1]], dtype=float))
        return tris

    def panel_polygon_for_span_interval(self, y0, y1):
        x0_le = self.domain.x_leading_edge_at(y0)
        x0_te = x0_le + self.domain.chord_at(y0)
        x1_le = self.domain.x_leading_edge_at(y1)
        x1_te = x1_le + self.domain.chord_at(y1)
        return np.array([[x0_le, y0], [x0_te, y0], [x1_te, y1], [x1_le, y1]], dtype=float)

    def span_edges_from_stations(self, eta_span_vals=None):
        if eta_span_vals is None:
            eta_span_vals = self.eta_span_comm
        eta = np.asarray(eta_span_vals, dtype=float).reshape(-1)
        if len(eta) == 0:
            raise ValueError("eta_span_vals must contain at least one station")
        if len(eta) == 1:
            return np.array([0.0, 1.0], dtype=float)

        edges = np.zeros((len(eta) + 1,), dtype=float)
        edges[0] = 0.0
        edges[-1] = 1.0
        edges[1:-1] = 0.5 * (eta[:-1] + eta[1:])
        edges = np.clip(edges, 0.0, 1.0)
        edges = np.maximum.accumulate(edges)
        edges[-1] = 1.0
        return edges

    def build_common_refinement_operator(self):
        eta_span_vals = self.eta_span_comm
        if len(eta_span_vals) == 0:
            raise ValueError("No span stations supplied to CRM operator")

        span_edges = self.span_edges_from_stations(eta_span_vals)
        panel_polys = []
        panel_areas = np.zeros((len(eta_span_vals),), dtype=float)
        for i_idx in range(len(eta_span_vals)):
            y0 = span_edges[i_idx] * self.domain.span
            y1 = span_edges[i_idx + 1] * self.domain.span
            poly = self.panel_polygon_for_span_interval(y0, y1)
            panel_polys.append(poly)
            panel_areas[i_idx] = max(self.polygon_area(poly), 1.0e-14)

        trial_space = FunctionSpace(self.domain.mesh, "CG", 1)
        dofmap = trial_space.dofmap()
        ndofs = trial_space.dim()
        n_panels = len(eta_span_vals)
        C = np.zeros((ndofs, n_panels), dtype=float)

        bary_rule, bary_weights = self.triangle_quadrature_rule()
        for cell in cells(self.domain.mesh):
            cell_index = cell.index()
            vert_ids = cell.entities(0)
            tri = self.domain.mesh.coordinates()[vert_ids, :2]
            tri_min = np.min(tri, axis=0)
            tri_max = np.max(tri, axis=0)
            local_dofs = dofmap.cell_dofs(cell_index)

            for j_idx, panel_poly in enumerate(panel_polys):
                panel_min = np.min(panel_poly, axis=0)
                panel_max = np.max(panel_poly, axis=0)
                if tri_max[0] < panel_min[0] - 1.0e-14 or tri_min[0] > panel_max[0] + 1.0e-14:
                    continue
                if tri_max[1] < panel_min[1] - 1.0e-14 or tri_min[1] > panel_max[1] + 1.0e-14:
                    continue

                clipped = self.clip_polygon_to_convex(tri, panel_poly)
                if len(clipped) < 3:
                    continue

                clipped_centroid = self.polygon_centroid(clipped)
                if not np.all(np.isfinite(clipped_centroid)):
                    continue

                for subtri in self.triangulate_polygon_fan(clipped):
                    sub_area = self.polygon_area(subtri)
                    if sub_area <= 1.0e-16:
                        continue
                    for k_idx in range(3):
                        bary_sub = bary_rule[k_idx]
                        p = (
                            bary_sub[0] * subtri[0]
                            + bary_sub[1] * subtri[1]
                            + bary_sub[2] * subtri[2]
                        )
                        bary_tri = self.barycentric_coords_triangle(p, tri)
                        if bary_tri is None:
                            continue
                        phi = self.cg1_triangle_basis(bary_tri)
                        wq = sub_area * bary_weights[k_idx]
                        C[local_dofs, j_idx] += wq * phi

        return C, panel_areas, panel_polys

    def build_edge_common_refinement_operator(self, xi, xi_eps=0.0):
        """Build a CG1-to-span-segment averaging operator along a chord line.

        The target segments are the dual span cells centred on the aerodynamic
        stations.  Integrating the CG1 basis over their intersections with
        structural triangles is the one-dimensional common-refinement analogue
        of the panel operator used for force transfer.
        """
        xi_eval = float(np.clip(float(xi), float(xi_eps), 1.0 - float(xi_eps)))
        span_edges = self.span_edges_from_stations(self.eta_span_comm)
        trial_space = FunctionSpace(self.domain.mesh, "CG", 1)
        dofmap = trial_space.dofmap()
        ndofs = trial_space.dim()
        C = np.zeros((ndofs, len(self.eta_span_comm)), dtype=float)
        lengths = np.zeros((len(self.eta_span_comm),), dtype=float)

        for j_idx in range(len(self.eta_span_comm)):
            y0 = span_edges[j_idx] * self.domain.span
            y1 = span_edges[j_idx + 1] * self.domain.span
            p0 = np.array(
                [self.domain.x_leading_edge_at(y0) + xi_eval * self.domain.chord_at(y0), y0],
                dtype=float,
            )
            p1 = np.array(
                [self.domain.x_leading_edge_at(y1) + xi_eval * self.domain.chord_at(y1), y1],
                dtype=float,
            )
            direction = p1 - p0
            lengths[j_idx] = max(float(np.linalg.norm(direction)), 1.0e-14)

            for cell in cells(self.domain.mesh):
                cell_index = cell.index()
                vert_ids = cell.entities(0)
                tri = self.domain.mesh.coordinates()[vert_ids, :2]
                bary0 = self.barycentric_coords_triangle(p0, tri)
                bary1 = self.barycentric_coords_triangle(p1, tri)
                if bary0 is None or bary1 is None:
                    continue
                slope = bary1 - bary0
                t_lo, t_hi = 0.0, 1.0
                valid = True
                for lam0, dlam in zip(bary0, slope):
                    if abs(dlam) <= 1.0e-14:
                        if lam0 < -1.0e-14:
                            valid = False
                            break
                    elif dlam > 0.0:
                        t_lo = max(t_lo, -lam0 / dlam)
                    else:
                        t_hi = min(t_hi, -lam0 / dlam)
                if not valid or t_hi - t_lo <= 1.0e-14:
                    continue
                # Integral of each affine CG1 basis function over the segment.
                basis_integral = lengths[j_idx] * (
                    bary0 * (t_hi - t_lo)
                    + 0.5 * slope * (t_hi * t_hi - t_lo * t_lo)
                )
                C[dofmap.cell_dofs(cell_index), j_idx] += basis_integral

        return C, lengths

    @staticmethod
    def panel_loads_to_solid_nodal_forces(panel_forces, C, panel_areas):
        panel_forces = np.asarray(panel_forces, dtype=float).reshape(-1, 3)
        densities = panel_forces / np.asarray(panel_areas, dtype=float).reshape(-1, 1)
        return C @ densities

    @staticmethod
    def nodal_displacements_to_panel_average(nodal_disp, C, panel_areas):
        nodal_disp = np.asarray(nodal_disp, dtype=float).reshape(-1, 3)
        return (C.T @ nodal_disp) / np.asarray(panel_areas, dtype=float).reshape(-1, 1)

    @staticmethod
    def build_moment_constrained_operator(
        C, panel_areas, panel_targets, dof_coords, feasibility_tol=1.0e-9
    ):
        """Correct CRM overlap weights to preserve each panel's force and moment.

        For panel j, the original CRM column defines nonnegative normalized
        weights ``w0 = C[:, j] / area[j]``.  This routine finds the closest
        nonnegative weights satisfying partition of unity and first spatial
        moments at the aerodynamic panel target.  The returned matrix has the
        same area-integrated units as C, so the existing force and adjoint
        displacement routines remain applicable.

        The equality constraints are solved independently per panel.  If a
        target is outside the convex hull of the structural nodes touched by
        that common-refinement panel, setup fails explicitly rather than
        claiming a moment-conservative transfer that cannot be realized.
        """
        C = np.asarray(C, dtype=float)
        areas = np.asarray(panel_areas, dtype=float).reshape(-1)
        targets = np.asarray(panel_targets, dtype=float).reshape((-1, 2))
        coords = np.asarray(dof_coords, dtype=float).reshape((-1, 2))
        if C.shape != (coords.shape[0], areas.shape[0]):
            raise ValueError("CRM matrix, panel areas, and structural coordinates have incompatible sizes")
        if targets.shape[0] != areas.shape[0]:
            raise ValueError("One physical target coordinate is required per CRM panel")

        C_moment = np.zeros_like(C)
        min_weight = np.inf
        max_constraint_residual = 0.0
        for panel_idx, (area, target) in enumerate(zip(areas, targets)):
            if not np.isfinite(area) or area <= 0.0:
                raise ValueError(f"CRM panel {panel_idx} has invalid area {area}")
            weights_all = C[:, panel_idx] / area
            support = np.flatnonzero(weights_all > 1.0e-14)
            if support.size < 3:
                raise ValueError(
                    f"CRM panel {panel_idx} has only {support.size} active structural nodes; "
                    "cannot constrain planar first moments"
                )

            points = coords[support]
            # Center at the requested CP and scale each coordinate so the
            # equality system is well conditioned even for small geometries.
            scale = np.ptp(points, axis=0)
            scale[scale <= 1.0e-14] = max(float(np.max(scale)), 1.0)
            normalized = (points - target[None, :]) / scale[None, :]
            Aeq = np.vstack((np.ones(support.size), normalized.T))
            beq = np.array([1.0, 0.0, 0.0], dtype=float)
            if np.linalg.matrix_rank(Aeq, tol=1.0e-12) < 3:
                raise ValueError(
                    f"CRM panel {panel_idx} support is geometrically degenerate; "
                    "cannot preserve both in-plane moment components"
                )

            w0 = weights_all[support]

            def objective(weights):
                delta = weights - w0
                return 0.5 * float(np.dot(delta, delta))

            def objective_jac(weights):
                return weights - w0

            result = minimize(
                objective,
                w0,
                jac=objective_jac,
                method="SLSQP",
                bounds=[(0.0, None)] * support.size,
                constraints={
                    "type": "eq",
                    "fun": lambda weights: Aeq @ weights - beq,
                    "jac": lambda weights: Aeq,
                },
                options={"ftol": 1.0e-13, "maxiter": 500, "disp": False},
            )
            corrected = np.asarray(result.x, dtype=float)
            residual = Aeq @ corrected - beq
            if (
                not result.success
                or not np.all(np.isfinite(corrected))
                or float(np.min(corrected)) < -feasibility_tol
                or float(np.linalg.norm(residual, ord=np.inf)) > feasibility_tol
            ):
                raise RuntimeError(
                    f"Moment-constrained CRM failed for panel {panel_idx}: "
                    f"success={result.success}, min_weight={np.min(corrected):.3e}, "
                    f"constraint_residual={np.linalg.norm(residual, ord=np.inf):.3e}, "
                    f"message={result.message}"
                )

            # Remove tiny optimizer-bound violations, then project once onto
            # the equalities using the strictly positive active set. This
            # tightens force/moment conservation close to floating precision.
            corrected[corrected < 0.0] = 0.0
            active = corrected > 1.0e-12
            if np.count_nonzero(active) < 3 or np.linalg.matrix_rank(Aeq[:, active]) < 3:
                raise RuntimeError(f"Moment-constrained CRM panel {panel_idx} lost equality rank")
            delta = Aeq[:, active].T @ np.linalg.solve(
                Aeq[:, active] @ Aeq[:, active].T,
                beq - Aeq[:, active] @ corrected[active],
            )
            corrected[active] += delta
            residual = Aeq @ corrected - beq
            if float(np.min(corrected)) < -1.0e-12:
                raise RuntimeError(
                    f"CRM equality refinement failed for panel {panel_idx}: "
                    f"negative corrected weight={np.min(corrected):.3e}"
                )
            corrected[corrected < 0.0] = 0.0
            residual = Aeq @ corrected - beq
            if float(np.linalg.norm(residual, ord=np.inf)) > 1.0e-11:
                raise RuntimeError(
                    f"CRM equality refinement failed for panel {panel_idx}: "
                    f"constraint_residual={np.linalg.norm(residual, ord=np.inf):.3e}"
                )

            C_moment[support, panel_idx] = area * corrected
            min_weight = min(min_weight, float(np.min(corrected)))
            max_constraint_residual = max(
                max_constraint_residual, float(np.linalg.norm(residual, ord=np.inf))
            )

        return C_moment, {
            "minimum_active_weight": min_weight,
            "maximum_constraint_residual": max_constraint_residual,
        }


__all__ = ["CommonRefinementMesh"]
