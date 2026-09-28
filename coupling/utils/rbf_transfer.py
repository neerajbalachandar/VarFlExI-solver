from scipy.spatial import cKDTree
import numpy as np


class RBFTransfer:
    def __init__(self, fluid_points, solid_points, radius, n_neighbors=24, A_diag=None):
        self.fluid_points = np.asarray(fluid_points, dtype=float)
        self.solid_points = np.asarray(solid_points, dtype=float)
        self.radius = float(radius)
        self.n_neighbors = int(n_neighbors)
        self.A_diag = (
            np.ones((self.fluid_points.shape[0],), dtype=float)
            if A_diag is None
            else np.asarray(A_diag, dtype=float).reshape(-1)
        )
        if self.A_diag.shape[0] != self.fluid_points.shape[0]:
            raise ValueError(
                "A_diag length must match the number of fluid transfer stations"
            )
        self.nbr_ids, self.nbr_w = self.build_local_rbf_map(
            self.fluid_points, self.solid_points, self.radius, n_neighbors=self.n_neighbors
        )
        self.S_lumped = self.compute_S_lumped(
            len(self.solid_points), self.nbr_ids, self.nbr_w, self.A_diag
        )

    @staticmethod
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
        q = dists / support_radius
        nbr_w = np.zeros_like(dists, dtype=float)
        inside = q < 1.0
        one_minus_q = 1.0 - q[inside]
        nbr_w[inside] = (one_minus_q ** 4) * (4.0 * q[inside] + 1.0)

        row_sum = np.sum(nbr_w, axis=1, keepdims=True)
        bad = np.where(row_sum[:, 0] <= 1.0e-16)[0]
        for bi in bad:
            nbr_w[bi, :] = 0.0
            nbr_w[bi, 0] = 1.0
        row_sum = np.sum(nbr_w, axis=1, keepdims=True)
        nbr_w /= np.maximum(row_sum, 1.0e-16)
        return nbr_ids, nbr_w

    @staticmethod
    def map_forces_to_solid(f_fluid, n_solid_nodes, nbr_ids, nbr_w):
        n_f, k = nbr_ids.shape
        out = np.zeros((n_solid_nodes, 3), dtype=float)
        for q_idx in range(k):
            contrib = nbr_w[:, q_idx : q_idx + 1] * f_fluid
            np.add.at(out, nbr_ids[:, q_idx], contrib)
        return out

    @staticmethod
    def compute_S_lumped(n_solid_nodes, nbr_ids, nbr_w, A_diag):
        _n_f, k = nbr_ids.shape
        A_diag = np.asarray(A_diag, dtype=float).reshape(-1)
        S = np.zeros((n_solid_nodes,), dtype=float)
        for q_idx in range(k):
            np.add.at(S, nbr_ids[:, q_idx], nbr_w[:, q_idx] * A_diag)
        return np.maximum(S, 1.0e-14)

    @staticmethod
    def apply_Tf_operator(Fa, n_solid_nodes, nbr_ids, nbr_w, A_diag, S_lumped):
        Fa = np.asarray(Fa, dtype=float).reshape(-1, 3)
        FaA = Fa * np.asarray(A_diag, dtype=float).reshape(-1, 1)
        rhs = RBFTransfer.map_forces_to_solid(FaA, n_solid_nodes, nbr_ids, nbr_w)
        Fs_coeff = rhs / np.asarray(S_lumped, dtype=float).reshape(-1, 1)
        return Fs_coeff, rhs

    def apply(self, Fa):
        return self.apply_Tf_operator(
            Fa,
            len(self.solid_points),
            self.nbr_ids,
            self.nbr_w,
            self.A_diag,
            self.S_lumped,
        )

    @staticmethod
    def relative_round_trip_error(reference, round_trip, epsilon_reg=1.0e-16):
        reference = np.asarray(reference, dtype=float).reshape(-1, 3)
        round_trip = np.asarray(round_trip, dtype=float).reshape(-1, 3)
        residual = float(np.linalg.norm((round_trip - reference).reshape(-1)))
        reference_norm = float(np.linalg.norm(reference.reshape(-1)))
        relative_error = residual / max(reference_norm + float(epsilon_reg), 1.0e-300)
        return residual, relative_error

    def round_trip_force_metrics(self, fluid_forces, epsilon_reg=1.0e-16):
        fluid_forces = np.asarray(fluid_forces, dtype=float).reshape(-1, 3)
        solid_forces, _rhs = self.apply(fluid_forces)
        fluid_forces_round_trip = self.apply_rbf_interpolation(
            solid_forces, self.nbr_ids, self.nbr_w
        ) * self.A_diag[:, None]
        residual, relative_error = self.relative_round_trip_error(
            fluid_forces, fluid_forces_round_trip, epsilon_reg=epsilon_reg
        )
        return residual, relative_error, fluid_forces_round_trip

    def round_trip_displacement_metrics(self, solid_displacements, epsilon_reg=1.0e-16):
        solid_displacements = np.asarray(solid_displacements, dtype=float).reshape(-1, 3)
        fluid_displacements = self.apply_rbf_interpolation(
            solid_displacements, self.nbr_ids, self.nbr_w
        )
        solid_displacements_round_trip, _rhs = self.apply(fluid_displacements)
        residual, relative_error = self.relative_round_trip_error(
            solid_displacements, solid_displacements_round_trip, epsilon_reg=epsilon_reg
        )
        return residual, relative_error, solid_displacements_round_trip

    @staticmethod
    def apply_rbf_interpolation(values, nbr_ids, nbr_w):
        values = np.asarray(values, dtype=float).reshape(-1, 3)
        return np.sum(nbr_w[:, :, None] * values[nbr_ids, :], axis=1)


__all__ = ["RBFTransfer"]
