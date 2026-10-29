import os
import sys
from time import perf_counter

print("cwd =", os.getcwd())
print("sys.path =")
for p in sys.path:
    print(" ", p)

from dolfin import *
import argparse
import csv
import json
import os
import socket

import numpy as np
import yaml
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from verification.mms.coupled_target import CoupledMMSTarget as PythonMMSTarget
from verification.mms.coupled_solid_source import (
    ReferenceForceFunctional, add_coupled_mms_source, _target_function,
)

from utils import SolidDomain, ReissnerMindlinModel
from coupling.utils.coupling_transfer import CouplingTransfer

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True


def load_yaml_config(config_dir, *candidate_names):
    for name in candidate_names:
        cfg_path = os.path.join(config_dir, name)
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as stream:
                return yaml.safe_load(stream)
    raise FileNotFoundError(
        f"could not find this config file in {config_dir}: {', '.join(candidate_names)}"
    )


def cfg_get(cfg, *keys, default=None):
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def optional_float(data, key, default=np.nan):
    try:
        value = data.get(key, default)
        return float(value)
    except (TypeError, ValueError):
        return default


def vector_norm(rows):
    arr = np.asarray(rows, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.linalg.norm(arr.reshape(-1)))


def residual_metrics(current, previous, epsilon_reg):
    reference_norm = vector_norm(current)
    if previous is None:
        return 0.0, reference_norm, 0.0
    current_arr = np.asarray(current, dtype=float)
    previous_arr = np.asarray(previous, dtype=float)
    if current_arr.shape != previous_arr.shape or current_arr.size == 0:
        return 0.0, reference_norm, 0.0
    residual = float(np.linalg.norm((current_arr - previous_arr).reshape(-1)))
    relative_error = residual / max(reference_norm + epsilon_reg, 1.0e-300)
    return residual, reference_norm, relative_error


class StructuralSolver:
    def __init__(self, config_path, mms_target_path=None, mms_output=None):
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.config_path = config_path
        self.solid_config = load_yaml_config(config_path, "solid_params.yaml")
        self.fluid_config = load_yaml_config(config_path, "fluid_params.yaml")
        self.coupling_config = load_yaml_config(config_path, "coupling_params.yaml")
        self.mms_target_path = mms_target_path
        self.mms_output = mms_output
        self.mms_target = None
        # The residual assembly is shared by ordinary and MMS solves; keep
        # the optional source unset for normal runs as well as MMS runs.
        self.mms_rhs_source = None
        self._newton_step_stats = {}

        self._setup_simulation_data()
        self._setup_domain()
        self._setup_model()
        self._setup_transfer()
        self._setup_output()
        self._setup_socket()
        self._setup_history()
        if self.mms_target_path:
            with open(self.mms_target_path, encoding="utf-8") as stream:
                self.mms_target = PythonMMSTarget.from_manifest(json.load(stream))
            self.reference_force = ReferenceForceFunctional(self.model, self.mms_target)
            self.mms_spaces = [self.model.V.sub(i).collapse() for i in range(3)]
            # The numerical solve starts from the target's declared initial
            # data. For the current ramp target q(0)=q_t(0)=0 while q_tt(0)
            # is generally nonzero; this is an initial condition, not a
            # replacement of any subsequent numerical state.
            self.model.q.assign(_target_function(self.model, self.mms_target, 0.0, 0, self.mms_spaces))
            self.model.q_old.assign(_target_function(self.model, self.mms_target, 0.0, 0, self.mms_spaces))
            self.model.v_old.assign(_target_function(self.model, self.mms_target, 0.0, 1, self.mms_spaces))
            self.model.a_old.assign(_target_function(self.model, self.mms_target, 0.0, 2, self.mms_spaces))
            out = self.mms_output or self.out_dir
            os.makedirs(out, exist_ok=True)
            self.mms_history = open(os.path.join(out, "solid_mms_history.csv"), "w", encoding="utf-8")
            self.mms_history.write("step,time_s,target_force,numerical_force,target_state,numerical_state,state_error_l2,source_l2\n")

    def _setup_simulation_data(self):
        self.T = float(self.coupling_config["total_time"])
        self.Nsteps = int(
            cfg_get(self.coupling_config, "n_steps", default=cfg_get(self.coupling_config, "nsteps"))
        )
        self.dt_value = self.T / self.Nsteps
        self.dt = Constant(self.dt_value)
        self.epsilon_reg = float(cfg_get(self.coupling_config, "epsilon_reg", default=1.0e-16))

    def _setup_domain(self):
        self.n_span_comm = int(
            cfg_get(
                self.coupling_config,
                "comm_nspan",
                default=cfg_get(
                    self.fluid_config,
                    "mesh",
                    "n_span",
                    default=cfg_get(self.fluid_config, "n_span_comm", default=80),
                ),
            )
        )
        self.domain = SolidDomain(
            span=cfg_get(self.solid_config, "span"),
            root_chord=cfg_get(self.solid_config, "root_chord"),
            tip_chord=cfg_get(self.solid_config, "tip_chord"),
            thickness_ratio=cfg_get(self.solid_config, "thickness_ratio"),
            leading_edge_sweep=cfg_get(self.solid_config, "leading_edge_sweep", default=0.0),
            nx=cfg_get(self.solid_config, "nx"),
            ny=cfg_get(self.solid_config, "ny"),
            n_span_comm=self.n_span_comm,
            span_sampling_mode=cfg_get(
                self.solid_config,
                "span_sampling_mode",
                default=cfg_get(self.coupling_config, "span_sampling", default="node-stride"),
            ),
            span_custom_stride=cfg_get(
                self.coupling_config,
                "custom_span_stride",
                default=os.getenv("COUPLING_SPAN_STRIDE"),
            ),
        )
        self.span = self.domain.span
        self.root_chord = self.domain.root_chord
        self.tip_chord = self.domain.tip_chord
        self.leading_edge_sweep = self.domain.leading_edge_sweep
        self.nx = self.domain.nx
        self.ny = self.domain.ny
        self.eta_span_comm = self.domain.eta_span_comm
        self.eta_span_comm_indices = self.domain.eta_span_comm_indices
        self.eta_cp = float(os.getenv("COUPLING_ETA_CP", "0.75"))
        self.eta_cp_comm = np.array([self.eta_cp], dtype=float)
        self.work_conservative_mode = bool(
            cfg_get(
                self.solid_config,
                "work_conservative_mode",
                default=cfg_get(self.coupling_config, "work_conservation_mode", default=True),
            )
        )
        self.edge_eval_xi_eps = float(os.getenv("COUPLING_EDGE_EVAL_XI_EPS", "1.0e-6"))
        self.enforce_chord_projection = bool(
            cfg_get(self.coupling_config, "enforce_chord_projection", default=False)
        )
        self.enforce_span_projection = bool(
            cfg_get(self.coupling_config, "enforce_span_projection", default=False)
        )
        self.DEBUG_IO = os.getenv("COUPLING_DEBUG_IO", "0").strip().lower() not in (
            "0",
            "false",
            "no",
        )

    def _setup_model(self):
        self.model = ReissnerMindlinModel(
            domain=self.domain,
            E=cfg_get(self.solid_config, "E"),
            nu=cfg_get(self.solid_config, "nu"),
            rho_s=cfg_get(self.solid_config, "rho_s"),
            eta_m=cfg_get(self.solid_config, "eta_m"),
            eta_k=cfg_get(self.solid_config, "eta_k"),
            kappa_shear=cfg_get(self.solid_config, "kappa_shear"),
            dt=self.dt,
            alpha_m=cfg_get(self.solid_config, "alpha_m", default=0.10),
            alpha_f=cfg_get(self.solid_config, "alpha_f", default=0.20),
            geometric_nonlinearity=cfg_get(
                self.solid_config, "geometric_nonlinearity", default=True
            ),
            include_quadratic_inplane_strain=cfg_get(
                self.solid_config, "include_quadratic_inplane_strain", default=True
            ),
        )
        self.model.build_residual_forms()

        self.newton_atol = float(
            cfg_get(
                self.solid_config,
                "newton_atol",
                default=os.getenv("SOLID_NEWTON_ATOL", "2.0e-4"),
            )
        )
        self.newton_rtol = float(
            cfg_get(
                self.solid_config,
                "newton_rtol",
                default=os.getenv("SOLID_NEWTON_RTOL", "3.0e-3"),
            )
        )
        self.newton_maxit = int(
            cfg_get(
                self.solid_config,
                "newton_maxiter",
                default=cfg_get(
                    self.solid_config,
                    "newton_maxit",
                    default=os.getenv("SOLID_NEWTON_MAXIT", "60"),
                ),
            )
        )
        self.force_ramp_iters = int(
            cfg_get(
                self.solid_config,
                "force_ramp_iters",
                default=os.getenv("SOLID_FORCE_RAMP_ITERS", "8"),
            )
        )
        self.newton_max_load_subdivisions = int(
            cfg_get(self.solid_config, "newton_max_load_subdivisions", default=5)
        )
        if self.newton_max_load_subdivisions < 0:
            raise ValueError("newton_max_load_subdivisions must be nonnegative")
        self.force_relax = cfg_get(self.coupling_config, "force_relax", default=1.0)
        self.nonlinear_solver = str(
            cfg_get(self.solid_config, "nonlinear_solver", default="newton_linesearch")
        ).strip().lower()
        if self.nonlinear_solver == "newton_line_search":
            self.nonlinear_solver = "newton_linesearch"
        if self.nonlinear_solver not in ("newton_linesearch", "regularized_newton"):
            raise ValueError(
                "nonlinear_solver must be 'newton_linesearch' or 'regularized_newton'"
            )
        self.newton_max_backtracks = int(
            cfg_get(self.solid_config, "newton_max_backtracks", default=12)
        )
        self.newton_armijo_c1 = float(
            cfg_get(self.solid_config, "newton_armijo_c1", default=1.0e-4)
        )
        self.regularized_newton_initial_lambda = float(
            cfg_get(self.solid_config, "regularized_newton_initial_lambda", default=1.0e-6)
        )
        self.regularized_newton_lambda_growth = float(
            cfg_get(self.solid_config, "regularized_newton_lambda_growth", default=10.0)
        )
        self.regularized_newton_lambda_decay = float(
            cfg_get(self.solid_config, "regularized_newton_lambda_decay", default=0.25)
        )
        self.regularized_newton_max_attempts = int(
            cfg_get(self.solid_config, "regularized_newton_max_attempts", default=8)
        )
        if self.regularized_newton_initial_lambda <= 0.0:
            raise ValueError("regularized_newton_initial_lambda must be positive")
        if self.regularized_newton_lambda_growth <= 1.0:
            raise ValueError("regularized_newton_lambda_growth must exceed one")
        if not 0.0 < self.regularized_newton_lambda_decay <= 1.0:
            raise ValueError("regularized_newton_lambda_decay must lie in (0, 1]")
        if self.regularized_newton_max_attempts < 1:
            raise ValueError("regularized_newton_max_attempts must be at least one")
        self._regularization_reference_matrix = None

        self.dofs_u_x, self.dofs_u_y, self.dofs_w = self.model.extract_dof_maps()

    def _setup_transfer(self):
        self.transfer = CouplingTransfer(
            domain=self.domain,
            model=self.model,
            eta_span_comm=self.eta_span_comm,
            eta_cp=self.eta_cp,
            force_transfer_mode=cfg_get(
                self.coupling_config, "force_transfer_mode", default="rbf"
            ),
            displacement_transfer_mode=cfg_get(
                self.coupling_config, "displacement_transfer_mode", default="rbf"
            ),
            rbf_radius=cfg_get(self.coupling_config, "rbf_radius", default=0.08),
            rbf_neighbors=cfg_get(self.coupling_config, "rbf_neighbors", default=24),
            max_abs_force_component=cfg_get(
                self.coupling_config, "max_abs_force_component", default=5.0e3
            ),
            edge_eval_xi_eps=self.edge_eval_xi_eps,
            work_conservative_mode=self.work_conservative_mode,
            enforce_chord_projection=self.enforce_chord_projection,
            enforce_span_projection=self.enforce_span_projection,
            debug_io=self.DEBUG_IO,
        )
        self.force_transfer_mode = self.transfer.force_transfer_mode
        self.displacement_transfer_mode = self.transfer.displacement_transfer_mode
        print(f"Solid force transfer mode: {self.force_transfer_mode}")
        print(f"Solid displacement transfer mode: {self.displacement_transfer_mode}")
        self.crm_transfer_matrix = self.transfer.crm_transfer_matrix
        self.crm_panel_areas = self.transfer.crm_panel_areas
        self.crm_panel_polys = self.transfer.crm_panel_polys
        self.cp_targets = self.transfer.cp_targets
        self.le_targets = self.transfer.le_targets
        self.te_targets = self.transfer.te_targets
        self.interface_node_ids = self.transfer.interface_node_ids
        self.interface_coords = self.transfer.interface_coords
        self.interface_tree = self.transfer.interface_tree
        self.nbr_ids = self.transfer.nbr_ids
        self.nbr_w = self.transfer.nbr_w
        self.A_diag = self.transfer.A_diag
        self.S_lumped = self.transfer.S_lumped

    def _setup_output(self):
        results_root = cfg_get(self.solid_config, "results_root", default=os.path.join(self.repo_root, "results"))
        solid_output_dir = cfg_get(
            self.solid_config,
            "output_dir",
            default=os.path.join(
                "solid", cfg_get(self.solid_config, "output_run_name", default="v18_reissner_mindlin_plate")
            ),
        )
        if os.path.isabs(results_root):
            base_results_root = results_root
        else:
            base_results_root = os.path.join(self.repo_root, results_root)
        if os.path.isabs(solid_output_dir):
            self.out_dir = solid_output_dir
        else:
            self.out_dir = os.path.join(base_results_root, solid_output_dir)
        os.makedirs(self.out_dir, exist_ok=True)

        self.xdmf_path = os.path.join(
            self.out_dir,
            cfg_get(self.solid_config, "xdmf_filename", default="elastodynamics-results.xdmf"),
        )
        self.xdmf_file = XDMFFile(self.xdmf_path)
        self.xdmf_file.parameters["flush_output"] = True
        self.xdmf_file.parameters["functions_share_mesh"] = True
        self.xdmf_file.parameters["rewrite_function_mesh"] = False

        self.mesh_pvd_path = os.path.join(
            self.out_dir, cfg_get(self.solid_config, "mesh_pvd_filename", default="solid_mesh.pvd")
        )
        self.q_pvd_path = os.path.join(
            self.out_dir, cfg_get(self.solid_config, "q_pvd_filename", default="plate_state.pvd")
        )
        self.sig_pvd_path = os.path.join(
            self.out_dir,
            cfg_get(self.solid_config, "sig_pvd_filename", default="membrane_stress.pvd"),
        )
        self.mesh_pvd = File(self.mesh_pvd_path)
        self.q_pvd = File(self.q_pvd_path)
        self.sig_pvd = File(self.sig_pvd_path)
        self.mesh_pvd << self.domain.mesh
        self.geometry_consistency_path = os.path.join(
            self.out_dir,
            cfg_get(self.solid_config, "geometry_consistency_filename", default="cp_geometry_consistency.txt"),
        )
        self.geometry_consistency_fp = open(self.geometry_consistency_path, "w")
        self.geometry_consistency_fp.write(
            "# step time station x y z u_ref_x u_ref_y u_ref_z u_geom_x u_geom_y u_geom_z "
            "err_x err_y err_z err_norm\\n"
        )
        self.state_history_path = os.path.join(self.out_dir, "solid_state_history.csv")
        self.state_history_fp = open(self.state_history_path, "w", newline="", encoding="utf-8")
        self.state_history_writer = csv.writer(self.state_history_fp)
        state_header = ["step", "time_s", "E_elas_J", "E_kin_J", "E_damp_J", "E_total_J",
                        "surface_strain_rms", "surface_strain_max", "membrane_stress_max_N_per_m",
                        "newton_iterations", "newton_residual_norm", "newton_relative_residual", "newton_load_stages"]
        for loc in ("tip", "midspan", "leading_edge_midspan"):
            for field in ("u", "v", "w", "theta_x", "theta_y"):
                for deriv in ("displacement", "velocity", "acceleration"):
                    state_header.append(f"{loc}_{field}_{deriv}")
        self.state_history_writer.writerow(state_header)

    def _setup_socket(self):
        print("Connecting solid to coupling server...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        coupling_host = str(
            cfg_get(self.coupling_config, "host", default=os.getenv("COUPLING_HOST", "127.0.0.1"))
        )
        coupling_port = int(
            cfg_get(self.coupling_config, "port", default=os.getenv("COUPLING_PORT", "9000"))
        )
        self.sock.connect((coupling_host, coupling_port))
        self.sock_file = self.sock.makefile("r")
        self.sock.sendall((json.dumps({"role": "solid"}) + "\n").encode())
        print("Solid connected.")

    def _setup_history(self):
        self.time = np.linspace(0.0, self.T, self.Nsteps + 1)
        self.u_tip = np.zeros((self.Nsteps + 1,), dtype=float)
        self.surface_strain_percent = np.zeros((self.Nsteps,), dtype=float)
        self.strain_increase_percentage_points = np.zeros((self.Nsteps,), dtype=float)
        self.initial_surface_strain_rms = self.model.compute_surface_strain_rms()
        self.energies = np.zeros((self.Nsteps + 1, 4), dtype=float)
        self.surface_strain_max = np.full((self.Nsteps,), np.nan, dtype=float)
        self.membrane_stress_max = np.full((self.Nsteps,), np.nan, dtype=float)
        self.newton_iterations = np.zeros((self.Nsteps,), dtype=int)
        self.newton_residual_norm = np.full((self.Nsteps,), np.nan, dtype=float)
        self.newton_relative_residual = np.full((self.Nsteps,), np.nan, dtype=float)
        self.newton_load_stages = np.zeros((self.Nsteps,), dtype=int)
        self.step_walltime = np.full((self.Nsteps,), np.nan, dtype=float)
        self.E_damp_acc = 0.0
        self.forces_prev = None
        self.incremental_work_rel_errors = np.full((self.Nsteps,), np.nan, dtype=float)
        self.incremental_work_Wf = np.full((self.Nsteps,), np.nan, dtype=float)
        self.incremental_work_Ws = np.full((self.Nsteps,), np.nan, dtype=float)
        self.incremental_work_defects = np.full((self.Nsteps,), np.nan, dtype=float)
        self.force_transfer_relative_errors = np.full((self.Nsteps,), np.nan, dtype=float)
        self.moment_transfer_relative_errors = np.full((self.Nsteps,), np.nan, dtype=float)
        self.force_transfer_residual_norms = np.full((self.Nsteps,), np.nan, dtype=float)
        self.moment_transfer_residual_norms = np.full((self.Nsteps,), np.nan, dtype=float)
        self.cp_geometry_rms_errors = np.full((self.Nsteps,), np.nan, dtype=float)
        self.cp_geometry_max_errors = np.full((self.Nsteps,), np.nan, dtype=float)
        self.cp_geometry_relative_rms_errors = np.full((self.Nsteps,), np.nan, dtype=float)
        self.force_residuals = np.full((self.Nsteps,), np.nan, dtype=float)
        self.force_reference_norms = np.full((self.Nsteps,), np.nan, dtype=float)
        self.force_relative_errors = np.full((self.Nsteps,), np.nan, dtype=float)
        self.force_transfer_apply_times = np.full((self.Nsteps,), np.nan, dtype=float)
        self.geometry_transfer_apply_times = np.full((self.Nsteps,), np.nan, dtype=float)
        self.solid_solve_times = np.full((self.Nsteps,), np.nan, dtype=float)
        self.solid_io_times = np.full((self.Nsteps,), np.nan, dtype=float)
        self.geometry_residuals = np.full((self.Nsteps,), np.nan, dtype=float)
        self.geometry_reference_norms = np.full((self.Nsteps,), np.nan, dtype=float)
        self.geometry_relative_errors = np.full((self.Nsteps,), np.nan, dtype=float)
        self.prev_geometry_sent = None
        self.ext_force_vec_template = self.model.q.vector().copy()
        self.ext_force_vec_template.zero()
        self.tip_x = self.domain.x_leading_edge_at(self.span) + self.eta_cp * self.domain.chord_at(self.span)
        self.tip_y = self.span - 1.0e-8
        self.zero_payload = [[0.0, 0.0, 0.0] for _ in range(self.transfer.cp_targets.shape[0])]
        self.cp_reference_payload = self.transfer.cp_targets.tolist()
        self.le_reference_payload = self.transfer.le_targets.tolist()
        self.te_reference_payload = self.transfer.te_targets.tolist()

    def _send_initial_geometry(self):
        init_msg = {
            "step": 0,
            "dt": self.dt_value,
            "ttot": self.T,
            "nsteps": self.Nsteps,
            "n_span": len(self.eta_span_comm),
            "n_chord": 1,
            "indexing": "span-major",
            "eta_span": self.eta_span_comm.tolist(),
            "eta_chord": self.eta_cp_comm.tolist(),
            "geometry": self.zero_payload,
            "geometry_le": self.zero_payload,
            "geometry_te": self.zero_payload,
            "geometry_cp_absolute": self.cp_reference_payload,
            "geometry_le_absolute": self.le_reference_payload,
            "geometry_te_absolute": self.te_reference_payload,
            "rotation": self.zero_payload,
            "rotation_le": self.zero_payload,
            "rotation_te": self.zero_payload,
            "surface_strain_percent": 100.0 * self.initial_surface_strain_rms,
            "strain_increase_percentage_points": 0.0,
        }
        self.sock.sendall((json.dumps(init_msg) + "\n").encode())
        print("Initial zero geometry sent.")

    def _assemble_equilibrium_residual(self, ext_force_vec, load_factor):
        """Return minus the fixed-load residual, with Dirichlet rows imposed."""
        residual = assemble(self.model.residual)
        rhs = residual.copy()
        rhs *= -1.0
        rhs.axpy(float(load_factor), ext_force_vec)
        if self.mms_rhs_source is not None:
            rhs.axpy(1.0, self.mms_rhs_source)
        for bc in self.model.bcs:
            bc.apply(rhs)
        return rhs

    def _regularization_reference(self):
        """Reference stiffness used by regularized Newton, assembled once."""
        if self._regularization_reference_matrix is None:
            K_ref = assemble(self.model.stiffness_form(self.model.dq_trial, self.model.q_test))
            for bc in self.model.bcs:
                bc.apply(K_ref)
            self._regularization_reference_matrix = K_ref
        return self._regularization_reference_matrix

    def _line_search(self, q_before, delta_q, ext_force_vec, load_factor, residual_norm):
        """Armijo backtracking on the residual-norm-squared merit function."""
        for backtrack in range(self.newton_max_backtracks + 1):
            step_fraction = 0.5 ** backtrack
            self.model.q.vector().set_local(q_before + step_fraction * delta_q)
            self.model.q.vector().apply("insert")
            trial_norm = self._assemble_equilibrium_residual(ext_force_vec, load_factor).norm("l2")
            sufficient_decrease = trial_norm <= (1.0 - self.newton_armijo_c1 * step_fraction) * residual_norm
            if np.isfinite(trial_norm) and sufficient_decrease:
                return True, step_fraction, trial_norm, backtrack
        self.model.q.vector().set_local(q_before)
        self.model.q.vector().apply("insert")
        return False, 0.0, residual_norm, self.newton_max_backtracks + 1

    def _solve_nonlinear_load_stage(self, ext_force_vec, load_factor, regularization_lambda):
        """Solve one fixed external-load continuation stage."""
        r0 = None
        r_norm = np.inf
        for newton_it in range(self.newton_maxit):
            A = assemble(self.model.jacobian)
            rhs = self._assemble_equilibrium_residual(ext_force_vec, load_factor)
            for bc in self.model.bcs:
                bc.apply(A)
            r_norm = rhs.norm("l2")
            if r0 is None:
                r0 = max(r_norm, 1.0e-16)
            r_rel = r_norm / r0
            if r_norm <= self.newton_atol or r_rel <= self.newton_rtol:
                self._newton_stage_stats.append({
                    "iterations": newton_it,
                    "residual_norm": float(r_norm),
                    "relative_residual": float(r_rel),
                })
                return regularization_lambda

            q_before = self.model.q.vector().get_local().copy()
            accepted = False
            if self.nonlinear_solver == "newton_linesearch":
                self.model.linear_solver.solve(A, self.model.dq_newton.vector(), rhs)
                accepted, step_fraction, trial_norm, backtracks = self._line_search(
                    q_before,
                    self.model.dq_newton.vector().get_local(),
                    ext_force_vec,
                    load_factor,
                    r_norm,
                )
            else:
                for regularization_attempt in range(self.regularized_newton_max_attempts):
                    A_reg = A.copy()
                    A_reg.axpy(regularization_lambda, self._regularization_reference(), False)
                    for bc in self.model.bcs:
                        bc.apply(A_reg)
                    self.model.linear_solver.solve(A_reg, self.model.dq_newton.vector(), rhs)
                    accepted, step_fraction, trial_norm, backtracks = self._line_search(
                        q_before,
                        self.model.dq_newton.vector().get_local(),
                        ext_force_vec,
                        load_factor,
                        r_norm,
                    )
                    if accepted:
                        regularization_lambda *= self.regularized_newton_lambda_decay
                        break
                    regularization_lambda *= self.regularized_newton_lambda_growth

            if not accepted:
                raise RuntimeError(
                    f"{self.nonlinear_solver} line search failed: load_factor={load_factor:.3e}, "
                    f"iteration={newton_it + 1}, residual={r_norm:.4e}, "
                    f"backtracks={backtracks}"
                )
            if self.DEBUG_IO:
                print(
                    f"{self.nonlinear_solver}: load={load_factor:.3f}, Newton={newton_it + 1}, "
                    f"step={step_fraction:.3e}, backtracks={backtracks}, "
                    f"residual={r_norm:.3e}->{trial_norm:.3e}"
                )

        raise RuntimeError(
            f"{self.nonlinear_solver} failed: load_factor={load_factor:.3e}, "
            f"residual={r_norm:.6e}, relative={r_rel:.6e}, "
            f"atol={self.newton_atol:.2e}, rtol={self.newton_rtol:.2e}, "
            f"maxit={self.newton_maxit}"
        )

    def _solve_newton_step(self, ext_force_vec):
        """Solve the step with bounded adaptive load continuation.

        A failed continuation stage restores its starting displacement and
        bisects that load interval. This gives Newton a closer initial guess
        without changing the physical time step or accepting a nonconverged
        state.
        """
        self._newton_stage_stats = []
        n_load_stages = max(1, self.force_ramp_iters)
        regularization_lambda = self.regularized_newton_initial_lambda
        pending = [(float(stage) / float(n_load_stages), 0)
                   for stage in range(1, n_load_stages + 1)]
        current_load_factor = 0.0
        load_subdivisions = 0
        while pending:
            load_factor, subdivision_depth = pending.pop(0)
            stage_start = self.model.q.vector().get_local().copy()
            try:
                regularization_lambda = self._solve_nonlinear_load_stage(
                    ext_force_vec, load_factor, regularization_lambda
                )
            except RuntimeError as exc:
                self.model.q.vector().set_local(stage_start)
                self.model.q.vector().apply("insert")
                if subdivision_depth >= self.newton_max_load_subdivisions:
                    raise RuntimeError(
                        f"Newton continuation failed at load_factor={load_factor:.6g} "
                        f"after {load_subdivisions} load-interval subdivisions; "
                        "state restored to the start of the failed interval"
                    ) from exc
                midpoint = 0.5 * (current_load_factor + load_factor)
                if midpoint <= current_load_factor or midpoint >= load_factor:
                    raise RuntimeError(
                        f"Newton continuation cannot subdivide load interval "
                        f"[{current_load_factor:.6g}, {load_factor:.6g}]"
                    ) from exc
                load_subdivisions += 1
                print(
                    f"Newton continuation retry: step load interval "
                    f"[{current_load_factor:.6g}, {load_factor:.6g}] failed; "
                    f"subdividing at {midpoint:.6g} "
                    f"({load_subdivisions}/{self.newton_max_load_subdivisions} retries)"
                )
                pending.insert(0, (load_factor, subdivision_depth + 1))
                pending.insert(0, (midpoint, subdivision_depth + 1))
                continue
            current_load_factor = load_factor
        stages = self._newton_stage_stats
        self._newton_step_stats = {
            "iterations": sum(stage["iterations"] for stage in stages),
            "residual_norm": stages[-1]["residual_norm"] if stages else float("nan"),
            "relative_residual": stages[-1]["relative_residual"] if stages else float("nan"),
            "load_stages": len(stages),
            "load_subdivisions": load_subdivisions,
        }

    def _sample_state_history(self, step, time_s, energies, strain_rms, strain_max, stress_max):
        """Write physical state/derivative samples at representative plate locations."""
        points = {
            "tip": (self.tip_x, self.tip_y),
            "midspan": (self.domain.x_leading_edge_at(0.5 * self.span) + 0.5 * self.domain.chord_at(0.5 * self.span), 0.5 * self.span),
            "leading_edge_midspan": (self.domain.x_leading_edge_at(0.5 * self.span), 0.5 * self.span),
        }
        row = [int(step), float(time_s), *map(float, energies), float(strain_rms),
               float(strain_max), float(stress_max), self._newton_step_stats.get("iterations", 0),
               self._newton_step_stats.get("residual_norm", float("nan")),
               self._newton_step_stats.get("relative_residual", float("nan")),
               self._newton_step_stats.get("load_stages", 0)]
        for x, y in points.values():
            point = Point(float(x), float(y))
            values = []
            for field in (self.model.q_old, self.model.v_old, self.model.a_old):
                try:
                    vals = np.asarray(field(point), dtype=float).reshape(-1)
                    values.append(vals[:5])
                except (RuntimeError, ValueError):
                    values.append(np.full(5, np.nan))
            for component in range(5):
                row.extend([float(values[0][component]), float(values[1][component]), float(values[2][component])])
        self.state_history_writer.writerow(row)
        self.state_history_fp.flush()

    def run(self):
        self._send_initial_geometry()

        for i_step in range(self.Nsteps):
            solid_step_time = float("nan")
            print(f"Solid step {i_step + 1}/{self.Nsteps}: waiting for force...")
            line = self.sock_file.readline()
            if line == "":
                raise RuntimeError("Coupling server disconnected while sending force data")

            data = json.loads(line)
            step_start = perf_counter()
            if self.mms_target is not None:
                if "target_force" not in data:
                    raise RuntimeError("MMS target run requires target_force in the fluid socket payload")
                target_force = np.asarray(data["target_force"], dtype=float)
                self.reference_force.set_panel_forces(target_force)
                physical_time = float(data.get("fluid_time_s", (i_step + 1) * self.dt_value))
                self.mms_rhs_source = add_coupled_mms_source(
                    self.model, self.mms_target, physical_time, self.reference_force,
                    self.mms_spaces,
                )
                if i_step == 0:
                    self._mms_target_force_current = target_force.copy()
            self.force_residuals[i_step] = optional_float(data, "force_residual")
            self.force_reference_norms[i_step] = optional_float(data, "force_reference_norm")
            self.force_relative_errors[i_step] = optional_float(data, "force_relative_error")
            forces, used_structured_force = self.transfer.parse_force_payload(
                data, len(self.eta_span_comm), 1, self.eta_span_comm, self.eta_cp_comm
            )
            if not np.isfinite(forces).all():
                raise RuntimeError(f"Non-finite force data at solid step {i_step + 1}")

            if i_step == 0:
                if used_structured_force:
                    print(
                        f"Solid: received structured force payload ({len(self.eta_span_comm)}x1)"
                    )
                else:
                    print("Solid: received legacy force payload and resampled to coupling stations")

            if self.forces_prev is None:
                forces_eff = forces.copy()
            else:
                forces_eff = self.force_relax * forces + (1.0 - self.force_relax) * self.forces_prev
            self.forces_prev = forces_eff.copy()

            nodal_forces = None
            force_transfer_apply_start = perf_counter()
            if self.force_transfer_mode == "crm":
                _, nodal_forces = self.transfer.apply_force_transfer(forces_eff)
            else:
                _, nodal_forces = self.transfer.apply_force_transfer(forces_eff)
            force_transfer_apply_time = perf_counter() - force_transfer_apply_start
            self.force_transfer_apply_times[i_step] = force_transfer_apply_time
            if not np.isfinite(nodal_forces).all():
                raise RuntimeError(f"Non-finite mapped nodal forces at solid step {i_step + 1}")

            ext_force_vec = self.ext_force_vec_template.copy()
            ext_force_vec.zero()
            if nodal_forces is not None:
                self.transfer.add_nodal_forces_to_rhs_plate(
                    ext_force_vec,
                    nodal_forces,
                    self.interface_node_ids,
                    self.dofs_u_x,
                    self.dofs_u_y,
                    self.dofs_w,
                )

            solid_solve_start = perf_counter()
            self._solve_newton_step(ext_force_vec)
            self.newton_iterations[i_step] = int(self._newton_step_stats.get("iterations", 0))
            self.newton_residual_norm[i_step] = self._newton_step_stats.get("residual_norm", np.nan)
            self.newton_relative_residual[i_step] = self._newton_step_stats.get("relative_residual", np.nan)
            self.newton_load_stages[i_step] = int(self._newton_step_stats.get("load_stages", 0))
            if self.mms_target is not None:
                num_state = self.model.q.vector().get_local()
                tgt_func = _target_function(
                    self.model, self.mms_target, physical_time, 0,
                    self.mms_spaces,
                )
                tgt_state = tgt_func.vector().get_local()
                srcnorm = self.mms_rhs_source.norm("l2") if self.mms_rhs_source is not None else 0.0
                self.mms_history.write(
                    f"{i_step+1},{physical_time:.16e},{json.dumps(data.get('target_force', []))},"
                    f"{json.dumps(data.get('force', []))},{json.dumps(tgt_state.tolist())},"
                    f"{json.dumps(num_state.tolist())},{np.linalg.norm(num_state-tgt_state):.16e},{srcnorm:.16e}\n"
                )
                self.mms_history.flush()
            solid_solve_time = perf_counter() - solid_solve_start
            self.solid_solve_times[i_step] = solid_solve_time
            interface_disp_cur = self.transfer.get_nodal_displacements_plate(
                self.model.q,
                self.interface_node_ids,
                self.dofs_u_x,
                self.dofs_u_y,
                self.dofs_w,
            )

            if self.work_conservative_mode and nodal_forces is not None:
                interface_disp_prev = self.transfer.get_nodal_displacements_plate(
                    self.model.q_old,
                    self.interface_node_ids,
                    self.dofs_u_x,
                    self.dofs_u_y,
                    self.dofs_w,
                )
                Wf, Ws, work_defect, rel_work_err = self.transfer.incremental_work_metrics(
                    interface_disp_cur - interface_disp_prev,
                    forces_eff,
                    nodal_forces,
                    epsilon_reg=self.epsilon_reg,
                )
                self.incremental_work_Wf[i_step] = Wf
                self.incremental_work_Ws[i_step] = Ws
                self.incremental_work_defects[i_step] = work_defect
                self.incremental_work_rel_errors[i_step] = rel_work_err
                (
                    _force_fluid,
                    _force_solid,
                    _moment_fluid,
                    _moment_solid,
                    force_residual,
                    moment_residual,
                    force_rel_error,
                    moment_rel_error,
                ) = self.transfer.wrench_transfer_metrics(
                    self.cp_targets,
                    forces_eff * self.A_diag[:, None],
                    (
                        np.column_stack(
                            (
                                self.transfer.crm_backend.force_dof_coords,
                                np.zeros(self.transfer.crm_backend.force_dof_coords.shape[0]),
                            )
                        )
                        if self.force_transfer_mode == "crm"
                        else self.interface_coords
                    ),
                    nodal_forces,
                    epsilon_reg=self.epsilon_reg,
                )
                self.force_transfer_residual_norms[i_step] = np.linalg.norm(force_residual)
                self.moment_transfer_residual_norms[i_step] = np.linalg.norm(moment_residual)
                self.force_transfer_relative_errors[i_step] = force_rel_error
                self.moment_transfer_relative_errors[i_step] = moment_rel_error
                if i_step == 0 or (i_step + 1) % 20 == 0:
                    print(
                        f"Transfer audit step {i_step + 1}: "
                        f"incremental-work rel_err={rel_work_err:.3e}, "
                        f"force/moment rel_err={force_rel_error:.3e}/{moment_rel_error:.3e}"
                    )

            force_transfer_msg = {
                "type": "force_transfer_diagnostics",
                "step": i_step + 1,
                "force_transfer_apply_time": force_transfer_apply_time,
                "incremental_work_Wf": self.incremental_work_Wf[i_step],
                "incremental_work_Ws": self.incremental_work_Ws[i_step],
                "incremental_work_defect": self.incremental_work_defects[i_step],
                "incremental_work_relative_error": self.incremental_work_rel_errors[i_step],
                "force_transfer_residual_norm": self.force_transfer_residual_norms[i_step],
                "force_transfer_relative_error": self.force_transfer_relative_errors[i_step],
                "moment_transfer_residual_norm": self.moment_transfer_residual_norms[i_step],
                "moment_transfer_relative_error": self.moment_transfer_relative_errors[i_step],
                "crm_setup_time": self.transfer.crm_setup_time,
                "rbf_setup_time": self.transfer.rbf_setup_time,
                "geometry_rbf_setup_time": self.transfer.geometry_rbf_setup_time,
                "geometry_crm_setup_time": self.transfer.geometry_crm_setup_time,
                "newton_iterations": self._newton_step_stats.get("iterations", 0),
                "newton_residual_norm": self._newton_step_stats.get("residual_norm", float("nan")),
                "newton_relative_residual": self._newton_step_stats.get("relative_residual", float("nan")),
                "newton_load_stages": self._newton_step_stats.get("load_stages", 0),
            }
            self.sock.sendall((json.dumps(force_transfer_msg) + "\n").encode())

            self.model.update_fields()
            surface_strain_rms = self.model.compute_surface_strain_rms()
            self.surface_strain_percent[i_step] = 100.0 * surface_strain_rms
            self.strain_increase_percentage_points[i_step] = 100.0 * (
                surface_strain_rms - self.initial_surface_strain_rms
            )
            t = self.time[i_step + 1]
            u_mem, w_mem, theta_mem = split(self.model.q_old)
            membrane_strain = (
                self.model.nonlinear_membrane_strain(u_mem, w_mem)
                if self.model.geometric_nonlinearity
                else self.model.membrane_strain(u_mem)
            )
            strain_field = project(membrane_strain, self.model.Vsig)
            stress_field = project(self.model.membrane_stress_from_strain(membrane_strain), self.model.Vsig)
            strain_values = strain_field.vector().get_local().reshape((-1, 4))
            stress_values = stress_field.vector().get_local().reshape((-1, 4))
            surface_strain_max = float(np.max(np.linalg.norm(strain_values, axis=1))) if strain_values.size else 0.0
            membrane_stress_max = float(np.max(np.linalg.norm(stress_values, axis=1))) if stress_values.size else 0.0
            self.surface_strain_max[i_step] = surface_strain_max
            self.membrane_stress_max[i_step] = membrane_stress_max
            self.model.sig.assign(stress_field)
            solid_io_start = perf_counter()
            self.xdmf_file.write(self.model.q, t)
            # The mixed Reissner-Mindlin state has value dimension five
            # (in-plane displacement[2], transverse displacement, rotation[2]);
            # DOLFIN's legacy VTK writer cannot serialize that mixed vector.
            # Export the physical three-component displacement to the PVD.
            plate_displacement = project(
                self.model.displacement_3d(self.model.q), self.model.Vt
            )
            plate_displacement.rename("PlateDisplacement", "")
            self.q_pvd << (plate_displacement, t)
            self.sig_pvd << (self.model.sig, t)
            solid_io_time = perf_counter() - solid_io_start
            self.solid_io_times[i_step] = solid_io_time

            E_elas = self.model.compute_strain_energy()
            E_kin = self.model.compute_kinetic_energy()
            self.E_damp_acc += self.model.compute_damping_increment(self.dt_value)
            E_tot = E_elas + E_kin + self.E_damp_acc
            self.energies[i_step + 1, :] = np.array([E_elas, E_kin, self.E_damp_acc, E_tot])
            self._sample_state_history(i_step + 1, t, self.energies[i_step + 1, :],
                                       surface_strain_rms, surface_strain_max, membrane_stress_max)

            try:
                vals = self.model.q(Point(self.tip_x, self.tip_y))
                self.u_tip[i_step + 1] = float(vals[2])
            except RuntimeError:
                self.u_tip[i_step + 1] = 0.0

            if i_step < self.Nsteps - 1:
                interface_disp_cur = self.transfer.get_nodal_displacements_plate(
                    self.model.q,
                    self.interface_node_ids,
                    self.dofs_u_x,
                    self.dofs_u_y,
                    self.dofs_w,
                )
                geometry_transfer_apply_start = perf_counter()
                u_le_arr, u_te_arr, _u_common_geom, _u_aero_geom = self.transfer.apply_geometry_transfer(
                    self.model.q,
                    fallback_tree=self.interface_tree,
                    fallback_vals=interface_disp_cur,
                )
                geometry_transfer_apply_time = perf_counter() - geometry_transfer_apply_start
                self.geometry_transfer_apply_times[i_step] = geometry_transfer_apply_time
                if self.enforce_chord_projection:
                    u_le_arr, u_te_arr, chord_len_cur, chord_len_ref = self.transfer.project_le_te_inextensible(
                        u_le_arr, u_te_arr, self.le_targets, self.te_targets
                    )
                    if self.DEBUG_IO and (i_step == 0 or (i_step + 1) % 20 == 0):
                        rel_ch = np.abs(chord_len_cur - chord_len_ref) / np.maximum(
                            chord_len_ref, 1.0e-14
                        )
                        print(
                            f"Chord projection step {i_step+1}: "
                            f"pre-proj rel chord err max/mean = {np.max(rel_ch):.3e}/{np.mean(rel_ch):.3e}"
                        )
                if self.enforce_span_projection:
                    u_le_arr, span_len_cur_le, span_len_ref_le = self.transfer.project_spanwise_inextensible_line(
                        u_le_arr, self.le_targets
                    )
                    u_te_arr, span_len_cur_te, span_len_ref_te = self.transfer.project_spanwise_inextensible_line(
                        u_te_arr, self.te_targets
                    )
                    if self.DEBUG_IO and (i_step == 0 or (i_step + 1) % 20 == 0):
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
                u_cp_arr = (1.0 - self.eta_cp) * u_le_arr + self.eta_cp * u_te_arr
                u_cp_reference = self.transfer.sample_vector_field_at_targets(
                    self.model.q,
                    self.cp_targets,
                    fallback_tree=self.interface_tree,
                    fallback_vals=interface_disp_cur,
                )
                cp_geometry_error = u_cp_arr - u_cp_reference
                cp_error_norm = np.linalg.norm(cp_geometry_error, axis=1)
                cp_reference_rms = np.sqrt(np.mean(np.sum(u_cp_reference ** 2, axis=1)))
                self.cp_geometry_rms_errors[i_step] = np.sqrt(np.mean(cp_error_norm ** 2))
                self.cp_geometry_max_errors[i_step] = np.max(cp_error_norm)
                self.cp_geometry_relative_rms_errors[i_step] = self.cp_geometry_rms_errors[i_step] / max(
                    cp_reference_rms, self.epsilon_reg
                )
                for station, (coord, u_ref, u_geom, err, err_norm) in enumerate(
                    zip(self.cp_targets, u_cp_reference, u_cp_arr, cp_geometry_error, cp_error_norm)
                ):
                    self.geometry_consistency_fp.write(
                        f"{i_step + 1} {self.time[i_step + 1]:.12e} {station} "
                        f"{coord[0]:.12e} {coord[1]:.12e} {coord[2]:.12e} "
                        f"{u_ref[0]:.12e} {u_ref[1]:.12e} {u_ref[2]:.12e} "
                        f"{u_geom[0]:.12e} {u_geom[1]:.12e} {u_geom[2]:.12e} "
                        f"{err[0]:.12e} {err[1]:.12e} {err[2]:.12e} {err_norm:.12e}\n"
                    )
                self.geometry_consistency_fp.flush()
                zero_rot = np.zeros_like(u_cp_arr)
                (
                    geometry_residual,
                    geometry_reference_norm,
                    geometry_relative_error,
                ) = residual_metrics(u_cp_arr, self.prev_geometry_sent, self.epsilon_reg)
                self.geometry_residuals[i_step] = geometry_residual
                self.geometry_reference_norms[i_step] = geometry_reference_norm
                self.geometry_relative_errors[i_step] = geometry_relative_error
                self.prev_geometry_sent = u_cp_arr.copy()

                if self.DEBUG_IO and (i_step == 0 or (i_step + 1) % 20 == 0):
                    print(
                        f"SEND step {i_step + 1} first LE/TE = {u_le_arr[0, :].tolist()} / {u_te_arr[0, :].tolist()}"
                    )
                    print(
                        f"SEND step {i_step + 1} last  LE/TE = {u_le_arr[-1, :].tolist()} / {u_te_arr[-1, :].tolist()}"
                    )

                solid_step_time = perf_counter() - step_start
                self.step_walltime[i_step] = solid_step_time
                msg_geo = json.dumps(
                    {
                        "step": i_step + 1,
                        "solid_time_s": float(self.time[i_step + 1]),
                        "dt": self.dt_value,
                        "ttot": self.T,
                        "nsteps": self.Nsteps,
                        "n_span": len(self.eta_span_comm),
                        "n_chord": 1,
                        "indexing": "span-major",
                        "eta_span": self.eta_span_comm.tolist(),
                        "eta_chord": self.eta_cp_comm.tolist(),
                        "geometry": u_cp_arr.tolist(),
                        "geometry_le": u_le_arr.tolist(),
                        "geometry_te": u_te_arr.tolist(),
                        "geometry_cp_absolute": (self.transfer.cp_targets + u_cp_arr).tolist(),
                        "geometry_le_absolute": (self.transfer.le_targets + u_le_arr).tolist(),
                        "geometry_te_absolute": (self.transfer.te_targets + u_te_arr).tolist(),
                        "rotation": zero_rot.tolist(),
                        "rotation_le": zero_rot.tolist(),
                        "rotation_te": zero_rot.tolist(),
                        "geometry_residual": geometry_residual,
                        "geometry_reference_norm": geometry_reference_norm,
                        "geometry_relative_error": geometry_relative_error,
                        "geometry_transfer_apply_time": geometry_transfer_apply_time,
                        "surface_strain_percent": self.surface_strain_percent[i_step],
                        "strain_increase_percentage_points": self.strain_increase_percentage_points[i_step],
                        "solid_solve_time": solid_solve_time,
                        "solid_io_time": solid_io_time,
                        "solid_step_time": solid_step_time,
                    }
                )
                self.sock.sendall((msg_geo + "\n").encode())
                print(f"Solid step {i_step + 1}/{self.Nsteps}: geometry sent.")
            else:
                solid_step_time = perf_counter() - step_start
                self.step_walltime[i_step] = solid_step_time

        self._finalize()

    def _finalize(self):
        self.geometry_consistency_fp.close()
        self.state_history_fp.close()
        self.sock_file.close()
        self.sock.close()
        print("Solid solver finished.")
        print(f"Solid field outputs: {self.xdmf_path}")
        print(f"Solid VTK outputs: {self.q_pvd_path}, {self.sig_pvd_path}, {self.mesh_pvd_path}")
        print(f"CP geometry consistency data: {self.geometry_consistency_path}")
        print(f"Structural state history: {self.state_history_path}")

        diag_csv = os.path.join(
            self.out_dir,
            cfg_get(self.solid_config, "diag_csv_filename", default="solid_v18_diagnostics.csv"),
        )
        with open(diag_csv, "w") as fp:
            fp.write(
                "step,time,u_tip,E_elas,E_kin,E_damp,E_tot,incremental_work_Wf,incremental_work_Ws,"
                "incremental_work_defect,incremental_work_relative_error,"
                "force_transfer_residual_norm,force_transfer_relative_error,"
                "moment_transfer_residual_norm,moment_transfer_relative_error,"
                "cp_geometry_rms_error,cp_geometry_max_error,cp_geometry_relative_rms_error,"
                "force_residual,force_reference_norm,force_relative_error,"
                "force_transfer_apply_time,"
                "geometry_residual,geometry_reference_norm,geometry_relative_error,geometry_transfer_apply_time,"
                "surface_strain_percent,strain_increase_percentage_points,"
                "solid_solve_time,solid_io_time,step_walltime,surface_strain_max,membrane_stress_max_N_per_m,"
                "newton_iterations,newton_residual_norm,newton_relative_residual,newton_load_stages\n"
            )
            for k_idx in range(self.Nsteps):
                fp.write(
                    f"{k_idx + 1},"
                    f"{self.time[k_idx + 1]:.12e},"
                    f"{self.u_tip[k_idx + 1]:.12e},"
                    f"{self.energies[k_idx + 1, 0]:.12e},"
                    f"{self.energies[k_idx + 1, 1]:.12e},"
                    f"{self.energies[k_idx + 1, 2]:.12e},"
                    f"{self.energies[k_idx + 1, 3]:.12e},"
                    f"{self.incremental_work_Wf[k_idx]:.12e},"
                    f"{self.incremental_work_Ws[k_idx]:.12e},"
                    f"{self.incremental_work_defects[k_idx]:.12e},"
                    f"{self.incremental_work_rel_errors[k_idx]:.12e},"
                    f"{self.force_transfer_residual_norms[k_idx]:.12e},"
                    f"{self.force_transfer_relative_errors[k_idx]:.12e},"
                    f"{self.moment_transfer_residual_norms[k_idx]:.12e},"
                    f"{self.moment_transfer_relative_errors[k_idx]:.12e},"
                    f"{self.cp_geometry_rms_errors[k_idx]:.12e},"
                    f"{self.cp_geometry_max_errors[k_idx]:.12e},"
                    f"{self.cp_geometry_relative_rms_errors[k_idx]:.12e},"
                    f"{self.force_residuals[k_idx]:.12e},"
                    f"{self.force_reference_norms[k_idx]:.12e},"
                    f"{self.force_relative_errors[k_idx]:.12e},"
                    f"{self.force_transfer_apply_times[k_idx]:.12e},"
                    f"{self.geometry_residuals[k_idx]:.12e},"
                    f"{self.geometry_reference_norms[k_idx]:.12e},"
                    f"{self.geometry_relative_errors[k_idx]:.12e},"
                    f"{self.geometry_transfer_apply_times[k_idx]:.12e},"
                    f"{self.surface_strain_percent[k_idx]:.12e},"
                    f"{self.strain_increase_percentage_points[k_idx]:.12e},"
                    f"{self.solid_solve_times[k_idx]:.12e},"
                    f"{self.solid_io_times[k_idx]:.12e},"
                    f"{self.step_walltime[k_idx]:.12e},"
                    f"{self.surface_strain_max[k_idx]:.12e},"
                    f"{self.membrane_stress_max[k_idx]:.12e},"
                    f"{self.newton_iterations[k_idx]},"
                    f"{self.newton_residual_norm[k_idx]:.12e},"
                    f"{self.newton_relative_residual[k_idx]:.12e},"
                    f"{self.newton_load_stages[k_idx]}\n"
                )


def main():
    parser=argparse.ArgumentParser(description="Solid solver for coupled FSI simulation")
    default_config_path=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "config")
    parser.add_argument("--config_path",type=str,default=default_config_path)
    parser.add_argument("--config-dir", dest="config_path", default=default_config_path)
    parser.add_argument("--mms-target", type=str, default=None)
    parser.add_argument("--mms-output", type=str, default=None)
    args = parser.parse_args()
    solver = StructuralSolver(args.config_path, args.mms_target, args.mms_output)
    solver.run()
if __name__ == "__main__":
    main()
