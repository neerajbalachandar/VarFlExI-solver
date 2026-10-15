import os
import sys
from time import perf_counter

print("cwd =", os.getcwd())
print("sys.path =")
for p in sys.path:
    print(" ", p)

from dolfin import *
import argparse
import json
import os
import socket

import numpy as np
import yaml

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
    def __init__(self, config_path):
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.config_path = config_path
        self.solid_config = load_yaml_config(config_path, "solid_params.yaml")
        self.fluid_config = load_yaml_config(config_path, "fluid_params.yaml")
        self.coupling_config = load_yaml_config(config_path, "coupling_params.yaml")

        self._setup_simulation_data()
        self._setup_domain()
        self._setup_model()
        self._setup_transfer()
        self._setup_output()
        self._setup_socket()
        self._setup_history()

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
                self.fluid_config,
                "mesh",
                "n_span",
                default=cfg_get(self.fluid_config, "n_span_comm", default=80),
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
        self.force_relax = cfg_get(self.coupling_config, "force_relax", default=1.0)

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
        print(f"Solid force transfer mode: {self.force_transfer_mode}")
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
        self.energies = np.zeros((self.Nsteps + 1, 4), dtype=float)
        self.step_walltime = np.full((self.Nsteps,), np.nan, dtype=float)
        self.E_damp_acc = 0.0
        self.forces_prev = None
        self.work_rel_errors = np.full((self.Nsteps,), np.nan, dtype=float)
        self.work_Wf = np.full((self.Nsteps,), np.nan, dtype=float)
        self.work_Ws = np.full((self.Nsteps,), np.nan, dtype=float)
        self.geometry_work_rel_errors = np.full((self.Nsteps,), np.nan, dtype=float)
        self.geometry_work_Wf = np.full((self.Nsteps,), np.nan, dtype=float)
        self.geometry_work_Ws = np.full((self.Nsteps,), np.nan, dtype=float)
        self.force_residuals = np.full((self.Nsteps,), np.nan, dtype=float)
        self.force_reference_norms = np.full((self.Nsteps,), np.nan, dtype=float)
        self.force_relative_errors = np.full((self.Nsteps,), np.nan, dtype=float)
        self.force_transfer_residuals = np.full((self.Nsteps,), np.nan, dtype=float)
        self.force_transfer_relative_errors = np.full((self.Nsteps,), np.nan, dtype=float)
        self.force_transfer_apply_times = np.full((self.Nsteps,), np.nan, dtype=float)
        self.displacement_transfer_residuals = np.full((self.Nsteps,), np.nan, dtype=float)
        self.displacement_transfer_relative_errors = np.full((self.Nsteps,), np.nan, dtype=float)
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
        }
        self.sock.sendall((json.dumps(init_msg) + "\n").encode())
        print("Initial zero geometry sent.")

    def _solve_newton_step(self, ext_force_vec):
        converged = False
        r0 = None
        r_rel = np.inf
        r_norm = np.inf
        for newton_it in range(self.newton_maxit):
            A = assemble(self.model.jacobian)
            R = assemble(self.model.residual)
            b = R.copy()
            b *= -1.0
            if self.force_ramp_iters > 0:
                ramp = min(1.0, float(newton_it + 1) / float(self.force_ramp_iters))
            else:
                ramp = 1.0
            b.axpy(ramp, ext_force_vec)
            for bc in self.model.bcs:
                bc.apply(A, b)
            r_norm = b.norm("l2")
            if r0 is None:
                r0 = max(r_norm, 1.0e-16)
            r_rel = r_norm / r0
            if r_norm <= self.newton_atol or r_rel <= self.newton_rtol:
                converged = True
                break
            self.model.linear_solver.solve(A, self.model.dq_newton.vector(), b)
            self.model.q.vector().axpy(1.0, self.model.dq_newton.vector())
            self.model.q.vector().apply("insert")
        if not converged:
            raise RuntimeError(
                f"Newton failed: residual={r_norm:.6e}, relative={r_rel:.6e}, "
                f"atol={self.newton_atol:.2e}, rtol={self.newton_rtol:.2e}, "
                f"maxit={self.newton_maxit}, ramp_iters={self.force_ramp_iters}"
            )

    def run(self):
        self._send_initial_geometry()

        for i_step in range(self.Nsteps):
            step_start = perf_counter()
            solid_step_time = float("nan")
            print(f"Solid step {i_step + 1}/{self.Nsteps}: waiting for force...")
            line = self.sock_file.readline()
            if line == "":
                raise RuntimeError("Coupling server disconnected while sending force data")

            data = json.loads(line)
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
            Fs_coeff = None
            force_transfer_apply_start = perf_counter()
            if self.force_transfer_mode == "crm":
                _, nodal_forces = self.transfer.apply_force_transfer(forces_eff)
            else:
                Fs_coeff, nodal_forces = self.transfer.apply_force_transfer(forces_eff)
            force_transfer_apply_time = perf_counter() - force_transfer_apply_start
            self.force_transfer_apply_times[i_step] = force_transfer_apply_time
            if not np.isfinite(nodal_forces).all():
                raise RuntimeError(f"Non-finite mapped nodal forces at solid step {i_step + 1}")

            (
                force_transfer_residual,
                force_transfer_relative_error,
                _structural_force_common,
            ) = self.transfer.force_transfer_metrics(
                forces_eff,
                nodal_forces,
                Fs_coeff=Fs_coeff,
                epsilon_reg=self.epsilon_reg,
            )
            self.force_transfer_residuals[i_step] = force_transfer_residual
            self.force_transfer_relative_errors[i_step] = force_transfer_relative_error
            interface_disp_prev = self.transfer.get_nodal_displacements_plate(
                self.model.q_old,
                self.interface_node_ids,
                self.dofs_u_x,
                self.dofs_u_y,
                self.dofs_w,
            )
            (
                displacement_transfer_residual,
                displacement_transfer_relative_error,
                _solid_displacement_round_trip,
            ) = self.transfer.displacement_transfer_metrics(
                interface_disp_prev, epsilon_reg=self.epsilon_reg
            )
            self.displacement_transfer_residuals[i_step] = displacement_transfer_residual
            self.displacement_transfer_relative_errors[i_step] = displacement_transfer_relative_error
            geometry_work_Wf = float("nan")
            geometry_work_Ws = float("nan")
            geometry_work_rel_error = float("nan")
            if self.work_conservative_mode and nodal_forces is not None:
                u_le_prev, u_te_prev, _u_common_prev, _u_geom_prev = self.transfer.apply_geometry_transfer(
                    self.model.q_old,
                    fallback_tree=self.interface_tree,
                    fallback_vals=interface_disp_prev,
                )
                u_cp_geom_prev = (1.0 - self.eta_cp) * u_le_prev + self.eta_cp * u_te_prev
                geometry_work_Wf = float(np.sum(u_cp_geom_prev * (forces_eff * self.A_diag[:, None])))
                geometry_work_Ws = float(np.sum(interface_disp_prev * nodal_forces))
                geometry_work_rel_error = abs(geometry_work_Wf - geometry_work_Ws) / max(
                    abs(geometry_work_Wf), abs(geometry_work_Ws), 1.0e-16
                )
                self.geometry_work_rel_errors[i_step] = geometry_work_rel_error
                self.geometry_work_Wf[i_step] = geometry_work_Wf
                self.geometry_work_Ws[i_step] = geometry_work_Ws
            force_transfer_msg = {
                "type": "force_transfer_diagnostics",
                "step": i_step + 1,
                "force_transfer_residual": force_transfer_residual,
                "force_transfer_relative_error": force_transfer_relative_error,
                "force_transfer_apply_time": force_transfer_apply_time,
                "fluid_force_round_trip_residual": force_transfer_residual,
                "fluid_force_round_trip_relative_error": force_transfer_relative_error,
                "solid_displacement_round_trip_residual": displacement_transfer_residual,
                "solid_displacement_round_trip_relative_error": displacement_transfer_relative_error,
                "crm_setup_time": self.transfer.crm_setup_time,
                "rbf_setup_time": self.transfer.rbf_setup_time,
                "geometry_rbf_setup_time": self.transfer.geometry_rbf_setup_time,
                "geometry_work_Wf": geometry_work_Wf,
                "geometry_work_Ws": geometry_work_Ws,
                "geometry_work_rel_error": geometry_work_rel_error,
            }
            self.sock.sendall((json.dumps(force_transfer_msg) + "\n").encode())

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
            solid_solve_time = perf_counter() - solid_solve_start
            self.solid_solve_times[i_step] = solid_solve_time

            if self.work_conservative_mode and nodal_forces is not None:
                interface_disp_prev = self.transfer.get_nodal_displacements_plate(
                    self.model.q_old,
                    self.interface_node_ids,
                    self.dofs_u_x,
                    self.dofs_u_y,
                    self.dofs_w,
                )
                if self.force_transfer_mode == "crm":
                    u_panel_prev = self.transfer.nodal_displacements_to_panel_average(
                        interface_disp_prev,
                        self.crm_transfer_matrix,
                        self.crm_panel_areas,
                    )
                    Wf = float(np.sum(u_panel_prev * (forces_eff * self.A_diag[:, None])))
                    Ws = float(np.sum(interface_disp_prev * nodal_forces))
                else:
                    u_cp_prev = self.transfer.sample_vector_field_at_targets(
                        self.model.q_old,
                        self.cp_targets,
                        fallback_tree=self.interface_tree,
                        fallback_vals=interface_disp_prev,
                    )
                    Wf = float(np.sum(u_cp_prev * (forces_eff * self.A_diag[:, None])))
                    Ws = float(np.sum(interface_disp_prev * (Fs_coeff * self.S_lumped[:, None])))
                rel_work_err = abs(Wf - Ws) / max(abs(Wf), abs(Ws), 1.0e-16)
                self.work_rel_errors[i_step] = rel_work_err
                self.work_Wf[i_step] = Wf
                self.work_Ws[i_step] = Ws

                u_le_prev, u_te_prev, _u_common_prev, _u_geom_prev = self.transfer.apply_geometry_transfer(
                    self.model.q_old,
                    fallback_tree=self.interface_tree,
                    fallback_vals=interface_disp_prev,
                )
                u_cp_geom_prev = (1.0 - self.eta_cp) * u_le_prev + self.eta_cp * u_te_prev
                Wf_geom = float(np.sum(u_cp_geom_prev * (forces_eff * self.A_diag[:, None])))
                Ws_geom = float(np.sum(interface_disp_prev * nodal_forces))
                rel_work_err_geom = abs(Wf_geom - Ws_geom) / max(
                    abs(Wf_geom), abs(Ws_geom), 1.0e-16
                )
                self.geometry_work_rel_errors[i_step] = rel_work_err_geom
                self.geometry_work_Wf[i_step] = Wf_geom
                self.geometry_work_Ws[i_step] = Ws_geom
                if i_step == 0 or (i_step + 1) % 20 == 0:
                    print(
                        f"Work audit step {i_step + 1}: "
                        f"Wf={Wf:.6e}, Ws={Ws:.6e}, rel_err={rel_work_err:.3e}, "
                        f"geom_rel_err={rel_work_err_geom:.3e}"
                    )

            self.model.update_fields()
            t = self.time[i_step + 1]
            solid_io_start = perf_counter()
            self.xdmf_file.write(self.model.q, t)
            solid_io_time = perf_counter() - solid_io_start
            self.solid_io_times[i_step] = solid_io_time

            E_elas = self.model.compute_strain_energy()
            E_kin = self.model.compute_kinetic_energy()
            self.E_damp_acc += self.model.compute_damping_increment(self.dt_value)
            E_tot = E_elas + E_kin + self.E_damp_acc
            self.energies[i_step + 1, :] = np.array([E_elas, E_kin, self.E_damp_acc, E_tot])

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
                        "solid_solve_time": solid_solve_time,
                        "solid_io_time": solid_io_time,
                        "solid_step_time": solid_step_time,
                    }
                )
                self.sock.sendall((msg_geo + "\n").encode())
                print(f"Solid step {i_step + 1}/{self.Nsteps}: geometry sent.")

        self._finalize()

    def _finalize(self):
        self.sock_file.close()
        self.sock.close()
        print("Solid solver finished.")
        print(f"Solid field outputs: {self.xdmf_path}")
        print(f"Solid VTK outputs: {self.q_pvd_path}, {self.sig_pvd_path}, {self.mesh_pvd_path}")

        diag_csv = os.path.join(
            self.out_dir,
            cfg_get(self.solid_config, "diag_csv_filename", default="solid_v18_diagnostics.csv"),
        )
        with open(diag_csv, "w") as fp:
            fp.write(
                "step,time,u_tip,E_elas,E_kin,E_damp,E_tot,work_Wf,work_Ws,work_rel_error,"
                "geometry_work_Wf,geometry_work_Ws,geometry_work_rel_error,"
                "force_residual,force_reference_norm,force_relative_error,"
                "force_transfer_residual,force_transfer_relative_error,force_transfer_apply_time,"
                "fluid_force_round_trip_residual,fluid_force_round_trip_relative_error,"
                "solid_displacement_round_trip_residual,solid_displacement_round_trip_relative_error,"
                "geometry_residual,geometry_reference_norm,geometry_relative_error,geometry_transfer_apply_time,"
                "solid_solve_time,solid_io_time,step_walltime\n"
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
                    f"{self.work_Wf[k_idx]:.12e},"
                    f"{self.work_Ws[k_idx]:.12e},"
                    f"{self.work_rel_errors[k_idx]:.12e},"
                    f"{self.geometry_work_Wf[k_idx]:.12e},"
                    f"{self.geometry_work_Ws[k_idx]:.12e},"
                    f"{self.geometry_work_rel_errors[k_idx]:.12e},"
                    f"{self.force_residuals[k_idx]:.12e},"
                    f"{self.force_reference_norms[k_idx]:.12e},"
                    f"{self.force_relative_errors[k_idx]:.12e},"
                    f"{self.force_transfer_residuals[k_idx]:.12e},"
                    f"{self.force_transfer_relative_errors[k_idx]:.12e},"
                    f"{self.force_transfer_apply_times[k_idx]:.12e},"
                    f"{self.force_transfer_residuals[k_idx]:.12e},"
                    f"{self.force_transfer_relative_errors[k_idx]:.12e},"
                    f"{self.displacement_transfer_residuals[k_idx]:.12e},"
                    f"{self.displacement_transfer_relative_errors[k_idx]:.12e},"
                    f"{self.geometry_residuals[k_idx]:.12e},"
                    f"{self.geometry_reference_norms[k_idx]:.12e},"
                    f"{self.geometry_relative_errors[k_idx]:.12e},"
                    f"{self.geometry_transfer_apply_times[k_idx]:.12e},"
                    f"{self.solid_solve_times[k_idx]:.12e},"
                    f"{self.solid_io_times[k_idx]:.12e},"
                    f"{self.step_walltime[k_idx]:.12e}\n"
                )


def main():
    parser=argparse.ArgumentParser(description="Solid solver for coupled FSI simulation")
    default_config_path=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "config")
    parser.add_argument("--config_path",type=str,default=default_config_path)
    args = parser.parse_args()
    solver = StructuralSolver(args.config_path)
    solver.run()
if __name__ == "__main__":
    main()
