#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.machinery
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import special

C0 = 299_792_458.0


def load_python_source(path: str | Path, module_name: str = "user_rcs_solver") -> ModuleType:
    src = str(Path(path).expanduser().resolve())
    loader = importlib.machinery.SourceFileLoader(module_name, src)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise ImportError(f"Could not create import spec for {src}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    loader.exec_module(module)
    return module


def db10(x: np.ndarray | float, floor: float = 1.0e-20) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    return 10.0 * np.log10(np.maximum(arr, floor))


# ---------------------------
# Exact 2D PEC circle theory
# ---------------------------
def exact_sigma_2d_pec_circle(freq_hz: float, radius_m: float, polarization: str, nmax: int | None = None) -> float:
    """
    Exact monostatic 2D scattering width of a PEC circular cylinder.

    Time convention matches the uploaded solver: exp(-j omega t).

    Polarization mapping used here is the standard 2D scalar BIE convention:
    - TM -> Dirichlet (Ez formulation on PEC)
    - TE -> Neumann   (Hz formulation on PEC)
    """

    pol = str(polarization).strip().upper()
    if pol not in {"TM", "TE"}:
        raise ValueError("polarization must be TE or TM")

    k = 2.0 * math.pi * float(freq_hz) / C0
    x = k * float(radius_m)
    if x <= 0.0:
        return 0.0

    if nmax is None:
        nmax = int(np.ceil(x + 10.0 * max(x, 1.0) ** (1.0 / 3.0) + 30.0))
    n = np.arange(-nmax, nmax + 1, dtype=int)

    if pol == "TM":
        jn = special.jv(n, x)
        hn = special.hankel2(n, x)
        coeff = jn / hn
    else:
        jn_p = 0.5 * (special.jv(n - 1, x) - special.jv(n + 1, x))
        hn_p = 0.5 * (special.hankel2(n - 1, x) - special.hankel2(n + 1, x))
        coeff = jn_p / hn_p

    # Backscatter from a circle is orientation-independent and corresponds to
    # scattering angle pi relative to incidence, giving the (-1)^n factor.
    series_sum = np.sum(np.power(-1.0, n) * coeff)
    amp = -4.0 * series_sum
    sigma_2d = (abs(amp) ** 2) / (4.0 * k)
    return float(np.real_if_close(sigma_2d))


# ---------------------------
# Geometry builders
# ---------------------------
def build_pec_circle_geometry(radius_m: float, n_edges: int) -> Dict[str, Any]:
    pts = [
        (radius_m * math.cos(2.0 * math.pi * i / n_edges), radius_m * math.sin(2.0 * math.pi * i / n_edges))
        for i in range(n_edges)
    ]
    point_pairs: List[Dict[str, float]] = []
    for i in range(n_edges):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n_edges]
        point_pairs.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    return {
        "title": "PEC circle benchmark",
        "segments": [
            {
                "name": "pec_circle",
                "properties": ["2", "1", "0", "0", "0", "0"],
                "point_pairs": point_pairs,
            }
        ],
        "ibcs": [],
        "dielectrics": [],
    }


def build_open_multimaterial_junction_geometry(length_m: float, panels_per_segment: int = 18) -> Dict[str, Any]:
    """
    Open junction audit geometry.

    Three open interfaces meet at one node:
    - air | eps1
    - eps1 | eps2
    - eps2 | eps3

    This is intentionally *not* a closed contour. It exercises the solver's
    topology handling, interface-aware node splitting, and junction constraints.
    """

    c60 = 0.5
    s60 = math.sqrt(3.0) / 2.0
    return {
        "title": "Open multi-material junction audit",
        "segments": [
            {
                "name": "if_air_1",
                "properties": ["3", str(panels_per_segment), "0", "0", "1", "0"],
                "point_pairs": [{"x1": -length_m, "y1": 0.0, "x2": 0.0, "y2": 0.0}],
            },
            {
                "name": "if_1_2",
                "properties": ["5", str(panels_per_segment), "0", "0", "1", "2"],
                "point_pairs": [{"x1": 0.0, "y1": 0.0, "x2": length_m * c60, "y2": length_m * s60}],
            },
            {
                "name": "if_2_3",
                "properties": ["5", str(panels_per_segment), "0", "0", "2", "3"],
                "point_pairs": [{"x1": 0.0, "y1": 0.0, "x2": length_m * c60, "y2": -length_m * s60}],
            },
        ],
        "ibcs": [],
        "dielectrics": [
            ["1", "2.5", "0.02", "1.0", "0.0"],
            ["2", "4.0", "0.05", "1.0", "0.0"],
            ["3", "6.0", "0.10", "1.0", "0.0"],
        ],
    }


# ---------------------------
# Solver execution helpers
# ---------------------------
def run_solver(
    solver_mod: ModuleType,
    geometry_snapshot: Dict[str, Any],
    frequencies_ghz: Iterable[float],
    elevations_deg: Iterable[float],
    polarization: str,
    basis_family: str = "linear",
    testing_family: str = "galerkin",
    max_panels: int = 20000,
) -> Dict[str, Any]:
    return solver_mod.solve_monostatic_rcs_2d(
        geometry_snapshot=geometry_snapshot,
        frequencies_ghz=[float(v) for v in frequencies_ghz],
        elevations_deg=[float(v) for v in elevations_deg],
        polarization=str(polarization).upper(),
        geometry_units="meters",
        parallel_elevations=False,
        reuse_angle_invariant_matrix=True,
        basis_family=basis_family,
        testing_family=testing_family,
        max_panels=int(max_panels),
    )


def samples_to_matrix(samples: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs = np.asarray(sorted({float(row["frequency_ghz"]) for row in samples}), dtype=float)
    angles = np.asarray(sorted({float(row["theta_inc_deg"]) for row in samples}), dtype=float)
    mat = np.full((len(freqs), len(angles)), np.nan, dtype=float)
    fidx = {v: i for i, v in enumerate(freqs)}
    aidx = {v: i for i, v in enumerate(angles)}
    for row in samples:
        mat[fidx[float(row["frequency_ghz"])], aidx[float(row["theta_inc_deg"])]] = float(row["rcs_linear"])
    return freqs, angles, mat


# ---------------------------
# Benchmarks
# ---------------------------
def frequency_sweep_benchmark(
    solver_mod: ModuleType,
    out_dir: Path,
    radius_m: float,
    n_edges: int,
    freqs_ghz: np.ndarray,
    angle_deg: float,
    polarization: str,
) -> Dict[str, Any]:
    geom = build_pec_circle_geometry(radius_m=radius_m, n_edges=n_edges)
    result = run_solver(
        solver_mod,
        geom,
        frequencies_ghz=freqs_ghz,
        elevations_deg=[angle_deg],
        polarization=polarization,
        basis_family="linear",
        testing_family="galerkin",
    )

    actual = np.asarray([float(row["rcs_linear"]) for row in result["samples"]], dtype=float)
    analytic = np.asarray(
        [exact_sigma_2d_pec_circle(freq_hz=f * 1.0e9, radius_m=radius_m, polarization=polarization) for f in freqs_ghz],
        dtype=float,
    )
    abs_err = np.abs(actual - analytic)
    rel_err = abs_err / np.maximum(np.abs(analytic), 1.0e-20)

    csv_path = out_dir / f"2d_circle_frequency_{polarization.lower()}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frequency_ghz",
            "actual_sigma_2d_m",
            "analytic_sigma_2d_m",
            "actual_db_sigma",
            "analytic_db_sigma",
            "abs_error",
            "rel_error",
        ])
        for vals in zip(freqs_ghz, actual, analytic, db10(actual), db10(analytic), abs_err, rel_err):
            writer.writerow([float(v) for v in vals])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(freqs_ghz, db10(analytic), label="Analytical")
    ax.plot(freqs_ghz, db10(actual), label="Solver")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("10log10(sigma_2d)")
    ax.set_title(f"2D PEC circle frequency benchmark ({polarization})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"2d_circle_frequency_{polarization.lower()}.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(freqs_ghz, rel_err * 100.0)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Relative error (%)")
    ax.set_title(f"2D PEC circle frequency error ({polarization})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"2d_circle_frequency_error_{polarization.lower()}.png", dpi=180)
    plt.close(fig)

    return {
        "polarization": polarization,
        "radius_m": float(radius_m),
        "n_edges": int(n_edges),
        "angle_deg": float(angle_deg),
        "max_abs_error": float(np.max(abs_err)),
        "mean_abs_error": float(np.mean(abs_err)),
        "max_rel_error": float(np.max(rel_err)),
        "mean_rel_error": float(np.mean(rel_err)),
        "solver_metadata": dict(result.get("metadata", {})),
        "csv": str(csv_path),
    }


def angle_sweep_benchmark(
    solver_mod: ModuleType,
    out_dir: Path,
    radius_m: float,
    n_edges: int,
    freq_ghz: float,
    angles_deg: np.ndarray,
    polarization: str,
) -> Dict[str, Any]:
    geom = build_pec_circle_geometry(radius_m=radius_m, n_edges=n_edges)
    result = run_solver(
        solver_mod,
        geom,
        frequencies_ghz=[freq_ghz],
        elevations_deg=angles_deg,
        polarization=polarization,
        basis_family="linear",
        testing_family="galerkin",
    )

    actual = np.asarray([float(row["rcs_linear"]) for row in result["samples"]], dtype=float)
    analytic_scalar = exact_sigma_2d_pec_circle(freq_hz=freq_ghz * 1.0e9, radius_m=radius_m, polarization=polarization)
    analytic = np.full_like(actual, analytic_scalar)
    abs_err = np.abs(actual - analytic)
    rel_err = abs_err / np.maximum(np.abs(analytic), 1.0e-20)

    csv_path = out_dir / f"2d_circle_angle_{polarization.lower()}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "theta_inc_deg",
            "actual_sigma_2d_m",
            "analytic_sigma_2d_m",
            "actual_db_sigma",
            "analytic_db_sigma",
            "abs_error",
            "rel_error",
        ])
        for vals in zip(angles_deg, actual, analytic, db10(actual), db10(analytic), abs_err, rel_err):
            writer.writerow([float(v) for v in vals])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(angles_deg, db10(analytic), label="Analytical")
    ax.plot(angles_deg, db10(actual), label="Solver")
    ax.set_xlabel("Incident angle (deg)")
    ax.set_ylabel("10log10(sigma_2d)")
    ax.set_title(f"2D PEC circle angle benchmark @ {freq_ghz:g} GHz ({polarization})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"2d_circle_angle_{polarization.lower()}.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(angles_deg, rel_err * 100.0)
    ax.set_xlabel("Incident angle (deg)")
    ax.set_ylabel("Relative error (%)")
    ax.set_title(f"2D PEC circle angle error @ {freq_ghz:g} GHz ({polarization})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"2d_circle_angle_error_{polarization.lower()}.png", dpi=180)
    plt.close(fig)

    return {
        "polarization": polarization,
        "radius_m": float(radius_m),
        "n_edges": int(n_edges),
        "frequency_ghz": float(freq_ghz),
        "max_abs_error": float(np.max(abs_err)),
        "mean_abs_error": float(np.mean(abs_err)),
        "max_rel_error": float(np.max(rel_err)),
        "mean_rel_error": float(np.mean(rel_err)),
        "solver_metadata": dict(result.get("metadata", {})),
        "csv": str(csv_path),
    }


def junction_topology_audit(
    solver_mod: ModuleType,
    out_dir: Path,
    length_m: float,
    freq_ghz: float,
    polarization: str,
) -> Dict[str, Any]:
    geom = build_open_multimaterial_junction_geometry(length_m=length_m)
    preflight = solver_mod.validate_geometry_snapshot_for_solver(geom, base_dir=str(Path.cwd()))

    result_linear = run_solver(
        solver_mod,
        geom,
        frequencies_ghz=[freq_ghz],
        elevations_deg=[0.0],
        polarization=polarization,
        basis_family="linear",
        testing_family="galerkin",
    )
    result_pulse = run_solver(
        solver_mod,
        geom,
        frequencies_ghz=[freq_ghz],
        elevations_deg=[0.0],
        polarization=polarization,
        basis_family="pulse",
        testing_family="collocation",
    )

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    color_map = {
        "if_air_1": "tab:blue",
        "if_1_2": "tab:orange",
        "if_2_3": "tab:green",
    }
    for seg in geom["segments"]:
        pair = seg["point_pairs"][0]
        x = [float(pair["x1"]), float(pair["x2"])]
        y = [float(pair["y1"]), float(pair["y2"])]
        ax.plot(x, y, linewidth=3, label=seg["name"], color=color_map.get(seg["name"], None))
    ax.scatter([0.0], [0.0], marker="o")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Open multi-material junction audit geometry")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "2d_open_junction_geometry.png", dpi=180)
    plt.close(fig)

    audit = {
        "purpose": "Open-segment and multi-material-junction topology audit",
        "preflight": dict(preflight),
        "linear_galerkin": {
            "rcs_linear": float(result_linear["samples"][0]["rcs_linear"]),
            "rcs_db": float(result_linear["samples"][0]["rcs_db"]),
            "metadata": dict(result_linear.get("metadata", {})),
        },
        "pulse_collocation": {
            "rcs_linear": float(result_pulse["samples"][0]["rcs_linear"]),
            "rcs_db": float(result_pulse["samples"][0]["rcs_db"]),
            "metadata": dict(result_pulse.get("metadata", {})),
        },
    }
    audit_path = out_dir / "2d_open_junction_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))
    return audit


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="2D physics audit against exact PEC circle theory and an open-junction topology audit.")
    parser.add_argument("--solver", default="/mnt/data/rcs_solver.txt", help="Path to the user's solver source file.")
    parser.add_argument("--outdir", default="./physics_audit_2d_out", help="Directory for plots and CSV/JSON outputs.")
    parser.add_argument("--radius-m", type=float, default=0.10, help="PEC circle radius in meters.")
    parser.add_argument("--n-edges", type=int, default=128, help="Polygon edges for the PEC circle geometry.")
    parser.add_argument("--freq-min-ghz", type=float, default=0.25)
    parser.add_argument("--freq-max-ghz", type=float, default=6.0)
    parser.add_argument("--freq-count", type=int, default=25)
    parser.add_argument("--angle-freq-ghz", type=float, default=1.0)
    parser.add_argument("--angle-step-deg", type=float, default=10.0)
    parser.add_argument("--angle-for-frequency-sweep", type=float, default=0.0)
    parser.add_argument("--junction-length-m", type=float, default=0.20)
    parser.add_argument("--junction-freq-ghz", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(args.outdir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    solver_mod = load_python_source(args.solver)
    freqs_ghz = np.linspace(float(args.freq_min_ghz), float(args.freq_max_ghz), int(args.freq_count))
    angles_deg = np.arange(-180.0, 180.0 + 0.5 * float(args.angle_step_deg), float(args.angle_step_deg))

    summary: Dict[str, Any] = {
        "solver": str(Path(args.solver).expanduser().resolve()),
        "circle_frequency_benchmarks": {},
        "circle_angle_benchmarks": {},
    }

    for pol in ("TM", "TE"):
        summary["circle_frequency_benchmarks"][pol] = frequency_sweep_benchmark(
            solver_mod=solver_mod,
            out_dir=out_dir,
            radius_m=float(args.radius_m),
            n_edges=int(args.n_edges),
            freqs_ghz=freqs_ghz,
            angle_deg=float(args.angle_for_frequency_sweep),
            polarization=pol,
        )
        summary["circle_angle_benchmarks"][pol] = angle_sweep_benchmark(
            solver_mod=solver_mod,
            out_dir=out_dir,
            radius_m=float(args.radius_m),
            n_edges=int(args.n_edges),
            freq_ghz=float(args.angle_freq_ghz),
            angles_deg=angles_deg,
            polarization=pol,
        )

    summary["open_junction_audit"] = junction_topology_audit(
        solver_mod=solver_mod,
        out_dir=out_dir,
        length_m=float(args.junction_length_m),
        freq_ghz=float(args.junction_freq_ghz),
        polarization="TM",
    )

    summary_path = out_dir / "2d_physics_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote 2D audit outputs to: {out_dir}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
