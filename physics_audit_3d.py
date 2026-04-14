#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import spherical_jn, spherical_yn

C0 = 299_792_458.0


def db10(x: np.ndarray | float, floor: float = 1.0e-20) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    return 10.0 * np.log10(np.maximum(arr, floor))


# ---------------------------
# Exact 3D PEC sphere theory
# ---------------------------
def mie_backscatter_pec_sphere(freq_hz: float, radius_m: float, nmax: int | None = None) -> float:
    """Exact monostatic RCS of a PEC sphere using Mie-series backscatter."""

    freq_hz = float(freq_hz)
    radius_m = float(radius_m)
    if freq_hz <= 0.0 or radius_m <= 0.0:
        return 0.0

    wavelength = C0 / freq_hz
    x = 2.0 * math.pi * radius_m / wavelength
    if nmax is None:
        nmax = int(np.ceil(x + 4.0 * max(x, 1.0) ** (1.0 / 3.0) + 15.0))

    n = np.arange(1, nmax + 1, dtype=int)
    psi = x * spherical_jn(n, x)
    psi_p = spherical_jn(n, x) + x * spherical_jn(n, x, derivative=True)
    chi = -x * spherical_yn(n, x)
    chi_p = -(spherical_yn(n, x) + x * spherical_yn(n, x, derivative=True))
    xi = psi + 1j * chi
    xi_p = psi_p + 1j * chi_p

    # PEC-sphere Mie coefficients.
    a_n = -psi_p / xi_p
    b_n = -psi / xi

    backscatter_sum = np.sum((2 * n + 1) * np.power(-1.0, n) * (a_n - b_n))
    sigma = (wavelength ** 2 / (4.0 * math.pi)) * abs(backscatter_sum) ** 2
    return float(np.real_if_close(sigma))


def rayleigh_pec_sphere(freq_hz: float, radius_m: float) -> float:
    """Low-frequency PEC sphere asymptote: sigma = 9*pi*a^2*(ka)^4."""

    k = 2.0 * math.pi * float(freq_hz) / C0
    a = float(radius_m)
    return 9.0 * math.pi * (a ** 2) * ((k * a) ** 4)


def optical_limit_pec_sphere(radius_m: float) -> float:
    """High-frequency optical-limit asymptote for monostatic PEC sphere."""

    a = float(radius_m)
    return math.pi * (a ** 2)


# ---------------------------
# Actual-data loading
# ---------------------------
def _pick_column(name_map: Dict[str, str], candidates: Tuple[str, ...]) -> str | None:
    for cand in candidates:
        key = cand.lower().strip()
        if key in name_map:
            return name_map[key]
    return None


def load_actual_csv(path: str | Path) -> Tuple[np.ndarray, np.ndarray, str]:
    path = Path(path).expanduser().resolve()
    rows = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        name_map = {name.lower().strip(): name for name in reader.fieldnames}
        freq_col = _pick_column(name_map, ("frequency_ghz", "freq_ghz", "frequency_hz", "freq_hz", "frequency", "freq"))
        db_col = _pick_column(name_map, ("rcs_dbsm", "sigma_dbsm", "dbsm"))
        lin_col = _pick_column(name_map, ("rcs_linear", "sigma_m2", "sigma", "rcs_m2"))
        if freq_col is None:
            raise ValueError("Could not find a frequency column in the actual-data CSV.")
        if db_col is None and lin_col is None:
            raise ValueError("Could not find an actual RCS column. Expected rcs_dbsm/dbsm or rcs_linear/sigma_m2.")

        freq_is_hz = "hz" in freq_col.lower() and "ghz" not in freq_col.lower()
        for row in reader:
            if not row:
                continue
            fval = float(row[freq_col])
            freq_ghz = fval / 1.0e9 if freq_is_hz else fval
            if db_col is not None and row.get(db_col, "") not in {"", None}:
                sigma = 10.0 ** (float(row[db_col]) / 10.0)
                sigma_kind = "dbsm"
            else:
                sigma = float(row[lin_col])
                sigma_kind = "linear"
            rows.append((freq_ghz, sigma))

    rows.sort(key=lambda x: x[0])
    freqs = np.asarray([r[0] for r in rows], dtype=float)
    sigma = np.asarray([r[1] for r in rows], dtype=float)
    return freqs, sigma, sigma_kind


# ---------------------------
# Main benchmark
# ---------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="3D physics audit against exact PEC sphere Mie backscatter.")
    parser.add_argument("--outdir", default="./physics_audit_3d_out", help="Directory for plots and CSV/JSON outputs.")
    parser.add_argument("--radius-m", type=float, default=0.10, help="PEC sphere radius in meters.")
    parser.add_argument("--freq-min-ghz", type=float, default=0.05)
    parser.add_argument("--freq-max-ghz", type=float, default=18.0)
    parser.add_argument("--freq-count", type=int, default=300)
    parser.add_argument("--actual-csv", default="", help="Optional CSV of actual 3D solver results to compare against theory.")
    args = parser.parse_args()

    out_dir = Path(args.outdir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    freqs_ghz = np.linspace(float(args.freq_min_ghz), float(args.freq_max_ghz), int(args.freq_count))
    freqs_hz = freqs_ghz * 1.0e9
    radius_m = float(args.radius_m)

    sigma_exact = np.asarray([mie_backscatter_pec_sphere(f, radius_m) for f in freqs_hz], dtype=float)
    sigma_rayleigh = np.asarray([rayleigh_pec_sphere(f, radius_m) for f in freqs_hz], dtype=float)
    sigma_optical = np.full_like(freqs_ghz, optical_limit_pec_sphere(radius_m), dtype=float)

    csv_path = out_dir / "3d_pec_sphere_reference.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frequency_ghz",
            "exact_sigma_m2",
            "exact_dbsm",
            "rayleigh_sigma_m2",
            "rayleigh_dbsm",
            "optical_sigma_m2",
            "optical_dbsm",
        ])
        for vals in zip(freqs_ghz, sigma_exact, db10(sigma_exact), sigma_rayleigh, db10(sigma_rayleigh), sigma_optical, db10(sigma_optical)):
            writer.writerow([float(v) for v in vals])

    summary: Dict[str, Any] = {
        "radius_m": radius_m,
        "reference_csv": str(csv_path),
        "actual_comparison": None,
    }

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.plot(freqs_ghz, db10(sigma_exact), label="Exact Mie")
    ax.plot(freqs_ghz, db10(sigma_rayleigh), label="Rayleigh asymptote")
    ax.plot(freqs_ghz, db10(sigma_optical), label="Optical limit")

    if str(args.actual_csv).strip():
        actual_freqs_ghz, actual_sigma, actual_kind = load_actual_csv(args.actual_csv)
        exact_on_actual = np.asarray([mie_backscatter_pec_sphere(f * 1.0e9, radius_m) for f in actual_freqs_ghz], dtype=float)
        abs_err = np.abs(actual_sigma - exact_on_actual)
        rel_err = abs_err / np.maximum(np.abs(exact_on_actual), 1.0e-20)
        summary["actual_comparison"] = {
            "actual_csv": str(Path(args.actual_csv).expanduser().resolve()),
            "actual_sigma_kind": actual_kind,
            "max_abs_error": float(np.max(abs_err)),
            "mean_abs_error": float(np.mean(abs_err)),
            "max_rel_error": float(np.max(rel_err)),
            "mean_rel_error": float(np.mean(rel_err)),
        }

        cmp_csv = out_dir / "3d_actual_vs_exact_comparison.csv"
        with cmp_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frequency_ghz",
                "actual_sigma_m2",
                "actual_dbsm",
                "exact_sigma_m2",
                "exact_dbsm",
                "abs_error",
                "rel_error",
            ])
            for vals in zip(actual_freqs_ghz, actual_sigma, db10(actual_sigma), exact_on_actual, db10(exact_on_actual), abs_err, rel_err):
                writer.writerow([float(v) for v in vals])
        summary["actual_comparison"]["comparison_csv"] = str(cmp_csv)

        ax.plot(actual_freqs_ghz, db10(actual_sigma), label="Actual")

        fig_err, ax_err = plt.subplots(figsize=(9.2, 4.8))
        ax_err.plot(actual_freqs_ghz, rel_err * 100.0)
        ax_err.set_xlabel("Frequency (GHz)")
        ax_err.set_ylabel("Relative error (%)")
        ax_err.set_title("3D PEC sphere exact-vs-actual relative error")
        ax_err.grid(True, alpha=0.3)
        fig_err.tight_layout()
        fig_err.savefig(out_dir / "3d_actual_vs_exact_error.png", dpi=180)
        plt.close(fig_err)

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("RCS (dBsm)")
    ax.set_title("3D PEC sphere monostatic benchmark")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "3d_pec_sphere_benchmark.png", dpi=180)
    plt.close(fig)

    summary_path = out_dir / "3d_physics_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote 3D audit outputs to: {out_dir}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
