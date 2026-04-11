from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from scipy import special as sp  # type: ignore
except Exception:
    sp = None

try:
    import mpmath as mp  # type: ignore
except Exception:
    mp = None

from rcs_solver import C0, solve_monostatic_rcs_2d

EPS = 1e-12


def _bessel_backend_name() -> str:
    if sp is not None:
        return "scipy"
    if mp is not None:
        return "mpmath"
    raise RuntimeError("solver_physics_validation requires scipy or mpmath.")



def _jv(n: int, x: float) -> complex:
    if sp is not None:
        return complex(sp.jv(n, x))
    return complex(mp.besselj(n, x))



def _h2v(n: int, x: float) -> complex:
    if sp is not None:
        return complex(sp.hankel2(n, x))
    return complex(mp.hankel2(n, x))



def _jvp(n: int, x: float) -> complex:
    if sp is not None:
        return complex(sp.jvp(n, x, 1))
    return 0.5 * (_jv(n - 1, x) - _jv(n + 1, x))



def _h2vp(n: int, x: float) -> complex:
    if sp is not None:
        return complex(sp.h2vp(n, x, 1))
    return 0.5 * (_h2v(n - 1, x) - _h2v(n + 1, x))



def _pec_cylinder_backscatter_width(radius_m: float, freq_ghz: float, pol: str) -> float:
    """
    Analytic 2D monostatic scattering width per unit length of a PEC circular cylinder.

    Uses the standard cylindrical-wave series with the same e^{-j omega t} convention.
    The returned quantity is the physical 2D scattering width sigma_2d.

    Boundary conditions:
    - TE => E_z = 0  (Dirichlet)
    - TM => dH_z/dn = 0  (Neumann)
    """

    k = 2.0 * math.pi * (freq_ghz * 1e9) / C0
    ka = k * radius_m
    nmax = max(20, int(math.ceil(ka + 10.0 * (ka ** (1.0 / 3.0)) + 20.0)))

    series = 0.0 + 0.0j
    for n in range(-nmax, nmax + 1):
        phase = complex((-1) ** n)
        if pol == "TE":
            denom = _h2v(n, ka)
            if abs(denom) <= EPS:
                continue
            coeff = _jv(n, ka) / denom
        else:
            denom = _h2vp(n, ka)
            if abs(denom) <= EPS:
                continue
            coeff = _jvp(n, ka) / denom
        series += coeff * phase

    return float((4.0 / max(k, EPS)) * (abs(series) ** 2))



def _circle_snapshot(radius_m: float, segment_count: int, n_prop: int) -> Dict[str, Any]:
    pts: List[Tuple[float, float]] = []
    for i in range(segment_count):
        t = 2.0 * math.pi * float(i) / float(segment_count)
        # Clockwise ordering so the solver's left-hand normal points outward.
        pts.append((radius_m * math.cos(-t), radius_m * math.sin(-t)))

    point_pairs = []
    for i in range(segment_count):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % segment_count]
        point_pairs.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    return {
        "title": "PEC circle validation",
        "segments": [
            {
                "name": "pec_circle",
                "seg_type": "2",
                "properties": ["2", str(int(n_prop)), "0", "0", "0", "0"],
                "point_pairs": point_pairs,
            }
        ],
        "ibcs": [],
        "dielectrics": [],
    }



def _run_case(
    radius_m: float,
    freq_ghz: float,
    pol: str,
    segment_count: int,
    n_prop: int,
    elevations_deg: List[float],
) -> Dict[str, Any]:
    snapshot = _circle_snapshot(radius_m=radius_m, segment_count=segment_count, n_prop=n_prop)
    result = solve_monostatic_rcs_2d(
        geometry_snapshot=snapshot,
        frequencies_ghz=[float(freq_ghz)],
        elevations_deg=[float(v) for v in elevations_deg],
        polarization=pol,
        geometry_units="meters",
        strict_quality_gate=False,
        compute_condition_number=True,
        rcs_normalization_mode="divide_by_k",
    )

    analytic_sigma = _pec_cylinder_backscatter_width(radius_m=radius_m, freq_ghz=freq_ghz, pol=pol)
    analytic_db = 10.0 * math.log10(max(analytic_sigma, EPS))

    samples = list(result.get("samples", []) or [])
    if not samples:
        raise RuntimeError("Solver returned no samples during physics validation.")

    sample_rows: List[Dict[str, Any]] = []
    errors_db: List[float] = []
    for row in samples:
        solver_sigma = float(row.get("rcs_linear", EPS))
        solver_db = float(row.get("rcs_db", 10.0 * math.log10(max(solver_sigma, EPS))))
        err_db = solver_db - analytic_db
        errors_db.append(err_db)
        sample_rows.append(
            {
                "theta_deg": float(row.get("theta_inc_deg", 0.0)),
                "solver_rcs_db": solver_db,
                "analytic_rcs_db": analytic_db,
                "error_db": err_db,
            }
        )

    err = np.asarray(errors_db, dtype=float)
    return {
        "frequency_ghz": float(freq_ghz),
        "polarization": pol,
        "analytic_sigma": analytic_sigma,
        "analytic_rcs_db": analytic_db,
        "solver_mean_rcs_db": float(np.mean([r["solver_rcs_db"] for r in sample_rows])),
        "rms_error_db": float(np.sqrt(np.mean(err * err))),
        "max_abs_error_db": float(np.max(np.abs(err))),
        "samples": sample_rows,
        "solver_metadata": result.get("metadata", {}),
    }



def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run physics validation cases for the 2D RCS solver.")
    parser.add_argument("--radius-m", type=float, default=0.0127, help="PEC cylinder radius in meters.")
    parser.add_argument("--frequencies-ghz", type=str, default="1,3,6", help="Comma-separated test frequencies.")
    parser.add_argument("--polarizations", type=str, default="TE,TM", help="Comma-separated solver polarizations.")
    parser.add_argument("--segment-count", type=int, default=180, help="Number of straight segments in the circle geometry.")
    parser.add_argument("--n-prop", type=int, default=1, help="Per-primitive panel density multiplier.")
    parser.add_argument("--angles-deg", type=str, default="0,45,90,135", help="Comma-separated monostatic angles.")
    parser.add_argument("--rms-limit-db", type=float, default=1.0, help="Pass threshold for RMS dB error.")
    parser.add_argument("--max-limit-db", type=float, default=2.0, help="Pass threshold for max absolute dB error.")
    parser.add_argument("--json", action="store_true", help="Emit full JSON instead of a compact text report.")
    args = parser.parse_args(argv)

    _ = _bessel_backend_name()
    freqs = [float(x) for x in args.frequencies_ghz.split(",") if str(x).strip()]
    pols = [str(x).strip().upper() for x in args.polarizations.split(",") if str(x).strip()]
    angles = [float(x) for x in args.angles_deg.split(",") if str(x).strip()]

    cases = []
    for pol in pols:
        for freq in freqs:
            cases.append(
                _run_case(
                    radius_m=float(args.radius_m),
                    freq_ghz=float(freq),
                    pol=pol,
                    segment_count=int(args.segment_count),
                    n_prop=int(args.n_prop),
                    elevations_deg=angles,
                )
            )

    rms_limit = float(args.rms_limit_db)
    max_limit = float(args.max_limit_db)
    passed = all(
        (float(case["rms_error_db"]) <= rms_limit) and (float(case["max_abs_error_db"]) <= max_limit)
        for case in cases
    )

    report = {
        "passed": bool(passed),
        "backend": _bessel_backend_name(),
        "limits": {"rms_error_db": rms_limit, "max_abs_error_db": max_limit},
        "cases": cases,
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if passed else "FAIL"
        print(f"Physics validation: {status}")
        print(f"Bessel backend: {report['backend']}")
        print(f"Limits: RMS <= {rms_limit:g} dB, max |error| <= {max_limit:g} dB")
        for case in cases:
            print(
                f"  {case['polarization']} @ {case['frequency_ghz']:g} GHz: "
                f"analytic={case['analytic_rcs_db']:.3f} dB, "
                f"solver_mean={case['solver_mean_rcs_db']:.3f} dB, "
                f"rms={case['rms_error_db']:.3f} dB, "
                f"max={case['max_abs_error_db']:.3f} dB"
            )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
