"""Convert audited raw runs into compact, numerical position-velocity maps.

Reads the raw files only. No simulation or PCA is run by this program.
The fixed histogram bounds are declared before looking at forecast scores.
"""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from twostream.io import C, SPECIES, frame_from_bytes, particles, fields, bin_particles

ROOT = Path(__file__).resolve().parent


def convert_run(raw_run, output, seed, position_bins, velocity_bins, v_limit, audit_run=None):
    files = sorted((raw_run / "diags/electron_track").glob("openpmd_*.json"))
    if not files:
        raise FileNotFoundError(f"No particle snapshots in {raw_run}")
    if output.exists():
        raise FileExistsError(f"Derived output already exists: {output}")
    _, _, initial_field = frame_from_bytes(
        (raw_run / "diags/fields" / files[0].name).read_bytes())
    initial = fields(initial_field)
    z_edges = np.linspace(*initial["bounds"], position_bins + 1)
    v_edges = np.linspace(-v_limit, v_limit, velocity_bins + 1)
    v_centres = 0.5 * (v_edges[:-1] + v_edges[1:])
    shape = (len(files), 2)
    phase = np.empty(shape + (position_bins, velocity_bins), dtype=np.float32)
    total_weight = np.empty(shape)
    mean_v = np.empty(shape)
    variance_v = np.empty(shape)
    binned_mean_v = np.empty(shape)
    binned_variance_v = np.empty(shape)
    kinetic = np.empty(shape)
    tail_loss = np.empty(shape)
    times, iterations, electric, magnetic, rho, current = [], [], [], [], [], []
    reference_ids = {}
    reference_weight = {}
    digest = hashlib.sha256()
    checksums = audit_run.get("sha256", {}) if audit_run else {}
    verified = 0

    def read(path):
        nonlocal verified
        payload = path.read_bytes()
        key = str(path.relative_to(raw_run))
        checksum = hashlib.sha256(payload).hexdigest()
        digest.update((key + ":" + checksum + "\n").encode())
        if checksums:
            if key not in checksums or checksum != checksums[key]:
                raise ValueError(f"Raw file differs from the audit: {key}")
            verified += 1
        return frame_from_bytes(payload)

    for index, path in enumerate(files):
        step, time, frame = read(path)
        fstep, ftime, fframe = read(raw_run / "diags/fields" / path.name)
        if step != fstep or time != ftime:
            raise ValueError("Particle and field times do not match")
        field = fields(fframe)
        np.testing.assert_array_equal(field["z"], initial["z"])
        times.append(time); iterations.append(step)
        electric.append(field["electric"]); magnetic.append(field["magnetic"])
        rho.append(field["rho"]); current.append(field["Jz"])
        for species_index, name in enumerate(SPECIES):
            p = particles(frame, name)
            if not all(np.isfinite(p[key]).all() for key in
                       ("position", "momentum", "velocity", "weight")):
                raise ValueError(f"Non-finite particle data: {name}, step {step}")
            ids = np.sort(p["id"])
            if len(ids) != len(np.unique(ids)):
                raise ValueError("Duplicate particle ID")
            if index == 0:
                reference_ids[name] = ids
                reference_weight[name] = float(p["weight"].sum())
            np.testing.assert_array_equal(ids, reference_ids[name])
            weight = p["weight"]
            total = float(weight.sum())
            np.testing.assert_allclose(total, reference_weight[name], rtol=1e-12)
            vz = p["velocity"][:, 2] / C
            hist, lost = bin_particles(p["position"][:, 2], vz, weight, z_edges, v_edges)
            if lost > 1e-12:
                raise ValueError(f"Histogram excludes {lost:g} of the weight: {name}, step {step}")
            phase[index, species_index] = hist
            total_weight[index, species_index] = total
            tail_loss[index, species_index] = lost
            mean = float(np.average(vz, weights=weight))
            mean_v[index, species_index] = mean
            variance_v[index, species_index] = np.average((vz - mean) ** 2, weights=weight)
            coarse_mean = float(np.sum(hist * v_centres[None, :]))
            binned_mean_v[index, species_index] = coarse_mean
            binned_variance_v[index, species_index] = np.sum(hist * (v_centres[None, :] - coarse_mean) ** 2)
            kinetic[index, species_index] = np.sum(weight * p["mass"] * C**2 * (p["gamma"] - 1))
        if index % 40 == 0 or index == len(files)-1:
            print(f"Seed {seed}: processed {index+1}/{len(files)} snapshots", flush=True)
    times = np.asarray(times)
    iterations = np.asarray(iterations, dtype=np.int64)
    if len(times) < 3 or not np.all(np.diff(times) > 0):
        raise ValueError("Insufficient or non-monotonic physical times")
    np.testing.assert_allclose(np.diff(times), np.diff(times)[0], rtol=1e-10, atol=0)
    np.testing.assert_allclose(phase.sum(axis=(-2,-1)), 1, rtol=2e-7, atol=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, phase_mass=phase, times_s=times, iterations=iterations,
        z_edges_m=z_edges, velocity_edges_over_c=v_edges,
        field_z_m=initial["z"], electric_field_V_m=np.asarray(electric),
        magnetic_field_T=np.asarray(magnetic), charge_density_C_m3=np.asarray(rho),
        current_density_A_m2=np.asarray(current), total_particle_weight=total_weight,
        true_mean_vz_over_c=mean_v, true_variance_vz_over_c_squared=variance_v,
        binned_mean_vz_over_c=binned_mean_v,
        binned_variance_vz_over_c_squared=binned_variance_v,
        particle_kinetic_energy_J_m2=kinetic, excluded_weight_fraction=tail_loss,
        seed=np.int64(seed), species=np.asarray(SPECIES),
    )
    return {
        "seed": seed, "file": output.name, "shape": list(phase.shape),
        "snapshots": len(times), "first_iteration": int(iterations[0]),
        "last_iteration": int(iterations[-1]), "final_time_s": float(times[-1]),
        "snapshot_interval_s": float(np.diff(times)[0]),
        "max_excluded_weight_fraction": float(tail_loss.max()),
        "max_abs_binned_mean_velocity_error_over_c": float(np.max(abs(binned_mean_v-mean_v))),
        "max_abs_binned_variance_error_over_c_squared": float(np.max(abs(binned_variance_v-variance_v))),
        "raw_files_verified_against_audit": verified,
        "raw_file_manifest_sha256": digest.hexdigest(),
        "processed_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "compressed_bytes": output.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1,2,3,4,5])
    parser.add_argument("--position-bins", type=int, default=64)
    parser.add_argument("--velocity-bins", type=int, default=256)
    parser.add_argument("--velocity-limit-over-c", type=float, default=.5)
    args = parser.parse_args()
    if args.position_bins < 2 or args.velocity_bins < 2 or args.velocity_limit_over_c <= 0:
        parser.error("Positive bounds and at least two bins per axis are required")
    args.output = args.output.expanduser().resolve()
    default_output = ROOT / "data/processed"
    manifest_path = (ROOT / "data/manifest.json" if args.output == default_output
                     else args.output / "manifest.json")
    if manifest_path.exists():
        parser.error(f"Manifest already exists: {manifest_path}; select a fresh --output directory")
    audit_bytes = args.audit.read_bytes() if args.audit else None
    audit = json.loads(audit_bytes) if audit_bytes else {}
    audit_runs = {r["seed"]:r for r in audit.get("runs", [])}
    results = []
    for seed in args.seeds:
        result = convert_run(args.raw_root / f"seed_{seed}",
                             args.output / f"seed_{seed}.npz", seed,
                             args.position_bins, args.velocity_bins,
                             args.velocity_limit_over_c, audit_runs.get(seed))
        result["file"] = str((args.output / result["file"]).relative_to(manifest_path.parent))
        results.append(result)
        print(f"Saved seed {seed}: {result['compressed_bytes']/1e6:.2f} MB", flush=True)
    manifest = {
        "schema_version": 1, "runs": results,
        "file_paths_relative_to": "manifest directory",
        "representation": "Per-stream weighted bin mass, normalised separately to one",
        "axes": ["time", "stream", "position", "physical_velocity_over_c"],
        "velocity_conversion": "v=p/(m*sqrt(1+sum((p/(m*c))**2))) using all momentum components",
        "position_bins": args.position_bins, "velocity_bins": args.velocity_bins,
        "velocity_bounds_over_c": [-args.velocity_limit_over_c, args.velocity_limit_over_c],
        "binning_status": "Initial analysis resolution; moment errors recorded; bin-resolution sensitivity still required",
        "data_source": f"{len(results)} fixed-parameter WarpX runs with different random seeds",
        "recorded_input_parameters": {"n0_per_stream_m3": 1e15, "PICMI_drift_parameter_over_c": .1,
                                      "PICMI_thermal_parameter_over_c": .01},
        "simulation_model": "1D electromagnetic Yee, Boris, periodic boundaries in current source",
        "provenance_qualification": "Full per-run original inputs were not archived; current source and audit are recorded separately",
        "pca_applied": False, "audit_sha256": hashlib.sha256(audit_bytes).hexdigest() if audit_bytes else None,
        "audit_qualifications": audit.get("qualifications", []),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Processed data complete; no simulation or PCA was run.", flush=True)


if __name__ == "__main__":
    main()
