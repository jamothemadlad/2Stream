"""Read the existing openPMD JSON diagnostics without importing WarpX."""

import json
from pathlib import Path
import numpy as np

C = 299792458.0
SPECIES = ("left_beam", "right_beam")


def attribute(node, name, default=None):
    return node.get("attributes", {}).get(name, {}).get("value", default)


def values(record, count=None, dtype=float):
    if "data" in record:
        result = np.asarray(record["data"], dtype=dtype)
    else:
        value = attribute(record, "value")
        if value is None or count is None:
            raise ValueError("Record has neither an array nor a supported constant")
        result = np.full(count, value, dtype=dtype)
    if count is not None and result.shape != (count,):
        raise ValueError(f"Unexpected record shape {result.shape}")
    if dtype is not np.uint64:
        result = result * float(attribute(record, "unitSI", 1.0))
    return result


def frame_from_bytes(payload):
    document = json.loads(payload)
    if len(document["data"]) != 1:
        raise ValueError("Expected one iteration per JSON file")
    iteration, frame = next(iter(document["data"].items()))
    time = float(attribute(frame, "time")) * float(attribute(frame, "timeUnitSI", 1.0))
    return int(iteration), time, frame


def read_frame(path):
    return frame_from_bytes(Path(path).read_bytes())


def momentum_to_velocity(momentum, mass):
    """SI p=m*gamma*v to physical velocity; never divide by macro-particle weight."""
    proper = np.asarray(momentum) / np.asarray(mass)[..., None]
    gamma = np.sqrt(1.0 + np.sum((proper / C) ** 2, axis=-1))
    return proper / gamma[..., None], gamma


def particles(frame, species):
    record = frame["particles"][species]
    count = len(record["id"]["data"])
    if attribute(record["momentum"], "macroWeighted") != 0:
        raise ValueError("This reader expects unweighted single-particle SI momentum")
    if attribute(record["momentum"], "unitDimension") != [1., 1., -1., 0., 0., 0., 0.]:
        raise ValueError("Unexpected momentum dimensions")
    position = np.column_stack([
        values(record["position"][axis], count)
        + values(record["positionOffset"][axis], count)
        for axis in "xyz"
    ])
    momentum = np.column_stack([values(record["momentum"][axis], count) for axis in "xyz"])
    mass = values(record["mass"], count)
    velocity, gamma = momentum_to_velocity(momentum, mass)
    return {
        "id": values(record["id"], count, np.uint64), "position": position,
        "momentum": momentum, "mass": mass, "velocity": velocity,
        "gamma": gamma, "weight": values(record["weighting"], count),
    }


def fields(frame):
    f = frame["fields"]
    electric = np.stack([values(f["E"][axis]) for axis in "xyz"])
    magnetic = np.stack([values(f["B"][axis]) for axis in "xyz"])
    count = electric.shape[1]
    dz = float(attribute(f["E"], "gridSpacing")[0]) * float(attribute(f["E"], "gridUnitSI", 1.0))
    lower = float(attribute(f["E"], "gridGlobalOffset")[0]) * float(attribute(f["E"], "gridUnitSI", 1.0))
    offset = float(attribute(f["E"]["z"], "position", [0.5])[0])
    result = {
        "electric": electric, "magnetic": magnetic,
        "rho": values(f["rho"]), "Jz": values(f["j"]["z"]),
        "z": lower + (np.arange(count) + offset) * dz,
        "bounds": (lower, lower + count * dz),
    }
    for name in ("electric", "magnetic", "rho", "Jz"):
        if not np.isfinite(result[name]).all():
            raise ValueError(f"Non-finite field: {name}")
    return result


def bin_particles(z, velocity_over_c, weight, z_edges, velocity_edges):
    outside = ((z < z_edges[0]) | (z > z_edges[-1])
               | (velocity_over_c < velocity_edges[0])
               | (velocity_over_c > velocity_edges[-1]))
    total = float(np.sum(weight))
    lost_fraction = float(np.sum(weight[outside]) / total)
    hist, _, _ = np.histogram2d(z, velocity_over_c,
                              bins=(z_edges, velocity_edges), weights=weight)
    return hist / total, lost_fraction
