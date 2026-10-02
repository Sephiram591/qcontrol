"""CPU vs GPU speed test of ``particle_helpers.drive_mw_hamiltonian`` (single particle).

Usage: python speedtest_drive_mw.py {cpu,gpu}
Set JAX_PLATFORMS=cpu externally for the CPU run (done by the driver command).
Writes speedtest_drive_mw_<backend>.json next to this file.
"""
import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from pulseseq.sequencing.waveform import AnalogPulse, Apodization, Shape

import qcontrol.snv.particle_helpers as ph
from qcontrol.snv.parameters import HyperfineNeighbor, GAMMA_FREQ
from qcontrol.snv.particle import SnVDifferentiableParams, SnVNonDiffParams, SnVParticle

label = sys.argv[1]
N_TIMED = int(__import__("os").environ.get("N_TIMED", 5))
import os
LENGTHS_NS = [int(x) for x in os.environ.get("LENGTHS", "100,200,300,400,500,600,700,800,900,1000").split(",")]
DETUNINGS_HZ = {"on_resonance": 0.0, "off_resonance_10MHz": 10e6}

B_target = jnp.asarray([0.0, 0.5, 1.4 / 0.85])
target_dipole_operator = jnp.asarray([0.0, 0.0, 1.0])
dipole_crystal_axes = jnp.asarray(
    [[1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1]], dtype=jnp.float64
)


def get_basic_particle(idx=0):  # mirrors testing_single_particle_impl.ipynb
    diffable = SnVDifferentiableParams(
        magnet_unit_magnitude=jnp.ones(3),
        magnet_axes_rotations=jnp.zeros((3, 3)),
        strain_params=jnp.asarray([0.0, 10.0, 30, 5, 15]),
        dark_count_rate=jnp.asarray(300.0),
        resonant_pump_coupling_rate=jnp.asarray(0.03),
        resonant_pump_pdl=jnp.asarray(1.0),
        resonant_pump_polarization=jnp.asarray(0.0),
        resonant_pump_phase=jnp.asarray(0.0),
        mode_field_orientation=jnp.asarray([[0.0, 0.0], [0.5 * jnp.pi, 0.5 * jnp.pi]]),
        transmission_out_diamond=0.0001 * jnp.ones(2),
        reflection_from_pic=jnp.zeros(2),
        eom_vpi_ratio=jnp.asarray(0.6),
        eom_vpi_bandwidth=jnp.asarray(1.5),
        mw_B_orientation=jnp.asarray([np.pi / 2, 0.0]),
        mw_B_magnitude=jnp.asarray(0.5 / 28),
        mw_B_bandwidth=jnp.asarray(1.5),
        spectral_diffusion_rate=jnp.asarray(1.0),
        polarization_drift_rate=jnp.asarray(1.0),
        resonant_pump_coupling_drift_rate=jnp.asarray(1.0),
    )
    nondiff = SnVNonDiffParams(
        dipole_crystal_axis_idx=jnp.asarray(idx, dtype=jnp.int32),
        hyperfine_neighbor_idx=jnp.asarray(HyperfineNeighbor.C13_FIRST_CLASS_0, dtype=jnp.int32),
        excited_state_lifetime=jnp.asarray(6.0),
        debye_waller_factor=jnp.asarray(0.6),
        quantum_efficiency=jnp.asarray(0.8),
        laser_frequency=jnp.asarray(GAMMA_FREQ - 2.5),
        diamond_lattice_100_orientation=jnp.asarray([0.0, 0.0]),
        diamond_lattice_011_orientation=jnp.asarray([jnp.pi / 2, 0.0]),
        nominal_magnet_axes=jnp.eye(3),
        sampling_rate=jnp.asarray(6.144),
        mu_B_GHz_per_T=jnp.asarray(13.996),
        dipole_crystal_axes=dipole_crystal_axes,
    )
    return SnVParticle(diffable=diffable, nondiff=nondiff)


def block(tree):
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree
    )


particle = get_basic_particle()
control_state = ph.SnVControlState.from_targets(particle, B_target, target_dipole_operator)
f_res = float(ph.get_emr_freqs(particle, control_state)[0]) * 1e9
print(f"[{label}] devices={jax.devices()} f_res={f_res/1e9:.6f} GHz", flush=True)

results = []
for tag, det in DETUNINGS_HZ.items():
    for L in LENGTHS_NS:
        pulse = AnalogPulse(
            length=L * 1e-9, apodization=Apodization.COSINE,
            apodization_length=0.1 * L * 1e-9, padding_length=0, shift=0, amplitude=1,
            frequency=f_res + det, frequency_chirp=0, shape=Shape.TRIANGLE,
            S21_correct=True, w_3db=2 * jnp.pi * 2e9, phase_offset=np.nan, name="",
        )
        t0 = time.perf_counter()
        out = ph.drive_mw_hamiltonian(particle, control_state, pulse,
                                      included_states=(0, 1, 2, 3), saveat_final_only=False)
        block(out)
        first = time.perf_counter() - t0  # includes JIT compile
        times = []
        for _ in range(N_TIMED):
            t0 = time.perf_counter()
            out = ph.drive_mw_hamiltonian(particle, control_state, pulse,
                                          included_states=(0, 1, 2, 3), saveat_final_only=False)
            block(out)
            times.append(time.perf_counter() - t0)
        pops = np.asarray(out[2])[:, -1]
        rec = dict(backend=label, drive=tag, length_ns=L, first_s=first,
                   median_s=float(np.median(times)), min_s=float(np.min(times)),
                   final_pops=pops.tolist())
        results.append(rec)
        print(f"[{label}] {tag:20s} L={L:5d} ns first={first:7.2f}s "
              f"median={rec['median_s']*1e3:9.2f} ms  pops={np.round(pops,4)}", flush=True)

Path(__file__).with_name(f"speedtest_drive_mw_{label}.json").write_text(json.dumps(results, indent=1))
