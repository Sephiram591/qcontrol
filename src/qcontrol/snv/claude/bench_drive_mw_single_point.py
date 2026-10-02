"""Before/after check for the single-point MW drive in ``_drive_mw_hamiltonian``.

Usage: python bench_drive_mw_single_point.py <label>
Env: LENGTHS (ns, comma list), N_TIMED. Uses S21_correct=False so results are
comparable across the S21 removal. Writes bench_drive_mw_single_point_<label>.json.
"""
import json
import os
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from pulseseq.sequencing.waveform import AnalogPulse, Apodization, Shape

import qcontrol.snv.particle_helpers as ph
from qcontrol.snv.pulseseq_interconnect import make_analog_pulse_time_array

import ast  # noqa: E402

# Reuse get_basic_particle from speedtest_drive_mw.py without running that script.
_src = Path(__file__).with_name("speedtest_drive_mw.py").read_text()
_keep = [
    node for node in ast.parse(_src).body
    if isinstance(node, (ast.Import, ast.ImportFrom))
    or (isinstance(node, ast.FunctionDef) and node.name == "get_basic_particle")
    or (isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "dipole_crystal_axes")
]
exec(compile(ast.Module(body=_keep, type_ignores=[]), "speedtest_drive_mw.py", "exec"))

label = sys.argv[1]
N_TIMED = int(os.environ.get("N_TIMED", 3))
LENGTHS_NS = [int(x) for x in os.environ.get("LENGTHS", "100,500").split(",")]
B_target = jnp.asarray([0.0, 0.5, 1.4 / 0.85])
target_dipole_operator = jnp.asarray([0.0, 0.0, 1.0])


def block(tree):
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree
    )


particle = get_basic_particle()
control_state = ph.SnVControlState.from_targets(particle, B_target, target_dipole_operator)
f_res = float(ph.get_emr_freqs(particle, control_state)[0]) * 1e9
print(f"[{label}] devices={jax.devices()} f_res={f_res/1e9:.6f} GHz", flush=True)


def make_pulse(L, apodization, shape):
    return AnalogPulse(
        length=L * 1e-9, apodization=apodization,
        apodization_length=0.1 * L * 1e-9, padding_length=0, shift=0, amplitude=1,
        frequency=f_res, frequency_chirp=0, shape=shape,
        S21_correct=False, w_3db=2 * jnp.pi * 2e9, phase_offset=np.nan, name="",
    )


results = []
cases = [(Apodization.COSINE, Shape.TRIANGLE), (Apodization.SQUARE, Shape.SINUSOID), (Apodization.GAUSSIAN, Shape.SINUSOID)]
for apod, shape in cases:
    for L in LENGTHS_NS:
        pulse = make_pulse(L, apod, shape)
        run = lambda: ph.drive_mw_hamiltonian(  # noqa: E731
            particle, control_state, pulse, included_states=(0, 1, 2, 3)
        )
        t0 = time.perf_counter()
        out = run()
        block(out)
        first = time.perf_counter() - t0
        times = []
        for _ in range(N_TIMED):
            t0 = time.perf_counter()
            out = run()
            block(out)
            times.append(time.perf_counter() - t0)
        pops = np.asarray(out[2])[:, -1]
        rec = dict(label=label, apod=str(apod), shape=str(shape), length_ns=L,
                   first_s=first, median_s=float(np.median(times)),
                   final_pops=pops.tolist())
        results.append(rec)
        print(f"[{label}] {apod!s:22s} {shape!s:12s} L={L:5d} ns first={first:6.2f}s "
              f"median={rec['median_s']:7.3f}s pops={np.round(pops, 6)}", flush=True)

# Gradient sanity check: d(final pop of state 3)/d(pulse amplitude), shortest pulse.
pulse = make_pulse(LENGTHS_NS[0], Apodization.COSINE, Shape.TRIANGLE)
sample_period = 1.0 / (float(particle.nondiff.sampling_rate) * 1e9)
tau = make_analog_pulse_time_array(pulse, sample_period, float(pulse.length) / 2.0)
leaves, treedef = jax.tree_util.tree_flatten(pulse)
psi0 = ph.jqt.basis(4, 0)


def final_pop(amplitude):
    p = jax.tree_util.tree_unflatten(treedef, leaves[:5] + [amplitude] + leaves[6:])
    _, _, pops = ph._drive_mw_hamiltonian(
        particle, control_state, p, tau, psi0, (0, 1, 2, 3), saveat_final_only=True
    )
    return pops[3, -1]


g = float(jax.grad(final_pop)(jnp.asarray(1.0)))
print(f"[{label}] grad d pop3 / d amplitude = {g:.6e}", flush=True)
results.append(dict(label=label, grad_pop3_amplitude=g))

Path(__file__).with_name(f"bench_drive_mw_single_point_{label}.json").write_text(
    json.dumps(results, indent=1)
)
