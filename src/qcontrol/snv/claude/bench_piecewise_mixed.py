"""Time and peak-memory benchmark of ``_drive_mw_hamiltonian_piecewise_mixed``.

Usage: python bench_piecewise_mixed.py BATCH CHUNK_SIZE [SUBSTEPS] [LENGTH_NS]
Runs ``saveat_final_only=True`` under ``jax.vmap`` over BATCH perturbed control
states and prints one JSON line. Peak memory does not depend on the pulse
length (final state only); time is linear in it, so the line also reports an
extrapolation to a 5 us pulse. Run each configuration in its own process so
``peak_bytes_in_use`` is per-configuration.
"""
import ast, json, sys, time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import jaxquantum as jqt
from pulseseq.sequencing.waveform import AnalogPulse, Apodization, Shape

import qcontrol.snv.particle_helpers as ph
from qcontrol.snv.pulseseq_interconnect import make_analog_pulse_time_array

_src = (Path(__file__).parent / "speedtest_drive_mw.py").read_text()
_keep = [
    n for n in ast.parse(_src).body
    if isinstance(n, (ast.Import, ast.ImportFrom))
    or (isinstance(n, ast.FunctionDef) and n.name == "get_basic_particle")
    or (isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "dipole_crystal_axes")
]
exec(compile(ast.Module(body=_keep, type_ignores=[]), "speedtest_drive_mw.py", "exec"))

batch, chunk = int(sys.argv[1]), int(sys.argv[2])
substeps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
length_ns = float(sys.argv[4]) if len(sys.argv) > 4 else 300.0
method = sys.argv[5] if len(sys.argv) > 5 else "pw"

particle = get_basic_particle()
cs = ph.SnVControlState.from_targets(
    particle, jnp.asarray([0.0, 0.5, 1.4 / 0.85]), jnp.asarray([0.0, 0.0, 1.0])
)
f_res = float(ph.get_emr_freqs(particle, cs)[0]) * 1e9
L = length_ns * 1e-9
pulse = AnalogPulse(
    length=L, apodization=Apodization.COSINE, apodization_length=0.1 * L,
    padding_length=0, shift=0, amplitude=1, frequency=f_res, frequency_chirp=0,
    shape=Shape.SINUSOID, S21_correct=False, w_3db=2 * np.pi * 2e9,
    phase_offset=np.nan, name="",
)
Ts = 1.0 / (float(particle.nondiff.sampling_rate) * 1e9)
tau = make_analog_pulse_time_array(pulse=pulse, sample_period=Ts, at_time=L / 2)
K = int(tau.shape[0])
inc = (0, 1, 2, 3)
rho0 = jnp.diag(jnp.asarray([0.4, 0.3, 0.2, 0.1])).astype(jnp.complex128)
cops = jnp.stack([jnp.sqrt(1e-4) * jnp.diag(jnp.asarray([1., -1., 1., -1.])).astype(jnp.complex128)])

key = jax.random.PRNGKey(0)
noise = 1.0 + 1e-3 * jax.random.normal(key, (batch, 3))
batched_cs = ph.SnVControlState(
    magnet_settings=cs.magnet_settings[None, :] * noise,
    waveplate_angles=jnp.broadcast_to(cs.waveplate_angles, (batch, 3)),
)


def one(c):
    if method == "dopri":
        _, _, pops = ph._drive_mw_hamiltonian_mixed(
            particle, c, pulse, tau, jqt.Qarray.create(rho0, dims=(4,)), inc,
            collapse_operators=cops, saveat_final_only=True)
        return pops
    _, _, pops = ph._drive_mw_hamiltonian_piecewise_mixed(
        particle, c, pulse, tau, rho0, inc, collapse_operators=cops,
        saveat_final_only=True, chunk_size=chunk, substeps=substeps,
    )
    return pops


run = jax.jit(jax.vmap(one))
dev = jax.devices()[0]
t0 = time.perf_counter()
out = run(batched_cs)
out.block_until_ready()
first = time.perf_counter() - t0
times = []
for _ in range(2):
    t0 = time.perf_counter()
    run(batched_cs).block_until_ready()
    times.append(time.perf_counter() - t0)
stats = dev.memory_stats() or {}
t = min(times)
print(json.dumps(dict(
    device=dev.device_kind, method=method, batch=batch, chunk_size=chunk, substeps=substeps,
    K=K, first_call_s=round(first, 2), run_s=round(t, 3),
    est_5us_s=round(t * 30720 / (K - 1), 1),
    peak_GB=round(stats.get("peak_bytes_in_use", float("nan")) / 1e9, 3),
    finite=bool(jnp.all(jnp.isfinite(out))),
)), flush=True)
