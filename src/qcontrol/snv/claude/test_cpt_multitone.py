"""Validate coherent population trapping (CPT) in ``multitone_optical_drive``.

This script drives a genuine Lambda system -- two ground sublevels coupled to
one common excited state -- and checks three things:

1. ``multitone_optical_drive`` (closed-form Liouvillian solve, single shared
   rotating frame per driven transition) and ``drive_excitation_hamiltonian``
   (direct time-domain ODE integration of a synthesized analog optical pulse)
   agree on the resulting state-population trajectories.
2. The population that survives at long times condenses onto the specific
   ground-state superposition ("dark state") predicted analytically from the
   two ground-excited optical coupling matrix elements actually selected for
   the chosen EOM tone -- not just onto "some" superposition.
3. Starting from the orthogonal ("bright") superposition, which couples
   maximally to the drive, population visibly pumps out through the excited
   state and decays into the dark state, the textbook CPT signature.

Two ground states of the reduced 4-level SnV lower orbital branch are exactly
degenerate here (a consequence of the chosen, highly symmetric B_target), so
a single EOM tone is simultaneously resonant with both g0->e and g1->e; that
is what makes this a Lambda system rather than two independent two-level
systems. Both legs are still independently driven, dipole-coupled transitions
with distinct (complex) coupling constants, so the CPT physics is genuine.

Do not confuse the coherent CPT dark *state* (a superposition of g0, g1)
tested here with the reduced-model's separate leakage/bookkeeping "dark"
population sink (the last basis index, absorbing decay to any ground state
outside ``included_states``) -- see ``get_dynamic_hamiltonian``'s docstring
in hamiltonian_jqt.py. They are unrelated despite the shared name.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp
import jaxquantum as jqt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pulseseq.sequencing.waveform import AnalogPulse, Apodization, Shape

import qcontrol.snv.particle_helpers as ph
import qcontrol.snv.parameters as params
from qcontrol.snv.parameters import HyperfineNeighbor
from qcontrol.snv.particle import (
    SnVParticle,
    SnVDifferentiableParams,
    SnVNonDiffParams,
)
from qcontrol.snv.pulseseq_interconnect import make_analog_pulse_time_array

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------
# One-particle setup (mirrors testing_single_particle_impl.ipynb's
# `get_basic_particle`, reproduced here so this script is standalone).
# -----------------------------------------------------------------------

B_TARGET = jnp.asarray([0.0, 0.2, 1.4 / 0.85])
TARGET_DIPOLE_OPERATOR = jnp.asarray([0.0, 0.0, 1.0])
DIAMOND_100 = jnp.asarray([0.0, 0.0])
DIAMOND_011 = jnp.asarray([jnp.pi / 2, 0.0])
NOMINAL_MAGNET_AXES = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
SAMPLING_RATE_GHZ = 6.144
DIPOLE_CRYSTAL_AXES = jnp.asarray(
    [[1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1]], dtype=jnp.float64
)


def get_basic_particle(laser_frequency, dipole_crystal_axis_idx=0) -> SnVParticle:
    diffable = SnVDifferentiableParams(
        magnet_unit_magnitude=jnp.ones(3),
        magnet_axes_rotations=jnp.zeros((3, 3)),
        strain_params=jnp.asarray([0.0, 10.0, 30.0, 5.0, 15.0]),
        dark_count_rate=jnp.asarray(0.001),
        resonant_pump_coupling_rate=jnp.asarray(0.03),
        resonant_pump_pdl=jnp.asarray(1.0),
        resonant_pump_polarization=jnp.asarray(0.0),
        resonant_pump_phase=jnp.asarray(0.0),
        mode_field_orientation=jnp.asarray([[0.0, 0.0], [0.5 * jnp.pi, 0.5 * jnp.pi]]),
        transmission_out_diamond=jnp.ones(2),
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
        dipole_crystal_axis_idx=jnp.asarray(dipole_crystal_axis_idx, dtype=jnp.int32),
        hyperfine_neighbor_idx=jnp.asarray(HyperfineNeighbor.NONEIGHBOR, dtype=jnp.int32),
        excited_state_lifetime=jnp.asarray(6.0),
        debye_waller_factor=jnp.asarray(0.6),
        quantum_efficiency=jnp.asarray(0.8),
        laser_frequency=jnp.asarray(laser_frequency),
        diamond_lattice_100_orientation=DIAMOND_100,
        diamond_lattice_011_orientation=DIAMOND_011,
        nominal_magnet_axes=NOMINAL_MAGNET_AXES,
        sampling_rate=jnp.asarray(SAMPLING_RATE_GHZ),
        mu_B_GHz_per_T=jnp.asarray(13.996),
        dipole_crystal_axes=DIPOLE_CRYSTAL_AXES,
    )
    return SnVParticle(diffable=diffable, nondiff=nondiff)


def find_lambda_transition(
    coupling_sq,
    branching,
    ple_freq,
    min_coupling=1e-6,
    min_branching=0.05,
    degeneracy_tol=1e-6,
):
    """Find the excited state with two distinct, comparably-coupled, exactly
    (or near-exactly) degenerate ground legs -- a genuine, *closed* Lambda
    system.

    Scans every (excited, ground_a, ground_b) triple, keeping only pairs
    whose two-photon (ground-ground) splitting is below ``degeneracy_tol``
    (so one EOM tone drives both legs at once), whose *excitation* couplings
    both exceed ``min_coupling`` (so both legs are actually driven), and
    whose spontaneous-emission *branching ratios* back into both grounds
    both exceed ``min_branching`` (so decay recycles population back into
    the driven {ga, gb} subspace instead of leaking out of it -- excitation
    strength and decay branching are set by different tensors in this model
    and need not track each other). Returns the triple that maximizes the
    weaker of the two couplings among qualifying candidates.
    """
    coupling_sq = np.asarray(coupling_sq)
    branching = np.asarray(branching)
    ple_freq = np.asarray(ple_freq)
    n_exc, n_gnd = coupling_sq.shape
    best = None
    for e in range(n_exc):
        for ga in range(n_gnd):
            for gb in range(ga + 1, n_gnd):
                ca, cb = coupling_sq[e, ga], coupling_sq[e, gb]
                if ca < min_coupling or cb < min_coupling:
                    continue
                ba, bb = branching[e, ga], branching[e, gb]
                if ba < min_branching or bb < min_branching:
                    continue
                split = abs(ple_freq[e, ga] - ple_freq[e, gb])
                if split > degeneracy_tol:
                    continue
                score = min(ca, cb)
                if best is None or score > best[0]:
                    best = (score, e, ga, gb, split)
    if best is None:
        raise RuntimeError("No degenerate, closed Lambda transition found.")
    return best


def main():
    # -------------------------------------------------------------------
    # 1. Build the particle/control state and locate a Lambda transition.
    # -------------------------------------------------------------------
    probe_particle = get_basic_particle(laser_frequency=params.GAMMA_FREQ - 2.5)
    control_state = ph.get_optimal_control_state(
        probe_particle, B_TARGET, TARGET_DIPOLE_OPERATOR
    )

    (E_gnd, _, _, _, E_exc, _, _, _, coupling_sq, branching) = ph.PLE_transitions(
        probe_particle, control_state
    )
    ple_freq_full = (
        E_exc[:, None] - E_gnd[None, :]
        + params.LEVEL_OFFSET
        + probe_particle.diffable.strain_params[0]
    )

    _, e_idx, g0_idx, g1_idx, split = find_lambda_transition(
        coupling_sq, branching, ple_freq_full
    )
    print(
        f"Lambda system: excited={e_idx}, grounds=({g0_idx}, {g1_idx}), "
        f"two-photon splitting={split:.3e} GHz\n"
        f"  coupling_sq[e,g0]={float(coupling_sq[e_idx, g0_idx]):.4e}, "
        f"coupling_sq[e,g1]={float(coupling_sq[e_idx, g1_idx]):.4e}\n"
        f"  branching e->g0={float(branching[e_idx, g0_idx]):.4f}, "
        f"e->g1={float(branching[e_idx, g1_idx]):.4f}"
    )

    included_states = tuple(sorted({0, 1, g0_idx, g1_idx, e_idx}))
    # `included_states` indexes ground and excited manifolds identically
    # (see multitone_optical_drive's docstring), so any excited index we
    # need (e_idx) drags the same-numbered ground state along for free.
    print(f"included_states={included_states}")

    reduced_dim = len(included_states)
    dimension = 2 * reduced_dim + 1
    local = {orig: i for i, orig in enumerate(included_states)}
    g0_local, g1_local, e_local = local[g0_idx], local[g1_idx], reduced_dim + local[e_idx]

    # -------------------------------------------------------------------
    # 2. Place a single EOM tone exactly resonant with both g0->e and
    #    g1->e (degenerate legs => one tone suffices for a Lambda drive).
    # -------------------------------------------------------------------
    eom_offset_ghz = 2.0
    resonant_freq = float(ple_freq_full[e_idx, g0_idx])
    laser_frequency = resonant_freq - eom_offset_ghz
    particle = get_basic_particle(laser_frequency=laser_frequency)
    # Re-solve the control state for the frequency-updated particle (the
    # optimum only depends on B_target/geometry, not laser_frequency, but
    # recompute for hygiene since `particle` is a new pytree).
    control_state = ph.get_optimal_control_state(
        particle, B_TARGET, TARGET_DIPOLE_OPERATOR
    )

    eom_freqs = jnp.asarray([eom_offset_ghz])
    eom_amplitudes = jnp.asarray([1.0])
    max_bessel_order = 4
    max_detuning = 1.0  # GHz; excludes any accidental far-off-resonant pick

    # -------------------------------------------------------------------
    # 3. Analytically predict the dark state from the actual selected
    #    complex Rabi couplings (the same quantities multitone_optical_drive
    #    itself builds internally).
    # -------------------------------------------------------------------
    (_, _, _, _, transition_offset_check) = ph.get_excitation_hamiltonian(
        particle, control_state, included_states=included_states
    )
    del transition_offset_check  # sanity only; unused below

    state_index = jnp.asarray(included_states, dtype=jnp.int32)
    (E_gnd2, _, _, _, E_exc2, _, _, _, coupling_sq2, _) = ph.PLE_transitions(
        particle, control_state
    )
    ple_freq2 = (
        E_exc2[:, None] - E_gnd2[None, :]
        + params.LEVEL_OFFSET
        + particle.diffable.strain_params[0]
    )
    reduced_ple_freq = ple_freq2[jnp.ix_(state_index, state_index)]
    reduced_coupling_sq = coupling_sq2[jnp.ix_(state_index, state_index)]

    selected_weight, selected_frequency, driven = ph._select_multitone_sidebands(
        reduced_coupling_sq,
        reduced_ple_freq,
        eom_freqs,
        eom_amplitudes,
        particle.nondiff.laser_frequency,
        particle.diffable.eom_vpi_ratio,
        particle.diffable.eom_vpi_bandwidth,
        particle.nondiff.excited_state_lifetime,
        max_bessel_order,
        48,
        max_detuning,
    )
    assert bool(driven[local[e_idx], g0_local]), "g0 leg not selected as driven"
    assert bool(driven[local[e_idx], g1_local]), "g1 leg not selected as driven"

    _, _, Hs_optical, c_ops, _ = ph.get_excitation_hamiltonian(
        particle, control_state, included_states=included_states
    )
    p_ge_block = Hs_optical[1].to_dense().data[
        reduced_dim : 2 * reduced_dim, 0:reduced_dim
    ]
    omega0 = complex(selected_weight[local[e_idx], g0_local] * p_ge_block[local[e_idx], g0_local])
    omega1 = complex(selected_weight[local[e_idx], g1_local] * p_ge_block[local[e_idx], g1_local])
    print(f"Selected Rabi couplings: Omega0={omega0:.4e}, Omega1={omega1:.4e}")

    dark_vec = np.array([omega1, -omega0])
    dark_vec /= np.linalg.norm(dark_vec)
    bright_vec = np.array([np.conj(omega0), np.conj(omega1)])
    bright_vec /= np.linalg.norm(bright_vec)
    print(f"Predicted dark state in (|g0>, |g1>): {dark_vec}")
    print(f"|<D|B>| (should be ~0): {abs(np.vdot(dark_vec, bright_vec)):.2e}")

    def embed_ground_dm(vec2):
        """Embed a 2-vector over (g0_local, g1_local) as a dim x dim density matrix."""
        rho = np.zeros((dimension, dimension), dtype=complex)
        idx = [g0_local, g1_local]
        for i, ii in enumerate(idx):
            for j, jj in enumerate(idx):
                rho[ii, jj] = vec2[i] * np.conj(vec2[j])
        return jqt.Qarray.create(jnp.asarray(rho), dims=(dimension,))

    rho_dark_predicted = embed_ground_dm(dark_vec)
    rho0 = embed_ground_dm(bright_vec)  # start fully bright: maximal CPT signature

    # -------------------------------------------------------------------
    # 4. Reference trajectory: direct time-domain integration of a
    #    synthesized single-tone optical (EOM) pulse via
    #    drive_excitation_hamiltonian.
    # -------------------------------------------------------------------
    # `included_states` spans two excited-state orbital branches ~2.3 THz
    # apart (see point 7 below), so this direct ODE integration is
    # deliberately kept short -- just long enough to validate agreement
    # with multitone_optical_drive, not to reach the dark-state steady
    # state (that part is shown cheaply, over a much longer window, using
    # multitone_optical_drive's closed-form solve alone; see below).
    pulse_length = float(os.environ.get("CPT_PULSE_LENGTH", 3e-7))  # seconds
    max_steps = int(float(os.environ.get("CPT_MAX_STEPS", 1e8)))
    rtol = float(os.environ.get("CPT_RTOL", 1e-5))
    atol = float(os.environ.get("CPT_ATOL", 1e-7))
    downsampling = int(os.environ.get("CPT_DOWNSAMPLING", 8))
    pulse = AnalogPulse(
        length=pulse_length,
        apodization=Apodization.SQUARE,
        apodization_length=0.0,
        padding_length=0.0,
        shift=0.0,
        amplitude=1.0,
        frequency=eom_offset_ghz * 1e9,
        frequency_chirp=0.0,
        shape=Shape.SINUSOID,
        S21_correct=False,
        w_3db=2 * jnp.pi * 2e9,
        phase_offset=np.nan,
        name="",
    )

    solver_options_args = (False, "Tsit5", max_steps, rtol, atol)

    print(
        f"Running drive_excitation_hamiltonian (direct time-domain reference); "
        f"pulse_length={pulse_length:.3e}s, max_steps={max_steps:.2e}, "
        f"rtol={rtol:.1e}, atol={atol:.1e} ...",
        flush=True,
    )
    t_start = time.perf_counter()
    states, filter_states, populations_direct = ph.drive_excitation_hamiltonian(
        particle,
        control_state,
        pulse,
        included_states=included_states,
        rho0=rho0,
        saveat_downsampling=downsampling,
        solver_options_args=solver_options_args,
    )
    populations_direct = jax.block_until_ready(np.asarray(populations_direct))
    print(f"  drive_excitation_hamiltonian took {time.perf_counter() - t_start:.1f} s", flush=True)

    tau = make_analog_pulse_time_array(
        pulse, 1.0 / float(particle.nondiff.sampling_rate) / 1e9, at_time=pulse.length / 2
    )
    tlist = np.asarray(tau[::downsampling])
    print(f"  {tlist.size} saved time points, t in [0, {tlist[-1] * 1e6:.3f}] us")

    # -------------------------------------------------------------------
    # 5. multitone_optical_drive on the exact same time grid.
    # -------------------------------------------------------------------
    print("Running multitone_optical_drive (closed-form Liouvillian)...")
    rho_traj, populations_multitone = ph.multitone_optical_drive(
        particle,
        control_state,
        rho0,
        eom_freqs,
        eom_amplitudes,
        max_bessel_order,
        jnp.asarray(tlist),
        included_states=included_states,
        max_detuning=max_detuning,
    )
    populations_multitone = np.asarray(populations_multitone)
    rho_traj = np.asarray(rho_traj)

    # -------------------------------------------------------------------
    # 6. Compare over the (short, ODE-cost-limited) window above.
    # -------------------------------------------------------------------
    abs_error = np.abs(populations_multitone - populations_direct)
    print(f"Max absolute population error: {abs_error.max():.3e}")
    print(f"RMS absolute population error: {np.sqrt(np.mean(abs_error ** 2)):.3e}")

    def dark_state_projection(rho_traj_local):
        rho_dark_np = np.asarray(rho_dark_predicted.to_dense().data)
        return np.real(np.einsum("ij,tji->t", rho_dark_np, rho_traj_local))

    dark_projection = dark_state_projection(rho_traj)

    # -------------------------------------------------------------------
    # 7. Long-time convergence to the CPT dark state: multitone_optical_drive
    #    is an exact closed-form Liouvillian solve (one eigendecomposition),
    #    so extending it to many pumping timescales costs almost nothing --
    #    unlike the direct ODE integration above, which is bottlenecked by
    #    having to numerically resolve the ~2.3 THz "bare" gap between the
    #    two excited-state orbital branches pulled into included_states, and
    #    is therefore kept to a short validation window.
    # -------------------------------------------------------------------
    long_factor = float(os.environ.get("CPT_LONG_FACTOR", 40))
    n_long = int(os.environ.get("CPT_LONG_POINTS", 800))
    tlist_long = np.linspace(0.0, tlist[-1] * long_factor, n_long)
    print(
        f"Running multitone_optical_drive over the long window "
        f"[0, {tlist_long[-1] * 1e9:.0f}] ns to show dark-state convergence...",
        flush=True,
    )
    rho_traj_long, populations_multitone_long = ph.multitone_optical_drive(
        particle,
        control_state,
        rho0,
        eom_freqs,
        eom_amplitudes,
        max_bessel_order,
        jnp.asarray(tlist_long),
        included_states=included_states,
        max_detuning=max_detuning,
    )
    populations_multitone_long = np.asarray(populations_multitone_long)
    rho_traj_long = np.asarray(rho_traj_long)
    dark_projection_long = dark_state_projection(rho_traj_long)

    final_ground_pop = (
        populations_multitone_long[g0_local, -1] + populations_multitone_long[g1_local, -1]
    )
    final_dark_frac = dark_projection_long[-1] / max(final_ground_pop, 1e-12)
    print(
        f"Final (t={tlist_long[-1] * 1e9:.0f} ns) dark-state projection "
        f"<D|rho|D> = {dark_projection_long[-1]:.4f} "
        f"(g0+g1 population = {final_ground_pop:.4f}, "
        f"fraction of ground population in |D> = {final_dark_frac:.4f})"
    )

    # -------------------------------------------------------------------
    # 8. Plot 1 (validation): populations vs time (both methods) + absolute
    #    error, over the short direct-integration window.
    # -------------------------------------------------------------------
    t_ns = tlist * 1e9
    labels = []
    for i, orig in enumerate(included_states):
        labels.append(f"g{orig}")
    for i, orig in enumerate(included_states):
        labels.append(f"e{orig}")
    labels.append("leak (outside included_states)")

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    cmap = plt.get_cmap("tab10")
    for i in range(dimension):
        color = cmap(i % 10)
        axes[0].plot(t_ns, populations_multitone[i], color=color, label=labels[i])
        axes[0].plot(t_ns, populations_direct[i], "--", color=color, alpha=0.6)
    axes[0].set_ylabel("Population")
    axes[0].set_title(
        "State populations vs time\n"
        "solid = multitone_optical_drive, dashed = drive_excitation_hamiltonian"
    )
    axes[0].legend(ncol=3, fontsize=7, loc="center right")

    for i in range(dimension):
        axes[1].plot(t_ns, abs_error[i], color=cmap(i % 10), label=labels[i])
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("Absolute error")
    axes[1].set_yscale("log")
    axes[1].set_title("|multitone_optical_drive - drive_excitation_hamiltonian|")
    axes[1].legend(ncol=3, fontsize=7, loc="center right")

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "cpt_multitone_validation.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved validation plot to {out_path}")

    # -------------------------------------------------------------------
    # 9. Plot 2 (CPT physics): long-time convergence onto the analytically
    #    predicted dark state, computed from multitone_optical_drive alone.
    # -------------------------------------------------------------------
    t_long_ns = tlist_long * 1e9
    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    ax2.plot(
        t_long_ns, dark_projection_long, color="black", linewidth=2,
        label=r"$\langle D|\rho(t)|D\rangle$ (predicted dark state)",
    )
    ax2.plot(
        t_long_ns,
        populations_multitone_long[g0_local] + populations_multitone_long[g1_local],
        color="tab:blue", linestyle=":", label="g0+g1 population",
    )
    for i, orig in ((g0_local, g0_idx), (g1_local, g1_idx), (e_local, e_idx)):
        ax2.plot(
            t_long_ns, populations_multitone_long[i], alpha=0.5,
            label=f"{'g' if i < reduced_dim else 'e'}{orig} population",
        )
    ax2.axhline(1.0, color="gray", linewidth=0.75, linestyle="--")
    ax2.set_xlabel("Time (ns)")
    ax2.set_ylabel("Population")
    ax2.set_title(
        "CPT: pumping from the bright state into the predicted dark state\n"
        f"|D> = ({dark_vec[0]:.3f})|g{g0_idx}> + ({dark_vec[1]:.3f})|g{g1_idx}>, "
        f"eom tone = {eom_offset_ghz:g} GHz (resonant with both g{g0_idx}->e{e_idx} "
        f"and g{g1_idx}->e{e_idx})"
    )
    ax2.legend(loc="center right", fontsize=8)
    fig2.tight_layout()
    out_path2 = os.path.join(OUTPUT_DIR, "cpt_dark_state_convergence.png")
    fig2.savefig(out_path2, dpi=150)
    print(f"Saved dark-state convergence plot to {out_path2}")


if __name__ == "__main__":
    main()
