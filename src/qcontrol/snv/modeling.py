import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple
import pulseseq
from pulseseq.sequencing.waveform import AnalogPulse, Apodization, CompositeWaveform, Shape
from qcontrol.snv.particle import SnVParticle
import qcontrol.snv.particle_helpers as ph
from qcontrol.snv.pulseseq_interconnect import (
    _unflatten_analog_pulse,
    _unflatten_composite_waveform,
    make_analog_pulse_time_array,
    make_composite_waveform_time_array,
    make_readout_bin_times_fn,
)
import jaxquantum as jqt

Array = jax.Array
PLE_LENGTH = 8e-6
INIT_LENGTH = 150e-6
MW_DRIVE_LENGTH = 5e-6
# Longest apodization the MW time window has room for (half at each end).
MW_DRIVE_APODIZATION_LENGTH = 0.1 * MW_DRIVE_LENGTH
MW_DRIVE_MAX_STEPS = 1_000_000

class PLEControlState(NamedTuple):
    """Physical controls applied to one or more SnV particles."""

    magnet_settings: Array
    # Shape (3,). Physical vector-magnet settings, in tesla.

    waveplate_angles: Array
    # Shape (3,). [QWP1, HWP, QWP2] angles, in radians.

    fcen: Array

    fwidth: Array
    @classmethod
    @jax.jit(static_argnames="cls")
    def from_targets(cls,
        particle: SnVParticle,
        B_target: Array,
        target_dipole_operator: Array,
        fwidth: Array
    ):
        """Return magnet and waveplate controls optimized for one particle.

        Parameters
        ----------
        particle
            Unbatched particle parameters.
        B_target : array_like, shape (3,)
            Target electron-Zeeman vector in the dipole frame, in GHz.
        target_dipole_operator : array_like, shape (3,)
            Target optical dipole-operator direction in the dipole frame.
        """
        base_state = ph.SnVControlState.from_targets(particle, B_target, target_dipole_operator)
        ple_freqs = ph.get_ple_freqs(particle, base_state)
        return cls(
            magnet_settings=base_state.magnet_settings,
            waveplate_angles=base_state.waveplate_angles,
            fcen=jnp.mean(ple_freqs) - particle.nondiff.laser_frequency,
            fwidth = fwidth
        )

class ReadoutControlState(NamedTuple):
    """Physical controls applied to one or more SnV particles."""

    magnet_settings: Array
    # Shape (3,). Physical vector-magnet settings, in tesla.

    waveplate_angles: Array
    # Shape (3,). [QWP1, HWP, QWP2] angles, in radians.

    fcen: Array
    # Shape (1,)

    length: Array
    # Shape (1,). Time, in seconds, at which the readout's density matrix is taken.

    rho0: Array

    @classmethod
    @jax.jit(static_argnames=["cls"])
    def from_targets(cls,
        particle: SnVParticle,
        B_target: Array,
        target_dipole_operator: Array,
        readout_index: Array,
        length: Array,
    ):
        """Return magnet and waveplate controls optimized for one particle.

        Parameters
        ----------
        particle
            Unbatched particle parameters.
        B_target : array_like, shape (3,)
            Target electron-Zeeman vector in the dipole frame, in GHz.
        target_dipole_operator : array_like, shape (3,)
            Target optical dipole-operator direction in the dipole frame.
        readout_index
            Index into the PLE frequencies of the readout tone.
        length : array_like, shape (1,)
            Time, in seconds, at which the readout's density matrix is taken.
        """
        dimension = 2 * 4 + 1  # default included_states=(0, 1, 2, 3)
        rho0 = sum(jqt.ket2dm(jqt.basis(dimension, i)) for i in range(4)) / 4.0
        base_state = ph.SnVControlState.from_targets(particle, B_target, target_dipole_operator)
        ple_freqs = ph.get_ple_freqs(particle, base_state)
        return cls(
            magnet_settings=base_state.magnet_settings,
            waveplate_angles=base_state.waveplate_angles,
            fcen=ple_freqs[readout_index] - particle.nondiff.laser_frequency,
            length=length,
            rho0=rho0
        )
    

class DualCyclicityControlState(NamedTuple):
    """Physical controls applied to one or more SnV particles."""

    magnet_settings: Array
    # Shape (3,). Physical vector-magnet settings, in tesla.

    waveplate_angles: Array
    # Shape (3,). [QWP1, HWP, QWP2] angles, in radians.

    fcen: Array # Size (2)

    length: Array
    # Shape (1)
    @classmethod
    @jax.jit(static_argnames="cls")
    def from_targets(cls,
        particle: SnVParticle,
        B_target: Array,
        target_dipole_operator: Array,
        length: Array,
    ):
        """Return magnet and waveplate controls optimized for one particle.

        Parameters
        ----------
        particle
            Unbatched particle parameters.
        B_target : array_like, shape (3,)
            Target electron-Zeeman vector in the dipole frame, in GHz.
        target_dipole_operator : array_like, shape (3,)
            Target optical dipole-operator direction in the dipole frame.
        """
        base_state = ph.SnVControlState.from_targets(particle, B_target, target_dipole_operator)
        ple_freqs = ph.get_ple_freqs(particle, base_state)
        return cls(
            magnet_settings=base_state.magnet_settings,
            waveplate_angles=base_state.waveplate_angles,
            fcen=jnp.asarray([(ple_freqs[0]+ple_freqs[1])/2,(ple_freqs[2]+ple_freqs[3])/2]),
            length=length
        )

class MWDriveControlState(NamedTuple):
    """Physical controls applied to one or more SnV particles."""

    magnet_settings: Array
    # Shape (3,). Physical vector-magnet settings, in tesla.

    waveplate_angles: Array
    # Shape (3,). [QWP1, HWP, QWP2] angles, in radians.

    fcen: Array
    # Shape (2,) 

    amps: Array
    # Shape (2,)

    fwidth: Array
    # Shape (2,)

    length: Array
    # Shape (1,)

    apodization_length: Array
    # Shape (1,)

    rho0: Array

    @classmethod
    @jax.jit(static_argnames=["cls"])
    def from_targets(cls,
        particle: SnVParticle,
        B_target: Array,
        target_dipole_operator: Array,
        length: Array,
        apodization_length: Array,
        mw_index: Array = jnp.asarray(0),
        fwidth: Array = jnp.asarray([0,0]),
    ):
        """Return magnet and waveplate controls optimized for one particle.

        Parameters
        ----------
        particle
            Unbatched particle parameters.
        B_target : array_like, shape (3,)
            Target electron-Zeeman vector in the dipole frame, in GHz.
        target_dipole_operator : array_like, shape (3,)
            Target optical dipole-operator direction in the dipole frame.
        """
        dimension = 4  # ground manifold only (default included_states=(0, 1, 2, 3))
        rho0 = sum(jqt.ket2dm(jqt.basis(dimension, i)) for i in range(dimension)) / dimension
        base_state = ph.SnVControlState.from_targets(particle, B_target, target_dipole_operator)
        amps = jnp.zeros(2).at[mw_index].set(1.0)
        amps /= jnp.linalg.norm(amps)
        emr_freqs = ph.get_emr_freqs(particle, base_state)
        return cls(
            magnet_settings=base_state.magnet_settings,
            waveplate_angles=base_state.waveplate_angles,
            fcen=emr_freqs,
            amps=amps,
            length=length,
            apodization_length=apodization_length,
            fwidth=fwidth,
            rho0=rho0
        )

def optimal_psb_readout_length(
    particle: SnVParticle,
    control_state: ph.SnVControlState,
    readout_fcen: Array,
    max_length: float = INIT_LENGTH,
    n_lengths: int = 512,
):
    """Return the PSB readout window maximizing electron-spin readout fidelity.

    Generalizes ``fidelity``/``optimal_integration_time`` in
    ``optimal_readout.ipynb`` to the full sum-of-exponentials fluorescence of
    both spin states. The readout drive is static in the multitone rotating
    frame, so each state's excited population is exactly
    ``sum_k b_k exp(lambda_k t)`` with ``lambda_k`` the eigenvalues of the
    readout Liouvillian, and its integral is closed form. Bright is the
    electron-down mixture (ground states 0, 1), dark the electron-up mixture
    (2, 3); the dark state's off-resonant fluorescence is kept. Dark counts
    and reflected pump light are a constant background, and the windowed
    count is classified with the optimal integer Poisson threshold.

    Parameters
    ----------
    particle
        Unbatched particle parameters.
    control_state
        Magnet and waveplate controls.
    readout_fcen
        Readout tone, relative to the laser frequency, in GHz.
    max_length
        Longest candidate window, in seconds.
    n_lengths
        Number of log-spaced candidate windows (static).

    Returns
    -------
    readout_length : jax.Array, shape (1,)
        Optimal window, in seconds. Gradients are stopped: the
        eigendecomposition has NaN derivatives for degenerate spectra.
    fidelity : jax.Array, scalar
        Windowed readout fidelity at that window.
    """
    included_states = (0, 1, 2, 3)
    dimension = 2 * len(included_states) + 1
    nondiff, diffable = particle.nondiff, particle.diffable
    L, inconsistent = ph._multitone_liouvillian(
        particle, control_state, readout_fcen, jnp.asarray(1.0), included_states,
        3, 48, 2 / nondiff.excited_state_lifetime,
    )
    eigenvalues, modes = jnp.linalg.eig(L)

    # Excited-population amplitude of each mode, for each initial state.
    excited_projector = jnp.zeros(dimension).at[4:8].set(1.0)
    excited_vec = jnp.diag(excited_projector).reshape(-1).astype(modes.dtype)
    rho0s = jnp.stack([
        ph.expand_excited_rho(jnp.diag(jnp.asarray(p)) / 2.0)
        for p in ([1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0])
    ])  # (bright, dark)
    coefficients = jnp.linalg.solve(modes, rho0s.reshape(2, -1).astype(modes.dtype).T)  # (d**2, 2)
    amplitudes = (excited_vec @ modes)[:, None] * coefficients  # (d**2, 2)

    # int_0^T exp(lambda t) dt, with the lambda -> 0 (steady-state) limit.
    lengths = jnp.geomspace(1e-9, max_length, n_lengths)
    lt = eigenvalues[:, None] * lengths[None, :]
    safe_eigenvalues = jnp.where(eigenvalues == 0, 1.0, eigenvalues)
    integrals = jnp.where(
        jnp.abs(lt) < 1e-8, lengths[None, :], jnp.expm1(lt) / safe_eigenvalues[:, None]
    )  # (d**2, n_lengths)
    excited_time = jnp.real(amplitudes.T @ integrals)  # (2, n_lengths), seconds

    # Same emission/collection chain as readout_model, PSB only.
    psb_counts = (
        excited_time * 1e9 / nondiff.excited_state_lifetime * nondiff.quantum_efficiency
        * (1 - nondiff.debye_waller_factor) * jnp.sum(diffable.transmission_out_diamond)
    )
    mode_couplings = ph.get_resonant_pump_coupling(particle=particle, control_state=control_state)
    reflected_count_rate = jnp.sum(
        diffable.reflection_from_pic * jnp.abs(mode_couplings) ** 2
        / (2 * jnp.pi * nondiff.laser_frequency)
    ) * 1e9
    background = (reflected_count_rate + diffable.dark_count_rate) * lengths
    lam_bright = jnp.maximum(psb_counts[0], psb_counts[1]) + background
    lam_dark = jnp.minimum(psb_counts[0], psb_counts[1]) + background

    # Optimal integer threshold for two Poisson hypotheses; bright if n >= nth.
    log_ratio = jnp.log(lam_bright / lam_dark)
    nth = jnp.floor((lam_bright - lam_dark) / jnp.where(log_ratio > 0, log_ratio, 1.0)) + 1
    p_dark_correct = jax.scipy.special.gammaincc(nth, lam_dark)  # P(n <= nth-1)
    p_bright_correct = jax.scipy.special.gammainc(nth, lam_bright)  # P(n >= nth)
    fidelities = 0.5 * (p_dark_correct + p_bright_correct)
    fidelities = jnp.where(inconsistent | ~jnp.isfinite(fidelities), 0.0, fidelities)

    best = jnp.argmax(fidelities)
    readout_length = jnp.where(inconsistent, jnp.nan, lengths[best])
    return (
        jax.lax.stop_gradient(jnp.atleast_1d(readout_length)),
        jax.lax.stop_gradient(fidelities[best]),
    )


class ODMRControlState(NamedTuple):
    """Physical controls applied to one or more SnV particles."""

    magnet_settings: Array
    # Shape (3,). Physical vector-magnet settings, in tesla.

    waveplate_angles: Array
    # Shape (3,). [QWP1, HWP, QWP2] angles, in radians.

    fcen: Array
    # Shape (2,). EMR (microwave) frequencies, in GHz.

    ple_fcen: Array
    # Shape (4,). Absolute PLE frequencies f0-f3, in GHz (f0, f1 electron
    # down; f2, f3 electron up).

    amps: Array
    # Shape (2,)

    fwidth: Array
    # Shape (2,)

    mw_length: Array
    # Shape (1,)

    readout_length: Array
    # Shape (1,)

    init_length: Array
    # Shape (1,)

    apodization_length: Array
    # Shape (1,)

    rho0: Array

    @classmethod
    @jax.jit(static_argnames=["cls"])
    def from_targets(cls,
        particle: SnVParticle,
        B_target: Array,
        target_dipole_operator: Array,
        mw_length: Array,
        init_length: Array,
        apodization_length: Array,
        mw_index: Array = jnp.asarray(0),
        fwidth: Array = jnp.asarray([0,0]),
    ):
        """Return magnet and waveplate controls optimized for one particle.

        Parameters
        ----------
        particle
            Unbatched particle parameters.
        B_target : array_like, shape (3,)
            Target electron-Zeeman vector in the dipole frame, in GHz.
        target_dipole_operator : array_like, shape (3,)
            Target optical dipole-operator direction in the dipole frame.

        `readout_length` is chosen by :func:`optimal_psb_readout_length`,
        with the readout tone at the mean of the two electron-down
        (ground-orbital) PLE lines, since both nuclear states are pumped.
        """
        dimension = 4  # ground manifold only (default included_states=(0, 1, 2, 3))
        rho0 = sum(jqt.ket2dm(jqt.basis(dimension, i)) for i in range(dimension)) / dimension
        base_state = ph.SnVControlState.from_targets(particle, B_target, target_dipole_operator)
        amps = jnp.zeros(2).at[mw_index].set(1.0)
        amps /= jnp.linalg.norm(amps)
        emr_freqs = ph.get_emr_freqs(particle, base_state)
        ple_freqs = ph.get_ple_freqs(particle, base_state)
        readout_fcen = jnp.mean(ple_freqs[:2]) - particle.nondiff.laser_frequency
        readout_length, _ = optimal_psb_readout_length(particle, base_state, readout_fcen)
        return cls(
            magnet_settings=base_state.magnet_settings,
            waveplate_angles=base_state.waveplate_angles,
            fcen=emr_freqs,
            ple_fcen=ple_freqs,
            amps=amps,
            mw_length=mw_length,
            readout_length=readout_length,
            init_length=init_length,
            apodization_length=apodization_length,
            fwidth=fwidth,
            rho0=rho0
        )
    

def psb_ple_model(awg):
    default_pulse = AnalogPulse(
        length=PLE_LENGTH,
        apodization=Apodization.COSINE,
        amplitude=1,
        frequency=0,
        frequency_chirp=0,
        shape=Shape.SINUSOID,
    )
    # bin_times, tau, and bin_indices depend only on awg/default_pulse (not
    # on particle or control_state), so they're computed once here, outside
    # model, rather than every jit trace.
    bin_times=make_readout_bin_times_fn(awg, default_pulse)
    bin_time = bin_times[1] - bin_times[0]  # bin_times is uniformly spaced.
    tau = make_analog_pulse_time_array(default_pulse, 1 / awg.sample_rate, at_time=default_pulse.length / 2)
    # For each tau sample, the index of the readout bin it falls into
    # (bin_times holds each bin's start time; the last bin absorbs any tau
    # past the final edge).
    bin_indices = jnp.clip(
        jnp.searchsorted(bin_times, tau, side="right") - 1,
        0,
        bin_times.shape[0] - 1,
    )

    @jax.jit
    def model(particle : SnVParticle, control_state : PLEControlState):
        # snv_control_state = SnVControlState(magnet_settings=control_state.magnet_settings, waveplate_angles=control_state.waveplate_angles)

        # Linear frequency chirp across the fixed tau grid, swept from
        # (fcen - fwidth/2) to (fcen + fwidth/2) over the pulse.
        factor = control_state.fwidth / 8e-6
        eom_freqs = tau * factor + (control_state.fcen - control_state.fwidth / 2)
        bin_freqs = bin_times * factor + (control_state.fcen - control_state.fwidth / 2)
        rates, branching_ratios = ph.scattering_rate(particle, control_state, eom_frequency=eom_freqs, max_bessel_order=4)
        # rates has shape (F, S, N_exc, N_gnd); sum sideband/excited/ground
        # axes and average over the 4 relevant transitions to get one
        # scattering rate per tau sample.
        rates_summed = rates.sum(axis=(-3, -2, -1))/4
        # Average rates_summed into readout-bin groups: each tau sample is
        # assigned to a bin via bin_indices (precomputed above from the
        # fixed tau/bin_times grids), summed per bin, then divided by the
        # number of samples landing in that bin.
        binned_rate_sums = jax.ops.segment_sum(rates_summed, bin_indices, num_segments=bin_times.shape[0])
        binned_sizes = jax.ops.segment_sum(jnp.ones_like(rates_summed), bin_indices, num_segments=bin_times.shape[0])
        # Convert the per-bin average rate (GHz) into expected counts over
        # the bin width (bin_time, in seconds; the 1e9 converts GHz -> 1/s).
        binned_excitations = binned_rate_sums / jnp.maximum(binned_sizes, 1) * bin_time * 1e9
        lossy_counts = binned_excitations*(1-particle.nondiff.debye_waller_factor)
        lossy_counts *= particle.nondiff.quantum_efficiency


        lossy_counts = particle.diffable.transmission_out_diamond[:, None]*lossy_counts[None, :]
        mode_couplings = ph.get_resonant_pump_coupling(particle=particle, control_state=control_state)
        reflected_count_rate = particle.diffable.reflection_from_pic*jnp.abs(mode_couplings)**2/(2*jnp.pi*particle.nondiff.laser_frequency)
        filtered_photons = lossy_counts + reflected_count_rate[:, None]*bin_time*1e9

        # TODO - Insert waveplate code for polarization extinction here, but probs not for PLE

        filtered_photons = jnp.sum(filtered_photons, axis=0)
        noisy_counts = filtered_photons + particle.diffable.dark_count_rate*bin_time
        lossy_counts = lossy_counts.at[:, -1].set(0)
        noisy_counts = noisy_counts.at[-1].set(0)
        return bin_freqs, binned_excitations, lossy_counts, noisy_counts
    return model, bin_times

def readout_model(awg):
    default_pulse = AnalogPulse(
        length=INIT_LENGTH,
        apodization=Apodization.COSINE,
        amplitude=1,
        frequency=0,
        frequency_chirp=0,
        shape=Shape.SINUSOID,
    )
    # bin_times and bin_widths depend only on awg/default_pulse (not on
    # particle or control_state), so they're computed once here, on the
    # host, rather than every jit trace.
    bin_times = make_readout_bin_times_fn(awg, default_pulse)
    bin_time  = bin_times[1] - bin_times[0]  # bin_times is uniformly spaced.
    # bin_times holds each bin's start time; the last bin runs to the end of
    # the pulse, so it is usually shorter than bin_time.
    pulse_end = float(default_pulse.length) + float(default_pulse.padding_length)
    bin_widths = np.diff(np.append(np.asarray(bin_times), pulse_end))

    @jax.jit
    def model(particle: SnVParticle, control_state: ReadoutControlState):
        """Simulate one constant readout pulse.

        `control_state.length` (seconds, may be traced) is when the returned
        density matrix is taken. Counts always cover the whole pulse.
        """
        state_time = jnp.squeeze(control_state.length)
        rho, pops = ph.multitone_optical_drive_binned(particle, control_state, control_state.rho0, control_state.fcen, jnp.asarray(1), bin_widths, state_time, max_detuning=2/particle.nondiff.excited_state_lifetime)

        # pops is (dimension, n_bins): ground 0-3, excited 4-7, dark 8.
        average_bin_pops = pops[4:8].sum(axis=0)
        binned_emissions = average_bin_pops*bin_time/particle.nondiff.excited_state_lifetime*particle.nondiff.quantum_efficiency*1e9
        binned_zpl_emissions = binned_emissions*particle.nondiff.debye_waller_factor
        binned_psb_emissions = binned_emissions*(1-particle.nondiff.debye_waller_factor)

        lossy_zpl_counts = particle.diffable.transmission_out_diamond[:, None]*binned_zpl_emissions[None, :]
        lossy_psb_counts = particle.diffable.transmission_out_diamond[:, None]*binned_psb_emissions[None, :]

        mode_couplings = ph.get_resonant_pump_coupling(particle=particle, control_state=control_state)
        reflected_count_rate = particle.diffable.reflection_from_pic*jnp.abs(mode_couplings)**2/(2*jnp.pi*particle.nondiff.laser_frequency)
        apd_zpl_photons = jnp.sum(lossy_zpl_counts + reflected_count_rate[:, None]*bin_time*1e9, axis=0)
        apd_psb_photons = jnp.sum(lossy_psb_counts + reflected_count_rate[:, None]*bin_time*1e9, axis=0)
        # TODO - Insert waveplate code for polarization extinction here

        apd_zpl_counts = apd_zpl_photons + particle.diffable.dark_count_rate*bin_time
        apd_psb_counts = apd_psb_photons + particle.diffable.dark_count_rate*bin_time
        return rho, apd_zpl_counts, apd_psb_counts, (binned_emissions, reflected_count_rate)
    return model, bin_times


def single_readout_model():
    """Build a readout model that counts the whole pulse as a single bin.

    Same drive and emission/collection chain as :func:`readout_model`, but
    the pulse is ``control_state.length`` long (may be traced) and its counts
    are integrated over the entire pulse rather than per AWG readout bin, so
    no awg is needed. One Van Loan exponential of ``[[L, I], [0, 0]] * length``
    gives both the propagator ``exp(L length)`` and the exact integral
    ``int_0^length exp(L s) ds`` (cf. ``ph.get_binned_density_matrix_evolution``,
    which needs concrete bin widths).
    """
    included_states = (0, 1, 2, 3)
    dimension = 2 * len(included_states) + 1

    @jax.jit
    def model(particle: SnVParticle, control_state: ReadoutControlState):
        """Simulate one constant readout pulse of length `control_state.length`.

        Returns the density matrix at the end of the pulse and the ZPL and
        PSB counts summed over the whole pulse.
        """
        length = jnp.squeeze(control_state.length)
        L, inconsistent = ph._multitone_liouvillian(
            particle, control_state, control_state.fcen, jnp.asarray(1.0), included_states,
            3, 48, 2 / particle.nondiff.excited_state_lifetime,
        )
        n = L.shape[0]
        block = (
            jnp.zeros((2 * n, 2 * n), dtype=L.dtype)
            .at[:n, :n].set(L * length)
            .at[:n, n:].set(jnp.eye(n, dtype=L.dtype) * length)
        )
        exp_block = jax.scipy.linalg.expm(block, max_squarings=64)
        rho0_vec = ph._to_dense(control_state.rho0).reshape(-1).astype(L.dtype)
        rho = (exp_block[:n, :n] @ rho0_vec).reshape(dimension, dimension)
        # int_0^length rho(s) ds, in seconds; ground 0-3, excited 4-7, dark 8.
        rho_integral = (exp_block[:n, n:] @ rho0_vec).reshape(dimension, dimension)
        excited_time = jnp.real(jnp.diagonal(rho_integral))[4:8].sum()
        rho = jnp.where(inconsistent, jnp.nan, rho)
        excited_time = jnp.where(inconsistent, jnp.nan, excited_time)

        emissions = excited_time/particle.nondiff.excited_state_lifetime*particle.nondiff.quantum_efficiency*1e9
        zpl_emissions = emissions*particle.nondiff.debye_waller_factor
        psb_emissions = emissions*(1-particle.nondiff.debye_waller_factor)

        lossy_zpl_counts = particle.diffable.transmission_out_diamond*zpl_emissions
        lossy_psb_counts = particle.diffable.transmission_out_diamond*psb_emissions

        mode_couplings = ph.get_resonant_pump_coupling(particle=particle, control_state=control_state)
        reflected_count_rate = particle.diffable.reflection_from_pic*jnp.abs(mode_couplings)**2/(2*jnp.pi*particle.nondiff.laser_frequency)
        apd_zpl_photons = jnp.sum(lossy_zpl_counts + reflected_count_rate*length*1e9)
        apd_psb_photons = jnp.sum(lossy_psb_counts + reflected_count_rate*length*1e9)
        # TODO - Insert waveplate code for polarization extinction here

        apd_zpl_counts = apd_zpl_photons + particle.diffable.dark_count_rate*length
        apd_psb_counts = apd_psb_photons + particle.diffable.dark_count_rate*length
        return rho, apd_zpl_counts, apd_psb_counts, (emissions, reflected_count_rate)
    return model


def dual_cyclicity_model(awg):
    model_readout, bin_times = readout_model(awg)

    dimension = 2 * 4 + 1  # default included_states=(0, 1, 2, 3)
    rho0 = sum(jqt.ket2dm(jqt.basis(dimension, i)) for i in range(4)) / 4.0

    @jax.jit
    def model(particle: SnVParticle, control_state: DualCyclicityControlState):
        """Pump at fcen[1], then read out at fcen[0] and at fcen[1].

        Each pulse's state is handed on at `control_state.length`. Returns
        the (ZPL, PSB) counts of both readouts and `valid_bins`, a boolean
        mask (same shape as `bin_times`) of the bins that start before
        `length`. A mask keeps the output shape fixed for jit/vmap, where a
        length-dependent slice of `bin_times` could not be traced.
        """
        length = jnp.squeeze(control_state.length)
        laser_frequency = particle.nondiff.laser_frequency

        readout_control = ReadoutControlState(
            magnet_settings=control_state.magnet_settings,
            waveplate_angles=control_state.waveplate_angles,
            fcen=control_state.fcen[1] - laser_frequency,
            length=control_state.length,
            rho0=rho0,
        )
        # Initialization: pump with the fcen[1] tone, then keep the state at `length`.
        rho_init, _, _, _ = model_readout(particle, readout_control)

        # Readouts from the initialized state at each tone.
        rho_post_f0, f0_zpl_counts, f0_psb_counts, _ = model_readout(
            particle,
            readout_control._replace(rho0=rho_init, fcen=control_state.fcen[0] - laser_frequency),
        )
        _, f1_zpl_counts, f1_psb_counts, _ = model_readout(
            particle,
            readout_control._replace(rho0=rho_post_f0, fcen=control_state.fcen[1] - laser_frequency),
        )
        valid_bins = bin_times < length
        return (f0_zpl_counts, f0_psb_counts), (f1_zpl_counts, f1_psb_counts), valid_bins
    return model, bin_times

def mw_drive_model(awg):
    """Build a model that applies a two-tone microwave drive and returns rho.

    The drive is integrated with ``_drive_mw_hamiltonian_mixed`` (Dopri5
    Lindblad solver), which was ~8x faster than the piecewise-constant
    propagator and far lighter on memory when batched over 4k particles on an
    L40S.

    The time grid starts at t = 0 and spans
    ``MW_DRIVE_LENGTH + MW_DRIVE_APODIZATION_LENGTH``. Each pulse starts at
    t = 0 and occupies ``length + apodization_length`` (the cosine ramps are
    centered on the edges of ``length``); a shorter pulse leaves free
    evolution for the rest of the window, and a longer one is truncated.
    """
    default_waveform = CompositeWaveform(
        [
            AnalogPulse(
                length=MW_DRIVE_LENGTH,
                apodization=Apodization.COSINE,
                apodization_length=MW_DRIVE_APODIZATION_LENGTH,
                amplitude=1,
                frequency=0,
                frequency_chirp=0,
                shape=Shape.SINUSOID,
            )
        ]
    )
    # tau depends only on awg/default_waveform, so it is built once on the
    # host. The composite's length includes the apodization ramps.
    tau = make_composite_waveform_time_array(
        default_waveform, 1 / awg.sample_rate, at_time=default_waveform.length / 2
    )

    @jax.jit
    def model(particle: SnVParticle, control_state: MWDriveControlState):
        """Drive both tones simultaneously and return the final rho.

        Tone ``i`` has frequency ``fcen[i]`` (GHz), amplitude ``amps[i]``, and
        is swept over ``fwidth[i]`` (GHz, total) across the pulse. Both tones
        share ``length`` and ``apodization_length`` and start, ramps
        included, at t = 0; a tone
        with zero amplitude contributes nothing. Returns the final density
        matrix in the retained ground eigenbasis, shape (4, 4).
        """
        length = jnp.squeeze(control_state.length)
        apodization_length = jnp.squeeze(control_state.apodization_length)

        def tone(i):
            # synthesize_analog_pulse sweeps frequency by chirp / 2 in total.
            return _unflatten_analog_pulse(
                None,
                (
                    length,
                    jnp.asarray(Apodization.COSINE, dtype=jnp.int32),
                    apodization_length,
                    jnp.zeros(()),
                    jnp.zeros(()),
                    control_state.amps[i],
                    control_state.fcen[i] * 1e9,
                    2.0 * control_state.fwidth[i] * 1e9,
                    jnp.asarray(Shape.SINUSOID, dtype=jnp.int32),
                    jnp.asarray(False),
                    jnp.zeros(()),
                    jnp.asarray(jnp.nan),
                ),
            )

        # Both tones sit on the composite center, which the drive places at
        # composite length / 2 = (length + apodization_length) / 2, so each
        # pulse, ramps included, spans [0, length + apodization_length].
        waveform = _unflatten_composite_waveform(
            ("tuple", None),
            (
                (tone(0), tone(1)),
                jnp.zeros(2),
                length + apodization_length,
                jnp.zeros(()),
            ),
        )
        snv_control_state = ph.SnVControlState(
            magnet_settings=control_state.magnet_settings,
            waveplate_angles=control_state.waveplate_angles,
        )
        states, _, _ = ph._drive_mw_hamiltonian_mixed(
            particle,
            snv_control_state,
            waveform,
            tau,
            control_state.rho0,
            (0, 1, 2, 3),
            saveat_final_only=True,
            # (progress_meter, solver, max_steps, rtol, atol): a 5 us drive
            # needs more than the default 100k adaptive steps.
            solver_options_args=(False, "Dopri5", MW_DRIVE_MAX_STEPS, 1e-5, 1e-7),
        )
        return states.to_dense().data[-1]
    return model

def odmr_model(awg):
    """Build the pulsed-ODMR model: init, norm, drain, zero, MW, readout, debug.

    Each optical step is one tone at the mean frequency of a PLE pair
    ((f0 + f1)/2 reads electron down, (f2 + f3)/2 pumps electron up -> down);
    the nuclear splitting is small enough that one tone addresses both
    nuclear states. Each step hands its density matrix to the next. The MW step works
    on the ground manifold, so the state is collapsed to 4x4 before it and
    expanded back to 9x9 after it.
    """
    model_mw_drive = mw_drive_model(awg)
    model_readout = single_readout_model()

    @jax.jit
    def model(particle: SnVParticle, control_state: ODMRControlState):
        """Return PSB counts per shot of the norm, zero, readout and debug bins.

        Each step's counts are integrated over its whole length as one bin
        (see :func:`single_readout_model`).
        """
        laser_frequency = particle.nondiff.laser_frequency
        ple = control_state.ple_fcen
        down_fcen = (ple[0] + ple[1]) / 2 - laser_frequency
        up_fcen = (ple[2] + ple[3]) / 2 - laser_frequency
        readout_length = control_state.readout_length
        init_length = control_state.init_length

        def readout(rho, fcen, length):
            rho, _, psb_counts, _ = model_readout(particle, ReadoutControlState(
                magnet_settings=control_state.magnet_settings,
                waveplate_angles=control_state.waveplate_angles,
                fcen=fcen,
                length=jnp.atleast_1d(jnp.squeeze(length)),
                rho0=rho,
            ))
            return rho, psb_counts

        # 1. Initialize: pump electron up -> down on f2/f3.
        rho, _ = readout(ph.expand_excited_rho(control_state.rho0), up_fcen, init_length)
        # 2. Norm bin on f0/f1.
        rho, norm_counts = readout(rho, down_fcen, readout_length)
        # 3. Norm drain: keep pumping on f0/f1 for the rest of init_length.
        rho, _ = readout(rho, down_fcen, init_length - readout_length)
        # 4. Zero bin on f0/f1.
        rho, zero_counts = readout(rho, down_fcen, readout_length)

        # 5. MW drive on the ground manifold.
        snv_control_state = ph.SnVControlState(
            magnet_settings=control_state.magnet_settings,
            waveplate_angles=control_state.waveplate_angles,
        )
        branching_ratios = ph.PLE_transitions(particle, snv_control_state)[-1]
        rho_ground = ph.collapse_excited_rho(rho, branching_ratios)
        rho_ground = model_mw_drive(particle, MWDriveControlState(
            magnet_settings=control_state.magnet_settings,
            waveplate_angles=control_state.waveplate_angles,
            fcen=control_state.fcen,
            amps=control_state.amps,
            fwidth=control_state.fwidth,
            length=control_state.mw_length,
            apodization_length=control_state.apodization_length,
            rho0=rho_ground,
        ))
        rho = ph.expand_excited_rho(rho_ground)

        # 6. Readout bin on f0/f1.
        rho, readout_counts = readout(rho, down_fcen, readout_length)
        # 7. Debug bin on f2/f3.
        _, debug_counts = readout(rho, up_fcen, readout_length)
        return norm_counts, zero_counts, readout_counts, debug_counts
    return model
