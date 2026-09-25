import jax
import jax.numpy as jnp
from typing import NamedTuple
import pulseseq
from pulseseq.sequencing.waveform import AnalogPulse, Apodization, Shape
from .particle import SnVParticle, SnVControlState
import qcontrol.snv.particle_helpers as ph
from qcontrol.snv.pulseseq_interconnect import make_analog_pulse_time_array, make_readout_bin_times_fn

Array = jax.Array
PLE_PULSE_LENGTH = 8e-6
INIT_PULSE_LENGTH = 150e-6

class PLEControlState(NamedTuple):
    """Physical controls applied to one or more SnV particles."""

    magnet_settings: Array
    # Shape (3,). Physical vector-magnet settings, in tesla.

    waveplate_angles: Array
    # Shape (3,). [QWP1, HWP, QWP2] angles, in radians.

    fcen: Array

    fwidth: Array

class ReadoutControlState(NamedTuple):
    """Physical controls applied to one or more SnV particles."""

    magnet_settings: Array
    # Shape (3,). Physical vector-magnet settings, in tesla.

    waveplate_angles: Array
    # Shape (3,). [QWP1, HWP, QWP2] angles, in radians.

    fcen: Array
    # Shape (1,)

    pulse_length: Array

    rho0: Array

class CyclicityControlState(NamedTuple):
    """Physical controls applied to one or more SnV particles."""

    magnet_settings: Array
    # Shape (3,). Physical vector-magnet settings, in tesla.

    waveplate_angles: Array
    # Shape (3,). [QWP1, HWP, QWP2] angles, in radians.

    fcen: Array # Size (2)

    pulse_length: Array

class ODMRControlState(NamedTuple):
    """Physical controls applied to one or more SnV particles."""

    magnet_settings: Array
    # Shape (3,). Physical vector-magnet settings, in tesla.

    waveplate_angles: Array
    # Shape (3,). [QWP1, HWP, QWP2] angles, in radians.

    fcen: Array

    fwidth: Array

    pulse_length: Array

def psb_ple_simulation(awg):
    default_pulse = AnalogPulse(
        length=PLE_PULSE_LENGTH,
        apodization=Apodization.COSINE,
        amplitude=1,
        frequency=0,
        frequency_chirp=0,
        shape=Shape.SINUSOID,
    )
    # bin_times, tau, and bin_indices depend only on awg/default_pulse (not
    # on particle or control_state), so they're computed once here, outside
    # simulate, rather than every jit trace.
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
    def simulate(particle : SnVParticle, control_state : PLEControlState):
        # snv_control_state = SnVControlState(magnet_settings=control_state.magnet_settings, waveplate_angles=control_state.waveplate_angles)

        # Linear frequency chirp across the fixed tau grid, swept from
        # (fcen - fwidth/2) to (fcen + fwidth/2) over the pulse.
        factor = control_state.fwidth / 8e-6
        eom_freqs = tau * factor + (control_state.fcen - control_state.fwidth / 2)
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
        binned_counts = binned_rate_sums/binned_sizes * bin_time * 1e9
        lossy_counts = binned_counts*(1-particle.nondiff.debye_waller_factor)
        lossy_counts *= particle.nondiff.quantum_efficiency
        lossy_counts *= particle.diffable.transmission_out_diamond


        noisy_filtered_counts = lossy_counts 

        noisy_filtered_counts += particle.diffable.dark_count_rate*bin_time

        mode_couplings = ph.get_resonant_pump_coupling(particle=particle, control_state=control_state)
        reflected_count_rate = particle.diffable.reflection_from_pic*jnp.abs(mode_couplings)**2/(2*jnp.pi*particle.nondiff.laser_frequency)
        noisy_filtered_counts += reflected_count_rate*bin_time*1e9
        # TODO - Insert waveplate code for polarization extinction here

        return binned_counts, lossy_counts, noisy_filtered_counts
    return simulate

def readout_simulation(awg):
    default_pulse = AnalogPulse(
        length=INIT_PULSE_LENGTH,
        apodization=Apodization.COSINE,
        amplitude=1,
        frequency=0,
        frequency_chirp=0,
        shape=Shape.SINUSOID,
    )
    # bin_times, tau, and bin_indices depend only on awg/default_pulse (not
    # on particle or control_state), so they're computed once here, outside
    # simulate, rather than every jit trace.
    bin_times=make_readout_bin_times_fn(awg, default_pulse)
    bin_time = bin_times[1] - bin_times[0]  # bin_times is uniformly spaced.
    tau = make_analog_pulse_time_array(default_pulse, 1 / awg.sample_rate, at_time=default_pulse.length / 2)
    bin_indices = jnp.clip(
        jnp.searchsorted(bin_times, tau, side="right") - 1,
        0,
        bin_times.shape[0] - 1,
    )

    @jax.jit
    def simulate(particle: SnVParticle, control_state: ReadoutControlState):
        rhos, pops = ph.multitone_optical_drive(particle, control_state, control_state.rho0, control_state.fcen, jnp.asarray(1), tau, max_detuning=2/particle.nondiff.excited_state_lifetime)

        excited_pops_summed = pops[:, 4:].sum(axis=(-1))
        binned_sums = jax.ops.segment_sum(excited_pops_summed, bin_indices, num_segments=bin_times.shape[0])
        binned_sizes = jax.ops.segment_sum(jnp.ones_like(excited_pops_summed), bin_indices, num_segments=bin_times.shape[0])
        average_bin_pops = binned_sums/binned_sizes
        binned_emissions = average_bin_pops*bin_time/particle.nondiff.excited_state_lifetime*particle.nondiff.quantum_efficiency*1e9
        binned_zpl_emissions = binned_emissions*particle.nondiff.debye_waller_factor
        binned_psb_emissions = binned_emissions*(1-particle.nondiff.debye_waller_factor)

        lossy_zpl_counts = particle.diffable.transmission_out_diamond*binned_zpl_emissions
        lossy_psb_counts = particle.diffable.transmission_out_diamond*binned_psb_emissions


        apd_zpl_counts = lossy_zpl_counts + particle.diffable.dark_count_rate*bin_time
        apd_psb_counts = lossy_psb_counts + particle.diffable.dark_count_rate*bin_time

        mode_couplings = ph.get_resonant_pump_coupling(particle=particle, control_state=control_state)
        reflected_count_rate = particle.diffable.reflection_from_pic*jnp.abs(mode_couplings)**2/(2*jnp.pi*particle.nondiff.laser_frequency)
        apd_zpl_counts += reflected_count_rate*bin_time*1e9
        apd_psb_counts += reflected_count_rate*bin_time*1e9
        # TODO - Insert waveplate code for polarization extinction here




def cyclicity_simulation(awg):
    simulate_readout = readout_simulation(awg)
    @jax.jit
    def simulate(particle: SnVParticle, control_state: CyclicityControlState):
        pass