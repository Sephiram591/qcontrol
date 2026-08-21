import jax
import jax.numpy as jnp
from pulseseq.sequencing.waveform import Waveform, CompositeWaveform, DigitalPulse, AnalogPulse
from pulseseq.sequencing.waveform import Apodization, Shape
from typing import Mapping, Any

# =============================================================================
# PyTree registration
# =============================================================================
#
# Important design choices:
#
# 1. Numerical pulse parameters are dynamic PyTree leaves.
#    Therefore, they can be traced, differentiated, stacked, and vmapped.
#
# 2. "name" is intentionally excluded.
#    It is human-readable metadata and should not affect JIT compilation.
#    Objects reconstructed internally by JAX therefore have name="".
#
# 3. tree_unflatten bypasses __init__.
#    Constructors often perform Python float(), bool(), enum conversion,
#    validation, and NumPy operations that cannot accept JAX tracers.
#
# 4. Waveform itself does not need registration because it is abstract.
#    JAX registration is type-specific and does not automatically propagate
#    from a parent class to its subclasses.
# =============================================================================


def _new_without_init(cls, **attributes):
    """Construct an object without calling its initializer."""
    obj = object.__new__(cls)

    for attribute_name, value in attributes.items():
        setattr(obj, attribute_name, value)

    return obj


# -----------------------------------------------------------------------------
# AnalogPulse
# -----------------------------------------------------------------------------


def _flatten_analog_pulse(pulse):
    """Flatten AnalogPulse into JAX-compatible numerical leaves."""

    children = (
        jnp.asarray(pulse.length),
        jnp.asarray(pulse.apodization, dtype=jnp.int32),
        jnp.asarray(pulse.apodization_length),
        jnp.asarray(pulse.padding_length),
        jnp.asarray(pulse.shift),
        jnp.asarray(pulse.amplitude),
        jnp.asarray(pulse.frequency),
        jnp.asarray(pulse.frequency_chirp),
        jnp.asarray(pulse.shape, dtype=jnp.int32),
        jnp.asarray(pulse.S21_correct, dtype=jnp.bool_),
        jnp.asarray(pulse.w_3db),
        jnp.asarray(pulse.phase_offset),
    )

    # No static auxiliary data are required.
    #
    # In particular, pulse.name is deliberately omitted because different
    # names should not create different PyTree structures or JIT compilations.
    auxiliary_data = None

    return children, auxiliary_data


def _unflatten_analog_pulse(auxiliary_data, children):
    """Reconstruct AnalogPulse without running AnalogPulse.__init__."""

    (
        length,
        apodization,
        apodization_length,
        padding_length,
        shift,
        amplitude,
        frequency,
        frequency_chirp,
        shape,
        S21_correct,
        w_3db,
        phase_offset,
    ) = children

    return _new_without_init(
        AnalogPulse,
        name="",
        length=length,
        apodization=apodization,
        apodization_length=apodization_length,
        padding_length=padding_length,
        shift=shift,
        amplitude=amplitude,
        frequency=frequency,
        frequency_chirp=frequency_chirp,
        shape=shape,
        S21_correct=S21_correct,
        w_3db=w_3db,
        phase_offset=phase_offset,
    )


# -----------------------------------------------------------------------------
# DigitalPulse
# -----------------------------------------------------------------------------


def _flatten_digital_pulse(pulse):
    """Flatten DigitalPulse into JAX-compatible numerical leaves."""

    children = (
        jnp.asarray(pulse.length),
        jnp.asarray(pulse.pulse, dtype=jnp.int32),
        jnp.asarray(pulse.pulse_length),
        jnp.asarray(pulse.pulse_max_frac),
    )

    auxiliary_data = None
    return children, auxiliary_data


def _unflatten_digital_pulse(auxiliary_data, children):
    """Reconstruct DigitalPulse without running DigitalPulse.__init__."""

    (
        length,
        pulse_type,
        pulse_length,
        pulse_max_frac,
    ) = children

    return _new_without_init(
        DigitalPulse,
        name="",
        length=length,
        pulse=pulse_type,
        pulse_length=pulse_length,
        pulse_max_frac=pulse_max_frac,
    )


# -----------------------------------------------------------------------------
# CompositeWaveform
# -----------------------------------------------------------------------------


def _flatten_composite_waveform(composite):
    """Flatten a CompositeWaveform, including its nested waveforms."""

    if isinstance(composite.waveforms, dict):
        waveform_container = tuple(composite.waveforms.values())

        # Dictionary keys describe structure rather than numerical data.
        container_metadata = (
            "dict",
            tuple(composite.waveforms.keys()),
        )

    elif isinstance(composite.waveforms, tuple):
        waveform_container = tuple(composite.waveforms)
        container_metadata = ("tuple", None)

    else:
        waveform_container = tuple(composite.waveforms)
        container_metadata = ("list", None)

    children = (
        waveform_container,
        jnp.asarray(composite.times),
        jnp.asarray(composite.length),
        jnp.asarray(composite.center),
    )

    # Container type and dictionary keys are static structural information.
    # The human-readable name remains excluded.
    auxiliary_data = container_metadata

    return children, auxiliary_data


def _unflatten_composite_waveform(auxiliary_data, children):
    """Reconstruct CompositeWaveform without running its initializer."""

    container_kind, dictionary_keys = auxiliary_data

    (
        waveform_container,
        times,
        length,
        center,
    ) = children

    if container_kind == "dict":
        waveforms = dict(zip(dictionary_keys, waveform_container))
    elif container_kind == "tuple":
        waveforms = tuple(waveform_container)
    else:
        waveforms = list(waveform_container)

    return _new_without_init(
        CompositeWaveform,
        name="",
        waveforms=waveforms,
        times=times,
        length=length,
        center=center,
    )


# -----------------------------------------------------------------------------
# Register every concrete waveform class.
# -----------------------------------------------------------------------------


def _register_pytree_once(cls, flatten_function, unflatten_function):
    """Register a PyTree class while tolerating repeated notebook execution."""

    try:
        jax.tree_util.register_pytree_node(
            cls,
            flatten_function,
            unflatten_function,
        )
    except ValueError as error:
        # This commonly happens when the registration cell is executed twice
        # without redefining the classes.
        if "Duplicate custom PyTreeDef type registration" not in str(error):
            raise


_register_pytree_once(
    AnalogPulse,
    _flatten_analog_pulse,
    _unflatten_analog_pulse,
)

_register_pytree_once(
    DigitalPulse,
    _flatten_digital_pulse,
    _unflatten_digital_pulse,
)

_register_pytree_once(
    CompositeWaveform,
    _flatten_composite_waveform,
    _unflatten_composite_waveform,
)


# =============================================================================
# PyTree utilities
# =============================================================================


def check_waveform_pytree(waveform):
    """Check that a waveform can be flattened and reconstructed.

    Parameters
    ----------
    waveform : AnalogPulse, DigitalPulse, or CompositeWaveform
        Waveform object to test.

    Returns
    -------
    rebuilt : Waveform
        Waveform reconstructed from its PyTree leaves.

    leaves : list
        Numerical leaves extracted from the waveform.

    tree_definition : PyTreeDef
        Static description of the waveform's nested structure.
    """
    leaves, tree_definition = jax.tree_util.tree_flatten(waveform)
    rebuilt = jax.tree_util.tree_unflatten(tree_definition, leaves)

    return rebuilt, leaves, tree_definition


def stack_waveforms(waveforms):
    """Stack homogeneous waveforms into one batched waveform PyTree.

    Parameters
    ----------
    waveforms : sequence of Waveform
        Waveforms with identical PyTree structures. For simple pulses, this
        normally means every waveform must have the same concrete class. For
        composites, every waveform must also contain the same number and types
        of component waveforms.

    Returns
    -------
    batched_waveform : Waveform
        A single waveform object whose numerical fields have an additional
        leading batch dimension.

    Notes
    -----
    A Python list of pulse objects is a PyTree, but its list entries are part
    of the static container structure. ``jax.vmap`` does not automatically
    interpret that list as a batch axis. This function converts the list into
    one object with batched numerical leaves.
    """
    waveforms = tuple(waveforms)

    if len(waveforms) == 0:
        raise ValueError("At least one waveform is required.")

    reference_structure = jax.tree_util.tree_structure(waveforms[0])

    for index, waveform in enumerate(waveforms[1:], start=1):
        waveform_structure = jax.tree_util.tree_structure(waveform)

        if waveform_structure != reference_structure:
            raise ValueError(
                "All waveforms must have identical PyTree structures. "
                f"Waveform 0 and waveform {index} have different structures. "
                "AnalogPulse, DigitalPulse, and differently structured "
                "CompositeWaveform objects cannot be stacked together."
            )

    return jax.tree_util.tree_map(
        lambda *values: jnp.stack(
            tuple(jnp.asarray(value) for value in values),
            axis=0,
        ),
        *waveforms,
    )

def make_analog_pulse_time_array(
    pulse: AnalogPulse,
    sample_period: float = 4e-11,
    at_time: float = 0.0,
) -> jax.Array:
    """Construct a fixed-shape time array for an analog pulse.

    This helper runs outside ``jax.jit`` because the number of samples depends
    on the numerical values of the pulse duration and sample period.

    Parameters
    ----------
    pulse : AnalogPulse
        Serialized analog-pulse parameters. The total pulse duration is

        ``pulse.length + pulse.padding_length``.

    sample_period : float, default=4e-11
        Time between adjacent samples, in seconds.

    at_time : float, default=0.0
        Time assigned to the center of the returned sample array, in seconds.

    Returns
    -------
    tau : jax.Array
        One-dimensional array containing the sample times, in seconds.
    """
    total_length = (
        float(pulse.length)
        + float(pulse.padding_length)
    )

    if sample_period <= 0:
        raise ValueError("sample_period must be positive.")

    number_of_samples = int(jnp.round(total_length / sample_period))

    if number_of_samples < 2:
        raise ValueError(
            "At least two time samples are required. Increase the pulse "
            "duration or decrease sample_period."
        )

    tau = jnp.arange(number_of_samples, dtype=jnp.float64) * sample_period
    tau = tau + at_time - jnp.mean(tau)
    return tau


def _triangle_carrier(argument: jax.Array) -> jax.Array:
    """Evaluate the triangular equivalent of ``-sawtooth(argument, 0.5)``."""
    cycle_fraction = jnp.mod(argument / (2.0 * jnp.pi), 1.0)
    return 4.0 * jnp.abs(cycle_fraction - 0.5) - 1.0


def _analytic_signal_envelope(signal: jax.Array) -> jax.Array:
    """Calculate a Hilbert-transform envelope using the FFT."""
    number_of_samples = signal.shape[0]

    if number_of_samples % 2 == 0:
        multiplier = jnp.concatenate(
            (
                jnp.ones((1,), dtype=signal.dtype),
                2.0 * jnp.ones(
                    (number_of_samples // 2 - 1,),
                    dtype=signal.dtype,
                ),
                jnp.ones((1,), dtype=signal.dtype),
                jnp.zeros(
                    (number_of_samples // 2 - 1,),
                    dtype=signal.dtype,
                ),
            )
        )
    else:
        multiplier = jnp.concatenate(
            (
                jnp.ones((1,), dtype=signal.dtype),
                2.0 * jnp.ones(
                    ((number_of_samples - 1) // 2,),
                    dtype=signal.dtype,
                ),
                jnp.zeros(
                    ((number_of_samples - 1) // 2,),
                    dtype=signal.dtype,
                ),
            )
        )

    analytic_signal = jnp.fft.ifft(jnp.fft.fft(signal) * multiplier)
    return jnp.abs(analytic_signal)


@jax.jit(static_argnames=("all_info",))
def synthesize_analog_pulse(
    pulse: AnalogPulse,
    tau: jax.Array,
    at_time: float | None = None,
    all_info: bool = False,
    dphase: float = 0.0,
):
    """Synthesize an analog pulse from serialized pulse parameters.

    Parameters
    ----------
    pulse : AnalogPulse
    tau : array_like
        One-dimensional array of absolute sample times, in seconds.

        Unlike the original NumPy method, this JIT-compiled function requires
        ``tau`` to be an array rather than a scalar sample period. JAX needs
        the number of output samples to be known when the function is
        compiled. Use ``make_analog_pulse_time_array`` outside ``jax.jit`` to
        create ``tau`` from a sample period.

    at_time : float or None, default=None
        Reference time corresponding to the nominal center of the pulse, in
        seconds. The actual envelope center is

        ``at_time + pickled_pulse["shift"]``.

        For a coherent pulse, this establishes the absolute phase reference.
        For an incoherent pulse, the carrier phase is referenced locally to
        ``at_time``. When ``None``, the mean of ``tau`` is used.

    all_info : bool, default=False
        If ``False``, return only the waveform. If ``True``, return the tuple
        ``(waveform, envelope, tau)``.

        This argument is static because changing it changes the return
        structure and therefore triggers a separate JAX compilation.

    dphase : float, default=0.0
        Additional carrier phase shift, in radians. Exponential pulses add
        ``pi / 2`` to this value, matching the original implementation.

    Returns
    -------
    waveform : jax.Array
        Synthesized waveform samples.

    envelope : jax.Array
        Pulse-envelope samples. Returned only when ``all_info=True``.

    tau : jax.Array
        Input time array. Returned only when ``all_info=True``.

    Notes
    -----
    The dictionary keys and their nested structure must remain unchanged
    between calls to avoid unnecessary JAX recompilation.
    """
    tau = jnp.asarray(tau)
    resolved_at_time = jnp.mean(tau) if at_time is None else jnp.asarray(at_time)

    length = pulse.length
    apodization = pulse.apodization
    apodization_length = pulse.apodization_length
    shift = pulse.shift
    amplitude = pulse.amplitude
    frequency = pulse.frequency
    frequency_chirp = pulse.frequency_chirp
    shape = pulse.shape
    s21_correct = pulse.S21_correct
    w_3db = pulse.w_3db
    phase_offset = pulse.phase_offset
    dphase = dphase

    center = resolved_at_time + shift
    dt = jnp.mean(jnp.diff(tau))
    start = center - length / 2.0
    end = center + length / 2.0

    def cosine_envelope(_):
        half_apodization = apodization_length / 2.0

        # Avoid division by zero in the unselected JAX branch.
        safe_apodization_length = jnp.where(
            apodization_length == 0.0,
            1.0,
            apodization_length,
        )

        start_ramp = (
            0.5
            + 0.5
            * jnp.sin(
                jnp.pi
                * (tau - start)
                / safe_apodization_length
            )
        )
        end_ramp = (
            0.5
            - 0.5
            * jnp.sin(
                jnp.pi
                * (tau - end)
                / safe_apodization_length
            )
        )

        envelope = jnp.zeros_like(tau)
        envelope = jnp.where(
            (tau >= start - half_apodization)
            & (tau < start + half_apodization),
            start_ramp,
            envelope,
        )
        envelope = jnp.where(
            (tau >= start + half_apodization)
            & (tau < end - half_apodization),
            1.0,
            envelope,
        )
        envelope = jnp.where(
            (tau >= end - half_apodization)
            & (tau < end + half_apodization),
            end_ramp,
            envelope,
        )
        return envelope

    def gaussian_envelope(_):
        safe_length = jnp.where(length == 0.0, 1.0, length)
        envelope = jnp.exp(
            -jnp.square(2.0 * (tau - center) / safe_length)
        )
        return jnp.where(length == 0.0, 0.0, envelope)

    def square_envelope(_):
        envelope = jnp.ones_like(tau)
        envelope = jnp.where(tau < start - dt, 0.0, envelope)
        envelope = jnp.where(tau > end, 0.0, envelope)
        return envelope

    def exponential_envelope(_):
        safe_length = jnp.where(length == 0.0, 1.0, length)
        exponential_start = center - length / 2.0
        local_time = tau - exponential_start

        envelope = jnp.exp(-local_time / safe_length)
        envelope = jnp.where(local_time < 0.0, 0.0, envelope)
        return jnp.where(length == 0.0, 0.0, envelope)

    envelope = jax.lax.switch(
        apodization,
        (
            cosine_envelope,
            gaussian_envelope,
            square_envelope,
            exponential_envelope,
        ),
        operand=None,
    )

    envelope = amplitude * envelope

    is_exponential = apodization == Apodization.EXPONENTIAL
    effective_dphase = dphase + jnp.where(
        is_exponential,
        jnp.pi / 2.0,
        0.0,
    )

    has_carrier = (frequency != 0.0) | (frequency_chirp != 0.0)
    is_coherent = (
        (~jnp.isnan(phase_offset))
        | (frequency_chirp != 0.0)
    ) & (amplitude != 0.0)

    coherent_reference = jnp.where(
        is_coherent,
        0.0,
        resolved_at_time,
    )

    safe_length = jnp.where(length == 0.0, 1.0, length)

    instantaneous_frequency = (
        frequency
        + frequency_chirp
        * (tau - center)
        / (2.0 * safe_length)
    )

    phase_reference = jnp.where(
        frequency_chirp != 0.0,
        center,
        coherent_reference,
    )

    finite_phase_offset = jnp.where(
        jnp.isnan(phase_offset),
        0.0,
        phase_offset,
    )

    argument = (
        2.0
        * jnp.pi
        * instantaneous_frequency
        * (tau - phase_reference)
        + effective_dphase
        + finite_phase_offset
    )

    sinusoidal_carrier = jnp.cos(argument)
    triangular_carrier = _triangle_carrier(argument)

    carrier = jax.lax.switch(
        shape,
        (
            lambda _: sinusoidal_carrier,
            lambda _: triangular_carrier,
        ),
        operand=None,
    )

    waveform = jnp.where(
        has_carrier,
        envelope * carrier,
        envelope,
    )

    # Protect against division by zero in the unselected branch.
    safe_w_3db = jnp.where(w_3db == 0.0, 1.0, w_3db)
    corrected_waveform = (
        waveform
        + jnp.gradient(waveform, dt) / safe_w_3db
    )
    corrected_waveform /= jnp.max(jnp.abs(corrected_waveform))
    corrected_envelope = _analytic_signal_envelope(
        corrected_waveform
    )
    corrected_envelope /= jnp.max(jnp.abs(corrected_envelope))

    apply_correction = s21_correct & (w_3db != 0.0)

    waveform = jnp.where(
        apply_correction,
        corrected_waveform,
        waveform,
    )
    envelope = jnp.where(
        apply_correction,
        corrected_envelope,
        envelope,
    )

    if all_info:
        return waveform, envelope, tau

    return waveform
