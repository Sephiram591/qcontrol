"""JAX-compatible physics helpers for one :class:`SnVParticle`.

Every public physics helper operates on one unbatched particle. Helpers that
model realized controls also take one explicit :class:`SnVControlState`, making
them directly compatible with ``jax.vmap(..., in_axes=(0, None))``. The two
optimization targets are explicit arguments only to ``get_B_settings``,
``get_waveplate_angles``, and ``get_optimal_control_state``.
"""

from __future__ import annotations

from collections import deque
from functools import partial
from typing import Literal

from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp_special
import jaxquantum as jqt
import numpy as np

from . import hamiltonian_jqt as qh_jqt
from . import parameters as params
from .jqt_ext import mesolve_components, sesolve_components
from .particle import SnVControlState, SnVParticle
from .pulseseq_interconnect import (
    make_analog_pulse_time_array,
    make_composite_waveform_time_array,
    synthesize_analog_pulse,
    synthesize_composite_waveform
)


Array = jax.Array
Frame = Literal["lab", "crystal", "dipole"]


# -----------------------------------------------------------------------------
# General numerical helpers
# -----------------------------------------------------------------------------


def lowpass_filter(omega_c, omega_r=0.0):
    """Construct a first-order low-pass filter.

    Parameters
    ----------
    omega_c
        Angular cutoff frequency, in rad/s.
    omega_r
        Rotating-frame angular frequency, in rad/s.

    Returns
    -------
    callable
        Filter compatible with ``sesolve_components``.
    """
    omega_c = jnp.asarray(omega_c)

    def filter_fn(t, z, u):
        return omega_c * (u - z), jnp.exp(-1j * omega_r * t) * z

    return filter_fn


def eom_lowpass_filter(omega_c, omega_r=0.0, Vpi_ratio=1.0):
    """Construct a low-pass filter followed by EOM phase modulation.

    Parameters
    ----------
    omega_c
        Angular cutoff frequency, in rad/s.
    omega_r
        Rotating-frame angular frequency, in rad/s.
    Vpi_ratio
        EOM drive amplitude in units of ``Vmax/Vpi``.

    Returns
    -------
    callable
        Filter compatible with ``mesolve_components``.
    """
    omega_c = jnp.asarray(omega_c)

    def filter_fn(t, z, u):
        return (
            omega_c * (u - z),
            jnp.exp(1j * (jnp.pi * Vpi_ratio * z - omega_r * t)),
        )

    return filter_fn


def cartesian_to_angles(vector):
    """Convert Cartesian vectors to polar and azimuthal angles.

    Parameters
    ----------
    vector : array_like, shape (..., 3)
        Cartesian vector or vectors.

    Returns
    -------
    theta, phi : jax.Array
        Polar and azimuthal angles, in radians.
    """
    vector = jnp.asarray(vector)
    theta = jnp.arctan2(jnp.linalg.norm(vector[..., :2], axis=-1), vector[..., 2])
    phi = jnp.arctan2(vector[..., 1], vector[..., 0])
    return theta, phi


def angles_to_cartesian(theta, phi):
    """Convert polar and azimuthal angles to Cartesian unit vectors."""
    return jnp.stack(
        (
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta),
        ),
        axis=-1,
    )


def normalize_vectors(vector):
    """Normalize vectors along their final Cartesian-component axis."""
    vector = jnp.asarray(vector)
    return vector / jnp.linalg.norm(vector, axis=-1, keepdims=True)


def _dipole_basis_crystal_from_axis(dipole_z_crystal):
    """Construct the paper-compatible local SnV frame.

    Parameters
    ----------
    dipole_z_crystal : array_like, shape (..., 3)
        Defect-axis direction in crystal coordinates.

    Returns
    -------
    jax.Array, shape (..., 3, 3)
        Matrix whose columns are local X, Y, and Z expressed in crystal
        coordinates. For a [111] defect this gives
        ``X=[2,-1,-1]/sqrt(6)``, ``Y=[0,1,-1]/sqrt(2)``, and
        ``Z=[1,1,1]/sqrt(3)``.
    """
    dipole_z_crystal = normalize_vectors(dipole_z_crystal)
    crystal_x = jnp.asarray([1.0, 0.0, 0.0], dtype=dipole_z_crystal.dtype)
    dipole_x_crystal = normalize_vectors(
        crystal_x
        - jnp.sum(crystal_x * dipole_z_crystal, axis=-1, keepdims=True)
        * dipole_z_crystal
    )
    dipole_y_crystal = normalize_vectors(
        jnp.cross(dipole_z_crystal, dipole_x_crystal)
    )
    return jnp.stack(
        (dipole_x_crystal, dipole_y_crystal, dipole_z_crystal), axis=-1
    )


@partial(jax.jit, static_argnums=(1, 2))
def _bessel_j_nonnegative_orders_integer_series(
    x,
    max_order: int,
    series_terms: int = 48,
):
    """Evaluate integer-order Bessel functions using a pure-JAX series.

    Parameters
    ----------
    x : array_like
        Bessel-function argument.
    max_order : int
        Largest nonnegative order. Must be static under JIT.
    series_terms : int
        Number of power-series terms. Must be static under JIT.

    Returns
    -------
    jax.Array, shape (max_order + 1, *x.shape)
        ``J_n(x)`` for ``n = 0, ..., max_order``.
    """
    x = jnp.asarray(x)
    orders = jnp.arange(max_order + 1, dtype=x.dtype)[:, None]
    k = jnp.arange(series_terms, dtype=x.dtype)[None, :]
    power = 2.0 * k + orders
    sign = jnp.where(jnp.arange(series_terms)[None, :] % 2 == 0, 1.0, -1.0)
    coeff = sign * jnp.exp(
        -jsp_special.gammaln(k + 1.0)
        - jsp_special.gammaln(k + orders + 1.0)
    )
    terms = (
        coeff[(...,) + (None,) * x.ndim]
        * (0.5 * x[None, None, ...]) ** power[(...,) + (None,) * x.ndim]
    )
    return jnp.sum(terms, axis=1)


def _validate_control_state(control_state: SnVControlState) -> SnVControlState:
    """Normalize and validate one physical control state."""
    if not isinstance(control_state, SnVControlState):
        raise TypeError("`control_state` must be an SnVControlState.")
    magnet_settings = jnp.asarray(control_state.magnet_settings)
    waveplate_angles = jnp.asarray(control_state.waveplate_angles)
    if magnet_settings.shape != (3,):
        raise ValueError("`control_state.magnet_settings` must have shape (3,).")
    if waveplate_angles.shape != (3,):
        raise ValueError("`control_state.waveplate_angles` must have shape (3,).")
    return SnVControlState(magnet_settings, waveplate_angles)


def _dipole_to_crystal(particle: SnVParticle, dtype) -> Array:
    n = particle.nondiff
    axis = jnp.asarray(n.dipole_crystal_axes, dtype=dtype)[
        n.dipole_crystal_axis_idx
    ]
    return _dipole_basis_crystal_from_axis(axis)


def _mode_vectors_dipole(particle: SnVParticle) -> Array:
    orientation = particle.diffable.mode_field_orientation
    mode_vectors_crystal = angles_to_cartesian(
        orientation[:, 0], orientation[:, 1]
    )
    return mode_vectors_crystal @ _dipole_to_crystal(
        particle, mode_vectors_crystal.dtype
    )


def _waveplate_jones(theta, delta) -> Array:
    c, s = jnp.cos(theta), jnp.sin(theta)
    phase_delay = jnp.exp(1j * jnp.asarray(delta, dtype=jnp.asarray(theta).dtype))
    return jnp.stack(
        (
            jnp.stack((c**2 + phase_delay * s**2, c * s * (1.0 - phase_delay))),
            jnp.stack((c * s * (1.0 - phase_delay), s**2 + phase_delay * c**2)),
        )
    )


def _source_jones(particle: SnVParticle) -> Array:
    d = particle.diffable
    source = jnp.stack(
        (
            jnp.cos(d.resonant_pump_polarization),
            jnp.exp(1j * d.resonant_pump_phase)
            * jnp.sin(d.resonant_pump_polarization),
        )
    )
    return source / jnp.linalg.norm(source)


def _stokes(jones: Array) -> Array:
    ex, ey = jones[0], jones[1]
    exey = ex * jnp.conj(ey)
    return jnp.stack(
        (jnp.abs(ex) ** 2 - jnp.abs(ey) ** 2, 2 * jnp.real(exey), 2 * jnp.imag(exey))
    )


# -----------------------------------------------------------------------------
# Coordinate frames and optimal controls
# -----------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("frame",))
def get_magnet_axes(particle: SnVParticle, frame: Frame = "lab") -> Array:
    """Return the three calibrated magnet-axis directions.

    Parameters
    ----------
    particle
        Unbatched particle parameters.
    frame : {'lab', 'crystal', 'dipole'}
        Output coordinate frame.

    Returns
    -------
    jax.Array, shape (3, 3)
        Rows index physical magnet axes; columns are Cartesian components.
    """
    if frame not in ("lab", "crystal", "dipole"):
        raise ValueError("`frame` must be 'lab', 'crystal', or 'dipole'.")

    d, n = particle.diffable, particle.nondiff
    rotation_vectors = jnp.asarray(d.magnet_axes_rotations)
    nominal_axes = jnp.asarray(n.nominal_magnet_axes, dtype=rotation_vectors.dtype)
    if rotation_vectors.shape != (3, 3) or nominal_axes.shape != (3, 3):
        raise ValueError("Magnet rotations and nominal axes must both have shape (3, 3).")

    axes_lab = normalize_vectors(nominal_axes)
    angle = jnp.linalg.norm(rotation_vectors, axis=-1)
    r_cross_v = jnp.cross(rotation_vectors, axes_lab)
    axes_lab = normalize_vectors(
        axes_lab
        + jnp.sinc(angle / jnp.pi)[:, None] * r_cross_v
        + 0.5
        * jnp.sinc(angle / (2.0 * jnp.pi))[:, None] ** 2
        * jnp.cross(rotation_vectors, r_cross_v)
    )
    if frame == "lab":
        return axes_lab

    lattice_100 = jnp.asarray(
        n.diamond_lattice_100_orientation, dtype=axes_lab.dtype
    )
    lattice_011 = jnp.asarray(
        n.diamond_lattice_011_orientation, dtype=axes_lab.dtype
    )
    if lattice_100.shape != (2,) or lattice_011.shape != (2,):
        raise ValueError("Diamond-lattice orientations must have shape (2,).")

    crystal_x_lab = normalize_vectors(angles_to_cartesian(*lattice_100))
    lattice_011_lab = angles_to_cartesian(*lattice_011)
    lattice_011_lab = normalize_vectors(
        lattice_011_lab - jnp.dot(lattice_011_lab, crystal_x_lab) * crystal_x_lab
    )
    crystal_z_minus_y_lab = normalize_vectors(
        jnp.cross(crystal_x_lab, lattice_011_lab)
    )
    crystal_y_lab = normalize_vectors(lattice_011_lab - crystal_z_minus_y_lab)
    crystal_z_lab = normalize_vectors(lattice_011_lab + crystal_z_minus_y_lab)
    crystal_to_lab = jnp.stack(
        (crystal_x_lab, crystal_y_lab, crystal_z_lab), axis=-1
    )
    axes_crystal = axes_lab @ crystal_to_lab
    if frame == "crystal":
        return axes_crystal
    return axes_crystal @ _dipole_to_crystal(particle, axes_crystal.dtype)


@jax.jit
def get_B_settings(particle: SnVParticle, B_target: Array) -> Array:
    """Calculate magnet settings that realize a target dipole-frame field.

    Parameters
    ----------
    particle
        Unbatched particle parameters.
    B_target : array_like, shape (3,)
        Target electron-Zeeman vector in the dipole frame, in GHz.

    Returns
    -------
    jax.Array, shape (3,)
        Physical vector-magnet settings, in tesla.
    """
    d, n = particle.diffable, particle.nondiff
    axes = get_magnet_axes(particle, frame="dipole")
    calibration = axes.T * d.magnet_unit_magnitude[None, :]
    B_target = jnp.asarray(B_target, dtype=calibration.dtype)
    if B_target.shape != (3,):
        raise ValueError("`B_target` must have shape (3,).")
    target_tesla = B_target / (n.mu_B_GHz_per_T * params.gS)
    return jnp.linalg.solve(calibration, target_tesla)


@jax.jit
def get_waveplate_angles(
    particle: SnVParticle,
    target_dipole_operator: Array,
) -> Array:
    """Calculate QWP-HWP-QWP angles for a target dipole operator.

    The target is orthogonally projected onto the reachable TE/TM mode span.
    PDL is then inverted at the field-amplitude level before an analytic
    QWP-HWP-QWP transformation is constructed.

    Parameters
    ----------
    particle
        Unbatched particle parameters.
    target_dipole_operator : array_like, shape (3,)
        Desired operator direction in the local dipole frame. Its magnitude is
        ignored.

    Returns
    -------
    jax.Array, shape (3,)
        ``[QWP1, HWP, QWP2]`` angles, in radians.
    """
    d = particle.diffable
    source = _source_jones(particle)
    mode_basis = _mode_vectors_dipole(particle).T
    target = jnp.asarray(target_dipole_operator)
    if target.shape != (3,):
        raise ValueError("`target_dipole_operator` must have shape (3,).")

    dtype = jnp.result_type(source.dtype, mode_basis.dtype, target.dtype)
    source, mode_basis, target = (
        source.astype(dtype),
        mode_basis.astype(dtype),
        target.astype(dtype),
    )
    target_norm = jnp.linalg.norm(target)
    target_direction = target / jnp.where(target_norm > 0, target_norm, 1.0)

    mode_amplitude = jnp.stack(
        (jnp.ones_like(d.resonant_pump_pdl), jnp.sqrt(10.0 ** (-d.resonant_pump_pdl / 10.0)))
    )
    mode_available = mode_amplitude > 0
    reachable_basis = mode_basis * mode_available[None, :].astype(dtype)
    amplitudes_after_pdl = jnp.linalg.pinv(reachable_basis) @ target_direction
    projected_target = reachable_basis @ amplitudes_after_pdl

    safe_mode_amplitude = jnp.where(mode_available, mode_amplitude, 1.0).astype(dtype)
    target_before_pdl = jnp.where(
        mode_available,
        amplitudes_after_pdl / safe_mode_amplitude,
        jnp.zeros_like(amplitudes_after_pdl),
    )
    before_norm = jnp.linalg.norm(target_before_pdl)
    projected_norm = jnp.linalg.norm(projected_target)
    tolerance = 32.0 * jnp.finfo(jnp.real(target_before_pdl).dtype).eps
    reachable = (
        (target_norm > 0)
        & (projected_norm > tolerance)
        & (before_norm > 0)
        & jnp.isfinite(before_norm)
    )
    target_jones = jnp.where(
        reachable,
        target_before_pdl / jnp.where(reachable, before_norm, 1.0),
        source,
    )

    s1, s2, s3 = _stokes(source)
    t1, t2, t3 = _stokes(target_jones)

    two_qwp1 = jnp.arctan2(s2, s1)
    qwp1 = 0.5 * two_qwp1
    nx, ny = jnp.cos(two_qwp1), jnp.sin(two_qwp1)
    projection = nx * s1 + ny * s2
    phi_linear_in = jnp.arctan2(
        ny * projection + nx * s3,
        nx * projection - ny * s3,
    )

    two_qwp2 = jnp.arctan2(t2, t1)
    qwp2 = 0.5 * two_qwp2
    nx, ny = jnp.cos(two_qwp2), jnp.sin(two_qwp2)
    projection = nx * t1 + ny * t2
    phi_linear_target = jnp.arctan2(
        ny * projection - nx * t3,
        nx * projection + ny * t3,
    )
    hwp = 0.25 * (phi_linear_target + phi_linear_in)
    return jnp.stack((qwp1 % jnp.pi, hwp % (0.5 * jnp.pi), qwp2 % jnp.pi))


@jax.jit
def get_optimal_control_state(
    particle: SnVParticle,
    B_target: Array,
    target_dipole_operator: Array,
) -> SnVControlState:
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
    return SnVControlState(
        magnet_settings=get_B_settings(particle, B_target),
        waveplate_angles=get_waveplate_angles(particle, target_dipole_operator),
    )


# -----------------------------------------------------------------------------
# Realized magnetic and optical fields
# -----------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("frame",))
def get_B_cartesian(
    particle: SnVParticle,
    control_state: SnVControlState,
    frame: Frame = "lab",
) -> Array:
    """Return the realized static magnetic-field vector, in tesla."""
    control_state = _validate_control_state(control_state)
    channel_fields = (
        control_state.magnet_settings * particle.diffable.magnet_unit_magnitude
    )
    return jnp.sum(channel_fields[:, None] * get_magnet_axes(particle, frame), axis=0)


@partial(jax.jit, static_argnames=("frame",))
def get_B_spherical(
    particle: SnVParticle,
    control_state: SnVControlState,
    frame: Frame = "dipole",
):
    """Return electron-Zeeman magnitude and field angles.

    Returns
    -------
    magnitude_GHz, theta, phi : jax.Array
        Electron-Zeeman magnitude in GHz and direction angles in radians.
    """
    B_tesla = get_B_cartesian(particle, control_state, frame)
    theta, phi = cartesian_to_angles(B_tesla)
    magnitude = (
        jnp.linalg.norm(B_tesla) * particle.nondiff.mu_B_GHz_per_T * params.gS
    )
    return magnitude, theta, phi


@jax.jit
def get_dipole_B_GHz(
    particle: SnVParticle, control_state: SnVControlState
) -> Array:
    """Return the electron-Zeeman field magnitude, in GHz."""
    return get_B_spherical(particle, control_state, frame="dipole")[0]


@partial(jax.jit, static_argnames=("frame",))
def get_B_theta(
    particle: SnVParticle,
    control_state: SnVControlState,
    frame: Frame = "dipole",
) -> Array:
    """Return the field polar angle in the selected frame, in radians."""
    return get_B_spherical(particle, control_state, frame)[1]


@partial(jax.jit, static_argnames=("frame",))
def get_B_phi(
    particle: SnVParticle,
    control_state: SnVControlState,
    frame: Frame = "dipole",
) -> Array:
    """Return the field azimuthal angle in the selected frame, in radians."""
    return get_B_spherical(particle, control_state, frame)[2]


@jax.jit
def get_resonant_pump_eta(
    particle: SnVParticle,
    control_state: SnVControlState,
) -> Array:
    """Return the complex resonant-pump coupling in the dipole frame.

    Returns
    -------
    jax.Array, shape (3,)
        Complex Cartesian coupling components, in GHz.
    """
    control_state = _validate_control_state(control_state)
    polarization = _source_jones(particle)
    qwp1, hwp, qwp2 = control_state.waveplate_angles
    for angle, delay in (
        (qwp1, 0.5 * jnp.pi),
        (hwp, jnp.pi),
        (qwp2, 0.5 * jnp.pi),
    ):
        polarization = _waveplate_jones(angle, delay) @ polarization
    polarization = polarization / jnp.linalg.norm(polarization)

    d = particle.diffable
    mode_amplitude = jnp.stack(
        (jnp.ones_like(d.resonant_pump_pdl), jnp.sqrt(10.0 ** (-d.resonant_pump_pdl / 10.0)))
    )
    mode_couplings = (
        d.resonant_pump_coupling_rate * polarization * mode_amplitude
    )
    return jnp.sum(mode_couplings[:, None] * _mode_vectors_dipole(particle), axis=0)


@jax.jit
def get_resonant_pump_coupling(
    particle: SnVParticle,
    control_state: SnVControlState,
) -> Array:
    """Return the complex resonant-pump coupling in the TE/TM frame.

    Returns
    -------
    jax.Array, shape (3,)
        Complex Cartesian coupling components, in GHz.
    """
    control_state = _validate_control_state(control_state)
    polarization = _source_jones(particle)
    qwp1, hwp, qwp2 = control_state.waveplate_angles
    for angle, delay in (
        (qwp1, 0.5 * jnp.pi),
        (hwp, jnp.pi),
        (qwp2, 0.5 * jnp.pi),
    ):
        polarization = _waveplate_jones(angle, delay) @ polarization
    polarization = polarization / jnp.linalg.norm(polarization)

    d = particle.diffable
    mode_amplitude = jnp.stack(
        (jnp.ones_like(d.resonant_pump_pdl), jnp.sqrt(10.0 ** (-d.resonant_pump_pdl / 10.0)))
    )
    mode_couplings = (
        d.resonant_pump_coupling_rate * polarization * mode_amplitude
    )
    return mode_couplings
# -----------------------------------------------------------------------------
# Static spectra and Hamiltonians
# -----------------------------------------------------------------------------


def _hyperfine_parameters(particle: SnVParticle, excited: bool):
    idx = particle.nondiff.hyperfine_neighbor_idx
    if excited:
        return (
            params.q_exc,
            params.L_exc,
            params.A_EXC_TENSORS[idx],
            params.AX_EXC_TENSORS[idx],
            params.AY_EXC_TENSORS[idx],
            params.delta_f_exc,
        )
    return (
        params.q,
        params.L,
        params.A_GND_TENSORS[idx],
        params.AX_GND_TENSORS[idx],
        params.AY_GND_TENSORS[idx],
        params.delta_f_gnd,
    )


@partial(jax.jit, static_argnames=("ground_state",))
def solve_hamiltonian(
    particle: SnVParticle,
    control_state: SnVControlState,
    ground_state: bool = True,
):
    """Solve the ground- or excited-state static Hamiltonian.

    Parameters
    ----------
    particle
        Unbatched particle parameters.
    control_state
        Applied physical controls. Only ``magnet_settings`` is used.
    ground_state
        Select the ground manifold when true and the excited manifold otherwise.
    """
    d, n = particle.diffable, particle.nondiff
    q, L, A, Ax, Ay, delta_f = _hyperfine_parameters(
        particle, excited=not ground_state
    )
    if ground_state:
        alpha, beta = d.strain_params[1], d.strain_params[2]
    else:
        alpha, beta = d.strain_params[3], d.strain_params[4]
    B, theta, phi = get_B_spherical(particle, control_state, frame="dipole")
    return qh_jqt.solve_hamiltonian(
        B,
        theta,
        phi,
        params.rg[n.hyperfine_neighbor_idx],
        q,
        A,
        Ax,
        Ay,
        L,
        alpha,
        beta,
        0.0,
        delta_f,
    )


@jax.jit
def get_folded_branching_ratios(
    particle: SnVParticle,
    control_state: SnVControlState,
):
    """Return spontaneous-emission branching ratios for one particle."""
    d, n = particle.diffable, particle.nondiff
    idx = n.hyperfine_neighbor_idx
    B, theta, phi = get_B_spherical(particle, control_state, frame="dipole")
    return qh_jqt.calculate_folded_branching_ratios(
        B,
        theta,
        phi,
        alpha=d.strain_params[1],
        beta=d.strain_params[2],
        alpha_exc=d.strain_params[3],
        beta_exc=d.strain_params[4],
        rg=params.rg[idx],
        A_gnd=params.A_GND_TENSORS[idx],
        Ax_gnd=params.AX_GND_TENSORS[idx],
        Ay_gnd=params.AY_GND_TENSORS[idx],
        A_exc=params.A_EXC_TENSORS[idx],
        Ax_exc=params.AX_EXC_TENSORS[idx],
        Ay_exc=params.AY_EXC_TENSORS[idx],
        delta_f_gnd=params.delta_f_gnd,
        delta_f_exc=params.delta_f_exc,
    )


@jax.jit
def PLE_transitions(
    particle: SnVParticle,
    control_state: SnVControlState,
):
    """Return polarization-resolved PLE transition intensities for one particle.

    Parameters
    ----------
    particle
        Unbatched particle parameters.
    control_state
        Applied magnet and QWP-HWP-QWP controls.

    Returns
    -------
    See :func:`hamiltonian_jqt.PLE_transitions`.
    """
    d, n = particle.diffable, particle.nondiff
    idx = n.hyperfine_neighbor_idx
    pump_eta = get_resonant_pump_eta(particle, control_state)
    B, theta, phi = get_B_spherical(particle, control_state, frame="dipole")
    return qh_jqt.PLE_transitions(
        B=B,
        theta=theta,
        phi=phi,
        eta_x=pump_eta[0],
        eta_y=pump_eta[1],
        eta_z=pump_eta[2],
        alpha=d.strain_params[1],
        beta=d.strain_params[2],
        alpha_exc=d.strain_params[3],
        beta_exc=d.strain_params[4],
        rg=params.rg[idx],
        q_gnd=params.q,
        A_gnd=params.A_GND_TENSORS[idx],
        Ax_gnd=params.AX_GND_TENSORS[idx],
        Ay_gnd=params.AY_GND_TENSORS[idx],
        L_gnd=params.L,
        upsilon_gnd=0.0,
        delta_f_gnd=params.delta_f_gnd,
        q_exc=params.q_exc,
        A_exc=params.A_EXC_TENSORS[idx],
        Ax_exc=params.AX_EXC_TENSORS[idx],
        Ay_exc=params.AY_EXC_TENSORS[idx],
        L_exc=params.L_exc,
        upsilon_exc=0.0,
        delta_f_exc=params.delta_f_exc,
    )


@jax.jit
def get_ple_freqs(
    particle: SnVParticle,
    control_state: SnVControlState,
) -> Array:
    """Return the four matched lower-orbital PLE frequencies, in GHz."""
    E_gnd = solve_hamiltonian(particle, control_state, ground_state=True)[0]
    E_exc = solve_hamiltonian(particle, control_state, ground_state=False)[0]
    return E_exc[:4] - E_gnd[:4] + params.LEVEL_OFFSET + particle.diffable.strain_params[0]


@jax.jit
def get_state_frequencies(
    particle: SnVParticle,
    control_state: SnVControlState,
) -> Array:
    """Return the relative frequency of all excited and ground states, with the ground states centered at 0"""
    E_gnd = solve_hamiltonian(particle, control_state, ground_state=True)[0]
    E_exc = solve_hamiltonian(particle, control_state, ground_state=False)[0]
    return E_gnd, E_exc + params.LEVEL_OFFSET + particle.diffable.strain_params[0]


@jax.jit
def get_emr_freqs(
    particle: SnVParticle,
    control_state: SnVControlState,
) -> Array:
    """Return the two electron-magnetic-resonance frequencies, in GHz."""
    E = solve_hamiltonian(particle, control_state, ground_state=True)[0]
    return jnp.stack((E[3] - E[0], E[2] - E[1]))


@jax.jit
def get_nmr_freqs(
    particle: SnVParticle,
    control_state: SnVControlState,
) -> Array:
    """Return the two nuclear-magnetic-resonance frequencies, in GHz."""
    E = solve_hamiltonian(particle, control_state, ground_state=True)[0]
    return jnp.stack((E[1] - E[0], E[3] - E[2]))


def get_init_timestep(
    particle: SnVParticle,
    control_state: SnVControlState,
    electron_state=0,
    nuclear_state=None,
    decay_target=0.001,
):
    """Raise until a physically specified initialization model is provided."""
    del particle, control_state, electron_state, nuclear_state, decay_target
    raise NotImplementedError(
        "`get_init_timestep` needs a specified pumping/decay model before a "
        "physically meaningful time step can be calculated."
    )


@partial(jax.jit, static_argnames=("included_states",))
def get_excitation_hamiltonian(
    particle: SnVParticle,
    control_state: SnVControlState,
    included_states=None,
):
    """Construct the optically driven dynamic Hamiltonian components."""
    d, n = particle.diffable, particle.nondiff
    idx = n.hyperfine_neighbor_idx
    B, theta, phi = get_B_spherical(particle, control_state, frame="dipole")
    pump_eta = get_resonant_pump_eta(particle, control_state)
    return qh_jqt.get_dynamic_hamiltonian(
        B=B,
        theta=theta,
        phi=phi,
        excited_state_lifetime=n.excited_state_lifetime,
        pump_eta_x=pump_eta[0],
        pump_eta_y=pump_eta[1],
        pump_eta_z=pump_eta[2],
        B_drive_strength=d.mw_B_magnitude * n.mu_B_GHz_per_T * params.gS,
        B_drive_theta=d.mw_B_orientation[0],
        B_drive_phi=d.mw_B_orientation[1],
        alpha=d.strain_params[1],
        beta=d.strain_params[2],
        alpha_exc=d.strain_params[3],
        beta_exc=d.strain_params[4],
        rg=params.rg[idx],
        A_gnd=params.A_GND_TENSORS[idx],
        Ax_gnd=params.AX_GND_TENSORS[idx],
        Ay_gnd=params.AY_GND_TENSORS[idx],
        A_exc=params.A_EXC_TENSORS[idx],
        Ax_exc=params.AX_EXC_TENSORS[idx],
        Ay_exc=params.AY_EXC_TENSORS[idx],
        delta_f_gnd=params.delta_f_gnd,
        delta_f_exc=params.delta_f_exc,
        included_states=included_states,
    )


@partial(jax.jit, static_argnames=("included_states",))
def get_ground_hamiltonian(
    particle: SnVParticle,
    control_state: SnVControlState,
    included_states=None,
):
    """Construct the microwave-driven ground-state Hamiltonian components."""
    d, n = particle.diffable, particle.nondiff
    idx = n.hyperfine_neighbor_idx
    B, theta, phi = get_B_spherical(particle, control_state, frame="dipole")
    return qh_jqt.get_ground_hamiltonian(
        B=B,
        theta=theta,
        phi=phi,
        B_drive_strength=d.mw_B_magnitude * n.mu_B_GHz_per_T * params.gS,
        B_drive_theta=d.mw_B_orientation[0],
        B_drive_phi=d.mw_B_orientation[1],
        alpha=d.strain_params[1],
        beta=d.strain_params[2],
        rg=params.rg[idx],
        A=params.A_GND_TENSORS[idx],
        Ax=params.AX_GND_TENSORS[idx],
        Ay=params.AY_GND_TENSORS[idx],
        delta_f=params.delta_f_gnd,
        included_states=included_states,
    )


# -----------------------------------------------------------------------------
# Optical scattering
# -----------------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=("max_bessel_order", "bessel_series_terms"),
)
def scattering_rate(
    particle: SnVParticle,
    control_state: SnVControlState,
    eom_frequency=0.0,
    max_bessel_order: int = 3,
    bessel_series_terms: int = 48,
):
    """Calculate transition- and sideband-resolved scattering rates.

    Parameters
    ----------
    particle
        Unbatched particle parameters.
    control_state
        Applied magnet and QWP-HWP-QWP controls.
    eom_frequency : float or array_like, shape (F,)
        EOM tone frequencies, in GHz. A scalar retains a length-one frequency
        axis.
    max_bessel_order
        Largest positive and negative sideband order.
    bessel_series_terms
        Number of terms in the pure-JAX Bessel series.

    Returns
    -------
    rates : jax.Array, shape (F, S, N_exc, N_gnd)
        Scattering rates in GHz, where ``S = 2*max_bessel_order + 1``.
    branching_ratios : jax.Array, shape (N_exc, N_gnd)
        Spontaneous-emission branching ratios.
    """
    d, n = particle.diffable, particle.nondiff
    eom_frequency = jnp.atleast_1d(jnp.asarray(eom_frequency))
    if eom_frequency.ndim != 1:
        raise ValueError("`eom_frequency` must be scalar or one-dimensional.")

    lifetime = n.excited_state_lifetime
    gamma = 1.0 / lifetime / (2.0 * jnp.pi)

    (
        E_gnd,
        _,
        _,
        _,
        E_exc,
        _,
        _,
        _,
        transition_coupling_squared,
        branching_ratios,
    ) = PLE_transitions(particle, control_state)
    ple_freqs = (
        E_exc[:, None]
        - E_gnd[None, :]
        + params.LEVEL_OFFSET
        + d.strain_params[0]
    )

    filtered_vpi_ratio = d.eom_vpi_ratio / jnp.sqrt(
        1.0 + (eom_frequency / d.eom_vpi_bandwidth) ** 2
    )
    modulation_index = jnp.pi * filtered_vpi_ratio
    sideband_orders = jnp.arange(
        -max_bessel_order, max_bessel_order + 1, dtype=gamma.dtype
    )
    J_nonnegative = _bessel_j_nonnegative_orders_integer_series(
        modulation_index,
        max_order=max_bessel_order,
        series_terms=bessel_series_terms,
    )
    J_sidebands = jnp.moveaxis(
        jnp.take(
            J_nonnegative,
            jnp.abs(sideband_orders).astype(jnp.int32),
            axis=0,
        ),
        0,
        -1,
    )
    sideband_coupling_squared = (
        J_sidebands[..., None, None] ** 2
        * transition_coupling_squared[None, None, :, :]
    )
    sideband_frequencies = (
        n.laser_frequency + eom_frequency[:, None] * sideband_orders[None, :]
    )
    detuning = sideband_frequencies[..., None, None] - ple_freqs[None, None, :, :]
    rates = sideband_coupling_squared / lifetime / (
        gamma**2 + 2.0 * sideband_coupling_squared + 4.0 * detuning**2
    )
    return rates, branching_ratios


# -----------------------------------------------------------------------------
# Time-domain evolution
# -----------------------------------------------------------------------------


def _solver_options(solver_options_args):
    if solver_options_args is None:
        return jqt.SolverOptions.create(
            progress_meter=False,
            solver="Dopri5",
            rtol=1e-5,
            atol=1e-7,
        )
    return jqt.SolverOptions.create(*solver_options_args)


def _populations(states, dimension: int) -> Array:
    return jnp.stack(
        tuple(
            jnp.real(jqt.overlap(jqt.basis(dimension, i).to_dm(), states))
            for i in range(dimension)
        ),
        axis=0,
    )


@partial(
    jax.jit,
    static_argnames=("included_states", "saveat_final_only", "solver_options_args"),
)
def _drive_mw_hamiltonian(
    particle: SnVParticle,
    control_state: SnVControlState,
    pulse,
    tau,
    psi0,
    included_states,
    saveat_final_only=False,
    solver_options_args=None,
):
    scale = 1e9
    dimension = len(included_states)
    sample_period = 1.0 / (particle.nondiff.sampling_rate * scale)
    pulse_center = pulse.length / 2

    def H_b_drive(t, args=None):
        del args
        local_times = jnp.stack((t - sample_period, t, t + sample_period))
        waveform, _, _ = synthesize_analog_pulse(
            pulse=pulse,
            tau=local_times,
            at_time=pulse_center,
            dphase=0.0,
            all_info=True,
        )
        return waveform[1]

    H0, Hb = get_ground_hamiltonian(
        particle, control_state, included_states=included_states
    )
    omega_c = 2.0 * jnp.pi * particle.diffable.mw_B_bandwidth * scale
    states, filter_states = sesolve_components(
        hamiltonians=(2.0 * jnp.pi * H0 * scale, 2.0 * jnp.pi * Hb * scale),
        coefficients=(1.0, H_b_drive),
        psi0=psi0,
        tlist=tau,
        saveat_tlist=tau[-2:] if saveat_final_only else tau,
        filters=(None, lowpass_filter(omega_c)),
        filter_y0s=(None, jnp.asarray(0.0, dtype=jnp.complex128)),
        return_filter_states=True,
        solver_options=_solver_options(solver_options_args),
    )
    return states, filter_states, _populations(states, dimension)


def drive_mw_hamiltonian(
    particle: SnVParticle,
    control_state: SnVControlState,
    pulse,
    included_states=(0, 1, 2, 3),
    psi0=None,
    saveat_final_only=False,
    solver_options_args=None,
):
    """Evolve one particle under a synthesized microwave pulse.

    Parameters
    ----------
    particle, control_state
        Particle parameters and applied controls.
    pulse
        Analog microwave pulse.
    included_states
        Static tuple of retained ground-manifold eigenstates.
    psi0
        Initial ket. Defaults to the first retained basis state.
    saveat_final_only
        Save only the final solver interval when true.
    solver_options_args
        Optional positional arguments for ``SolverOptions.create``.

    Returns
    -------
    states, filter_states, populations
        Solver states, internal filter states, and basis populations.
    """
    included_states = tuple(included_states)
    solver_options_args = (
        None if solver_options_args is None else tuple(solver_options_args)
    )
    if psi0 is None:
        psi0 = jqt.basis(len(included_states), 0)
    scale = 1e9
    sample_period = 1.0 / float(np.asarray(particle.nondiff.sampling_rate) * scale)
    tau = make_analog_pulse_time_array(
        pulse=pulse,
        sample_period=sample_period,
        at_time=float(np.asarray(pulse.length)) / 2.0,
    )
    return _drive_mw_hamiltonian(
        particle,
        control_state,
        pulse,
        tau,
        psi0,
        included_states,
        saveat_final_only=saveat_final_only,
        solver_options_args=solver_options_args,
    )


@partial(
    jax.jit,
    static_argnames=("included_states", "saveat_downsampling", "solver_options_args"),
)
def _drive_excitation_hamiltonian(
    particle: SnVParticle,
    control_state: SnVControlState,
    optical_pulse,
    tau,
    rho0,
    included_states,
    saveat_downsampling=1,
    solver_options_args=None,
):
    scale = 1e9
    dimension = 2 * len(included_states) + 1
    sample_period = 1.0 / (particle.nondiff.sampling_rate * scale)
    pulse_center = optical_pulse.length / 2

    def H_optical_drive(t, args=None):
        del args
        local_times = jnp.stack((t - sample_period, t, t + sample_period))
        waveform, _, _ = synthesize_composite_waveform(
            composite=optical_pulse,
            tau=local_times,
            at_time=pulse_center,
            dphase=0.0,
            all_info=True,
        )
        return waveform[1]

    def H_dag_optical_drive(t, args=None):
        return jnp.conj(H_optical_drive(t, args))

    H0, _, Hs_optical, c_ops, transition_offset = get_excitation_hamiltonian(
        particle, control_state, included_states=included_states
    )
    omega_r = 2.0 * jnp.pi * (
        params.LEVEL_OFFSET
        + particle.diffable.strain_params[0]
        - particle.nondiff.laser_frequency
        + transition_offset
    ) * scale
    omega_c = 2.0 * jnp.pi * particle.diffable.eom_vpi_bandwidth * scale
    saveat_tlist = (
        tau[-2:] if saveat_downsampling is None else tau[::saveat_downsampling]
    )
    states, filter_states = mesolve_components(
        hamiltonians=(
            2.0 * jnp.pi * H0 * scale,
            2.0 * jnp.pi * Hs_optical[0] * scale,
            2.0 * jnp.pi * Hs_optical[1] * scale,
        ),
        coefficients=(1.0, H_optical_drive, H_dag_optical_drive),
        rho0=rho0,
        tlist=tau,
        saveat_tlist=saveat_tlist,
        filters=(
            None,
            eom_lowpass_filter(
                omega_c, +omega_r, particle.diffable.eom_vpi_ratio
            ),
            eom_lowpass_filter(
                omega_c, -omega_r, -particle.diffable.eom_vpi_ratio
            ),
        ),
        filter_y0s=(
            None,
            jnp.asarray(0.0, dtype=jnp.complex128),
            jnp.asarray(0.0, dtype=jnp.complex128),
        ),
        collapse_operators=c_ops * jnp.sqrt(scale),
        return_filter_states=True,
        solver_options=_solver_options(solver_options_args),
    )
    return states, filter_states, _populations(states, dimension)


def drive_excitation_hamiltonian(
    particle: SnVParticle,
    control_state: SnVControlState,
    optical_pulse,
    included_states=(0, 1, 2, 3),
    rho0=None,
    saveat_downsampling=1,
    solver_options_args=None,
):
    """Evolve one particle under a synthesized optical pulse.

    Parameters
    ----------
    particle, control_state
        Particle parameters and applied controls.
    optical_pulse
        Analog optical/EOM pulse.
    included_states
        Static tuple of retained states from each orbital manifold.
    rho0
        Initial density matrix. Defaults to the first basis state.
    saveat_downsampling
        Save every Nth time-grid point; ``None`` saves only the final interval.
    solver_options_args
        Optional positional arguments for ``SolverOptions.create``.

    Returns
    -------
    states, filter_states, populations
        Solver states, internal filter states, and basis populations.
    """
    included_states = tuple(included_states)
    solver_options_args = (
        None if solver_options_args is None else tuple(solver_options_args)
    )
    dimension = 2 * len(included_states) + 1
    if rho0 is None:
        rho0 = jqt.ket2dm(jqt.basis(dimension, 0))
    scale = 1e9
    sample_period = 1.0 / float(np.asarray(particle.nondiff.sampling_rate) * scale)
    tau = make_composite_waveform_time_array(
        composite=optical_pulse,
        sample_period=sample_period,
        at_time=float(np.asarray(optical_pulse.length)) / 2.0,
    )
    return _drive_excitation_hamiltonian(
        particle,
        control_state,
        optical_pulse,
        tau,
        rho0,
        included_states,
        saveat_downsampling=saveat_downsampling,
        solver_options_args=solver_options_args,
    )


# -----------------------------------------------------------------------------
# Multitone optical (EOM sideband) driving
# -----------------------------------------------------------------------------


def _select_multitone_sidebands(
    transition_coupling_squared,
    ple_freqs,
    eom_freqs,
    eom_amplitudes,
    laser_frequency,
    eom_vpi_ratio,
    eom_vpi_bandwidth,
    lifetime,
    max_bessel_order: int = 3,
    bessel_series_terms: int = 48,
    max_detuning=jnp.inf,
):
    """Select each ground-excited transition's own best optical line, independently.

    All EOM tones drive one phase modulator, so the optical field is
    ``exp(i * sum_k beta_k sin(Omega_k t))``. By Jacobi-Anger this is a
    product of one Bessel comb per tone, i.e. a grid of mixing products
    labelled by one sideband order per tone, ``(n_1, ..., n_F)``, at frequency
    ``laser_frequency + sum_k n_k * eom_freqs[k]`` with real amplitude
    ``prod_k J_{n_k}(beta_k)``. A tone left at order zero therefore still
    contributes its carrier factor ``J_0(beta_k)``. Mixing products that land
    on exactly the same frequency (commensurate tones) are summed coherently
    into one line.

    For every line, calculates the scattering rate that line alone would
    produce on every transition (its "solo" rate, ignoring every other line).
    Each transition independently picks whichever line maximizes its own solo
    rate, with no regard for what any other transition picks -- several
    transitions may end up sharing the same line. A transition is left
    undriven if it has no dipole coupling at all
    (``transition_coupling_squared == 0``), since such a transition has
    nothing meaningful to select regardless of line, or if its own best line
    still leaves it detuned by ``max_detuning`` or more -- since the
    rotating-frame construction downstream always treats a selected line as
    exactly resonant, silently forcing a wildly detuned "best available" line
    onto a transition would misrepresent it as driven. Two or more transitions
    sharing a line can still close a loop in the drive graph (e.g. if they
    also share a level); :func:`_assign_rotating_frame_shifts` is the actual
    guard against that, raising if a loop has nonzero net loop detuning.

    Parameters
    ----------
    transition_coupling_squared : jax.Array, shape (N_exc, N_gnd)
        Polarization-projected excitation strengths from
        :func:`hamiltonian_jqt.PLE_transitions`.
    ple_freqs : jax.Array, shape (N_exc, N_gnd)
        Absolute transition frequencies, in GHz.
    eom_freqs : jax.Array, shape (F,)
        EOM tone frequencies, in GHz.
    eom_amplitudes : jax.Array, shape (F,)
        Per-tone drive amplitude, as a multiplicative factor on
        ``eom_vpi_ratio``. Enters only through each tone's modulation index.
    laser_frequency, eom_vpi_ratio, eom_vpi_bandwidth, lifetime
        Scalar particle parameters, as used by :func:`scattering_rate`.
    max_bessel_order
        Largest positive and negative sideband order per tone. The grid has
        ``(2 * max_bessel_order + 1) ** F`` mixing products.
    bessel_series_terms
        Number of terms in the pure-JAX Bessel series.
    max_detuning
        Largest allowed ``|selected line frequency - ple_freqs|``, in GHz, for
        a transition's own best line to still count as driven. Defaults to no
        limit.

    Returns
    -------
    selected_weight : jax.Array, shape (N_exc, N_gnd)
        Signed field amplitude ``prod_k J_{n_k}(beta_k)`` of the selected
        line, or zero for a transition that is not driven.
    selected_frequency : jax.Array, shape (N_exc, N_gnd)
        Absolute frequency of the selected line, in GHz.
    driven : jax.Array, shape (N_exc, N_gnd), bool
        Whether the transition has dipole coupling and is within
        ``max_detuning`` of its own best line.
    selected_orders : jax.Array, shape (N_exc, N_gnd, F), int
        Mixing orders ``(n_1, ..., n_F)`` of the selected line.
    """
    gamma = 1.0 / lifetime / (2.0 * jnp.pi)
    num_tones = eom_freqs.shape[0]
    filtered_ratio = eom_vpi_ratio * eom_amplitudes / jnp.sqrt(
        1.0 + (eom_freqs / eom_vpi_bandwidth) ** 2
    )
    modulation_index = jnp.pi * filtered_ratio

    # Signed single-tone Bessel factors, J_{-n}(x) = (-1)^n J_n(x).
    single_orders = np.arange(-max_bessel_order, max_bessel_order + 1)
    J_nonnegative = _bessel_j_nonnegative_orders_integer_series(
        modulation_index,
        max_order=max_bessel_order,
        series_terms=bessel_series_terms,
    )  # (max_bessel_order + 1, F)
    parity = np.where((single_orders < 0) & (single_orders % 2 == 1), -1.0, 1.0)
    J_signed = J_nonnegative[np.abs(single_orders)] * parity[:, None]  # (S, F)

    # Full mixing grid: one sideband order per tone.
    mixing_orders = np.stack(
        np.meshgrid(*([single_orders] * num_tones), indexing="ij"), axis=-1
    ).reshape(-1, num_tones)  # (M, F)
    line_amplitude = jnp.prod(
        J_signed[mixing_orders + max_bessel_order, np.arange(num_tones)[None, :]],
        axis=-1,
    )  # (M,)
    line_offset = jnp.asarray(mixing_orders, dtype=eom_freqs.dtype) @ eom_freqs

    # Coherently merge mixing products that share a frequency.
    same_line = jnp.abs(line_offset[:, None] - line_offset[None, :]) < 1e-9
    weight = same_line.astype(line_amplitude.dtype) @ line_amplitude  # (M,)
    line_frequency = laser_frequency + line_offset  # (M,)

    detuning = line_frequency[:, None, None] - ple_freqs[None, :, :]
    coupling_squared = (
        weight[:, None, None] ** 2 * transition_coupling_squared[None, :, :]
    )
    solo_rate = coupling_squared / lifetime / (
        gamma**2 + 2.0 * coupling_squared + 4.0 * detuning**2
    )

    num_lines = weight.shape[0]
    num_exc, num_gnd = transition_coupling_squared.shape
    flat_rate = solo_rate.reshape(num_lines, num_exc * num_gnd)
    flat_detuning = detuning.reshape(num_lines, num_exc * num_gnd)

    best_line_for_transition = jnp.argmax(flat_rate, axis=0)
    transition_index = jnp.arange(num_exc * num_gnd)
    selected_detuning = flat_detuning[best_line_for_transition, transition_index]
    driven_flat = (transition_coupling_squared.reshape(num_exc * num_gnd) > 0) & (
        jnp.abs(selected_detuning) < max_detuning
    )

    selected_weight = jnp.where(
        driven_flat, weight[best_line_for_transition], 0.0
    ).reshape(num_exc, num_gnd)
    selected_frequency = line_frequency[best_line_for_transition].reshape(
        num_exc, num_gnd
    )
    selected_orders = jnp.asarray(mixing_orders)[best_line_for_transition].reshape(
        num_exc, num_gnd, num_tones
    )
    driven = driven_flat.reshape(num_exc, num_gnd)
    return selected_weight, selected_frequency, driven, selected_orders


def _assign_rotating_frame_shifts(
    driven,
    selected_frequency,
    num_ground,
    num_excited,
    edge_labels=None,
    tolerance=1e-6,
):
    """Assign per-level rotating-frame frequency shifts to a drive graph.

    Shifts are propagated along a breadth-first spanning tree of each
    connected component. A closed loop in the drive graph is allowed as long
    as it is frequency-consistent, i.e. the alternating sum of its line
    frequencies vanishes (for example two degenerate ground levels driven to
    two degenerate excited levels by one line). Only an inconsistent loop,
    whose net loop detuning no rotating frame can remove, is rejected.

    Parameters
    ----------
    driven : array_like, shape (num_excited, num_ground), bool
        Which ground-excited pairs are driven.
    selected_frequency : array_like, shape (num_excited, num_ground)
        Absolute frequency, in GHz, of the line driving each driven pair.
    num_ground, num_excited : int
        Number of retained ground and excited levels.
    edge_labels : array_like of str, shape (num_excited, num_ground), optional
        Per-pair description included in the error message for an
        inconsistent loop.
    tolerance
        Largest net loop detuning, in GHz, still treated as consistent.

    Returns
    -------
    shift_ground, shift_excited : np.ndarray
        Per-level frequency shifts, in GHz, such that
        ``shift_excited[e] - shift_ground[g] == selected_frequency[e, g]``
        for every driven pair. Each connected component of the drive graph is
        assigned shifts relative to an arbitrary zero-shift ground reference
        within that component; undriven levels keep a zero shift.

    Raises
    ------
    ValueError
        If the selected drive graph contains a loop with nonzero net loop
        detuning, so no time-independent rotating frame exists. The message
        lists every edge of the offending loop.
    """
    driven = np.asarray(driven)
    selected_frequency = np.asarray(selected_frequency)

    edges_from_ground = [[] for _ in range(num_ground)]
    edges_from_excited = [[] for _ in range(num_excited)]
    for e in range(num_excited):
        for g in range(num_ground):
            if driven[e, g]:
                edges_from_ground[g].append(e)
                edges_from_excited[e].append(g)

    shift = {}
    parent = {}
    for start in range(num_ground):
        root = ("g", start)
        if root in shift:
            continue
        shift[root] = 0.0
        parent[root] = None
        queue = deque([root])
        while queue:
            node = queue.popleft()
            kind, index = node
            if kind == "g":
                neighbors = [("e", e) for e in edges_from_ground[index]]
            else:
                neighbors = [("g", g) for g in edges_from_excited[index]]
            for neighbor in neighbors:
                e, g = (neighbor[1], index) if kind == "g" else (index, neighbor[1])
                frequency = selected_frequency[e, g]
                sign = 1.0 if kind == "g" else -1.0
                if neighbor not in shift:
                    shift[neighbor] = shift[node] + sign * frequency
                    parent[neighbor] = node
                    queue.append(neighbor)
                    continue
                mismatch = shift[("e", e)] - shift[("g", g)] - frequency
                if abs(mismatch) > tolerance:
                    raise ValueError(
                        _describe_inconsistent_loop(
                            parent, node, neighbor, mismatch, edge_labels
                        )
                    )

    shift_ground = np.asarray([shift.get(("g", g), 0.0) for g in range(num_ground)])
    shift_excited = np.asarray(
        [shift.get(("e", e), 0.0) for e in range(num_excited)]
    )
    return shift_ground, shift_excited


def _describe_inconsistent_loop(parent, node, neighbor, mismatch, edge_labels):
    """Format the loop closed by the non-tree edge ``node``-``neighbor``."""

    def path_to_root(n):
        path = []
        while n is not None:
            path.append(n)
            n = parent[n]
        return path

    path_node = path_to_root(node)
    path_neighbor = path_to_root(neighbor)
    common = set(path_node) & set(path_neighbor)
    head = [n for n in path_node if n not in common]
    tail = [n for n in path_neighbor if n not in common]
    ancestor = next(n for n in path_node if n in common)
    # Walk node -> ... -> ancestor -> ... -> neighbor -> node.
    loop = head + [ancestor] + tail[::-1] + [node]

    lines = []
    for a, b in zip(loop[:-1], loop[1:]):
        e, g = (a[1], b[1]) if a[0] == "e" else (b[1], a[1])
        label = "" if edge_labels is None else f"  {edge_labels[e][g]}"
        lines.append(f"  {a[0]}{a[1]} -> {b[0]}{b[1]}:{label}")
    return (
        "multitone_optical_drive found a closed loop in the selected drive "
        f"graph with net loop detuning {mismatch:+.6g} GHz, so no "
        "time-independent rotating frame exists:\n"
        + "\n".join(lines)
        + "\nReduce `max_detuning`, `max_bessel_order`, or change "
        "`eom_freqs`/`eom_amplitudes` so one of these transitions is no "
        "longer driven."
    )


def _to_dense(operator) -> Array:
    """Return a plain dense array for a ``Qarray`` or array-like operator."""
    if isinstance(operator, jqt.Qarray):
        return operator.to_dense().data
    return jnp.asarray(operator)


@jax.jit
def get_multitone_liouvillian(H, c_ops) -> Array:
    """Build the row-major vectorized Liouvillian for a static ``H``/``c_ops``.

    Only valid for a genuinely time-independent Hamiltonian and collapse
    operators, such as the effective Hamiltonian built by
    :func:`multitone_optical_drive`.

    Parameters
    ----------
    H : jaxquantum.Qarray or array_like, shape (d, d)
        Time-independent Hamiltonian.
    c_ops : jaxquantum.Qarray or array_like, shape (K, d, d)
        Batch of collapse operators.

    Returns
    -------
    jax.Array, shape (d**2, d**2)
        Liouvillian ``L`` such that ``d(vec(rho))/dt = L @ vec(rho)``, with
        ``vec`` the default (row-major) flatten of a ``(d, d)`` matrix.

    Notes
    -----
    Because ``H`` is Hermitian and every ``C_k^dag @ C_k`` is Hermitian, the
    row-major and the more commonly quoted column-major vectorized
    Liouvillian coincide as matrices; only the meaning of ``vec`` (and hence
    of ``rho0.reshape(-1)`` in :func:`get_density_matrix_trajectory`)
    differs.
    """
    H_data = _to_dense(H)
    c_ops_data = _to_dense(c_ops)
    dim = H_data.shape[-1]
    identity = jnp.eye(dim, dtype=H_data.dtype)

    hamiltonian_term = -1j * (
        jnp.kron(H_data, identity) - jnp.kron(identity, H_data.T)
    )

    c_ops_dag = jnp.conj(jnp.swapaxes(c_ops_data, -1, -2))
    c_dag_c = jnp.einsum("kij,kjl->kil", c_ops_dag, c_ops_data)
    decay_sum = jnp.sum(c_dag_c, axis=0)

    jump_term = jnp.sum(
        jax.vmap(jnp.kron)(c_ops_data, jnp.conj(c_ops_data)), axis=0
    )
    dissipator = (
        jump_term
        - 0.5 * jnp.kron(decay_sum, identity)
        - 0.5 * jnp.kron(identity, decay_sum.T)
    )

    return hamiltonian_term + dissipator


def get_density_matrix_trajectory(L, rho0):
    """Build rho(t) from a time-independent Liouvillian and initial state.

    Diagonalizes `L` once and returns a callable that evaluates the
    trajectory at arbitrary times via the closed-form eigen-expansion,
    rather than numerically integrating. Only valid for a genuinely
    time-independent `L` (e.g. from `get_multitone_liouvillian`) -- a
    Floquet monodromy matrix's eigenvalues only give the once-per-period
    envelope, not a plain exp(lambda*t) trajectory.

    Parameters
    ----------
    L : jax.Array, shape (d**2, d**2)
        Vectorized (row-major) Liouvillian.
    rho0 : array-like, shape (d, d)
        Initial density matrix.

    Returns
    -------
    rho_t : callable
        `rho_t(t)`, `t` scalar or shape (n_t,), returns the density
        matrix (shape (d, d)) or stacked matrices (shape (n_t, d, d)).
    """
    dim = int(round(L.shape[0] ** 0.5))
    eigenvalues, modes = jnp.linalg.eig(L)
    rho0_vec = jnp.asarray(rho0).reshape(-1).astype(modes.dtype)
    coefficients = jnp.linalg.solve(modes, rho0_vec)

    def rho_t(t):
        t = jnp.atleast_1d(jnp.asarray(t, dtype=eigenvalues.dtype))
        weights = coefficients[:, None] * jnp.exp(eigenvalues[:, None] * t[None, :])
        vec = modes @ weights  # (d**2, n_t)
        rho = jnp.moveaxis(vec.reshape(dim, dim, -1), -1, 0)  # (n_t, d, d)
        return rho[0] if jnp.ndim(t) == 0 or t.shape[0] == 1 else rho

    return rho_t


def multitone_optical_drive(
    particle: SnVParticle,
    control_state: SnVControlState,
    rho0,
    eom_freqs,
    eom_amplitudes,
    tlist,
    included_states=(0, 1, 2, 3),
    max_bessel_order: int = 3,
    bessel_series_terms: int = 48,
    max_detuning=jnp.inf,
):
    """Evolve one particle under several simultaneous EOM-sideband tones.

    All EOM tones phase-modulate the same laser, producing a grid of optical
    mixing products ``(n_1, ..., n_F)`` with amplitude ``prod_k J_{n_k}(beta_k)``.
    Every retained ground-excited transition is assigned the single line that
    alone would drive it fastest -- its largest "solo" scattering rate,
    ignoring every other line -- via :func:`_select_multitone_sidebands`. The
    resulting per-transition detunings are then removed with one frequency
    shift per retained level (:func:`_assign_rotating_frame_shifts`), chosen
    so every selected line becomes exactly static in the shifted frame. This
    is only possible because every closed loop in the selected drive graph
    has zero net loop detuning (a tree trivially satisfies this); the caller
    is responsible for choosing ``eom_freqs``/``eom_amplitudes``/
    ``max_bessel_order``/``max_detuning`` so that holds, and for keeping tones far enough
    apart that no transition sits near two distinct mixing products.
    Because the effective Hamiltonian, collapse operators, and hence the
    Liouvillian are all genuinely time-independent, the trajectory is solved
    in closed form via one eigendecomposition
    (:func:`get_multitone_liouvillian`, :func:`get_density_matrix_trajectory`)
    rather than by numerically integrating an ODE, unlike the time-dependent
    pulse propagation in :func:`drive_excitation_hamiltonian`.

    Parameters
    ----------
    particle, control_state
        Particle parameters and applied controls.
    rho0
        Initial density matrix in the ``included_states`` reduced basis
        (``2 * len(included_states) + 1`` dimensional, including the dark
        state).
    eom_freqs : array_like, shape (F,)
        EOM tone frequencies, in GHz.
    eom_amplitudes : array_like, shape (F,)
        Per-tone drive amplitude, as a multiplicative factor on
        ``particle.diffable.eom_vpi_ratio``.
    max_bessel_order
        Largest positive and negative sideband order considered per tone;
        ``(2 * max_bessel_order + 1) ** F`` mixing products are searched.
    tlist
        One-dimensional array of times, in seconds, at which to save the
        solution.
    included_states
        Static tuple of retained matched ground/excited eigenstate indices.
    bessel_series_terms
        Number of terms in the pure-JAX Bessel series.
    max_detuning
        Largest allowed detuning, in GHz, between a transition and its own
        best available sideband for that transition to be driven at all (see
        :func:`_select_multitone_sidebands`). Defaults to no limit.

    Returns
    -------
    rho_trajectory : jax.Array, shape (len(tlist), dimension, dimension)
        Density matrix at every time in ``tlist``.
    populations : jax.Array, shape (dimension, len(tlist))
        Real basis populations at every time in ``tlist``.
    """
    included_states = tuple(included_states)
    d, n = particle.diffable, particle.nondiff

    eom_freqs = jnp.atleast_1d(jnp.asarray(eom_freqs))
    eom_amplitudes = jnp.atleast_1d(jnp.asarray(eom_amplitudes))
    if eom_freqs.ndim != 1 or eom_freqs.shape != eom_amplitudes.shape:
        raise ValueError(
            "`eom_freqs` and `eom_amplitudes` must be one-dimensional arrays "
            "of the same length."
        )
    tlist = jnp.asarray(tlist)

    (
        E_gnd,
        _,
        _,
        _,
        E_exc,
        _,
        _,
        _,
        transition_coupling_squared,
        _,
    ) = PLE_transitions(particle, control_state)
    ple_freqs = (
        E_exc[:, None] - E_gnd[None, :] + params.LEVEL_OFFSET + d.strain_params[0]
    )

    state_index = jnp.asarray(included_states, dtype=jnp.int32)
    reduced_ple_freqs = ple_freqs[jnp.ix_(state_index, state_index)]
    reduced_coupling_squared = transition_coupling_squared[
        jnp.ix_(state_index, state_index)
    ]

    selected_weight, selected_frequency, driven, selected_orders = _select_multitone_sidebands(
        reduced_coupling_squared,
        reduced_ple_freqs,
        eom_freqs,
        eom_amplitudes,
        n.laser_frequency,
        d.eom_vpi_ratio,
        d.eom_vpi_bandwidth,
        n.excited_state_lifetime,
        max_bessel_order,
        bessel_series_terms,
        max_detuning,
    )

    reduced_dim = len(included_states)
    line_offset = np.asarray(selected_frequency - n.laser_frequency)
    line_detuning = np.asarray(selected_frequency - reduced_ple_freqs)
    orders = np.asarray(selected_orders)
    edge_labels = [
        [
            f"line = laser {line_offset[e, g]:+.6f} GHz, mixing orders "
            f"{tuple(int(k) for k in orders[e, g])}, transition detuning "
            f"{line_detuning[e, g]:+.6f} GHz"
            for g in range(reduced_dim)
        ]
        for e in range(reduced_dim)
    ]
    shift_ground, shift_excited = _assign_rotating_frame_shifts(
        driven, selected_frequency, reduced_dim, reduced_dim, edge_labels
    )

    _, _, Hs_optical, c_ops, _ = get_excitation_hamiltonian(
        particle, control_state, included_states=included_states
    )
    dimension = 2 * reduced_dim + 1

    p_ge_block = Hs_optical[1].to_dense().data[
        reduced_dim : 2 * reduced_dim, 0:reduced_dim
    ]
    dtype = p_ge_block.dtype
    coupling_block = jnp.asarray(selected_weight, dtype=dtype) * p_ge_block

    E_gnd_reduced = E_gnd[state_index]
    E_exc_reduced = E_exc[state_index] + params.LEVEL_OFFSET + d.strain_params[0]
    diagonal = jnp.concatenate(
        [
            E_gnd_reduced - jnp.asarray(shift_ground, dtype=E_gnd_reduced.dtype),
            E_exc_reduced - jnp.asarray(shift_excited, dtype=E_exc_reduced.dtype),
            jnp.zeros((1,), dtype=E_gnd_reduced.dtype),
        ]
    ).astype(dtype)

    H_eff = (
        jnp.zeros((dimension, dimension), dtype=dtype)
        .at[jnp.diag_indices(dimension)]
        .set(diagonal)
        .at[reduced_dim : 2 * reduced_dim, 0:reduced_dim]
        .set(coupling_block)
        .at[0:reduced_dim, reduced_dim : 2 * reduced_dim]
        .set(jnp.conj(coupling_block).T)
    )
    H_eff = jqt.Qarray.create(H_eff, dims=(dimension,))

    scale = 1e9
    L = get_multitone_liouvillian(2.0 * jnp.pi * H_eff * scale, c_ops * jnp.sqrt(scale))
    rho_t = get_density_matrix_trajectory(L, _to_dense(rho0))
    rho_trajectory = jnp.reshape(rho_t(tlist), (-1, dimension, dimension))
    populations = jnp.real(
        jnp.diagonal(rho_trajectory, axis1=-2, axis2=-1)
    ).T
    return rho_trajectory, populations


__all__ = [
    "lowpass_filter",
    "eom_lowpass_filter",
    "cartesian_to_angles",
    "angles_to_cartesian",
    "normalize_vectors",
    "get_magnet_axes",
    "get_B_settings",
    "get_waveplate_angles",
    "get_optimal_control_state",
    "get_B_cartesian",
    "get_B_spherical",
    "get_dipole_B_GHz",
    "get_B_theta",
    "get_B_phi",
    "get_resonant_pump_eta",
    "scattering_rate",
    "solve_hamiltonian",
    "get_folded_branching_ratios",
    "get_ple_freqs",
    "get_emr_freqs",
    "get_nmr_freqs",
    "get_init_timestep",
    "get_excitation_hamiltonian",
    "get_ground_hamiltonian",
    "drive_mw_hamiltonian",
    "drive_excitation_hamiltonian",
    "get_multitone_liouvillian",
    "get_density_matrix_trajectory",
    "multitone_optical_drive",
]
