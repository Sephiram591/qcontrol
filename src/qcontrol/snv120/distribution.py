from typing import NamedTuple
from enum import IntEnum
import jax
import jax.numpy as jnp
import numpy as np
import jax.scipy.special as jsp_special
import qcontrol.snv120.parameters as params
import qcontrol.snv120.hamiltonian_jqt as qh_jqt
import jaxquantum as jqt
from pulseseq.sequencing.waveform import AnalogPulse, Apodization, Shape, DigitalPulse, DigitalType
import qcontrol.snv120.pulseseq_interconnect
from qcontrol.snv120.pulseseq_interconnect import stack_waveforms, make_analog_pulse_time_array, synthesize_analog_pulse

from pulseseq.sequencing.waveform import AnalogPulse, Apodization, Shape, DigitalPulse, DigitalType

def lowpass_filter(omega_c):
    """Return a first-order low-pass filter.

    The transfer function is

        H(s) = omega_c / (s + omega_c).

    Args:
        omega_c: Angular cutoff frequency in radians per second.

    Returns:
        A filter function compatible with ``sesolve_components``.
    """
    omega_c = jnp.asarray(omega_c)

    def filter_fn(t, z, u):
        del t
        dz_dt = omega_c * (u - z)
        return dz_dt, z

    return filter_fn

def cartesian_to_angles(vector):
    theta = jnp.arctan2(
        jnp.linalg.norm(vector[..., :2], axis=-1),
        vector[..., 2],
    )
    phi = jnp.arctan2(
        vector[..., 1],
        vector[..., 0],
    )
    return theta, phi

def angles_to_cartesian(theta, phi):
    return jnp.stack(
        [
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta),
        ],
        axis=-1,
    )


def normalize_vectors(vector):
    """Normalize vectors along their final Cartesian-component axis."""
    vector = jnp.asarray(vector)
    return vector / jnp.linalg.norm(vector, axis=-1, keepdims=True)

@jax.jit(static_argnums=(1, 2))
def _bessel_j_nonnegative_orders_integer_series(
    x,
    max_order: int,
    series_terms: int = 48,
):
    """
    Pure-JAX integer-order Bessel J_n(x) for n = 0, ..., max_order.

    Returns
    -------
    jax.Array
        Shape (max_order + 1, *x.shape)

    Notes
    -----
    This avoids relying on scipy.special.jv, which is not JIT-compatible.
    max_order and series_terms must be static Python integers.
    """
    x = jnp.asarray(x)

    orders = jnp.arange(max_order + 1, dtype=x.dtype)[:, None]
    k = jnp.arange(series_terms, dtype=x.dtype)[None, :]

    power = 2.0 * k + orders
    sign = jnp.where((jnp.arange(series_terms)[None, :] % 2) == 0, 1.0, -1.0)

    coeff = sign * jnp.exp(
        -jsp_special.gammaln(k + 1.0)
        -jsp_special.gammaln(k + orders + 1.0)
    )

    # Shape: (max_order + 1, series_terms, *x.shape)
    terms = (
        coeff[(...,) + (None,) * x.ndim]
        * (0.5 * x[None, None, ...]) ** power[(...,) + (None,) * x.ndim]
    )

    return jnp.sum(terms, axis=1)

class SnVConstants(NamedTuple):
    B_target: jnp.ndarray                   # (3,) target electron-Zeeman vector in the dipole frame, units of GHz
    resonant_pump_polarization_target: jnp.ndarray # (TE, TM) fraction of resonant pump coupling to the TE and TM modes
    mw_amps: float                    # microwave amp setting for RFSoC
    optical_amps: float               # optical amp setting for RFSoC
    laser_frequency: float            # laser frequency, units of GHz
    diamond_lattice_100_orientation: jnp.ndarray # Best estimate of (theta, phi) for the diamond lattice [100] axis in the lab frame
    diamond_lattice_011_orientation: jnp.ndarray # Best estimate of (theta, phi) for the diamond lattice [011] axis in the lab frame
    nominal_magnet_axes: jnp.ndarray        # (3, 3) nominal magnet-axis unit vectors in the lab frame
    sampling_rate: float                    # awg sampling rate, units of Hz
    # Physical constants
    mu_B_GHz_per_T = 13.996 # [GHz/T]
    g_e = 2.0
    dipole_crystal_axes = jnp.asarray([ # Cartesian Unit vector relative to crystal frame
        [1,1,1],
        [1,-1,1],
        [-1,1,1],
        [-1,-1,1]
    ])


class SnV120Distribution(NamedTuple):
    """
    Theta_i (0 < i < N) parameters for the SnV120 qubit.
    """
    weights: jnp.ndarray # N
    constants: SnVConstants

    # Energy level and cyclicity parameters
    magnet_unit_magnitude: jnp.ndarray   # (N, 3) multiplicative field calibration for each physical magnet axis
    magnet_axes_rotations: jnp.ndarray   # (N, 3, 3): (magnet_axis, Cartesian rotation-vector component)
                                         # Cartesian components are (rx, ry, rz), in radians.
    dipole_crystal_axis_idx: jnp.ndarray # (N) dipole crystal axis index

    # resonant_pump_E_angles: jnp.ndarray # (N, 2, 2), (theta, phi) of the electric field of the resonant pump for TE and TM modes in the crystal frame

    # Hamiltonian parameters
    alpha: jnp.ndarray     # N
    beta: jnp.ndarray      # N
    alpha_exc: jnp.ndarray # N
    beta_exc: jnp.ndarray  # N
    hyperfine_neighbor_idx: jnp.ndarray # N, HyperfineNeighbor enum values
    
    excited_state_lifetime: jnp.ndarray # N, in units of nanoseconds
    # Collection efficiency parameters
    debye_waller_factor: jnp.ndarray  # N
    quantum_efficiency: jnp.ndarray   # N
    optical_transmission: jnp.ndarray # N
    dark_count_rate: jnp.ndarray      # N

    # Resonant pump coupling
    resonant_pump_coupling_rate: jnp.ndarray # (N, 2), in units of coupling rate, or GHz for TE and TM modes
    resonant_pump_polarization: jnp.ndarray  # (N, 1), Angle of the resonant pump polarization with respect to the TE mode without factoring in the waveplates
    eom_vpi_ratio: jnp.ndarray               # N, in units of Vmax/Vpi
    eom_vpi_bandwidth: jnp.ndarray           # N, in units of GHz

    # Transmission line dynamic B field
    mw_B_orientation: jnp.ndarray # (N, 2), transmission line B field orientation (theta, phi) relative to the dipole axis
    mw_B_magnitude: jnp.ndarray   # N, in units of Tesla
    mw_B_bandwidth: jnp.ndarray    # N, in units of GHz

    # Drift and diffusion
    spectral_diffusion_rate: jnp.ndarray  # N, in units of Hz/sqrt(s)
    polarization_drift_rate: jnp.ndarray  # N, in units of rads/sqrt(s)
    resonant_pump_coupling_drift_rate: jnp.ndarray # N, in units of rads/sqrt(s)

    def _prepare_indices(self, idx):
        """Normalize public scalar/array/None indices for the jitted batch cores."""
        scalar_idx = idx is not None and jnp.asarray(idx).ndim == 0

        if idx is None:
            idx_array = jnp.arange(
                self.weights.shape[0],
                dtype=jnp.int32,
            )
        else:
            idx_array = jnp.atleast_1d(
                jnp.asarray(idx, dtype=jnp.int32)
            )

        return idx_array, scalar_idx

    def _select_scalar_parameter_batch(self, value, idx):
        """Select a shared scalar or particle-resolved scalar parameter.

        Parameters
        ----------
        value : scalar or jax.Array, shape (N,)
            A value shared by every distribution member, or one value per
            member of the complete distribution.
        idx : jax.Array, shape (K,)
            Distribution indices selected for the current batch.

        Returns
        -------
        jax.Array, shape (K,)
            Values aligned with ``idx``.

        Raises
        ------
        ValueError
            If ``value`` is neither scalar nor one-dimensional.

        Notes
        -----
        Array rank is static during JAX tracing, so this Python branch does not
        introduce data-dependent control flow into a jitted batch method.
        """
        value = jnp.asarray(value)
        idx = jnp.asarray(idx, dtype=jnp.int32)

        if value.ndim == 0:
            return jnp.broadcast_to(value, idx.shape)
        if value.ndim == 1:
            return value[idx]

        raise ValueError(
            "Expected a scalar or a one-dimensional particle parameter; "
            f"received shape {value.shape}."
        )
        
    @jax.jit(static_argnames=("frame",))
    def get_magnet_axes_batch(self, idx, frame="lab"):
        """Return physical magnet-axis unit vectors for a one-dimensional index batch.

        Each magnet-axis rotation is represented by a Cartesian rotation vector

            r = (rx, ry, rz),

        expressed in the lab frame. The direction of ``r`` is the rotation axis and
        its magnitude is the rotation angle in radians:

            angle = ||r||.

        The rotations are applied exactly using Rodrigues' rotation formula; no
        small-angle approximation is made.

        Parameters
        ----------
        idx : jax.Array, shape (K,)
            Distribution indices. The length ``K`` is static for each compiled
            executable.

        frame : {"lab", "crystal", "dipole"}, optional
            Output coordinate frame. This string is a static JIT argument, so
            each frame has its own cached compilation.

        Returns
        -------
        jax.Array, shape (K, 3, 3)
            Cartesian unit vectors. The final dimensions are

                (magnet_axis, Cartesian_component).

        Notes
        -----
        ``self.magnet_axes_rotations`` has shape ``(N, 3, 3)``, with dimensions

            (distribution_member, magnet_axis, rotation_vector_component),

        where the final dimension is

            (rx, ry, rz).

        All rotation-vector components are in radians.
        """
        if frame not in ("lab", "crystal", "dipole"):
            raise ValueError(
                f"Invalid frame {frame!r}; expected "
                "'lab', 'crystal', or 'dipole'."
            )

        idx = jnp.asarray(idx, dtype=jnp.int32)

        if idx.ndim != 1:
            raise ValueError(
                "`idx` must be a one-dimensional integer array."
            )

        # -------------------------------------------------------------------------
        # Select rotation vectors
        # -------------------------------------------------------------------------
        #
        # Shape:
        #     (K, magnet_axis=3, xyz=3)
        #
        # Each vector is
        #
        #     r = angle * rotation_axis
        #
        # so that
        #
        #     angle = ||r||.
        #
        rotation_vectors_lab = self.magnet_axes_rotations[idx]

        # -------------------------------------------------------------------------
        # Start from the nominal physical magnet axes
        # -------------------------------------------------------------------------

        nominal_magnet_axes = jnp.asarray(
            self.constants.nominal_magnet_axes,
            dtype=rotation_vectors_lab.dtype,
        )

        # Expected nominal shape:
        #
        #     (3, 3)
        #
        # corresponding to
        #
        #     (magnet_axis, Cartesian_component).
        #
        # Broadcast the shared calibration over the selected distribution members.
        magnet_axes_lab = jnp.broadcast_to(
            nominal_magnet_axes,
            (
                idx.shape[0],
                *nominal_magnet_axes.shape,
            ),
        )

        magnet_axes_lab = normalize_vectors(
            magnet_axes_lab
        )

        # -------------------------------------------------------------------------
        # Apply the rotation vectors exactly
        # -------------------------------------------------------------------------
        #
        # For a rotation vector
        #
        #     r = theta * k
        #
        # Rodrigues' formula can be written without explicitly constructing the
        # unit rotation axis k:
        #
        #     v_rot
        #       = v
        #       + [sin(theta) / theta] * (r x v)
        #       + [(1 - cos(theta)) / theta^2]
        #           * (r x (r x v)).
        #
        # This representation is particularly convenient for rotation vectors.
        #
        # We evaluate the two coefficients using sinc identities so that the
        # expression remains finite and differentiable at theta = 0.
        #
        # JAX defines
        #
        #     jnp.sinc(x) = sin(pi*x) / (pi*x),
        #
        # therefore
        #
        #     sin(theta) / theta
        #         = sinc(theta / pi),
        #
        # and
        #
        #     (1 - cos(theta)) / theta^2
        #         = 1/2 * sinc(theta / (2*pi))^2.
        #

        # Shape:
        #     (K, 3)
        rotation_angle = jnp.linalg.norm(
            rotation_vectors_lab,
            axis=-1,
        )

        # Shape:
        #     (K, 3, 1)
        sin_theta_over_theta = jnp.sinc(
            rotation_angle / jnp.pi
        )[..., None]

        # Shape:
        #     (K, 3, 1)
        one_minus_cos_over_theta_squared = (
            0.5
            * jnp.sinc(
                rotation_angle / (2.0 * jnp.pi)
            )[..., None] ** 2
        )

        # First-order cross-product term:
        #
        #     r x v
        #
        # Shape:
        #     (K, 3, 3)
        r_cross_v = jnp.cross(
            rotation_vectors_lab,
            magnet_axes_lab,
        )

        # Second-order cross-product term:
        #
        #     r x (r x v)
        #
        # Shape:
        #     (K, 3, 3)
        r_cross_r_cross_v = jnp.cross(
            rotation_vectors_lab,
            r_cross_v,
        )

        magnet_axes_lab = (
            magnet_axes_lab
            + sin_theta_over_theta
            * r_cross_v
            + one_minus_cos_over_theta_squared
            * r_cross_r_cross_v
        )

        # An exact rotation preserves norm. Renormalizing here only removes small
        # floating-point accumulation errors.
        magnet_axes_lab = normalize_vectors(
            magnet_axes_lab
        )

        # -------------------------------------------------------------------------
        # Lab frame requested
        # -------------------------------------------------------------------------

        if frame == "lab":
            return magnet_axes_lab

        # -------------------------------------------------------------------------
        # Construct the crystal basis in lab coordinates
        # -------------------------------------------------------------------------
        #
        # Reconstruct a right-handed crystal basis using the measured [100] and
        # [011] crystal directions in the lab frame.
        #

        lattice_100_angles = jnp.asarray(
            self.constants.diamond_lattice_100_orientation,
            dtype=magnet_axes_lab.dtype,
        )

        lattice_011_angles = jnp.asarray(
            self.constants.diamond_lattice_011_orientation,
            dtype=magnet_axes_lab.dtype,
        )

        # Crystal [100] direction expressed in lab coordinates.
        crystal_x_lab = normalize_vectors(
            angles_to_cartesian(
                lattice_100_angles[..., 0],
                lattice_100_angles[..., 1],
            )
        )

        # Measured crystal [011] direction expressed in lab coordinates.
        lattice_011_lab = angles_to_cartesian(
            lattice_011_angles[..., 0],
            lattice_011_angles[..., 1],
        )

        # -------------------------------------------------------------------------
        # Enforce the expected crystal orthogonality
        # -------------------------------------------------------------------------
        #
        # Ideally
        #
        #     [100] . [011] = 0.
        #
        # Measurement uncertainty may violate this slightly, so perform one
        # Gram-Schmidt step.
        #

        lattice_011_lab = normalize_vectors(
            lattice_011_lab
            - jnp.sum(
                lattice_011_lab * crystal_x_lab,
                axis=-1,
                keepdims=True,
            )
            * crystal_x_lab
        )

        # -------------------------------------------------------------------------
        # Recover crystal y and z
        # -------------------------------------------------------------------------
        #
        # In crystal coordinates,
        #
        #     [100] x [011]
        #         = ([001] - [010]) / sqrt(2),
        #
        # while
        #
        #     [011]
        #         = ([010] + [001]) / sqrt(2).
        #

        crystal_z_minus_y_lab = normalize_vectors(
            jnp.cross(
                crystal_x_lab,
                lattice_011_lab,
            )
        )

        crystal_y_lab = normalize_vectors(
            lattice_011_lab
            - crystal_z_minus_y_lab
        )

        crystal_z_lab = normalize_vectors(
            lattice_011_lab
            + crystal_z_minus_y_lab
        )

        # Columns are crystal basis vectors expressed in lab coordinates:
        #
        #     C = [x_crystal_lab, y_crystal_lab, z_crystal_lab].
        #
        # Shape:
        #     (3, 3)
        crystal_to_lab = jnp.stack(
            [
                crystal_x_lab,
                crystal_y_lab,
                crystal_z_lab,
            ],
            axis=-1,
        )

        # -------------------------------------------------------------------------
        # Lab -> crystal coordinates
        # -------------------------------------------------------------------------
        #
        # For row-vector convention,
        #
        #     v_crystal = v_lab @ C.
        #
        # Shape:
        #     (K, 3, 3)
        magnet_axes_crystal = jnp.matmul(
            magnet_axes_lab,
            crystal_to_lab,
        )

        if frame == "crystal":
            return magnet_axes_crystal

        # -------------------------------------------------------------------------
        # Construct the dipole basis in crystal coordinates
        # -------------------------------------------------------------------------

        # The selected <111> crystal direction defines the dipole-frame z axis.
        #
        # Shape:
        #     (K, 3)
        dipole_z_crystal = normalize_vectors(
            jnp.asarray(
                self.constants.dipole_crystal_axes,
                dtype=magnet_axes_crystal.dtype,
            )[
                self.dipole_crystal_axis_idx[idx]
            ]
        )

        # Crystal [001].
        crystal_z = jnp.asarray(
            [0.0, 0.0, 1.0],
            dtype=magnet_axes_crystal.dtype,
        )

        # Use crystal [001] to fix the otherwise arbitrary rotation about the
        # selected <111> dipole axis.
        #
        # y_dipole = z_crystal x z_dipole
        dipole_y_crystal = normalize_vectors(
            jnp.cross(
                crystal_z,
                dipole_z_crystal,
            )
        )

        # Complete the right-handed basis.
        #
        # x_dipole = y_dipole x z_dipole
        dipole_x_crystal = normalize_vectors(
            jnp.cross(
                dipole_y_crystal,
                dipole_z_crystal,
            )
        )

        # Columns are dipole basis vectors expressed in crystal coordinates.
        #
        # Shape:
        #     (K, 3, 3)
        dipole_to_crystal = jnp.stack(
            [
                dipole_x_crystal,
                dipole_y_crystal,
                dipole_z_crystal,
            ],
            axis=-1,
        )

        # -------------------------------------------------------------------------
        # Crystal -> dipole coordinates
        # -------------------------------------------------------------------------
        #
        # For the row-vector convention,
        #
        #     v_dipole = v_crystal @ D.
        #
        return jnp.matmul(
            magnet_axes_crystal,
            dipole_to_crystal,
        )

    def get_magnet_axes(self, idx=None, frame="lab"):
        """Return magnet-axis unit vectors with scalar-index convenience.

        A scalar index returns shape ``(3, 3)``. ``None`` or a one-dimensional
        index array returns shape ``(K, 3, 3)``.
        """
        idx_array, scalar_idx = self._prepare_indices(idx)
        result = self.get_magnet_axes_batch(idx_array, frame=frame)
        return result[0] if scalar_idx else result

    @jax.jit
    def get_B_settings_batch(self, idx):
        """Return nominal magnet settings for a one-dimensional index batch.

        Parameters
        ----------
        idx : jax.Array, shape (K,)
            Distribution indices.

        Returns
        -------
        jax.Array, shape (K, 3)
            Nominal magnet settings in tesla.
        """
        idx = jnp.asarray(idx, dtype=jnp.int32)
        if idx.ndim != 1:
            raise ValueError("`idx` must be a one-dimensional integer array.")

        B_unit_directions = self.get_magnet_axes_batch(
            idx,
            frame="dipole",
        )
        magnet_unit_magnitude = self.magnet_unit_magnitude[idx]

        # calibration_matrix[..., :, i] is the true dipole-frame field vector
        # generated by one nominal tesla on physical magnet channel i.
        calibration_matrix = (
            jnp.swapaxes(B_unit_directions, -1, -2)
            * magnet_unit_magnitude[..., None, :]
        )

        B_target_GHz = jnp.asarray(
            self.constants.B_target,
            dtype=calibration_matrix.dtype,
        )
        if B_target_GHz.ndim == 1:
            B_target_GHz = jnp.broadcast_to(
                B_target_GHz,
                (idx.shape[0], B_target_GHz.shape[0]),
            )
        elif B_target_GHz.ndim == 2:
            B_target_GHz = B_target_GHz[idx]
        else:
            raise ValueError("`B_target` must have shape (3,) or (N, 3).")

        B_target_T = B_target_GHz / (
            self.constants.mu_B_GHz_per_T * self.constants.g_e
        )

        return jnp.linalg.solve(
            calibration_matrix,
            B_target_T[..., None],
        )[..., 0]

    def get_B_settings(self, idx=None):
        """Return nominal magnet settings in tesla.

        A scalar index returns shape ``(3,)``. ``None`` or a one-dimensional
        index array returns shape ``(K, 3)``.
        """
        idx_array, scalar_idx = self._prepare_indices(idx)
        result = self.get_B_settings_batch(idx_array)
        return result[0] if scalar_idx else result

    @jax.jit(static_argnames=("frame",))
    def get_B_cartesian_batch(self, idx, frame="lab"):
        """Return the realized magnetic-field vectors in tesla.

        Parameters
        ----------
        idx : jax.Array, shape (K,)
            Distribution indices.
        frame : {"lab", "crystal", "dipole"}, optional
            Output coordinate frame; static under JIT.

        Returns
        -------
        jax.Array, shape (K, 3)
            Magnetic-field Cartesian components in tesla.
        """
        idx = jnp.asarray(idx, dtype=jnp.int32)
        if idx.ndim != 1:
            raise ValueError("`idx` must be a one-dimensional integer array.")

        B_settings = self.get_B_settings_batch(idx)
        B_axes = self.get_magnet_axes_batch(idx, frame=frame)

        return jnp.sum(
            B_settings[..., :, None]
            * self.magnet_unit_magnitude[idx, :, None]
            * B_axes,
            axis=-2,
        )

    def get_B_cartesian(self, idx=None, frame="lab"):
        """Return realized magnetic-field vectors in tesla.

        A scalar index returns shape ``(3,)``. ``None`` or a one-dimensional
        index array returns shape ``(K, 3)``.
        """
        idx_array, scalar_idx = self._prepare_indices(idx)
        result = self.get_B_cartesian_batch(idx_array, frame=frame)
        return result[0] if scalar_idx else result

    @jax.jit(static_argnames=("frame",))
    def get_B_spherical_batch(self, idx, frame="dipole"):
        """Return field magnitude in GHz and direction angles for an index batch."""
        B_T = self.get_B_cartesian_batch(idx, frame=frame)
        theta, phi = cartesian_to_angles(B_T)
        magnitude_GHz = jnp.linalg.norm(B_T, axis=-1) * (
            self.constants.mu_B_GHz_per_T * self.constants.g_e
        )
        return magnitude_GHz, theta, phi

    def get_dipole_B_GHz(self, idx=None):
        """Return magnetic-field magnitude in electron-Zeeman GHz."""
        idx_array, scalar_idx = self._prepare_indices(idx)
        magnitude_GHz, _, _ = self.get_B_spherical_batch(
            idx_array,
            frame="dipole",
        )
        return magnitude_GHz[0] if scalar_idx else magnitude_GHz

    def get_B_theta(self, idx=None, frame="dipole"):
        """Return the field polar angle in the selected frame, in radians."""
        idx_array, scalar_idx = self._prepare_indices(idx)
        _, theta, _ = self.get_B_spherical_batch(idx_array, frame=frame)
        return theta[0] if scalar_idx else theta

    def get_B_phi(self, idx=None, frame="dipole"):
        """Return the field azimuthal angle in the selected frame, in radians."""
        idx_array, scalar_idx = self._prepare_indices(idx)
        _, _, phi = self.get_B_spherical_batch(idx_array, frame=frame)
        return phi[0] if scalar_idx else phi

    def get_eta(self, idx=None):
        """Return normalized resonant-pump Jones vectors ``[TE, TM]``.

        A scalar index returns shape ``(2,)``. ``None`` or a one-dimensional
        index array returns shape ``(K, 2)``.
        """
        idx_array, scalar_idx = self._prepare_indices(idx)
        polarization = jnp.asarray(self.constants.resonant_pump_polarization_target)
        selected = jnp.broadcast_to(
            polarization,
            (idx_array.shape[0], polarization.shape[0]),
        )

        selected = selected / jnp.linalg.norm(
            selected,
            axis=-1,
            keepdims=True,
        )
        return selected[0] if scalar_idx else selected

    @jax.jit(
        static_argnames=(
            "max_bessel_order",
            "bessel_series_terms",
            "sum_counts",
        )
    )
    def scattering_rate_batch(
        self,
        idx,
        eom_frequency=0.0,
        max_bessel_order: int = 2,
        bessel_series_terms: int = 48,
        sum_counts: bool = True,
    ):
        """Calculate optical scattering rates for an EOM-frequency sweep.

        The optical field is treated as a phase-modulated carrier with sidebands
        at integer multiples of ``eom_frequency``. The contribution from each
        sideband is calculated independently and may optionally be summed over
        both sidebands and optical transitions.

        Parameters
        ----------
        idx : jax.Array
            One-dimensional array containing the selected distribution indices.
            If ``N`` particles are selected, this array has shape ``(N,)``.

        eom_frequency : float or jax.Array, optional
            EOM modulation frequency in GHz.

            A scalar input is converted into a frequency sweep of length one. An
            array input must be one-dimensional and is interpreted as a frequency
            sweep with shape ``(F,)``.

            Every frequency in the sweep is evaluated for every selected
            distribution member.

        max_bessel_order : int, optional
            Largest positive and negative EOM sideband order to include. Included
            sideband orders are

            ``-max_bessel_order, ..., 0, ..., +max_bessel_order``.

            This argument must be static under ``jax.jit`` because it determines
            the number of sidebands and therefore the compiled array shapes.

        bessel_series_terms : int, optional
            Number of terms retained in the pure-JAX Bessel-function series.

            This argument must be static under ``jax.jit``.

        sum_counts : bool, optional
            If ``True``, sum the scattering rates over the sideband and transition
            axes. If ``False``, return the individual contribution from every
            sideband and optical transition.

            This argument must be static under ``jax.jit`` because it changes the
            output shape.

        Returns
        -------
        jax.Array
            If ``sum_counts=True``, returns an array with shape ``(N, F)``.

            If ``sum_counts=False``, returns an array with shape
            ``(N, F, S, T)``, where

            - ``N`` is the number of selected distribution members,
            - ``F`` is the number of EOM frequencies,
            - ``S = 2 * max_bessel_order + 1`` is the number of sidebands,
            - ``T`` is the number of optical transitions.

        Notes
        -----
        A scalar ``eom_frequency`` is deliberately retained as a length-one
        frequency sweep, so the second output axis is always the frequency axis.
        """
        # Normalize the distribution indices to a one-dimensional integer array.
        #
        # Shape: (N,)
        idx = jnp.atleast_1d(
            jnp.asarray(idx, dtype=jnp.int32)
        )

        # Normalize the EOM frequency to a one-dimensional sweep.
        #
        # Scalar input:
        #     () -> (1,)
        #
        # Sweep input:
        #     (F,) -> (F,)
        eom_frequency = jnp.atleast_1d(
            jnp.asarray(eom_frequency)
        )

        # Homogeneous optical linewidth for each selected distribution member.
        #
        # The excited-state lifetime is stored in ns, so its inverse has units
        # of GHz. Division by 2*pi converts the decay rate to an ordinary-frequency
        # linewidth under the convention used by the scattering-rate expression.
        #
        # Shape: (N,)
        gamma = (
            1.0
            / self.excited_state_lifetime[idx]
            / (2.0 * jnp.pi)
        )

        # Unmodulated optical coupling rate for each selected distribution member.
        #
        # Shape: (N,)
        omega = jnp.sum(
            self.resonant_pump_coupling_rate[idx]
            * self.constants.resonant_pump_polarization_target[None, :],
            axis=-1,
        )
        # Low-frequency EOM voltage-to-Vpi ratio.
        #
        # Shape: (N,)
        unfiltered_vpi_ratio = self.eom_vpi_ratio[idx]

        # EOM Vpi response bandwidth.
        #
        # Shape: (N,)
        eom_bandwidth = self.eom_vpi_bandwidth[idx]

        # Spin-preserving optical transition frequencies.
        #
        # Shape: (N, T)
        ple_freqs = self.get_ple_freqs(idx=idx)

        # Convert the target laser frequency to the same dtype as gamma.
        laser_frequency = jnp.asarray(
            self.constants.laser_frequency,
            dtype=gamma.dtype,
        )

        # Support either one shared laser frequency or one laser frequency per
        # full-distribution member.
        if laser_frequency.ndim == 0:
            # Shared scalar laser frequency.
            #
            # Shape: () -> (N,)
            laser_frequency = jnp.broadcast_to(
                laser_frequency,
                gamma.shape,
            )
        elif laser_frequency.ndim == 1:
            # Select the laser frequency corresponding to each selected particle.
            #
            # Shape: (N,)
            laser_frequency = laser_frequency[idx]
        else:
            raise ValueError(
                "`laser_frequency` must be scalar or have shape (N,)."
            )

        # Apply the finite-bandwidth EOM response independently for each frequency
        # and each distribution member.
        #
        # eom_frequency[None, :] has shape (1, F).
        # eom_bandwidth[:, None] has shape (N, 1).
        #
        # Broadcasting produces shape (N, F).
        filtered_vpi_ratio = (
            unfiltered_vpi_ratio[:, None]
            / jnp.sqrt(
                1.0
                + (
                    eom_frequency[None, :]
                    / eom_bandwidth[:, None]
                ) ** 2
            )
        )

        # Phase-modulation index beta = pi * Vmax / Vpi.
        #
        # Shape: (N, F)
        modulation_index = (
            jnp.pi * filtered_vpi_ratio
        )

        # Integer sideband orders.
        #
        # Shape: (S,), where S = 2 * max_bessel_order + 1
        sideband_orders = jnp.arange(
            -max_bessel_order,
            max_bessel_order + 1,
            dtype=gamma.dtype,
        )

        # The Bessel helper calculates only nonnegative integer orders. Negative
        # orders use the same squared magnitude because
        #
        # J_{-n}(beta) = (-1)^n J_n(beta).
        #
        # Shape: (S,)
        abs_sideband_orders = jnp.abs(
            sideband_orders
        ).astype(jnp.int32)

        # Calculate J_n(beta) for n = 0, ..., max_bessel_order.
        #
        # Input shape:
        #     modulation_index: (N, F)
        #
        # Output shape:
        #     (max_bessel_order + 1, N, F)
        J_nonnegative = (
            _bessel_j_nonnegative_orders_integer_series(
                modulation_index,
                max_order=max_bessel_order,
                series_terms=bessel_series_terms,
            )
        )

        # Select the Bessel coefficient associated with each positive and negative
        # sideband order.
        #
        # After jnp.take:
        #     (S, N, F)
        #
        # After jnp.moveaxis:
        #     (N, F, S)
        J_sidebands = jnp.moveaxis(
            jnp.take(
                J_nonnegative,
                abs_sideband_orders,
                axis=0,
            ),
            0,
            -1,
        )

        # Effective squared optical coupling for each sideband.
        #
        # omega[:, None, None] has shape (N, 1, 1).
        # J_sidebands has shape (N, F, S).
        #
        # Output shape: (N, F, S)
        sideband_coupling_squared = (
            omega[:, None, None] ** 2
            * J_sidebands**2
        )

        # Optical frequency of each EOM sideband.
        #
        # laser_frequency[:, None, None] has shape (N, 1, 1).
        # eom_frequency[None, :, None] has shape (1, F, 1).
        # sideband_orders[None, None, :] has shape (1, 1, S).
        #
        # Output shape: (N, F, S)
        sideband_frequencies = (
            laser_frequency[:, None, None]
            + eom_frequency[None, :, None]
            * sideband_orders[None, None, :]
        )

        # Detuning between every EOM sideband and every optical transition.
        #
        # sideband_frequencies[..., None] has shape (N, F, S, 1).
        # ple_freqs[:, None, None, :] has shape (N, 1, 1, T).
        #
        # Output shape: (N, F, S, T)
        detuning = (
            sideband_frequencies[..., None]
            - ple_freqs[:, None, None, :]
        )

        # Expand gamma across frequency, sideband, and transition axes.
        #
        # Shape: (N, 1, 1, 1)
        gamma_expanded = gamma[:, None, None, None]

        # Expand the squared sideband coupling across the transition axis.
        #
        # Shape: (N, F, S, 1)
        coupling_expanded = (
            sideband_coupling_squared[..., None]
        )

        # Saturated Lorentzian scattering rate for every frequency, particle,
        # sideband, and optical transition.
        #
        # Shape: (N, F, S, T)
        sideband_rates = (
            coupling_expanded
            / self.excited_state_lifetime[idx, None, None, None]
            / (
                gamma_expanded**2
                + 2.0 * coupling_expanded
                + 4.0 * detuning**2
            )
        )

        if sum_counts:
            # Sum over the sideband and transition axes.
            #
            # Input shape:  (N, F, S, T)
            # Output shape: (N, F)
            return jnp.sum(
                sideband_rates,
                axis=(-2, -1),
            )

        # Return the complete sideband- and transition-resolved scattering rates.
        #
        # Shape: (N, F, S, T)
        return sideband_rates


    def scattering_rate(
        self,
        idx=None,
        eom_frequency=0.0,
        max_bessel_order: int = 2,
        bessel_series_terms: int = 48,
        sum_counts: bool = True,
    ):
        """Calculate optical scattering rates with scalar-index convenience.

        This method normalizes ``idx`` and delegates the numerical calculation to
        :meth:`scattering_rate_batch`. If a scalar distribution index is supplied,
        the singleton distribution axis is removed while the frequency axis is
        preserved.

        Parameters
        ----------
        idx : int or jax.Array or None, optional
            Distribution index or indices to evaluate.

            If ``None``, all distribution members are evaluated. If a scalar index
            is provided, the distribution axis is removed from the returned array.
            If an array is provided, the distribution axis is retained.

        eom_frequency : float or jax.Array, optional
            Scalar EOM frequency or one-dimensional frequency sweep in GHz.

            A scalar input is represented internally as a length-one frequency
            sweep.

        max_bessel_order : int, optional
            Largest positive and negative EOM sideband order to include.

        bessel_series_terms : int, optional
            Number of terms retained in the Bessel-function series.

        sum_counts : bool, optional
            If ``True``, sum over sidebands and optical transitions. If ``False``,
            return the full sideband- and transition-resolved result.

        Returns
        -------
        jax.Array
            For batched ``idx`` and ``sum_counts=True``, the shape is ``(N, F)``.

            For scalar ``idx`` and ``sum_counts=True``, the shape is ``(F,)``.

            For batched ``idx`` and ``sum_counts=False``, the shape is
            ``(N, F, S, T)``.

            For scalar ``idx`` and ``sum_counts=False``, the shape is
            ``(F, S, T)``.
        """
        # Record whether the caller supplied one scalar distribution index.
        scalar_idx = (
            idx is not None
            and jnp.asarray(idx).ndim == 0
        )

        if idx is None:
            # Evaluate every member of the distribution.
            #
            # Shape: (N,)
            idx = jnp.arange(
                self.weights.shape[0],
                dtype=jnp.int32,
            )
        else:
            # Normalize scalar and batched indices to a one-dimensional array.
            #
            # Scalar:
            #     () -> (1,)
            #
            # Batched:
            #     (N,) -> (N,)
            idx = jnp.atleast_1d(
                jnp.asarray(idx, dtype=jnp.int32)
            )

        rates = self.scattering_rate_batch(
            idx=idx,
            eom_frequency=eom_frequency,
            max_bessel_order=max_bessel_order,
            bessel_series_terms=bessel_series_terms,
            sum_counts=sum_counts,
        )

        if scalar_idx:
            # The distribution axis is axis 0 in both output layouts:
            #
            # sum_counts=True:  (1, F) -> (F,)
            # sum_counts=False: (1, F, S, T) -> (F, S, T)
            rates = rates[0]

        return rates
        
    @jax.jit(static_argnames=("ground_state",))
    def solve_hamiltonian_batch(self, idx, ground_state=True, experiment_idx=None):
        """Solve the static Hamiltonian for a one-dimensional particle batch.

        ``ground_state`` is static because it selects different constants and
        parameter arrays. ``experiment_idx`` is dynamic; ``None`` and a scalar
        index trace as two different pytree structures without making every
        index value a separate compilation-cache key.

        The backend evaluates one parameter point at a time.  This method maps
        that scalar backend over the selected distribution members and supplies
        the shared nuclear Zeeman ratio, zero iso-orbital coupling, and the
        particle-resolved asymmetric-Ham correction explicitly.
        """
        idx = jnp.asarray(idx, dtype=jnp.int32)

        if ground_state:
            q = params.q
            L = params.L
            A = params.A_GND_TENSORS[self.hyperfine_neighbor_idx[idx]]
            Ax = params.AX_GND_TENSORS[self.hyperfine_neighbor_idx[idx]]
            Ay = params.AY_GND_TENSORS[self.hyperfine_neighbor_idx[idx]]
            alpha = self.alpha[idx]
            beta = self.beta[idx]
            delta_f = self._select_scalar_parameter_batch(
                params.delta_f_gnd,
                idx,
            )
        else:
            q = params.q_exc
            L = params.L_exc
            A = params.A_EXC_TENSORS[self.hyperfine_neighbor_idx[idx]]
            Ax = params.AX_EXC_TENSORS[self.hyperfine_neighbor_idx[idx]]
            Ay = params.AY_EXC_TENSORS[self.hyperfine_neighbor_idx[idx]]
            alpha = self.alpha_exc[idx]
            beta = self.beta_exc[idx]
            delta_f = self._select_scalar_parameter_batch(
                params.delta_f_exc,
                idx,
            )

        if experiment_idx is not None:
            experiment_idx = jnp.atleast_1d(
                jnp.asarray(experiment_idx, dtype=jnp.int32)
            )
            B, theta, phi = self.get_B_spherical_batch(
                experiment_idx,
                frame="dipole",
            )
            B = B[0]
            theta = theta[0]
            phi = phi[0]
            in_axes = (
                None, None, None,
                0, None,
                0, 0, 0,
                None,
                0, 0,
                None,
                0,
            )
        else:
            B, theta, phi = self.get_B_spherical_batch(
                idx,
                frame="dipole",
            )
            in_axes = (
                0, 0, 0,
                0, None,
                0, 0, 0,
                None,
                0, 0,
                None,
                0,
            )

        return jax.vmap(
            qh_jqt.solve_hamiltonian,
            in_axes=in_axes,
        )(
            B,
            theta,
            phi,
            params.rg[self.hyperfine_neighbor_idx[idx]],
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

    def solve_hamiltonian(self, idx=None, ground_state=True, experiment_idx=None):
        scalar = idx is not None and jnp.asarray(idx).ndim == 0

        if idx is None:
            idx = jnp.arange(self.weights.shape[0], dtype=jnp.int32)
        else:
            idx = jnp.atleast_1d(jnp.asarray(idx, dtype=jnp.int32))

        result = self.solve_hamiltonian_batch(idx, ground_state, experiment_idx)

        if scalar:
            return jax.tree_util.tree_map(lambda x: x[0], result)

        return result

    @jax.jit
    def get_cyclicity_batch(self, idx, experiment_idx=None):
        """Calculate spontaneous-emission branching ratios for a batch."""
        def calculate_one(
            B,
            theta,
            phi,
            alpha,
            beta,
            alpha_exc,
            beta_exc,
            rg,
            A_gnd,
            Ax_gnd,
            Ay_gnd,
            A_exc,
            Ax_exc,
            Ay_exc,
            delta_f_gnd,
            delta_f_exc,
        ):
            return qh_jqt.calculate_spontaneous_cyclicity(
                B,
                theta,
                phi,
                alpha=alpha,
                beta=beta,
                alpha_exc=alpha_exc,
                beta_exc=beta_exc,
                rg=rg,
                A_gnd=A_gnd,
                Ax_gnd=Ax_gnd,
                Ay_gnd=Ay_gnd,
                A_exc=A_exc,
                Ax_exc=Ax_exc,
                Ay_exc=Ay_exc,
                delta_f_gnd=delta_f_gnd,
                delta_f_exc=delta_f_exc,
                included_states=None,
            )

        idx = jnp.asarray(idx, dtype=jnp.int32)
        delta_f_gnd = self._select_scalar_parameter_batch(
            params.delta_f_gnd,
            idx,
        )
        delta_f_exc = self._select_scalar_parameter_batch(
            params.delta_f_exc,
            idx,
        )

        if experiment_idx is not None:
            experiment_idx = jnp.atleast_1d(
                jnp.asarray(experiment_idx, dtype=jnp.int32)
            )
            B, theta, phi = self.get_B_spherical_batch(
                experiment_idx,
                frame="dipole",
            )
            return jax.vmap(
                calculate_one,
                in_axes=(
                    None, None, None,
                    0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0,
                    0, 0,
                ),
            )(
                B[0],
                theta[0],
                phi[0],
                self.alpha[idx],
                self.beta[idx],
                self.alpha_exc[idx],
                self.beta_exc[idx],
                params.rg[self.hyperfine_neighbor_idx[idx]],
                params.A_GND_TENSORS[self.hyperfine_neighbor_idx[idx]],
                params.AX_GND_TENSORS[self.hyperfine_neighbor_idx[idx]],
                params.AY_GND_TENSORS[self.hyperfine_neighbor_idx[idx]],
                params.A_EXC_TENSORS[self.hyperfine_neighbor_idx[idx]],
                params.AX_EXC_TENSORS[self.hyperfine_neighbor_idx[idx]],
                params.AY_EXC_TENSORS[self.hyperfine_neighbor_idx[idx]],
                delta_f_gnd,
                delta_f_exc,
            )

        B, theta, phi = self.get_B_spherical_batch(
            idx,
            frame="dipole",
        )
        return jax.vmap(calculate_one)(
            B,
            theta,
            phi,
            self.alpha[idx],
            self.beta[idx],
            self.alpha_exc[idx],
            self.beta_exc[idx],
            params.rg[self.hyperfine_neighbor_idx[idx]],
            params.A_GND_TENSORS[self.hyperfine_neighbor_idx[idx]],
            params.AX_GND_TENSORS[self.hyperfine_neighbor_idx[idx]],
            params.AY_GND_TENSORS[self.hyperfine_neighbor_idx[idx]],
            params.A_EXC_TENSORS[self.hyperfine_neighbor_idx[idx]],
            params.AX_EXC_TENSORS[self.hyperfine_neighbor_idx[idx]],
            params.AY_EXC_TENSORS[self.hyperfine_neighbor_idx[idx]],
            delta_f_gnd,
            delta_f_exc,
        )

    def get_cyclicity(self, idx=None, experiment_idx=None):
        scalar = idx is not None and jnp.asarray(idx).ndim == 0

        if idx is None:
            idx = jnp.arange(self.weights.shape[0], dtype=jnp.int32)
        else:
            idx = jnp.atleast_1d(jnp.asarray(idx, dtype=jnp.int32))

        result = self.get_cyclicity_batch(idx, experiment_idx)

        if scalar:
            return jax.tree_util.tree_map(lambda x: x[0], result)

        return result

    def get_ple_freqs(self, idx=None, experiment_idx=None):
        E, Eref, U, U_states, alignment = self.solve_hamiltonian(
            idx=idx,
            ground_state=True,
            experiment_idx=experiment_idx,
        )
        E_exc, Eref_exc, U_exc, U_exc_states, alignment_exc = self.solve_hamiltonian(
            idx=idx,
            ground_state=False,
            experiment_idx=experiment_idx,
        )

        # Spin-preserving ground-orbital transition frequencies.
        return E_exc[..., :4] - E[..., :4] + params.LEVEL_OFFSET

    def get_emr_freqs(self, idx=None, experiment_idx=None):
        """
        Returns the electron magnetic resonance frequencies.
        """
        E, Eref, U, U_states, alignment = self.solve_hamiltonian(
            idx=idx,
            ground_state=True,
            experiment_idx=experiment_idx,
        )
        return jnp.stack(
            [
                E[..., 2] - E[..., 0],
                E[..., 3] - E[..., 1],
            ],
            axis=-1,
        )

    def get_nmr_freqs(self, idx=None, experiment_idx=None):
        """
        Returns the nuclear magnetic resonance frequencies.
        """
        E, Eref, U, U_states, alignment = self.solve_hamiltonian(
            idx=idx,
            ground_state=True,
            experiment_idx=experiment_idx,
        )
        return jnp.stack(
            [
                E[..., 1] - E[..., 0],
                E[..., 3] - E[..., 2],
            ],
            axis=-1,
        )

    def get_init_timestep(
        self,
        idx=0,
        electron_state=0,
        nuclear_state=None,
        decay_target=0.001,
    ):
        """Return an initialization time step.

        This routine was incomplete in the supplied source: it calculated a pair
        of excitation frequencies but never used ``nuclear_state`` or
        ``decay_target`` and always returned ``None``. An explicit exception is
        safer than silently propagating an invalid time step.
        """
        raise NotImplementedError(
            "`get_init_timestep` needs a specified pumping/decay model before "
            "a physically meaningful time step can be calculated."
        )

    @jax.jit(static_argnames=("included_states",))
    def get_excitation_hamiltonian_batch(self, idx, included_states=None):
        """Construct dynamic Hamiltonians for a one-dimensional index batch.

        ``included_states`` must be ``None`` or a hashable static value such as
        a tuple, because it changes the returned operator structure.  The public
        frontend continues to accept the existing two-angle optical and
        microwave orientation representations; this adapter converts them to
        the scalar Cartesian/angular components required by the backend.
        """
        def calculate_one(
            B,
            theta,
            phi,
            laser_frequency,
            resonant_pump_polarization,
            excited_state_lifetime,
            B_drive_strength,
            B_drive_orientation,
            alpha,
            beta,
            alpha_exc,
            beta_exc,
            rg,
            A_gnd,
            Ax_gnd,
            Ay_gnd,
            A_exc,
            Ax_exc,
            Ay_exc,
            delta_f_gnd,
            delta_f_exc,
        ):
            # Preserve the previous frontend convention: the two pump values
            # are interpreted as spherical (theta, phi) angles and converted to
            # Cartesian dipole components before entering the scalar backend.
            pump_eta = angles_to_cartesian(
                resonant_pump_polarization[0],
                resonant_pump_polarization[1],
            )

            return qh_jqt.get_dynamic_hamiltonian(
                B=B,
                theta=theta,
                phi=phi,
                excited_ground_split=params.LEVEL_OFFSET - laser_frequency,
                excited_state_lifetime=excited_state_lifetime,
                pump_eta_x=pump_eta[0],
                pump_eta_y=pump_eta[1],
                pump_eta_z=pump_eta[2],
                B_drive_strength=B_drive_strength,
                B_drive_theta=B_drive_orientation[0],
                B_drive_phi=B_drive_orientation[1],
                alpha=alpha,
                beta=beta,
                alpha_exc=alpha_exc,
                beta_exc=beta_exc,
                rg=rg,
                A_gnd=A_gnd,
                Ax_gnd=Ax_gnd,
                Ay_gnd=Ay_gnd,
                A_exc=A_exc,
                Ax_exc=Ax_exc,
                Ay_exc=Ay_exc,
                delta_f_gnd=delta_f_gnd,
                delta_f_exc=delta_f_exc,
                included_states=included_states,
            )

        idx = jnp.asarray(idx, dtype=jnp.int32)
        B, theta, phi = self.get_B_spherical_batch(
            idx,
            frame="dipole",
        )

        laser_frequency = self._select_scalar_parameter_batch(
            self.constants.laser_frequency,
            idx,
        )
        resonant_pump_polarization = jnp.asarray(
            self.constants.resonant_pump_polarization_target
        )
        resonant_pump_polarization = jnp.broadcast_to(
            resonant_pump_polarization,
            (idx.shape[0], resonant_pump_polarization.shape[0]),
        )
        delta_f_gnd = self._select_scalar_parameter_batch(
            params.delta_f_gnd,
            idx,
        )
        delta_f_exc = self._select_scalar_parameter_batch(
            params.delta_f_exc,
            idx,
        )

        return jax.vmap(calculate_one)(
            B,
            theta,
            phi,
            laser_frequency,
            resonant_pump_polarization,
            self.excited_state_lifetime[idx],
            self.mw_B_magnitude[idx],
            self.mw_B_orientation[idx],
            self.alpha[idx],
            self.beta[idx],
            self.alpha_exc[idx],
            self.beta_exc[idx],
            params.rg[self.hyperfine_neighbor_idx[idx]],
            params.A_GND_TENSORS[self.hyperfine_neighbor_idx[idx]],
            params.AX_GND_TENSORS[self.hyperfine_neighbor_idx[idx]],
            params.AY_GND_TENSORS[self.hyperfine_neighbor_idx[idx]],
            params.A_EXC_TENSORS[self.hyperfine_neighbor_idx[idx]],
            params.AX_EXC_TENSORS[self.hyperfine_neighbor_idx[idx]],
            params.AY_EXC_TENSORS[self.hyperfine_neighbor_idx[idx]],
            delta_f_gnd,
            delta_f_exc,
        )

    def get_excitation_hamiltonian(self, idx=None, included_states=None):
        scalar = idx is not None and jnp.asarray(idx).ndim == 0

        if idx is None:
            idx = jnp.arange(self.weights.shape[0], dtype=jnp.int32)
        else:
            idx = jnp.atleast_1d(jnp.asarray(idx, dtype=jnp.int32))

        result = self.get_excitation_hamiltonian_batch(idx, included_states)

        if scalar:
            return jax.tree_util.tree_map(lambda x: x[0], result)

        return result


    
    @jax.jit(static_argnames=("included_states",))
    def get_ground_hamiltonian_batch(self, idx, included_states=None):
        """Construct ground-state Hamiltonians for an index batch.

        ``included_states`` must be ``None`` or a hashable static value such as
        a tuple, because it changes the returned operator structure.  The
        frontend retains its two-angle microwave orientation while this adapter
        passes separate scalar angles to the backend.
        """
        def calculate_one(
            B,
            theta,
            phi,
            B_drive_strength,
            B_drive_orientation,
            alpha,
            beta,
            rg,
            A_gnd,
            Ax_gnd,
            Ay_gnd,
            delta_f_gnd,
        ):
            return qh_jqt.get_ground_hamiltonian(
                B=B,
                theta=theta,
                phi=phi,
                B_drive_strength=B_drive_strength,
                B_drive_theta=B_drive_orientation[0],
                B_drive_phi=B_drive_orientation[1],
                alpha=alpha,
                beta=beta,
                rg=rg,
                A=A_gnd,
                Ax=Ax_gnd,
                Ay=Ay_gnd,
                delta_f=delta_f_gnd,
                included_states=included_states,
            )

        idx = jnp.asarray(idx, dtype=jnp.int32)
        B, theta, phi = self.get_B_spherical_batch(
            idx,
            frame="dipole",
        )
        delta_f_gnd = self._select_scalar_parameter_batch(
            params.delta_f_gnd,
            idx,
        )

        return jax.vmap(calculate_one)(
            B,
            theta,
            phi,
            self.mw_B_magnitude[idx],
            self.mw_B_orientation[idx],
            self.alpha[idx],
            self.beta[idx],
            params.rg[self.hyperfine_neighbor_idx[idx]],
            params.A_GND_TENSORS[self.hyperfine_neighbor_idx[idx]],
            params.AX_GND_TENSORS[self.hyperfine_neighbor_idx[idx]],
            params.AY_GND_TENSORS[self.hyperfine_neighbor_idx[idx]],
            delta_f_gnd,
        )

    def get_ground_hamiltonian(self, idx=None, included_states=None):
        scalar = idx is not None and jnp.asarray(idx).ndim == 0

        if idx is None:
            idx = jnp.arange(self.weights.shape[0], dtype=jnp.int32)
        else:
            idx = jnp.atleast_1d(jnp.asarray(idx, dtype=jnp.int32))

        result = self.get_ground_hamiltonian_batch(idx, included_states)

        if scalar:
            return jax.tree_util.tree_map(lambda x: x[0], result)

        return result

    @jax.jit(static_argnames=("included_states", "saveat_final_only", "solver_options_args"))
    def _drive_mw_hamiltonian_batch(
        self,
        pulse,
        idx,
        tau,
        psi0,
        included_states,
        saveat_final_only=False,
        solver_options_args=None,
        scale=1e9
    ):
        """Run Hamiltonian evolutions for a batch of distribution indices.

        Each distribution member is solved independently, and the results are
        stacked along a leading batch dimension.

        Args:
            pulse: Analog pulse to apply.
            idx: One-dimensional array of distribution member indices.
            tau: Fixed-shape time array in seconds.
            psi0: Initial quantum state shared by all distribution members.
            included_states: Static tuple identifying the included eigenstates.

        Returns:
            A tuple ``(states, filter_states, populations)`` where every array leaf
            has a leading batch dimension corresponding to ``idx``.
        """
        idx = jnp.atleast_1d(jnp.asarray(idx, dtype=jnp.int32))

        dimension = len(included_states)
        sampling_rate = jnp.asarray(self.constants.sampling_rate)*scale
        sample_period = 1.0 / sampling_rate
        pulse_center = pulse.length / 2

        # This coefficient is identical for every distribution member. Only the
        # Hamiltonians and filter cutoff frequency depend on the member index.
        def H_b_drive(t, args=None):
            del args

            local_times = jnp.stack(
                (
                    t - sample_period,
                    t,
                    t + sample_period,
                )
            )

            waveform, _, _ = synthesize_analog_pulse(
                pulse=pulse,
                tau=local_times,
                at_time=pulse_center,
                dphase=0.0,
                all_info=True,
            )

            return waveform[1]

        # These are static because included_states is static.
        projectors = tuple(
            jqt.basis(dimension, state_index).to_dm()
            for state_index in range(dimension)
        )
        if saveat_final_only:
            saveat_tlist = tau[-2:]#SaveAt(t1=True)
        else:
            saveat_tlist = tau
        if solver_options_args is None:
            solver_options = jqt.SolverOptions.create(
                progress_meter=False,
                solver="Dopri5",
                rtol=1e-5,
                atol=1e-7,
            )
        else:
            solver_options = jqt.SolverOptions.create(*solver_options_args)

        def solve_single(single_idx):
            """Solve the evolution for one scalar distribution index."""
            H0, Hb = self.get_ground_hamiltonian(
                idx=single_idx,
                included_states=included_states,
            )

            H0_rads_s = 2.0 * jnp.pi * H0 * scale
            Hb_rads_s = 2.0 * jnp.pi * Hb * scale

            omega_c = (
                2.0
                * jnp.pi
                * self.mw_B_bandwidth[single_idx]
                * scale
            )

            states, filter_states = jqt.sesolve_components(
                hamiltonians=(
                    H0_rads_s,
                    Hb_rads_s,
                ),
                coefficients=(
                    1.0,
                    H_b_drive,
                ),
                psi0=psi0,
                tlist=tau,
                saveat_tlist=saveat_tlist,
                filters=(
                    None,
                    lowpass_filter(omega_c),
                ),
                filter_y0s=(
                    None,
                    jnp.asarray(0.0, dtype=jnp.complex128),
                ),
                return_filter_states=True,
                solver_options=solver_options,
            )

            populations = jnp.stack(
                tuple(
                    jnp.real(jqt.overlap(projector, states))
                    for projector in projectors
                ),
                axis=0,
            )

            return states, filter_states, populations

        # Each invocation receives a scalar index. All output array leaves are
        # stacked along axis 0.
        return jax.vmap(solve_single, in_axes=0, out_axes=0)(idx)


    def drive_mw_hamiltonian(
        self,
        idx,
        pulse,
        included_states=(0, 1, 2, 3),
        psi0=None,
        saveat_final_only=False,
        solver_options_args=None,
        scale=1e9
    ):
        """Construct the time grid and run one or more Hamiltonian evolutions.

        Args:
            dist: Distribution containing the Hamiltonian parameters.
            idx: Scalar index, array of indices, or ``None`` for all members.
            pulse: Analog pulse to apply.
            included_states: Tuple identifying the included eigenstates.
            psi0: Initial quantum state. Defaults to the first basis state.
            scale: Scale factor for the time array.
        Returns:
            A tuple containing the quantum states, filter states, and populations.

            For a scalar ``idx``, the leading batch dimension is removed. For an
            array or ``None``, the leading dimension corresponds to distribution
            members.
        """
        idx_array, scalar_idx = self._prepare_indices(idx)
        dimension = len(included_states)

        if psi0 is None:
            psi0 = jqt.basis(dimension, 0)

        # Keep time-grid construction outside JIT because its output length is
        # determined using Python values.
        sampling_rate = float(
            np.asarray(self.constants.sampling_rate)*scale
        )
        sample_period = 1.0 / sampling_rate

        tau = make_analog_pulse_time_array(
            pulse=pulse,
            sample_period=sample_period,
            at_time=float(np.asarray(pulse.length)) / 2.0,
        )

        states, filter_states, populations = self._drive_mw_hamiltonian_batch(
            pulse=pulse,
            idx=idx_array,
            tau=tau,
            psi0=psi0,
            included_states=tuple(included_states),
            saveat_final_only=saveat_final_only,
            solver_options_args=solver_options_args,
            scale=scale,
        )

        if scalar_idx:
            states = states[0]

            # filter_states is a pytree organized by Hamiltonian/filter component.
            # Index every array leaf instead of indexing the outer component list.
            filter_states = jax.tree_util.tree_map(
                lambda leaf: leaf[0],
                filter_states,
            )

            populations = populations[0]

        return states, filter_states, populations

