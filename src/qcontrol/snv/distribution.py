from typing import NamedTuple
from enum import IntEnum
from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import numpy as np
import jax.scipy.special as jsp_special
import qcontrol.snv120.parameters as params
import qcontrol.snv120.hamiltonian_jqt as qh_jqt
from qcontrol.snv120.jqt_ext import sesolve_components, mesolve_components
import jaxquantum as jqt
from pulseseq.sequencing.waveform import AnalogPulse, Apodization, Shape, DigitalPulse, DigitalType
import qcontrol.snv120.pulseseq_interconnect
from qcontrol.snv120.pulseseq_interconnect import stack_waveforms, make_analog_pulse_time_array, synthesize_analog_pulse

from pulseseq.sequencing.waveform import AnalogPulse, Apodization, Shape, DigitalPulse, DigitalType

def lowpass_filter(omega_c, omega_r=0.0):
    """Return a first-order low-pass filter, where the DC frequency lies at -omega_r.

    The transfer function is

        H(s) = omega_c / (s + omega_c).

    Args:
        omega_c: Angular cutoff frequency in radians per second.
        omega_r: Rotating frame frequency in radians per second.
    Returns:
        A filter function compatible with ``sesolve_components``.
    """
    omega_c = jnp.asarray(omega_c)

    def filter_fn(t, z, u):
        dz_dt = omega_c * (u - z)
        z_out = jnp.exp(-1j*omega_r*t) * z
        return dz_dt, z_out

    return filter_fn

def eom_lowpass_filter(omega_c, omega_r=0.0, Vpi_ratio=1.0):
    """Return a first-order low-pass filter, where the DC frequency lies at -omega_r, followed by EOM phase modulation.

    The transfer function is

        H(s) = omega_c / (s + omega_c).

    Args:
        omega_c: Angular cutoff frequency in radians per second.
        omega_r: Rotating frame frequency in radians per second.
        Vpi_ratio: V_max/V_pi ratio for the EOM.  The output of the filter is multiplied by this factor before being converted to a phase modulation.
    Returns:
        A filter function compatible with ``sesolve_components``.
    """
    omega_c = jnp.asarray(omega_c)

    def filter_fn(t, z, u):
        dz_dt = omega_c * (u - z)
        z_out = jnp.exp(1j*(jnp.pi*Vpi_ratio*z - omega_r*t))
        return dz_dt, z_out

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


def _dipole_basis_crystal_from_axis(dipole_z_crystal):
    """Construct the paper-compatible local SnV frame in crystal coordinates.

    The hyperfine tensors in Table XII of Mohseni *et al.* use the frame shown
    in their Fig. 8(e).  For a [111] defect this is

        X = [2, -1, -1] / sqrt(6),
        Y = [0,  1, -1] / sqrt(2),
        Z = [1,  1,  1] / sqrt(3).

    Equivalently, X is the normalized projection of crystal [100] onto the
    plane perpendicular to the selected defect axis, and Y = Z x X.  This
    definition generalizes consistently to every selected <111> orientation.

    Parameters
    ----------
    dipole_z_crystal : array_like, shape (..., 3)
        Selected defect-axis directions in crystal coordinates.

    Returns
    -------
    jax.Array, shape (..., 3, 3)
        Matrix whose columns are the local X, Y, and Z basis vectors expressed
        in crystal coordinates.
    """
    dipole_z_crystal = normalize_vectors(dipole_z_crystal)

    crystal_x = jnp.asarray(
        [1.0, 0.0, 0.0],
        dtype=dipole_z_crystal.dtype,
    )

    dipole_x_crystal = normalize_vectors(
        crystal_x
        - jnp.sum(
            crystal_x * dipole_z_crystal,
            axis=-1,
            keepdims=True,
        )
        * dipole_z_crystal
    )

    dipole_y_crystal = normalize_vectors(
        jnp.cross(
            dipole_z_crystal,
            dipole_x_crystal,
        )
    )

    return jnp.stack(
        [
            dipole_x_crystal,
            dipole_y_crystal,
            dipole_z_crystal,
        ],
        axis=-1,
    )

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
    target_dipole_operator: jnp.ndarray # (x,y,z) vector in the local dipole frame. May be impossible to reach given only 2 modes, but the closest orthogonal projection will be solved for
    laser_frequency: float            # laser frequency, units of GHz
    diamond_lattice_100_orientation: jnp.ndarray # Best estimate of (theta, phi) for the diamond lattice [100] axis in the lab frame
    diamond_lattice_011_orientation: jnp.ndarray # Best estimate of (theta, phi) for the diamond lattice [011] axis in the lab frame
    nominal_magnet_axes: jnp.ndarray        # (3, 3) nominal magnet-axis unit vectors in the lab frame
    sampling_rate: float                    # awg sampling rate, units of GHz
    # Physical constants
    mu_B_GHz_per_T = 13.996 # [GHz/T]
    dipole_crystal_axes = jnp.asarray([ # Cartesian Unit vector relative to crystal frame
        [1,1,1],
        [1,-1,1],
        [-1,1,1],
        [-1,-1,1]
    ])
    # tperp: float # Perpendicular strain susceptibility, units of GHz/strain
    # tpar: float # Parallel strain susceptibility, units of GHz/strain
    # f_gnd: float # Transverse orbital mixing strain susceptibility, units of GHz/strain
    # d_gnd: float # Shear orbital splitting strain susceptibility, units of GHz/strain
    # f_exc: float # Transverse orbital mixing strain susceptibility, units of GHz/strain
    # d_exc: float # Shear orbital splitting strain susceptibility, units of GHz/strain

class SnVControlState(NamedTuple):
    """
    Control state for the SnV120 qubit.
    """
    magnet_settings: jnp.ndarray # (3,) physical magnet settings in tesla
    waveplate_angles: jnp.ndarray # (3,) [QWP1, HWP, QWP2] angles in radians

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

    # Hamiltonian parameters
    strain_params: jnp.ndarray # (N, 5) zpl_shift, alpha, beta, alpha_exc, beta_exc. Old def: epsilon strain tensor in the dipole frame. Order is (xx, yy, zz, xy, xz, yz)
    hyperfine_neighbor_idx: jnp.ndarray # N, HyperfineNeighbor IDs
    
    excited_state_lifetime: jnp.ndarray # N, in units of nanoseconds
    # Collection efficiency parameters
    debye_waller_factor: jnp.ndarray  # N
    quantum_efficiency: jnp.ndarray   # N
    optical_transmission: jnp.ndarray # N
    dark_count_rate: jnp.ndarray      # N

    # Resonant pump coupling
    resonant_pump_coupling_rate: jnp.ndarray # (N), in units of GHz for the TE mode.
    resonant_pump_pdl : jnp.ndarray # (N), dB lowering of the TM mode relative to the TE mode due to polarization-dependent loss in the PIC and diamond. If negative, the TM mode is stronger than the TE mode.
    resonant_pump_polarization: jnp.ndarray  # N, Angle of the resonant pump polarization with respect to the TE mode without factoring in the waveplates
    resonant_pump_phase: jnp.ndarray  # N, Phase difference between the TE and TM modes without factoring in the waveplates
    # transmission_to_diamond: jnp.ndarray # (N, 2), Transmission from the edge coupler to the diamond for the TE and TM modes
    mode_field_orientation: jnp.ndarray # (N, 2 (TE/TM), 2 (theta,phi)), Electric field orientation for the TE and TM modes in the crystal frame
    transmission_out_diamond: jnp.ndarray # (N, 2), Transmission from the diamond to the edge coupler for the TE and TM modes
    reflection_from_pic: jnp.ndarray # (N, 2), Reflection of the resonant pump from the PIC to the APD for the TE and TM modes

    # EOM drive settings
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

    # @jax.jit
    # def get_zpl_shift(self, idx):
        # eps_xx = self.strain_tensor[idx, 0]
        # eps_yy = self.strain_tensor[idx, 1]
        # eps_zz = self.strain_tensor[idx, 2]
        # return self.constants.tperp * (eps_xx + eps_yy) + self.constants.tpar * eps_zz

    # @jax.jit
    # def get_alpha_beta(self, idx):
        # eps_xx = self.strain_tensor[idx, 0]
        # eps_yy = self.strain_tensor[idx, 1]
        # eps_xy = self.strain_tensor[idx, 3]
        # eps_xz = self.strain_tensor[idx, 4]
        # eps_yz = self.strain_tensor[idx, 5]
        # alpha_gnd = -self.constants.d_gnd * (eps_xx - eps_yy) - self.constants.f_gnd * eps_xz
        # alpha_exc = -self.constants.d_exc * (eps_xx - eps_yy) - self.constants.f_exc * eps_xz
        # beta_gnd = 2*self.constants.d_gnd * eps_xy - self.constants.f_gnd * eps_yz
        # beta_exc = 2*self.constants.d_exc * eps_xy - self.constants.f_exc * eps_yz
        # return alpha_gnd, beta_gnd, alpha_exc, beta_exc

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

    def _prepare_control_state(
        self,
        control_state: SnVControlState | None,
    ) -> SnVControlState | None:
        """Validate and normalize a shared physical control state.

        A supplied control state is shared by every distribution member selected
        by ``idx``.  ``None`` retains the per-member-optimal behavior: each
        selected member uses its own magnet settings and waveplate angles.
        """
        if control_state is None:
            return None

        if not isinstance(control_state, SnVControlState):
            raise TypeError(
                "`control_state` must be an SnVControlState or None."
            )

        magnet_settings = jnp.asarray(control_state.magnet_settings)
        waveplate_angles = jnp.asarray(control_state.waveplate_angles)

        if magnet_settings.ndim != 1 or magnet_settings.shape[0] != 3:
            raise ValueError(
                "`control_state.magnet_settings` must have shape (3,)."
            )

        # The optical model uses a QWP-HWP-QWP sequence, hence three angles.
        if waveplate_angles.ndim != 1 or waveplate_angles.shape[0] != 3:
            raise ValueError(
                "`control_state.waveplate_angles` must have shape (3,) "
                "for [QWP1, HWP, QWP2]."
            )

        return SnVControlState(
            magnet_settings=magnet_settings,
            waveplate_angles=waveplate_angles,
        )

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

        # The selected <111> crystal direction defines the dipole-frame Z axis.
        # X and Y follow the Fig. 8(e) convention used by the carbon hyperfine
        # tensors in Table XII.  For the canonical [111] orientation this gives
        # X=[2,-1,-1]/sqrt(6) and Y=[0,1,-1]/sqrt(2).
        #
        # Shape:
        #     (K, 3)
        dipole_z_crystal = jnp.asarray(
            self.constants.dipole_crystal_axes,
            dtype=magnet_axes_crystal.dtype,
        )[
            self.dipole_crystal_axis_idx[idx]
        ]

        # Columns are dipole basis vectors expressed in crystal coordinates.
        #
        # Shape:
        #     (K, 3, 3)
        dipole_to_crystal = _dipole_basis_crystal_from_axis(
            dipole_z_crystal
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
            self.constants.mu_B_GHz_per_T * params.gS
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
    def get_B_cartesian_batch(
        self,
        idx,
        frame="lab",
        control_state: SnVControlState | None = None,
    ):
        """Return realized magnetic-field vectors in tesla.

        Parameters
        ----------
        idx : jax.Array, shape (K,)
            Distribution indices.
        frame : {"lab", "crystal", "dipole"}, optional
            Output coordinate frame; static under JIT.
        control_state : SnVControlState or None, optional
            Explicit physical controls shared by every selected member. This
            method uses ``control_state.magnet_settings``. If ``None``, each
            member in ``idx`` uses its own optimal magnet settings.

        Returns
        -------
        jax.Array, shape (K, 3)
            Magnetic-field Cartesian components in tesla.
        """
        idx = jnp.asarray(idx, dtype=jnp.int32)
        if idx.ndim != 1:
            raise ValueError("`idx` must be a one-dimensional integer array.")

        control_state = self._prepare_control_state(control_state)

        if control_state is None:
            # Preserve the original per-member-optimal behavior.
            B_settings = self.get_B_settings_batch(idx)
        else:
            # Apply one explicit physical vector-magnet setting to every model
            # member. Each member still supplies its own magnet calibration.
            shared_B_settings = control_state.magnet_settings
            B_settings = jnp.broadcast_to(
                shared_B_settings,
                (idx.shape[0], shared_B_settings.shape[0]),
            )

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
    def get_B_spherical_batch(
        self,
        idx,
        frame="dipole",
        control_state: SnVControlState | None = None,
    ):
        """Return field magnitude in GHz and direction angles for an index batch."""
        B_T = self.get_B_cartesian_batch(
            idx,
            frame=frame,
            control_state=control_state,
        )
        theta, phi = cartesian_to_angles(B_T)
        magnitude_GHz = jnp.linalg.norm(B_T, axis=-1) * (
            self.constants.mu_B_GHz_per_T * params.gS
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

    def get_resonant_pump_eta(
        self,
        idx=None,
        control_state: SnVControlState | None = None,
    ):
        """Return complex dipole-frame resonant pump couplings in GHz.

        Parameters
        ----------
        idx : int, array_like, or None, optional
            Distribution indices. If None, return results for all members.
            A scalar index returns shape ``(3,)``. A one-dimensional index 
            array returns shape ``(K, 3)``.
        control_state : SnVControlState or None, optional
            Explicit physical controls shared by every selected member. This
            method uses ``control_state.waveplate_angles`` and propagates those
            angles through each member's own source polarization, PDL, mode
            directions, and coupling rate. If ``None``, each member uses its
            own optimal QWP-HWP-QWP angles.

        Returns
        -------
        jax.Array, shape (..., 3)
            Complex dipole-frame pump couplings in GHz.
        """
        idx_array, scalar_idx = self._prepare_indices(idx)
        result = self._get_resonant_pump_eta_batch(
            idx_array,
            control_state=control_state,
        )
        return result[0] if scalar_idx else result

    @jax.jit
    def _get_resonant_pump_eta_batch(
        self,
        idx,
        control_state: SnVControlState | None = None,
    ):
        """Return complex dipole-frame pump couplings, shape (N, 3), in GHz.

        If ``control_state`` is supplied, its explicit QWP-HWP-QWP angles
        are shared across the full ``idx`` batch. All optical-device parameters
        remain resolved by ``idx`` so that each model member predicts the
        realized coupling under those common physical settings. If it is
        ``None``, each member uses its own optimal waveplate angles.
        """
        idx = jnp.atleast_1d(jnp.asarray(idx, dtype=jnp.int32))
        if idx.ndim != 1:
            raise ValueError("`idx` must be a one-dimensional integer array.")

        control_state = self._prepare_control_state(control_state)

        # Initial Jones vector in the TE/TM basis:
        #
        #     [cos(alpha), exp(i * phase) sin(alpha)].
        pump_alpha = self.resonant_pump_polarization[idx]
        pump_phase = self.resonant_pump_phase[idx]
        polarization = jnp.stack(
            [
                jnp.cos(pump_alpha),
                jnp.exp(1j * pump_phase) * jnp.sin(pump_alpha),
            ],
            axis=-1,
        )

        # QWP-HWP-QWP settings that produce the required pre-PDL state.
        if control_state is None:
            # Preserve the original per-member-optimal behavior.
            waveplate_angles = self.get_waveplate_angles_batch(idx)
        else:
            shared_waveplate_angles = control_state.waveplate_angles
            waveplate_angles = jnp.broadcast_to(
                shared_waveplate_angles,
                (idx.shape[0], shared_waveplate_angles.shape[0]),
            )

        def waveplate_jones(theta, delta):
            """Return W(theta, delta) for a batch of waveplate angles."""
            c = jnp.cos(theta)
            s = jnp.sin(theta)
            phase_delay = jnp.exp(
                1j * jnp.asarray(delta, dtype=theta.dtype)
            )

            return jnp.stack(
                [
                    jnp.stack(
                        [
                            c**2 + phase_delay * s**2,
                            c * s * (1.0 - phase_delay),
                        ],
                        axis=-1,
                    ),
                    jnp.stack(
                        [
                            c * s * (1.0 - phase_delay),
                            s**2 + phase_delay * c**2,
                        ],
                        axis=-1,
                    ),
                ],
                axis=-2,
            )

        QWP1 = waveplate_jones(
            waveplate_angles[:, 0],
            0.5 * jnp.pi,
        )
        HWP = waveplate_jones(
            waveplate_angles[:, 1],
            jnp.pi,
        )
        QWP2 = waveplate_jones(
            waveplate_angles[:, 2],
            0.5 * jnp.pi,
        )

        # eta_final = QWP2 @ HWP @ QWP1 @ eta_initial.
        polarization = jnp.einsum(
            "nij,nj->ni",
            QWP1,
            polarization,
        )
        polarization = jnp.einsum(
            "nij,nj->ni",
            HWP,
            polarization,
        )
        polarization = jnp.einsum(
            "nij,nj->ni",
            QWP2,
            polarization,
        )

        # The waveplates are unitary. This only removes numerical roundoff.
        polarization = polarization / jnp.linalg.norm(
            polarization,
            axis=-1,
            keepdims=True,
        )

        # TE/TM electric-field directions in crystal coordinates.
        #
        # Shape: (N, 2, 3).
        mode_vectors_crystal = angles_to_cartesian(
            self.mode_field_orientation[idx, :, 0],
            self.mode_field_orientation[idx, :, 1],
        )

        # Construct the local SnV dipole frame.
        dipole_z_crystal = jnp.asarray(
            self.constants.dipole_crystal_axes,
            dtype=mode_vectors_crystal.dtype,
        )[self.dipole_crystal_axis_idx[idx]]

        dipole_to_crystal = _dipole_basis_crystal_from_axis(
            dipole_z_crystal
        )

        # Under the row-vector convention used elsewhere in the class,
        #
        #     v_dipole = v_crystal @ dipole_to_crystal.
        #
        # Shape: (N, 2, 3).
        mode_vectors_dipole = jnp.matmul(
            mode_vectors_crystal,
            dipole_to_crystal,
        )

        # PDL is a power ratio, so sqrt(PDL) is the corresponding
        # field-amplitude factor for the TM mode.
        tm_te_ratio = jnp.power(10.0, -self.resonant_pump_pdl[idx]/10)
        mode_amplitude = jnp.stack(
            [
                jnp.ones_like(tm_te_ratio),
                jnp.sqrt(tm_te_ratio),
            ],
            axis=-1,
        )

        # Complex TE/TM coupling amplitudes:
        #
        #     [
        #         Omega_TE * eta_TE,
        #         Omega_TE * sqrt(PDL) * eta_TM,
        #     ].
        #
        # Shape: (N, 2).
        mode_couplings = (
            self.resonant_pump_coupling_rate[idx][:, None]
            * polarization
            * mode_amplitude
        )

        # Coherently combine TE and TM in the local dipole frame:
        #
        #     pump_eta =
        #         Omega_TE * eta_TE * e_TE
        #         + Omega_TE * sqrt(PDL) * eta_TM * e_TM.
        #
        # Do not normalize this vector. Its magnitude carries the physical
        # optical coupling in GHz.
        return jnp.sum(
            mode_couplings[..., None] * mode_vectors_dipole,
            axis=-2,
        )

    @jax.jit(
        static_argnames=(
            "max_bessel_order",
            "bessel_series_terms",
        )
    )
    def scattering_rate_batch(
        self,
        idx,
        eom_frequency=0.0,
        control_state: SnVControlState | None = None,
        max_bessel_order: int = 2,
        bessel_series_terms: int = 48,
    ):
        """Calculate transition- and sideband-resolved scattering rates in GHz.

        The retained optical transitions are the same four matched lower-orbital
        transitions used by get_ple_freqs:

            ground state t -> excited state t, t = 0, 1, 2, 3.

        For transition t, the carrier coupling is calculated from the actual
        dipole matrix element:

            Omega_t^2 =
                |sum_j pump_eta[j] <exc_t|p_j|gnd_t>|^2.

        Sideband n therefore has squared coupling

            Omega_n,t^2 = J_n(beta)^2 * Omega_t^2.

        A supplied ``control_state`` applies one explicit vector-magnet setting
        and one explicit QWP-HWP-QWP setting to the entire ``idx`` batch. If it
        is ``None``, each member uses its own optimal controls.

        Returns
        -------
        jax.Array
            shape (N, F, S, N_exc, N_gnd)

            where F is the number of EOM frequencies and
            S = 2 * max_bessel_order + 1.
        """
        idx = jnp.atleast_1d(
            jnp.asarray(idx, dtype=jnp.int32)
        )
        eom_frequency = jnp.atleast_1d(
            jnp.asarray(eom_frequency)
        )

        if idx.ndim != 1:
            raise ValueError(
                "`idx` must be a one-dimensional integer array."
            )
        if eom_frequency.ndim != 1:
            raise ValueError(
                "`eom_frequency` must be scalar or one-dimensional."
            )

        lifetime = self.excited_state_lifetime[idx]

        # The lifetime is in ns, so 1/lifetime is in GHz. This preserves the
        # ordinary-frequency linewidth convention used by the existing rate
        # expression.
        gamma = (
            1.0
            / lifetime
            / (2.0 * jnp.pi)
        )

        # Complex pump coupling in the local dipole frame.
        #
        # This already includes:
        #   - initial TE/TM polarization and phase,
        #   - QWP-HWP-QWP propagation,
        #   - PDL,
        #   - resonant_pump_coupling_rate,
        #   - TE/TM mode directions.
        #
        # Shape: (N, 3).
        pump_eta = self._get_resonant_pump_eta_batch(
            idx,
            control_state=control_state,
        )

        # Static Hamiltonian inputs.
        B, theta, phi = self.get_B_spherical_batch(
            idx,
            control_state=control_state,
            frame="dipole",
        )
        neighbor_idx = self.hyperfine_neighbor_idx[idx]

        delta_f_gnd = self._select_scalar_parameter_batch(
            params.delta_f_gnd,
            idx,
        )
        delta_f_exc = self._select_scalar_parameter_batch(
            params.delta_f_exc,
            idx,
        )

        def transition_data_one(
            B_one,
            theta_one,
            phi_one,
            pump_eta_one,
            zpl_shift_one,
            alpha_one,
            beta_one,
            alpha_exc_one,
            beta_exc_one,
            rg_one,
            A_gnd_one,
            Ax_gnd_one,
            Ay_gnd_one,
            A_exc_one,
            Ax_exc_one,
            Ay_exc_one,
            delta_f_gnd_one,
            delta_f_exc_one,
        ):
            (
                E,
                _,
                _,
                _,
                E_exc,
                _,
                _,
                _,
                transition,
                branching_ratios,
            ) = qh_jqt.PLE_transitions(
                B=B_one,
                theta=theta_one,
                phi=phi_one,
                eta_x=pump_eta_one[0],
                eta_y=pump_eta_one[1],
                eta_z=pump_eta_one[2],
                alpha=alpha_one,
                beta=beta_one,
                alpha_exc=alpha_exc_one,
                beta_exc=beta_exc_one,
                rg=rg_one,
                q_gnd=params.q,
                A_gnd=A_gnd_one,
                Ax_gnd=Ax_gnd_one,
                Ay_gnd=Ay_gnd_one,
                L_gnd=params.L,
                upsilon_gnd=0.0,
                delta_f_gnd=delta_f_gnd_one,
                q_exc=params.q_exc,
                A_exc=A_exc_one,
                Ax_exc=Ax_exc_one,
                Ay_exc=Ay_exc_one,
                L_exc=params.L_exc,
                upsilon_exc=0.0,
                delta_f_exc=delta_f_exc_one,
            )

            # Preserve the existing scattering-rate transition set:
            #
            #     g_0 -> e_0
            #     g_1 -> e_1
            #     g_2 -> e_2
            #     g_3 -> e_3

            frequencies = (
                E_exc[:, None]
                - E[None, :]
                + params.LEVEL_OFFSET
                + zpl_shift_one
            )

            # qh_jqt.PLE_transitions calculates
            #
            #     transition[l, k]
            #       = |sum_j pump_eta[j] <exc_l|p_j|gnd_k>|^2.
            #
            # Because pump_eta carries units of GHz, these selected elements
            # are the transition-specific squared couplings in GHz^2.

            return frequencies, transition, branching_ratios

        # Both the frequencies and dipole-resolved couplings come from the
        # same eigensystem calculation.
        #
        # ple_freqs:                   (N, N_exc, N_gnd)
        # transition_coupling_squared: (N, N_exc, N_gnd)
        # branching_ratios:             (N, N_exc, N_gnd)
        ple_freqs, transition_coupling_squared, branching_ratios = jax.vmap(
            transition_data_one
        )(
            B,
            theta,
            phi,
            pump_eta,
            self.strain_params[idx, 0],
            self.strain_params[idx, 1],
            self.strain_params[idx, 2],
            self.strain_params[idx, 3],
            self.strain_params[idx, 4],
            params.rg[neighbor_idx],
            params.A_GND_TENSORS[neighbor_idx],
            params.AX_GND_TENSORS[neighbor_idx],
            params.AY_GND_TENSORS[neighbor_idx],
            params.A_EXC_TENSORS[neighbor_idx],
            params.AX_EXC_TENSORS[neighbor_idx],
            params.AY_EXC_TENSORS[neighbor_idx],
            delta_f_gnd,
            delta_f_exc,
        )

        # Apply the finite-bandwidth EOM response.
        # Shape: (N, F).
        filtered_vpi_ratio = (
            self.eom_vpi_ratio[idx, None]
            / jnp.sqrt(
                1.0
                + (
                    eom_frequency[None, :]
                    / self.eom_vpi_bandwidth[idx, None]
                ) ** 2
            )
        )

        # Phase-modulation index beta = pi * Vmax/Vpi.
        #
        # Shape: (N, F).
        modulation_index = (
            jnp.pi * filtered_vpi_ratio
        )

        # Shape: (S,).
        sideband_orders = jnp.arange(
            -max_bessel_order,
            max_bessel_order + 1,
            dtype=gamma.dtype,
        )

        # Shape:
        #     (max_bessel_order + 1, N, F).
        J_nonnegative = (
            _bessel_j_nonnegative_orders_integer_series(
                modulation_index,
                max_order=max_bessel_order,
                series_terms=bessel_series_terms,
            )
        )

        # J_{-n}(beta) differs only by a sign for integer n, so its squared
        # amplitude equals that of J_n(beta).
        #
        # Shape: (N, F, S).
        J_sidebands = jnp.moveaxis(
            jnp.take(
                J_nonnegative,
                jnp.abs(sideband_orders).astype(jnp.int32),
                axis=0,
            ),
            0,
            -1,
        )

        # Apply each sideband coefficient to each transition-specific coupling:
        #
        #     Omega_n,t^2 = J_n(beta)^2 * Omega_t^2.
        #
        # Shape: (N, F, S, N_exc, N_gnd).
        sideband_coupling_squared = (
            J_sidebands[..., None, None] ** 2
            * transition_coupling_squared[:, None, None, :, :]
        )

        laser_frequency = jnp.asarray(
            self.constants.laser_frequency,
            dtype=gamma.dtype,
        )

        if laser_frequency.ndim == 0:
            laser_frequency = jnp.broadcast_to(
                laser_frequency,
                gamma.shape,
            )
        elif laser_frequency.ndim == 1:
            laser_frequency = laser_frequency[idx]
        else:
            raise ValueError(
                "`laser_frequency` must be scalar or have shape (N,)."
            )

        # Shape: (N, F, S).
        sideband_frequencies = (
            laser_frequency[:, None, None]
            + eom_frequency[None, :, None]
            * sideband_orders[None, None, :]
        )

        # Shape: (N, F, S, N_exc, N_gnd).
        detuning = (
            sideband_frequencies[..., None, None]
            - ple_freqs[:, None, None, :, :]
        )

        # The dipole-resolved coupling enters both the numerator and the
        # power-broadening term in the denominator.
        #
        # Shape: (N, F, S, N_exc, N_gnd).
        sideband_rates = (
            sideband_coupling_squared 
            / lifetime[:, None, None, None, None]
            / (
                gamma[:, None, None, None, None] ** 2
                + 2.0 * sideband_coupling_squared
                + 4.0 * detuning**2
            )
        )

        return sideband_rates, branching_ratios

    def scattering_rate(
        self,
        idx=None,
        eom_frequency=0.0,
        control_state: SnVControlState | None = None,
        max_bessel_order: int = 2,
        bessel_series_terms: int = 48,
    ):
        """Calculate optical scattering rates in GHz with scalar-index convenience.

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

        control_state : SnVControlState or None, optional
            Explicit vector-magnet and QWP-HWP-QWP controls shared by every
            selected member. If ``None``, each member uses its own optimal
            controls.

        max_bessel_order : int, optional
            Largest positive and negative EOM sideband order to include.

        bessel_series_terms : int, optional
            Number of terms retained in the Bessel-function series.

        Returns
        -------
        jax.Array
            Units of GHz.

            For batched ``idx``, the shape is
            ``(N, F, S, N_exc, N_gnd)``.

            For scalar ``idx``, the shape is
            ``(F, S, N_exc, N_gnd)``.
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

        rates, branching_ratios = self.scattering_rate_batch(
            idx=idx,
            eom_frequency=eom_frequency,
            control_state=control_state,
            max_bessel_order=max_bessel_order,
            bessel_series_terms=bessel_series_terms,
        )

        if scalar_idx:
            # The distribution axis is axis 0:
            # (1, F, S, N_exc, N_gnd) -> (F, S, N_exc, N_gnd)
            rates = rates[0]
            branching_ratios = branching_ratios[0]

        return rates, branching_ratios
        
    @jax.jit(static_argnames=("ground_state",))
    def solve_hamiltonian_batch(
        self,
        idx,
        ground_state=True,
        control_state: SnVControlState | None = None,
    ):
        """Solve the static Hamiltonian for a one-dimensional particle batch.

        ``ground_state`` is static because it selects different constants and
        parameter arrays. A supplied ``control_state`` provides one explicit
        vector-magnet setting that is applied to every member in ``idx`` using
        each member's own magnet calibration. If it is ``None``, every member
        uses its own optimal magnet settings.

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
            alpha = self.strain_params[idx, 1]
            beta = self.strain_params[idx, 2]
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
            alpha = self.strain_params[idx, 3]
            beta = self.strain_params[idx, 4]
            delta_f = self._select_scalar_parameter_batch(
                params.delta_f_exc,
                idx,
            )

        B, theta, phi = self.get_B_spherical_batch(
            idx,
            frame="dipole",
            control_state=control_state,
        )

        return jax.vmap(
            qh_jqt.solve_hamiltonian,
            in_axes=(
                0, 0, 0,
                0, None,
                0, 0, 0,
                None,
                0, 0,
                None,
                0,
            ),
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

    def solve_hamiltonian(
        self,
        idx=None,
        ground_state=True,
        control_state: SnVControlState | None = None,
    ):
        scalar = idx is not None and jnp.asarray(idx).ndim == 0

        if idx is None:
            idx = jnp.arange(self.weights.shape[0], dtype=jnp.int32)
        else:
            idx = jnp.atleast_1d(jnp.asarray(idx, dtype=jnp.int32))

        result = self.solve_hamiltonian_batch(
            idx,
            ground_state=ground_state,
            control_state=control_state,
        )

        if scalar:
            return jax.tree_util.tree_map(lambda x: x[0], result)

        return result

    @jax.jit
    def get_folded_branching_ratios_batch(
        self,
        idx,
        control_state: SnVControlState | None = None,
    ):
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
            return qh_jqt.calculate_folded_branching_ratios(
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

        B, theta, phi = self.get_B_spherical_batch(
            idx,
            frame="dipole",
            control_state=control_state,
        )
        return jax.vmap(calculate_one)(
            B,
            theta,
            phi,
            self.strain_params[idx, 1],
            self.strain_params[idx, 2],
            self.strain_params[idx, 3],
            self.strain_params[idx, 4],
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

    def get_folded_branching_ratios(
        self,
        idx=None,
        control_state: SnVControlState | None = None,
    ):
        scalar = idx is not None and jnp.asarray(idx).ndim == 0

        if idx is None:
            idx = jnp.arange(self.weights.shape[0], dtype=jnp.int32)
        else:
            idx = jnp.atleast_1d(jnp.asarray(idx, dtype=jnp.int32))

        result = self.get_folded_branching_ratios_batch(
            idx,
            control_state=control_state,
        )

        if scalar:
            return jax.tree_util.tree_map(lambda x: x[0], result)

        return result
    
    def get_ple_freqs(
        self,
        idx=None,
        control_state: SnVControlState | None = None,
    ):
        """Return the four matched lower-orbital PLE transition frequencies.

        The returned transitions are

            ground state t -> excited state t,  t = 0, 1, 2, 3,

        with frequencies

            f_t = E_exc[t] - E_gnd[t] + LEVEL_OFFSET + zpl_shift,

        in GHz.

        Parameters
        ----------
        idx : int, array_like, or None, optional
            Distribution members to evaluate.

            - Scalar index: returns shape ``(4,)``.
            - One-dimensional index array: returns shape ``(K, 4)``.
            - ``None``: evaluates all members and returns shape ``(N, 4)``.

        control_state : SnVControlState or None, optional
            Explicit physical controls shared by every selected member. This
            calculation uses ``control_state.magnet_settings``. If ``None``,
            each member uses its own optimal magnet settings.

        Returns
        -------
        jax.Array
            PLE transition frequencies in GHz.
        """
        # Normalize all inputs to an explicit one-dimensional batch. This avoids
        # relying on the scalar convenience behavior of solve_hamiltonian().
        idx_array, scalar_idx = self._prepare_indices(idx)

        if idx_array.ndim != 1:
            raise ValueError(
                "`idx` must be a scalar, a one-dimensional integer array, or None."
            )

        (
            E_gnd,
            _,
            _,
            _,
            _,
        ) = self.solve_hamiltonian_batch(
            idx=idx_array,
            ground_state=True,
            control_state=control_state,
        )

        (
            E_exc,
            _,
            _,
            _,
            _,
        ) = self.solve_hamiltonian_batch(
            idx=idx_array,
            ground_state=False,
            control_state=control_state,
        )

        # E_gnd and E_exc have shape (K, num_states).
        #
        # The explicit trailing singleton dimension is essential:
        #
        #     zpl_shift: (K,)    -> incorrect/ambiguous broadcasting
        #     zpl_shift: (K, 1)  -> one shift applied to all four transitions
        #                            belonging to each distribution member.
        zpl_shift_GHz = self.strain_params[idx_array, 0][:, None]

        ple_freqs_GHz = (
            E_exc[:, :4]
            - E_gnd[:, :4]
            + params.LEVEL_OFFSET
            + zpl_shift_GHz
        )

        # Preserve the class's scalar-index convenience convention.
        return ple_freqs_GHz[0] if scalar_idx else ple_freqs_GHz
    
    def get_emr_freqs(
        self,
        idx=None,
        control_state: SnVControlState | None = None,
    ):
        """
        Returns the electron magnetic resonance frequencies.
        """
        E, Eref, U, U_states, alignment = self.solve_hamiltonian(
            idx=idx,
            ground_state=True,
            control_state=control_state,
        )
        return jnp.stack(
            [
                E[..., 2] - E[..., 0],
                E[..., 3] - E[..., 1],
            ],
            axis=-1,
        )

    def get_nmr_freqs(
        self,
        idx=None,
        control_state: SnVControlState | None = None,
    ):
        """
        Returns the nuclear magnetic resonance frequencies.
        """
        E, Eref, U, U_states, alignment = self.solve_hamiltonian(
            idx=idx,
            ground_state=True,
            control_state=control_state,
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
    def get_excitation_hamiltonian_batch(
        self,
        idx,
        included_states=None,
        control_state: SnVControlState | None = None,
    ):
        """Construct dynamic Hamiltonians for a one-dimensional index batch.

        The resonant-pump vector is built by
        :meth:`_get_resonant_pump_eta_batch`, so the dynamic Hamiltonian and
        scattering-rate model use exactly the same QWP-HWP-QWP, PDL, TE/TM
        mode-direction, and coupling-rate handling.

        If ``control_state`` is supplied, its explicit vector-magnet settings
        and QWP-HWP-QWP angles are used for every selected distribution member.
        Each member in ``idx`` still contributes its own physical calibration
        and Hamiltonian parameters. If it is ``None``, each member uses its own
        optimal controls.
        """

        def calculate_one(
            B,
            theta,
            phi,
            pump_eta,
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
            return qh_jqt.get_dynamic_hamiltonian(
                B=B,
                theta=theta,
                phi=phi,
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
        if idx.ndim != 1:
            raise ValueError("`idx` must be a one-dimensional integer array.")

        B, theta, phi = self.get_B_spherical_batch(
            idx,
            frame="dipole",
            control_state=control_state,
        )
        pump_eta = self._get_resonant_pump_eta_batch(
            idx,
            control_state=control_state,
        )

        neighbor_idx = self.hyperfine_neighbor_idx[idx]
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
            pump_eta,
            self.excited_state_lifetime[idx],
            self.mw_B_magnitude[idx]*self.constants.mu_B_GHz_per_T*params.gS,
            self.mw_B_orientation[idx],
            self.strain_params[idx, 1],
            self.strain_params[idx, 2],
            self.strain_params[idx, 3],
            self.strain_params[idx, 4],
            params.rg[neighbor_idx],
            params.A_GND_TENSORS[neighbor_idx],
            params.AX_GND_TENSORS[neighbor_idx],
            params.AY_GND_TENSORS[neighbor_idx],
            params.A_EXC_TENSORS[neighbor_idx],
            params.AX_EXC_TENSORS[neighbor_idx],
            params.AY_EXC_TENSORS[neighbor_idx],
            delta_f_gnd,
            delta_f_exc,
        )

    def get_excitation_hamiltonian(
        self,
        idx=None,
        included_states=None,
        control_state: SnVControlState | None = None,
    ):
        scalar = idx is not None and jnp.asarray(idx).ndim == 0

        if idx is None:
            idx = jnp.arange(self.weights.shape[0], dtype=jnp.int32)
        else:
            idx = jnp.atleast_1d(jnp.asarray(idx, dtype=jnp.int32))

        result = self.get_excitation_hamiltonian_batch(
            idx,
            included_states=included_states,
            control_state=control_state,
        )

        if scalar:
            return jax.tree_util.tree_map(lambda x: x[0], result)

        return result


    
    @jax.jit(static_argnames=("included_states",))
    def get_ground_hamiltonian_batch(
        self,
        idx,
        included_states=None,
        control_state: SnVControlState | None = None,
    ):
        """Construct ground-state Hamiltonians for an index batch.

        ``included_states`` must be ``None`` or a hashable static value such as
        a tuple, because it changes the returned operator structure.  The
        frontend retains its two-angle microwave orientation while this adapter
        passes separate scalar angles to the backend.

        If ``control_state`` is supplied, its explicit vector-magnet settings
        are applied to every selected distribution member using that member's
        own magnet calibration. If it is ``None``, each member uses its own
        optimal magnet settings.
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
            control_state=control_state,
        )
        delta_f_gnd = self._select_scalar_parameter_batch(
            params.delta_f_gnd,
            idx,
        )

        return jax.vmap(calculate_one)(
            B,
            theta,
            phi,
            self.mw_B_magnitude[idx]*self.constants.mu_B_GHz_per_T*params.gS,
            self.mw_B_orientation[idx],
            self.strain_params[idx, 1],
            self.strain_params[idx, 2],
            params.rg[self.hyperfine_neighbor_idx[idx]],
            params.A_GND_TENSORS[self.hyperfine_neighbor_idx[idx]],
            params.AX_GND_TENSORS[self.hyperfine_neighbor_idx[idx]],
            params.AY_GND_TENSORS[self.hyperfine_neighbor_idx[idx]],
            delta_f_gnd,
        )

    def get_ground_hamiltonian(
        self,
        idx=None,
        included_states=None,
        control_state: SnVControlState | None = None,
    ):
        scalar = idx is not None and jnp.asarray(idx).ndim == 0

        if idx is None:
            idx = jnp.arange(self.weights.shape[0], dtype=jnp.int32)
        else:
            idx = jnp.atleast_1d(jnp.asarray(idx, dtype=jnp.int32))

        result = self.get_ground_hamiltonian_batch(
            idx,
            included_states=included_states,
            control_state=control_state,
        )

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
        control_state: SnVControlState | None = None,
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
            control_state: Explicit physical controls shared by every member
                in ``idx``. Only ``magnet_settings`` is used here. If ``None``,
                each member uses its own optimal magnet settings.

        Returns:
            A tuple ``(states, filter_states, populations)`` where every array leaf
            has a leading batch dimension corresponding to ``idx``.
        """
        scale=1e9 # Convert from GHz to Hz for the time array.
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
                control_state=control_state,
            )

            H0_rads_s = 2.0 * jnp.pi * H0 * scale
            Hb_rads_s = 2.0 * jnp.pi * Hb * scale

            omega_c = (
                2.0
                * jnp.pi
                * self.mw_B_bandwidth[single_idx]
                * scale
            )

            states, filter_states = sesolve_components(
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
        control_state: SnVControlState | None = None,
    ):
        """Construct the time grid and run one or more Hamiltonian evolutions.

        Args:
            dist: Distribution containing the Hamiltonian parameters.
            idx: Scalar index, array of indices, or ``None`` for all members.
            pulse: Analog pulse to apply.
            included_states: Tuple identifying the included eigenstates.
            psi0: Initial quantum state. Defaults to the first basis state.
            scale: Scale factor for the time array.
            control_state: Explicit physical controls shared by every member
                selected by ``idx``. Only ``magnet_settings`` is used here. If
                ``None``, each member uses its own optimal magnet settings.
        Returns:
            A tuple containing the quantum states, filter states, and populations.

            For a scalar ``idx``, the leading batch dimension is removed. For an
            array or ``None``, the leading dimension corresponds to distribution
            members.
        """
        scale=1e9 # Convert from GHz to Hz for the time array.
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
            control_state=control_state,
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


    @jax.jit(static_argnames=("included_states", "saveat_downsampling", "solver_options_args"))
    def _drive_excitation_hamiltonian_batch(
        self,
        optical_pulse,
        idx,
        tau,
        rho0,
        included_states,
        saveat_downsampling=1,
        solver_options_args=None,
        control_state: SnVControlState | None = None,
    ):
        """Run Hamiltonian evolutions for a batch of distribution indices.

        Each distribution member is solved independently, and the results are
        stacked along a leading batch dimension.

        Args:
            optical_pulse: Analog pulse to apply to the optical hamiltonian.
            b_pulse: Analog pulse to apply to the magnetic field hamiltonian.
            idx: One-dimensional array of distribution member indices.
            tau: Fixed-shape time array in seconds.
            psi0: Initial quantum state shared by all distribution members.
            included_states: Static tuple identifying the included eigenstates.
            control_state: Explicit vector-magnet and QWP-HWP-QWP controls
                shared by every member in ``idx``. If ``None``, each member uses
                its own optimal controls.

        Returns:
            A tuple ``(states, filter_states, populations)`` where every array leaf
            has a leading batch dimension corresponding to ``idx``.
        """
        scale=1e9 # Convert from GHz to Hz for the time array.
        idx = jnp.atleast_1d(jnp.asarray(idx, dtype=jnp.int32))

        dimension = len(included_states)*2 +1
        sampling_rate = jnp.asarray(self.constants.sampling_rate)*scale
        sample_period = 1.0 / sampling_rate
        pulse_center = optical_pulse.length / 2

        # These are static because included_states is static.
        projectors = tuple(
            jqt.basis(dimension, state_index).to_dm()
            for state_index in range(dimension)
        )
        if saveat_downsampling is None:
            saveat_tlist = tau[-2:]#SaveAt(t1=True)
        else:
            saveat_tlist = tau[::saveat_downsampling]
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
            H0, Hb, Hs_optical, c_ops, optical_transition_offset = self.get_excitation_hamiltonian(
                idx=single_idx,
                included_states=included_states,
                control_state=control_state,
            )
            omega_r = 2*jnp.pi*(params.LEVEL_OFFSET  + self.strain_params[single_idx, 0] - self.constants.laser_frequency + optical_transition_offset) * scale

            # This coefficient is identical for every distribution member. Only the
            # Hamiltonians and filter cutoff frequency depend on the member index.
            def H_optical_drive(t, args=None):
                del args

                local_times = jnp.stack(
                    (
                        t - sample_period,
                        t,
                        t + sample_period,
                    )
                )

                waveform, _, _ = synthesize_analog_pulse(
                    pulse=optical_pulse,
                    tau=local_times,
                    at_time=pulse_center,
                    dphase=0.0,
                    all_info=True,
                )

                return waveform[1]
            def H_dag_optical_drive(t, args=None):
                return jnp.conj(H_optical_drive(t, args))
            H0_rads_s = 2.0 * jnp.pi * H0 * scale
            Hb_rads_s = 2.0 * jnp.pi * Hb * scale
            Hs_optical_rads_s = [
                2.0*jnp.pi*Hs_optical[0] * scale,
                2.0*jnp.pi*Hs_optical[1] * scale,
            ]
            c_ops = c_ops * jnp.sqrt(scale)

            omega_c = (
                2.0
                * jnp.pi
                * self.eom_vpi_bandwidth[single_idx]
                * scale
            )
            states, filter_states = mesolve_components(
                hamiltonians=(
                    H0_rads_s,
                    Hs_optical_rads_s[0],
                    Hs_optical_rads_s[1],
                ),
                coefficients=(
                    1.0,
                    H_optical_drive,
                    H_dag_optical_drive,
                ),
                rho0=rho0,
                tlist=tau,
                saveat_tlist=saveat_tlist,
                filters=(
                    None,
                    eom_lowpass_filter(omega_c, +omega_r, self.eom_vpi_ratio[single_idx]),
                    eom_lowpass_filter(omega_c, -omega_r, -self.eom_vpi_ratio[single_idx]),
                ),
                filter_y0s=(
                    None,
                    jnp.asarray(0.0, dtype=jnp.complex128),
                    jnp.asarray(0.0, dtype=jnp.complex128),
                ),
                collapse_operators=c_ops,
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


    def drive_excitation_hamiltonian(
        self,
        idx,
        optical_pulse,
        included_states=(0, 1, 2, 3),
        rho0=None,
        saveat_downsampling=1,
        solver_options_args=None,
        control_state: SnVControlState | None = None,
    ):
        """Construct the time grid and run one or more Hamiltonian evolutions.

        Args:
            dist: Distribution containing the Hamiltonian parameters.
            idx: Scalar index, array of indices, or ``None`` for all members.
            optical_pulse: Analog pulse to apply to the optical axis.
            included_states: Tuple identifying the included eigenstates.
            rho0: Initial quantum state. Defaults to the first basis state.
            scale: Scale factor for the time array.
            control_state: Explicit vector-magnet and QWP-HWP-QWP controls
                shared by every member selected by ``idx``. If ``None``, each
                member uses its own optimal controls.
        Returns:
            A tuple containing the quantum states, filter states, and populations.

            For a scalar ``idx``, the leading batch dimension is removed. For an
            array or ``None``, the leading dimension corresponds to distribution
            members.
        """
        scale=1e9 # Convert from GHz to Hz for the time array.
        idx_array, scalar_idx = self._prepare_indices(idx)
        dimension = len(included_states)*2 + 1 

        if rho0 is None:
            rho0 = jqt.ket2dm(jqt.basis(dimension, 0))

        # Keep time-grid construction outside JIT because its output length is
        # determined using Python values.
        sampling_rate = float(
            np.asarray(self.constants.sampling_rate)*scale
        )
        sample_period = 1.0 / sampling_rate

        tau = make_analog_pulse_time_array(
            pulse=optical_pulse,
            sample_period=sample_period,
            at_time=float(np.asarray(optical_pulse.length)) / 2.0,
        )

        states, filter_states, populations = self._drive_excitation_hamiltonian_batch(
            optical_pulse=optical_pulse,
            idx=idx_array,
            tau=tau,
            rho0=rho0,
            included_states=tuple(included_states),
            saveat_downsampling=saveat_downsampling,
            solver_options_args=solver_options_args,
            control_state=control_state,
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
    @jax.jit
    def get_waveplate_angles_batch(self, idx):
        """Calculate QWP-HWP-QWP angles that make the realized pump dipole
        operator as close as possible to ``constants.target_dipole_operator``.

        The TE and TM electric-field vectors span at most a two-dimensional
        subspace of the three-component dipole-operator space. For each selected
        distribution member, this method therefore:

            1. Converts the TE/TM mode directions from crystal coordinates to
               the local SnV dipole frame.
            2. Orthogonally projects ``target_dipole_operator`` onto the
               reachable TE/TM mode span using a Moore-Penrose pseudoinverse.
            3. Converts the projected post-PDL TE/TM coefficients into the
               pre-PDL Jones vector that the waveplates must produce.
            4. Uses the analytic QWP-HWP-QWP construction to transform the
               current input Jones vector into that pre-PDL target.

        ``target_dipole_operator`` is interpreted in the local dipole frame.
        Its overall magnitude is ignored because the waveplates control a
        normalized Jones state, while ``resonant_pump_coupling_rate`` controls
        the overall optical-coupling scale.

        Returns
        -------
        jax.Array, shape (K, 3)
            ``[QWP1 angle, HWP angle, QWP2 angle]`` in radians.
        """
        idx = jnp.asarray(idx, dtype=jnp.int32)

        if idx.ndim != 1:
            raise ValueError(
                "`idx` must be a one-dimensional integer array."
            )

        # ------------------------------------------------------------------
        # Input Jones vector before the waveplates
        # ------------------------------------------------------------------
        #
        #     source = [cos(alpha), exp(i * phase) sin(alpha)].
        #
        alpha = self.resonant_pump_polarization[idx]
        phase = self.resonant_pump_phase[idx]

        source = jnp.stack(
            [
                jnp.cos(alpha),
                jnp.exp(1j * phase) * jnp.sin(alpha),
            ],
            axis=-1,
        )

        source = source / jnp.linalg.norm(
            source,
            axis=-1,
            keepdims=True,
        )

        # ------------------------------------------------------------------
        # TE/TM mode vectors in the local dipole frame
        # ------------------------------------------------------------------
        #
        # mode_field_orientation stores (theta, phi) in crystal coordinates.
        #
        # Shape:
        #     mode_vectors_crystal: (K, 2, 3)
        #
        mode_vectors_crystal = angles_to_cartesian(
            self.mode_field_orientation[idx, :, 0],
            self.mode_field_orientation[idx, :, 1],
        )

        dipole_z_crystal = jnp.asarray(
            self.constants.dipole_crystal_axes,
            dtype=mode_vectors_crystal.dtype,
        )[self.dipole_crystal_axis_idx[idx]]

        # Columns are the local dipole-frame basis vectors expressed in
        # crystal coordinates.
        #
        # Shape:
        #     (K, 3, 3)
        #
        dipole_to_crystal = _dipole_basis_crystal_from_axis(
            dipole_z_crystal
        )

        # Under the row-vector convention used by
        # _get_resonant_pump_eta_batch,
        #
        #     v_dipole = v_crystal @ dipole_to_crystal.
        #
        # Shape:
        #     (K, 2, 3)
        #
        mode_vectors_dipole = jnp.matmul(
            mode_vectors_crystal,
            dipole_to_crystal,
        )

        # Arrange the TE/TM vectors as columns:
        #
        #     M = [e_TE, e_TM].
        #
        # A post-PDL TE/TM coefficient vector ``a`` therefore produces
        #
        #     eta = M @ a.
        #
        # Shape:
        #     (K, 3, 2)
        #
        mode_basis = jnp.swapaxes(
            mode_vectors_dipole,
            -1,
            -2,
        )

        # ------------------------------------------------------------------
        # Desired dipole operator
        # ------------------------------------------------------------------
        #
        # Support either one shared target, shape (3,), or one target per
        # complete distribution member, shape (N, 3).
        #
        target_dipole_operator = jnp.asarray(
            self.constants.target_dipole_operator
        )

        if target_dipole_operator.ndim == 1:
            if target_dipole_operator.shape[0] != 3:
                raise ValueError(
                    "`target_dipole_operator` must have shape "
                    "(3,) or (N, 3)."
                )

            target_dipole_operator = jnp.broadcast_to(
                target_dipole_operator,
                (idx.shape[0], 3),
            )

        elif target_dipole_operator.ndim == 2:
            if target_dipole_operator.shape[-1] != 3:
                raise ValueError(
                    "`target_dipole_operator` must have shape "
                    "(3,) or (N, 3)."
                )

            target_dipole_operator = target_dipole_operator[idx]

        else:
            raise ValueError(
                "`target_dipole_operator` must have shape "
                "(3,) or (N, 3)."
            )

        # Preserve complex target components even though the mode-direction
        # vectors themselves are real.
        working_dtype = jnp.result_type(
            source.dtype,
            mode_basis.dtype,
            target_dipole_operator.dtype,
        )

        source = source.astype(working_dtype)
        mode_basis = mode_basis.astype(working_dtype)
        target_dipole_operator = target_dipole_operator.astype(
            working_dtype
        )

        # Only the target direction is controllable through polarization.
        target_operator_norm = jnp.linalg.norm(
            target_dipole_operator,
            axis=-1,
            keepdims=True,
        )

        safe_target_operator_norm = jnp.where(
            target_operator_norm > 0.0,
            target_operator_norm,
            jnp.ones_like(target_operator_norm),
        )

        target_direction = (
            target_dipole_operator
            / safe_target_operator_norm
        )

        # ------------------------------------------------------------------
        # Project the target onto the reachable TE/TM mode span
        # ------------------------------------------------------------------
        #
        # PDL is a power ratio, so the corresponding field-amplitude matrix is
        #
        #     D = diag(1, sqrt(PDL)).
        #
        tm_te_ratio = jnp.power(10.0, -self.resonant_pump_pdl[idx]/10)
        mode_amplitude = jnp.stack(
            [
                jnp.ones_like(tm_te_ratio),
                jnp.sqrt(tm_te_ratio),
            ],
            axis=-1,
        )

        # A mode with exactly zero field transmission cannot contribute to the
        # realized operator. For positive PDL, both columns remain present and
        # this is exactly the span formed by mode_field_orientation.
        mode_available = mode_amplitude > 0.0

        reachable_mode_basis = (
            mode_basis
            * mode_available[:, None, :].astype(working_dtype)
        )

        # Least-squares post-PDL TE/TM amplitudes:
        #
        #     a = M^+ @ target_direction.
        #
        # The pseudoinverse correctly handles nonorthogonal mode vectors and
        # remains defined if the TE and TM vectors become linearly dependent.
        #
        target_mode_amplitudes_after_pdl = jnp.einsum(
            "nij,nj->ni",
            jnp.linalg.pinv(reachable_mode_basis),
            target_direction,
        )

        # Explicit orthogonal projection:
        #
        #     target_projected
        #         = M @ M^+ @ target_direction.
        #
        target_projected = jnp.einsum(
            "nij,nj->ni",
            reachable_mode_basis,
            target_mode_amplitudes_after_pdl,
        )

        # ------------------------------------------------------------------
        # Convert post-PDL coefficients into a pre-PDL Jones target
        # ------------------------------------------------------------------
        #
        # _get_resonant_pump_eta_batch later calculates
        #
        #     a_after = D @ Jones_before.
        #
        # Therefore the waveplates should produce
        #
        #     Jones_before proportional to D^+ @ a_after.
        #
        safe_mode_amplitude = jnp.where(
            mode_available,
            mode_amplitude,
            jnp.ones_like(mode_amplitude),
        ).astype(working_dtype)

        target_before_pdl = (
            target_mode_amplitudes_after_pdl
            / safe_mode_amplitude
        )

        target_before_pdl = jnp.where(
            mode_available,
            target_before_pdl,
            jnp.zeros_like(target_before_pdl),
        )

        # Normalize the desired pre-PDL Jones state.
        target_before_pdl_norm = jnp.linalg.norm(
            target_before_pdl,
            axis=-1,
            keepdims=True,
        )

        target_projected_norm = jnp.linalg.norm(
            target_projected,
            axis=-1,
            keepdims=True,
        )

        real_dtype = jnp.real(target_before_pdl).dtype

        projection_tolerance = (
            32.0 * jnp.finfo(real_dtype).eps
        )

        # If the requested operator has no numerically resolvable component in
        # the reachable mode span, leave the input polarization unchanged as a
        # deterministic finite fallback.
        has_reachable_target = (
            (target_operator_norm > 0.0)
            & (target_projected_norm > projection_tolerance)
            & (target_before_pdl_norm > 0.0)
            & jnp.isfinite(target_before_pdl_norm)
        )

        safe_target_before_pdl_norm = jnp.where(
            has_reachable_target,
            target_before_pdl_norm,
            jnp.ones_like(target_before_pdl_norm),
        )

        target = (
            target_before_pdl
            / safe_target_before_pdl_norm
        )

        target = jnp.where(
            has_reachable_target,
            target,
            source,
        )

        # ------------------------------------------------------------------
        # Jones -> Stokes
        # ------------------------------------------------------------------
        #
        #     S1 = |Ex|^2 - |Ey|^2
        #     S2 = 2 Re(Ex Ey*)
        #     S3 = 2 Im(Ex Ey*)
        #
        def stokes(E):
            Ex = E[..., 0]
            Ey = E[..., 1]
            ExEy = Ex * jnp.conj(Ey)

            return jnp.stack(
                [
                    jnp.abs(Ex) ** 2 - jnp.abs(Ey) ** 2,
                    2.0 * jnp.real(ExEy),
                    2.0 * jnp.imag(ExEy),
                ],
                axis=-1,
            )

        s_in = stokes(source)
        s_target = stokes(target)

        S1, S2, S3 = (
            s_in[..., 0],
            s_in[..., 1],
            s_in[..., 2],
        )

        T1, T2, T3 = (
            s_target[..., 0],
            s_target[..., 1],
            s_target[..., 2],
        )

        # ------------------------------------------------------------------
        # QWP1: convert input to linear polarization
        # ------------------------------------------------------------------
        #
        # For a QWP at q,
        #
        #     S3_out = sin(2q) S1 - cos(2q) S2.
        #
        two_qwp1 = jnp.arctan2(S2, S1)
        qwp1_angle = 0.5 * two_qwp1

        nx = jnp.cos(two_qwp1)
        ny = jnp.sin(two_qwp1)

        projection = nx * S1 + ny * S2

        S1_linear = (
            nx * projection
            - ny * S3
        )

        S2_linear = (
            ny * projection
            + nx * S3
        )

        phi_linear_in = jnp.arctan2(
            S2_linear,
            S1_linear,
        )

        # ------------------------------------------------------------------
        # QWP2: determine which linear state produces the target
        # ------------------------------------------------------------------
        #
        # Propagate the target backwards through QWP2 and choose the QWP angle
        # that makes the resulting state linear.
        #
        two_qwp2 = jnp.arctan2(T2, T1)
        qwp2_angle = 0.5 * two_qwp2

        nx = jnp.cos(two_qwp2)
        ny = jnp.sin(two_qwp2)

        projection = nx * T1 + ny * T2

        T1_linear = (
            nx * projection
            + ny * T3
        )

        T2_linear = (
            ny * projection
            - nx * T3
        )

        phi_linear_target = jnp.arctan2(
            T2_linear,
            T1_linear,
        )

        # ------------------------------------------------------------------
        # HWP: rotate one linear polarization into the other
        # ------------------------------------------------------------------
        #
        # A HWP maps equatorial Stokes angle phi according to
        #
        #     phi_out = 4h - phi_in.
        #
        hwp_angle = 0.25 * (
            phi_linear_target
            + phi_linear_in
        )

        # ------------------------------------------------------------------
        # Canonical physical angle ranges
        # ------------------------------------------------------------------
        qwp1_angle = jnp.mod(
            qwp1_angle,
            jnp.pi,
        )

        hwp_angle = jnp.mod(
            hwp_angle,
            0.5 * jnp.pi,
        )

        qwp2_angle = jnp.mod(
            qwp2_angle,
            jnp.pi,
        )

        return jnp.stack(
            [
                qwp1_angle,
                hwp_angle,
                qwp2_angle,
            ],
            axis=-1,
        )


    def get_waveplate_angles(self, idx=None):
        """Calculate QWP-HWP-QWP angles with scalar-index convenience.

        Parameters
        ----------
        idx : jax.Array, scalar integer, or None
            Distribution indices.

        Returns
        -------
        jax.Array
            Scalar idx:
                shape (3,)

            Array/None:
                shape (K, 3)

            Final axis is

                [QWP1 angle, HWP angle, QWP2 angle]

            in radians.
        """
        idx_array, scalar_idx = self._prepare_indices(idx)

        result = self.get_waveplate_angles_batch(idx_array)

        return result[0] if scalar_idx else result

    def get_optimal_control_state(self, idx=0) -> SnVControlState:
        """Return the controls optimized for one distribution member.

        This replaces the former index-selected shared-control behavior.
        The selected member determines both:

        - the physical vector-magnet settings that realize ``constants.B_target``;
        - the QWP-HWP-QWP angles that best realize
          ``constants.target_dipole_operator``.

        Parameters
        ----------
        idx : int or length-one array_like, optional
            Distribution member whose model is used to calculate the controls.

        Returns
        -------
        SnVControlState
            ``magnet_settings`` has shape ``(3,)`` in tesla and
            ``waveplate_angles`` has shape ``(3,)`` in radians, ordered as
            ``[QWP1, HWP, QWP2]``.
        """
        if idx is None:
            raise ValueError(
                "`idx` must identify exactly one distribution member; "
                "it cannot be None."
            )

        idx_array = jnp.atleast_1d(
            jnp.asarray(idx, dtype=jnp.int32)
        )

        if idx_array.ndim != 1 or idx_array.shape[0] != 1:
            raise ValueError(
                "`idx` must be a scalar integer or a length-one integer array."
            )

        return SnVControlState(
            magnet_settings=self.get_B_settings_batch(idx_array)[0],
            waveplate_angles=self.get_waveplate_angles_batch(idx_array)[0],
        )
