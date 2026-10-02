from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp


Array = jax.Array


class SnVDifferentiableParams(NamedTuple):
    """Continuously differentiable parameters for one SnV particle.

    Notes
    -----
    Every documented shape is for one particle. Scalar parameters should be
    zero-dimensional floating-point or complex JAX arrays when possible.
    """

    magnet_unit_magnitude: Array
    # Shape (3,). Multiplicative calibration of the three magnet channels.

    magnet_axes_rotations: Array
    # Shape (3, 3): (magnet axis, lab-frame rotation-vector component).

    strain_params: Array
    # Shape (5,): [zpl_shift, alpha, beta, alpha_exc, beta_exc], in GHz.

    resonant_pump_coupling_rate: Array
    # Scalar TE-mode resonant-pump coupling rate, in GHz.

    resonant_pump_pdl: Array
    # Scalar TM attenuation relative to TE, in dB; may be negative.

    resonant_pump_polarization: Array
    # Scalar source-polarization angle relative to TE, in radians.

    resonant_pump_phase: Array
    # Scalar source phase difference between TE and TM, in radians.

    mode_field_orientation: Array
    # Shape (2, 2): (TE/TM, theta/phi), in the crystal frame.

    transmission_out_diamond: Array
    # Shape (2,): TE/TM transmission from diamond to APD.

    reflection_from_pic: Array
    # Shape (2, 2): (ZPL/PSB, TE/TM) resonant-pump reflection from PIC to APD. Constant of proportionality - photons/resonant_pump_coupling_rate**2

    dark_count_rate: Array
    # Scalar detector dark-count rate.

    eom_vpi_ratio: Array
    # Scalar EOM drive amplitude in units of Vmax/Vpi.

    eom_vpi_bandwidth: Array
    # Scalar EOM bandwidth, in GHz.

    mw_B_orientation: Array
    # Shape (2,): microwave-field (theta, phi) in the dipole frame.

    mw_B_magnitude: Array
    # Scalar microwave magnetic-field magnitude, in tesla.

    mw_B_bandwidth: Array
    # Scalar microwave magnetic-field bandwidth, in GHz.

    spectral_diffusion_rate: Array
    # Scalar spectral-diffusion rate, in Hz/sqrt(s).

    polarization_drift_rate: Array
    # Scalar polarization-drift rate, in rad/sqrt(s).

    resonant_pump_coupling_drift_rate: Array
    # Scalar resonant-pump coupling drift rate, in rad/sqrt(s).


class SnVNonDiffParams(NamedTuple):
    """Parameters held fixed when differentiating one SnV particle.

    Notes
    -----
    These remain ordinary JAX pytree leaves, so a batched ``SnVParticle`` can
    still be passed directly to ``jax.vmap``. They include categorical values,
    fixed device parameters, and the former ``SnVConstants`` fields. The two
    optimization targets are deliberately absent: ``B_target`` and
    ``target_dipole_operator`` are explicit helper arguments.
    """

    dipole_crystal_axis_idx: Array
    # Scalar integer index into dipole_crystal_axes.

    hyperfine_neighbor_idx: Array
    # Scalar integer HyperfineNeighbor ID.

    excited_state_lifetime: Array
    # Scalar excited-state lifetime, in ns.

    debye_waller_factor: Array
    # Scalar Debye-Waller factor.

    quantum_efficiency: Array
    # Scalar quantum efficiency.

    laser_frequency: Array
    # Scalar optical carrier frequency, in GHz.

    diamond_lattice_100_orientation: Array
    # Shape (2,): lab-frame (theta, phi) of crystal [100], in radians.

    diamond_lattice_011_orientation: Array
    # Shape (2,): lab-frame (theta, phi) of crystal [011], in radians.

    nominal_magnet_axes: Array
    # Shape (3, 3): nominal magnet-axis unit vectors in the lab frame.

    sampling_rate: Array
    # Scalar AWG sampling rate, in GHz.

    mu_B_GHz_per_T: Array
    # Scalar Bohr magneton in GHz/T, conventionally 13.996.

    dipole_crystal_axes: Array
    # Shape (4, 3): allowed <111> defect axes in crystal coordinates.


class SnVParticle(NamedTuple):
    """Complete unbatched parameterization of one SnV particle."""

    diffable: SnVDifferentiableParams
    nondiff: SnVNonDiffParams

    def with_diffable(self, diffable: SnVDifferentiableParams) -> SnVParticle:
        """Return a copy with new differentiable parameters."""
        return SnVParticle(diffable=diffable, nondiff=self.nondiff)


class SnVDistribution(NamedTuple):
    """Weighted structure-of-arrays container of SnV particles."""

    particles: SnVParticle
    weights: Array

    @classmethod
    def from_particles(
        cls,
        particles: Sequence[SnVParticle],
        weights: Array | None = None,
    ) -> SnVDistribution:
        """Stack single-particle pytrees into a batched distribution.

        Parameters
        ----------
        particles
            Nonempty sequence of unbatched particles.
        weights
            Optional one-dimensional particle weights. Uniform weights are used
            when omitted.
        """
        particles = tuple(particles)
        if not particles:
            raise ValueError("`particles` must contain at least one SnVParticle.")

        stacked = jax.tree_util.tree_map(
            lambda *leaves: jnp.stack(leaves, axis=0), *particles
        )
        count = len(particles)
        if weights is None:
            weights = jnp.full((count,), 1.0 / count)
        else:
            weights = jnp.asarray(weights)
            if weights.shape != (count,):
                raise ValueError(
                    "`weights` must contain one value per particle; "
                    f"received {weights.shape} for {count} particles."
                )
        return cls(particles=stacked, weights=weights)

    def particle(self, index: int | Array) -> SnVParticle:
        """Extract one particle and remove the leading batch axis."""
        return jax.tree_util.tree_map(lambda leaf: leaf[index], self.particles)

    def subset(self, indices: Array) -> SnVDistribution:
        """Return a selected particle subset."""
        indices = jnp.asarray(indices, dtype=jnp.int32)
        if indices.ndim != 1:
            raise ValueError("`indices` must be a one-dimensional integer array.")
        return SnVDistribution(
            particles=jax.tree_util.tree_map(
                lambda leaf: leaf[indices], self.particles
            ),
            weights=self.weights[indices],
        )

    @property
    def size(self) -> int:
        """Number of particles."""
        return self.weights.shape[0]

    def normalized_weights(self) -> Array:
        """Return weights normalized to sum to one."""
        return self.weights / jnp.sum(self.weights)