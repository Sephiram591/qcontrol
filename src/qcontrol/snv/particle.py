from __future__ import annotations

from typing import Any, Callable, Literal, NamedTuple, Sequence

import jax
import jax.numpy as jnp


Array = jax.Array
PyTree = Any
JacobianMode = Literal["fwd", "rev"]


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


class SnVControlState(NamedTuple):
    """Physical controls applied to one or more SnV particles."""

    magnet_settings: Array
    # Shape (3,). Physical vector-magnet settings, in tesla.

    waveplate_angles: Array
    # Shape (3,). [QWP1, HWP, QWP2] angles, in radians.


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


def stop_gradient_tree(tree: PyTree) -> PyTree:
    """Apply ``stop_gradient`` to every pytree leaf."""
    return jax.tree_util.tree_map(jax.lax.stop_gradient, tree)


def diffable_form(
    helper: Callable[[SnVParticle, SnVControlState], PyTree],
) -> Callable[[SnVDifferentiableParams, SnVNonDiffParams, SnVControlState], PyTree]:
    """Expose the differentiable subtree as a helper's first argument."""

    def helper_from_diffable(diffable, nondiff, control_state):
        return helper(SnVParticle(diffable, nondiff), control_state)

    return helper_from_diffable


ParticleHelper = Callable[[SnVParticle, SnVControlState], PyTree]
ParticleLoss = Callable[[SnVParticle, SnVControlState], Array]


def make_batched_helper(helper: ParticleHelper) -> Callable:
    """Vectorize a helper over particles under one shared control state."""
    return jax.jit(jax.vmap(helper, in_axes=(0, None), out_axes=0))


def make_batched_helper_with_particle_controls(helper: ParticleHelper) -> Callable:
    """Vectorize a helper over matched particle and control batches."""
    return jax.jit(jax.vmap(helper, in_axes=(0, 0), out_axes=0))


def make_single_particle_jacobian(
    helper: ParticleHelper, *, mode: JacobianMode = "rev"
) -> Callable:
    """Differentiate one helper with respect to ``SnVDifferentiableParams``."""
    transformed = diffable_form(helper)
    if mode == "rev":
        jacobian = jax.jacrev(transformed, argnums=0)
    elif mode == "fwd":
        jacobian = jax.jacfwd(transformed, argnums=0)
    else:
        raise ValueError("`mode` must be 'fwd' or 'rev'.")
    return jax.jit(jacobian)


def make_batched_particle_jacobian(
    helper: ParticleHelper, *, mode: JacobianMode = "rev"
) -> Callable:
    """Return one independent parameter Jacobian per particle."""
    transformed = diffable_form(helper)
    if mode == "rev":
        jacobian_one = jax.jacrev(transformed, argnums=0)
    elif mode == "fwd":
        jacobian_one = jax.jacfwd(transformed, argnums=0)
    else:
        raise ValueError("`mode` must be 'fwd' or 'rev'.")
    return jax.jit(jax.vmap(jacobian_one, in_axes=(0, 0, None), out_axes=0))


def make_single_control_jacobian(
    helper: ParticleHelper, *, mode: JacobianMode = "rev"
) -> Callable:
    """Differentiate one helper with respect to its control state."""
    if mode == "rev":
        jacobian = jax.jacrev(helper, argnums=1)
    elif mode == "fwd":
        jacobian = jax.jacfwd(helper, argnums=1)
    else:
        raise ValueError("`mode` must be 'fwd' or 'rev'.")
    return jax.jit(jacobian)


def make_batched_control_jacobian(
    helper: ParticleHelper, *, mode: JacobianMode = "rev"
) -> Callable:
    """Return one independent control Jacobian per particle, under one shared
    control state."""
    if mode == "rev":
        jacobian_one = jax.jacrev(helper, argnums=1)
    elif mode == "fwd":
        jacobian_one = jax.jacfwd(helper, argnums=1)
    else:
        raise ValueError("`mode` must be 'fwd' or 'rev'.")
    return jax.jit(jax.vmap(jacobian_one, in_axes=(0, None), out_axes=0))


def make_batched_control_jacobian_with_particle_controls(
    helper: ParticleHelper, *, mode: JacobianMode = "rev"
) -> Callable:
    """Return one independent control Jacobian per matched particle/control pair."""
    if mode == "rev":
        jacobian_one = jax.jacrev(helper, argnums=1)
    elif mode == "fwd":
        jacobian_one = jax.jacfwd(helper, argnums=1)
    else:
        raise ValueError("`mode` must be 'fwd' or 'rev'.")
    return jax.jit(jax.vmap(jacobian_one, in_axes=(0, 0), out_axes=0))


def make_batched_particle_value_and_grad(loss: ParticleLoss) -> Callable:
    """Return one real scalar loss and parameter gradient per particle."""
    value_and_grad_one = jax.value_and_grad(diffable_form(loss), argnums=0)
    return jax.jit(
        jax.vmap(value_and_grad_one, in_axes=(0, 0, None), out_axes=(0, 0))
    )


def make_weighted_distribution_value_and_grad(loss: ParticleLoss) -> Callable:
    """Differentiate a weighted loss with respect to all particle parameters."""
    loss_batch = jax.vmap(loss, in_axes=(0, None), out_axes=0)

    def objective(diffable, nondiff, weights, control_state):
        losses = loss_batch(SnVParticle(diffable, nondiff), control_state)
        weights = weights / jnp.sum(weights)
        return jnp.sum(weights * losses)

    return jax.jit(jax.value_and_grad(objective, argnums=0))


def make_particle_control_value_and_grad(loss: ParticleLoss) -> Callable:
    """Differentiate one particle's scalar loss with respect to its control state."""
    return jax.jit(jax.value_and_grad(loss, argnums=1))


def make_weighted_control_value_and_grad(loss: ParticleLoss) -> Callable:
    """Differentiate a weighted particle loss with respect to shared controls."""
    loss_batch = jax.vmap(loss, in_axes=(0, None), out_axes=0)

    def objective(particles, weights, control_state):
        weights = weights / jnp.sum(weights)
        return jnp.sum(weights * loss_batch(particles, control_state))

    return jax.jit(jax.value_and_grad(objective, argnums=2))


def example_vector_helper(
    particle: SnVParticle, control_state: SnVControlState
) -> Array:
    """Small vector-valued example using both parameter subtrees."""
    d, n = particle.diffable, particle.nondiff
    factors = jnp.asarray([1.0, -1.0, 0.5, -0.5], dtype=d.strain_params.dtype)
    return jnp.stack(
        [
            jnp.sum(d.magnet_unit_magnitude * control_state.magnet_settings),
            d.strain_params[0]
            + factors[n.dipole_crystal_axis_idx] * d.strain_params[1],
        ]
    )


def example_scalar_loss(
    particle: SnVParticle, control_state: SnVControlState
) -> Array:
    """Real scalar example suitable for gradient transformations."""
    return jnp.sum(example_vector_helper(particle, control_state) ** 2)
