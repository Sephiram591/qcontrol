from __future__ import annotations

from typing import Any, Callable, Literal, NamedTuple, Sequence

import jax
import jax.numpy as jnp


Array = jax.Array
PyTree = Any
JacobianMode = Literal["fwd", "rev"]


# ============================================================================
# Per-particle parameters
# ============================================================================
#
# Every shape documented below is the shape for ONE particle.
#
# SnVDistribution stores a batched SnVParticle pytree. Every array leaf in
# `diffable` and `nondiff` is stacked along an additional leading particle
# axis, allowing the complete particle pytree to be passed directly to vmap.
# ============================================================================


class SnVDifferentiableParams(NamedTuple):
    """
    Continuously differentiable parameters for one SnV particle.

    Every field should be a floating-point or complex JAX array. Scalar
    parameters should preferably be zero-dimensional JAX arrays rather than
    Python floats.
    """

    # ------------------------------------------------------------------------
    # Energy-level and cyclicity parameters
    # ------------------------------------------------------------------------

    magnet_unit_magnitude: Array
    # Shape: (3,)
    # Multiplicative field calibration for each physical magnet axis.

    magnet_axes_rotations: Array
    # Shape: (3, 3)
    # Axes:
    #     (magnet_axis, Cartesian rotation-vector component)
    #
    # Cartesian components are (rx, ry, rz), in radians.

    # ------------------------------------------------------------------------
    # Hamiltonian parameters
    # ------------------------------------------------------------------------

    strain_params: Array
    # Shape: (5,)
    # Current definition:
    #     [zpl_shift, alpha, beta, alpha_exc, beta_exc]
    #
    # Old definition:
    #     epsilon strain tensor in the dipole frame, ordered
    #     (xx, yy, zz, xy, xz, yz).


    optical_transmission: Array
    # Shape: ()

    dark_count_rate: Array
    # Shape: ()

    # ------------------------------------------------------------------------
    # Resonant-pump coupling
    # ------------------------------------------------------------------------

    resonant_pump_coupling_rate: Array
    # Shape: ()
    # Resonant-pump coupling rate in GHz for the TE mode.

    resonant_pump_pdl: Array
    # Shape: ()
    # dB lowering of the TM mode relative to the TE mode due to
    # polarization-dependent loss in the PIC and diamond.
    #
    # If negative, the TM mode is stronger than the TE mode.

    resonant_pump_polarization: Array
    # Shape: ()
    # Angle of the resonant-pump polarization with respect to the TE mode,
    # without factoring in the waveplates.

    resonant_pump_phase: Array
    # Shape: ()
    # Phase difference between the TE and TM modes without factoring in the
    # waveplates.

    mode_field_orientation: Array
    # Shape: (2, 2)
    # Axes:
    #     (TE/TM, theta/phi)
    #
    # Electric-field orientation for the TE and TM modes in the crystal frame.

    transmission_out_diamond: Array
    # Shape: (2,)
    # Transmission from the diamond to the edge coupler for the TE and TM
    # modes.

    reflection_from_pic: Array
    # Shape: (2,)
    # Reflection of the resonant pump from the PIC to the APD for the TE and
    # TM modes.

    # ------------------------------------------------------------------------
    # EOM drive settings
    # ------------------------------------------------------------------------

    eom_vpi_ratio: Array
    # Shape: ()
    # EOM drive amplitude in units of Vmax/Vpi.

    eom_vpi_bandwidth: Array
    # Shape: ()
    # EOM bandwidth in GHz.

    # ------------------------------------------------------------------------
    # Transmission-line dynamic magnetic field
    # ------------------------------------------------------------------------

    mw_B_orientation: Array
    # Shape: (2,)
    # Transmission-line magnetic-field orientation (theta, phi) relative to
    # the dipole axis.

    mw_B_magnitude: Array
    # Shape: ()
    # Transmission-line magnetic-field magnitude in tesla.

    mw_B_bandwidth: Array
    # Shape: ()
    # Transmission-line magnetic-field bandwidth in GHz.

    # ------------------------------------------------------------------------
    # Drift and diffusion
    # ------------------------------------------------------------------------

    spectral_diffusion_rate: Array
    # Shape: ()
    # Spectral-diffusion rate in Hz/sqrt(s).

    polarization_drift_rate: Array
    # Shape: ()
    # Polarization-drift rate in radians/sqrt(s).

    resonant_pump_coupling_drift_rate: Array
    # Shape: ()
    # Resonant-pump coupling drift rate in radians/sqrt(s).


class SnVNonDiffParams(NamedTuple):
    """
    Nondifferentiable categorical parameters for one SnV particle.

    These values remain ordinary JAX array leaves so that they can vary between
    particles and participate in vmap, but they are not included in the
    argument with respect to which grad or jacobian transformations are taken.
    """

    # ------------------------------------------------------------------------
    # Energy-level and cyclicity parameters
    # ------------------------------------------------------------------------

    dipole_crystal_axis_idx: Array
    # Shape: ()
    # Integer dipole-crystal-axis index.

    # ------------------------------------------------------------------------
    # Hamiltonian parameters
    # ------------------------------------------------------------------------

    hyperfine_neighbor_idx: Array
    # Shape: ()
    # Integer HyperfineNeighbor ID.


    # ------------------------------------------------------------------------
    # Emission Characteristics
    # ------------------------------------------------------------------------


    excited_state_lifetime: Array
    # Shape: ()
    # Excited-state lifetime in nanoseconds.

    # ------------------------------------------------------------------------
    # Collection-efficiency parameters
    # ------------------------------------------------------------------------

    debye_waller_factor: Array
    # Shape: ()

    quantum_efficiency: Array
    # Shape: ()


class SnVParticle(NamedTuple):
    """
    Complete parameterization of one SnV particle.

    A single particle contains unbatched parameter leaves. SnVDistribution
    stores the same pytree structure with its leaves stacked along a leading
    particle axis.
    """

    diffable: SnVDifferentiableParams
    nondiff: SnVNonDiffParams

    def with_diffable(
        self,
        diffable: SnVDifferentiableParams,
    ) -> SnVParticle:
        """
        Return a particle with replaced differentiable parameters.

        The nondifferentiable categorical parameters are preserved.
        """
        return SnVParticle(
            diffable=diffable,
            nondiff=self.nondiff,
        )


class SnVControlState(NamedTuple):
    """
    Physical control settings applied to an SnV particle.
    """

    magnet_settings: Array
    # Shape: (3,)
    # Physical magnet settings in tesla.

    waveplate_angles: Array
    # Shape: (3,)
    # QWP-HWP-QWP angles in radians:
    #     [QWP1 angle, HWP angle, QWP2 angle]


# ============================================================================
# Batched particle container
# ============================================================================
#
# SnVDistribution uses a structure-of-arrays representation:
#
#     distribution.particles
#
# is one SnVParticle pytree whose leaves have a leading particle axis. It is
# not a Python list of independently allocated SnVParticle objects.
#
# This representation makes the complete particle argument directly compatible
# with:
#
#     jax.vmap(helper, in_axes=(0, None))
# ============================================================================


class SnVDistribution(NamedTuple):
    """
    Weighted container of SnV particles.
    """

    particles: SnVParticle
    # Batched SnVParticle pytree. Every parameter leaf has a leading particle
    # axis.

    weights: Array
    # One scalar weight for each particle.

    @classmethod
    def from_particles(
        cls,
        particles: Sequence[SnVParticle],
        weights: Array | None = None,
    ) -> SnVDistribution:
        """
        Stack a Python sequence of single-particle objects into the batched
        pytree representation used by JAX.
        """
        particles = tuple(particles)

        if not particles:
            raise ValueError(
                "`particles` must contain at least one SnVParticle."
            )

        stacked_particles = jax.tree_util.tree_map(
            lambda *leaves: jnp.stack(leaves, axis=0),
            *particles,
        )

        particle_count = len(particles)

        if weights is None:
            weights = jnp.full(
                (particle_count,),
                1.0 / particle_count,
            )
        else:
            weights = jnp.asarray(weights)

            if weights.shape != (particle_count,):
                raise ValueError(
                    "`weights` must contain exactly one value for each "
                    f"particle. Received {weights.shape} for "
                    f"{particle_count} particles."
                )

        return cls(
            particles=stacked_particles,
            weights=weights,
        )

    def particle(
        self,
        index: int | Array,
    ) -> SnVParticle:
        """
        Extract one single-particle pytree.

        A scalar index removes the leading particle axis from every parameter
        leaf.
        """
        return jax.tree_util.tree_map(
            lambda leaf: leaf[index],
            self.particles,
        )

    def subset(
        self,
        indices: Array,
    ) -> SnVDistribution:
        """
        Return a selected subset of the distribution.
        """
        indices = jnp.asarray(
            indices,
            dtype=jnp.int32,
        )

        if indices.ndim != 1:
            raise ValueError(
                "`indices` must be a one-dimensional integer array."
            )

        return SnVDistribution(
            particles=jax.tree_util.tree_map(
                lambda leaf: leaf[indices],
                self.particles,
            ),
            weights=self.weights[indices],
        )

    @property
    def size(self) -> int:
        """
        Return the number of particles.
        """
        return self.weights.shape[0]

    def normalized_weights(self) -> Array:
        """
        Return weights normalized to sum to one.
        """
        return self.weights / jnp.sum(self.weights)


# ============================================================================
# Basic pytree utilities
# ============================================================================


def stop_gradient_tree(tree: PyTree) -> PyTree:
    """
    Apply stop_gradient to every numerical leaf in a pytree.
    """
    return jax.tree_util.tree_map(
        jax.lax.stop_gradient,
        tree,
    )


def diffable_form(
    helper: Callable[
        [SnVParticle, SnVControlState],
        PyTree,
    ],
) -> Callable[
    [
        SnVDifferentiableParams,
        SnVNonDiffParams,
        SnVControlState,
    ],
    PyTree,
]:
    """
    Rewrite a particle helper

        helper(particle, control_state)

    as

        helper_from_diffable(diffable, nondiff, control_state).

    This makes the differentiable parameter subtree the top-level argument for
    grad, jacfwd, and jacrev while keeping integer categorical parameters fixed.
    """

    def helper_from_diffable(
        diffable: SnVDifferentiableParams,
        nondiff: SnVNonDiffParams,
        control_state: SnVControlState,
    ) -> PyTree:
        particle = SnVParticle(
            diffable=diffable,
            nondiff=nondiff,
        )

        return helper(
            particle,
            control_state,
        )

    return helper_from_diffable


# ============================================================================
# Helper transformation factories
# ============================================================================


ParticleHelper = Callable[
    [SnVParticle, SnVControlState],
    PyTree,
]

ParticleLoss = Callable[
    [SnVParticle, SnVControlState],
    Array,
]


def make_batched_helper(
    helper: ParticleHelper,
) -> Callable[
    [SnVParticle, SnVControlState],
    PyTree,
]:
    """
    Vectorize a one-particle helper under one shared control state.

    The first argument is a batched SnVParticle pytree. The second argument is
    one SnVControlState applied to every particle.
    """
    return jax.jit(
        jax.vmap(
            helper,
            in_axes=(
                0,
                None,
            ),
            out_axes=0,
        )
    )


def make_batched_helper_with_particle_controls(
    helper: ParticleHelper,
) -> Callable[
    [SnVParticle, SnVControlState],
    PyTree,
]:
    """
    Vectorize a helper with one independently specified control state per
    particle.

    Every control-state leaf must have the same leading particle axis as the
    SnVParticle leaves.
    """
    return jax.jit(
        jax.vmap(
            helper,
            in_axes=(
                0,
                0,
            ),
            out_axes=0,
        )
    )


def make_single_particle_jacobian(
    helper: ParticleHelper,
    *,
    mode: JacobianMode = "rev",
) -> Callable[
    [
        SnVDifferentiableParams,
        SnVNonDiffParams,
        SnVControlState,
    ],
    PyTree,
]:
    """
    Construct the Jacobian of a one-particle helper with respect to all fields
    in SnVDifferentiableParams.

    The nondifferentiable categorical parameters and physical controls are held
    fixed.

    For complex-valued helper outputs, first expose real and imaginary
    components explicitly or reduce the output to a real scalar objective.
    """
    helper_from_diffable = diffable_form(
        helper
    )

    if mode == "rev":
        jacobian = jax.jacrev(
            helper_from_diffable,
            argnums=0,
        )
    elif mode == "fwd":
        jacobian = jax.jacfwd(
            helper_from_diffable,
            argnums=0,
        )
    else:
        raise ValueError(
            "`mode` must be 'fwd' or 'rev'."
        )

    return jax.jit(
        jacobian
    )


def make_batched_particle_jacobian(
    helper: ParticleHelper,
    *,
    mode: JacobianMode = "rev",
) -> Callable[
    [
        SnVDifferentiableParams,
        SnVNonDiffParams,
        SnVControlState,
    ],
    PyTree,
]:
    """
    Construct one independent differentiable-parameter Jacobian per particle.

    This calculates each particle's helper derivative with respect to that same
    particle's differentiable parameters. It does not construct cross-particle
    Jacobian blocks.
    """
    helper_from_diffable = diffable_form(
        helper
    )

    if mode == "rev":
        jacobian_one = jax.jacrev(
            helper_from_diffable,
            argnums=0,
        )
    elif mode == "fwd":
        jacobian_one = jax.jacfwd(
            helper_from_diffable,
            argnums=0,
        )
    else:
        raise ValueError(
            "`mode` must be 'fwd' or 'rev'."
        )

    return jax.jit(
        jax.vmap(
            jacobian_one,
            in_axes=(
                0,
                0,
                None,
            ),
            out_axes=0,
        )
    )


def make_batched_particle_value_and_grad(
    loss: ParticleLoss,
) -> Callable[
    [
        SnVDifferentiableParams,
        SnVNonDiffParams,
        SnVControlState,
    ],
    tuple[Array, SnVDifferentiableParams],
]:
    """
    Return one scalar loss and one differentiable-parameter gradient pytree per
    particle.

    `loss` must return one real scalar for one particle.
    """
    loss_from_diffable = diffable_form(
        loss
    )

    value_and_grad_one = jax.value_and_grad(
        loss_from_diffable,
        argnums=0,
    )

    return jax.jit(
        jax.vmap(
            value_and_grad_one,
            in_axes=(
                0,
                0,
                None,
            ),
            out_axes=(
                0,
                0,
            ),
        )
    )


def make_weighted_distribution_value_and_grad(
    loss: ParticleLoss,
) -> Callable[
    [
        SnVDifferentiableParams,
        SnVNonDiffParams,
        Array,
        SnVControlState,
    ],
    tuple[Array, SnVDifferentiableParams],
]:
    """
    Differentiate a weighted scalar distribution objective with respect to all
    particle-resolved differentiable parameters.

    The returned gradient has the same pytree structure as the batched
    SnVDifferentiableParams argument.
    """
    loss_batch = jax.vmap(
        loss,
        in_axes=(
            0,
            None,
        ),
        out_axes=0,
    )

    def objective(
        diffable: SnVDifferentiableParams,
        nondiff: SnVNonDiffParams,
        weights: Array,
        control_state: SnVControlState,
    ) -> Array:
        particles = SnVParticle(
            diffable=diffable,
            nondiff=nondiff,
        )

        losses = loss_batch(
            particles,
            control_state,
        )

        normalized_weights = (
            weights / jnp.sum(weights)
        )

        return jnp.sum(
            normalized_weights * losses
        )

    return jax.jit(
        jax.value_and_grad(
            objective,
            argnums=0,
        )
    )


def make_weighted_control_value_and_grad(
    loss: ParticleLoss,
) -> Callable[
    [
        SnVParticle,
        Array,
        SnVControlState,
    ],
    tuple[Array, SnVControlState],
]:
    """
    Differentiate a weighted distribution objective with respect to one shared
    physical control state.
    """
    loss_batch = jax.vmap(
        loss,
        in_axes=(
            0,
            None,
        ),
        out_axes=0,
    )

    def objective(
        particles: SnVParticle,
        weights: Array,
        control_state: SnVControlState,
    ) -> Array:
        losses = loss_batch(
            particles,
            control_state,
        )

        normalized_weights = (
            weights / jnp.sum(weights)
        )

        return jnp.sum(
            normalized_weights * losses
        )

    return jax.jit(
        jax.value_and_grad(
            objective,
            argnums=2,
        )
    )


# ============================================================================
# Example single-particle helper and loss
# ============================================================================


def example_vector_helper(
    particle: SnVParticle,
    control_state: SnVControlState,
) -> Array:
    """
    Representative vector-valued single-particle helper.

    Production helpers such as get_B_cartesian, get_ple_freqs, and
    get_folded_branching_ratios should follow this same signature.
    """
    diffable = particle.diffable
    nondiff = particle.nondiff

    realized_channel_fields = (
        diffable.magnet_unit_magnitude
        * control_state.magnet_settings
    )

    orientation_factor_table = jnp.asarray(
        [
            1.0,
            -1.0,
            0.5,
            -0.5,
        ],
        dtype=diffable.strain_params.dtype,
    )

    orientation_factor = orientation_factor_table[
        nondiff.dipole_crystal_axis_idx
    ]

    return jnp.stack(
        [
            jnp.sum(realized_channel_fields),
            diffable.strain_params[0]
            + orientation_factor
            * diffable.strain_params[1],
        ]
    )


def example_scalar_loss(
    particle: SnVParticle,
    control_state: SnVControlState,
) -> Array:
    """
    Representative real scalar single-particle objective.
    """
    value = example_vector_helper(
        particle,
        control_state,
    )

    return jnp.sum(
        value**2
    )


# ============================================================================
# Intended usage
# ============================================================================
#
# Single-particle helper:
#
#     value = example_vector_helper(
#         particle,
#         control_state,
#     )
#
#
# Single-particle Jacobian with respect to SnVDifferentiableParams:
#
#     jacobian_one_fn = make_single_particle_jacobian(
#         example_vector_helper,
#         mode="rev",
#     )
#
#     jacobian_one = jacobian_one_fn(
#         particle.diffable,
#         particle.nondiff,
#         control_state,
#     )
#
# For the example helper, the single-particle strain Jacobian has shape:
#
#     jacobian_one.strain_params.shape == (2, 5)
#
#
# Evaluate the helper for the complete distribution under one shared control:
#
#     helper_batch_fn = make_batched_helper(
#         example_vector_helper
#     )
#
#     values = helper_batch_fn(
#         distribution.particles,
#         control_state,
#     )
#
#
# Calculate independent per-particle Jacobians:
#
#     jacobian_batch_fn = make_batched_particle_jacobian(
#         example_vector_helper,
#         mode="rev",
#     )
#
#     jacobians = jacobian_batch_fn(
#         distribution.particles.diffable,
#         distribution.particles.nondiff,
#         control_state,
#     )
#
#
# Calculate independent per-particle scalar gradients:
#
#     value_and_grad_batch_fn = make_batched_particle_value_and_grad(
#         example_scalar_loss
#     )
#
#     losses, gradients = value_and_grad_batch_fn(
#         distribution.particles.diffable,
#         distribution.particles.nondiff,
#         control_state,
#     )
#
#
# Differentiate one weighted distribution objective with respect to every
# particle's differentiable parameters:
#
#     distribution_value_and_grad_fn = (
#         make_weighted_distribution_value_and_grad(
#             example_scalar_loss
#         )
#     )
#
#     weighted_loss, distribution_gradient = (
#         distribution_value_and_grad_fn(
#             distribution.particles.diffable,
#             distribution.particles.nondiff,
#             distribution.weights,
#             control_state,
#         )
#     )
#
#
# Differentiate the weighted distribution objective with respect to the shared
# physical control state:
#
#     control_value_and_grad_fn = make_weighted_control_value_and_grad(
#         example_scalar_loss
#     )
#
#     weighted_loss, control_gradient = control_value_and_grad_fn(
#         distribution.particles,
#         distribution.weights,
#         control_state,
#     )