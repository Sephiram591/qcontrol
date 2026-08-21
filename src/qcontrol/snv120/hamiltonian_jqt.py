"""JAXQuantum implementation of the SnV dynamic-Jahn--Teller model.

This module builds and analyzes the ground and excited electronic manifolds of
an SnV center with dynamic-Jahn--Teller (DJT) hyperfine coupling. Every
numerical function evaluates exactly one magnetic-field, strain, polarization,
or frequency point. Parameter sweeps should therefore be expressed outside the
module with :func:`jax.vmap`.

The tensor-product order of one electronic manifold is

``orbital x electron x nuclear``.

The low-level Hamiltonian uses the same frequency-like magnetic-field units as
the original implementation. In particular, a field component such as ``bz``
is already scaled into the electron-Zeeman energy convention. All Hamiltonian
couplings are expected in mutually consistent frequency units, conventionally
GHz in the accompanying parameter module.

Hyperfine interface
-------------------
Every Hamiltonian, eigensystem, optical, cyclicity, and dynamics helper accepts
the Cartesian tensor triplet ``A``, ``Ax``, and ``Ay``; none of those APIs
accepts ``Aperp``, ``Apar``, ``A1``, or ``A2``. The standalone
:func:`djt_hyperfine_tensors` utility is retained only to construct a tensor
triplet from the legacy scalar parameterization.

For a manifold-specific public helper, passing ``A=None`` with ``Ax`` and
``Ay`` also omitted inserts the corresponding default DJT tensors obtained from
:func:`djt_hyperfine_tensors` and :mod:`qcontrol.snv120.parameters`. With a custom ``A``,
omitted ``Ax`` or ``Ay`` tensors are replaced by zero matrices.

JAX conventions
---------------
All physical model arguments are explicit; no function uses ``**kwargs``.
Array-valued hyperfine tensors describe one parameter point and have shape
``(3, 3)``. Use ``in_axes=None`` to hold them fixed under :func:`jax.vmap`, or
map over a leading batch dimension to vary them point by point.

Notes
-----
JAX 64-bit mode is enabled before importing :mod:`jax.numpy`. This is important
because the model combines THz-scale spin-orbit energies with sub-MHz dynamic
hyperfine terms.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple

from jax import config

# Configure precision before importing jax.numpy. Calling this after importing
# jax.numpy can leave arrays created during module initialization at 32-bit
# precision.
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jaxquantum as jqt

import qcontrol.snv120.parameters as params


# Package-level defaults.  SnV120Distribution supplies explicit A/Ax/Ay
# tensors for every particle, so the legacy scalar DJT values are only
# fallback defaults for standalone backend use.  getattr keeps this module
# compatible with parameter files that predate delta_f or the scalar DJT
# parameterization.
_G_S = getattr(params, "gS", getattr(params, "g_e", 2.0))
_DEFAULT_DELTA_F_GND = getattr(params, "delta_f_gnd", 0.0)
_DEFAULT_DELTA_F_EXC = getattr(params, "delta_f_exc", 0.0)
_DEFAULT_APERP_GND = getattr(params, "Aperp_gnd", 0.0)
_DEFAULT_APAR_GND = getattr(params, "Apar_gnd", 0.0)
_DEFAULT_A1_GND = getattr(params, "A1_gnd", 0.0)
_DEFAULT_A2_GND = getattr(params, "A2_gnd", 0.0)
_DEFAULT_APERP_EXC = getattr(params, "Aperp_exc", 0.0)
_DEFAULT_APAR_EXC = getattr(params, "Apar_exc", 0.0)
_DEFAULT_A1_EXC = getattr(params, "A1_exc", 0.0)
_DEFAULT_A2_EXC = getattr(params, "A2_exc", 0.0)


__all__ = [
    "djt_hyperfine_tensors",
    "create_hamiltonian_nuclear",
    "create_B_hamiltonian",
    "build_single_manifold_hamiltonian",
    "build_ground_hamiltonian",
    "build_excited_hamiltonian",
    "solve_hamiltonian",
    "solve_ground_hamiltonian",
    "solve_excited_hamiltonian",
    "calculate_cyclicity",
    "calculate_spontaneous_cyclicity",
    "calculate_cyclicity_spinflip",
    "PLE_transitions",
    "PLE_spectrum",
    "PLE_spectrum_with_cyclicity",
    "get_dynamic_hamiltonian",
    "get_ground_hamiltonian",
    "get_excited_hamiltonian",
]


# -----------------------------------------------------------------------------
# Dense-array and angular-momentum helpers
# -----------------------------------------------------------------------------


def _spin_operator(j: float, axis: str) -> jqt.Qarray:
    """Construct one Cartesian angular-momentum operator.

    Parameters
    ----------
    j : float
        Spin quantum number. ``j`` must be a non-negative integer or
        half-integer.
    axis : {'x', 'y', 'z'}
        Cartesian component of the angular-momentum operator.

    Returns
    -------
    jaxquantum.Qarray
        Dense angular-momentum operator with Hilbert-space dimension
        ``2 * j + 1``. The basis order is
        ``|j, j>, |j, j - 1>, ..., |j, -j>``.

    Raises
    ------
    ValueError
        If ``j`` is negative, if ``j`` is not integer or half-integer, or if
        ``axis`` is not one of ``'x'``, ``'y'``, and ``'z'``.

    Notes
    -----
    The raising operator is constructed first from

    ``<j, m + 1|J+|j, m> = sqrt(j(j + 1) - m(m + 1))``.

    The Cartesian operators then follow from ``Jx = (J+ + J-)/2`` and
    ``Jy = (J+ - J-)/(2i)``.
    """
    j_float = float(j)
    two_j = int(round(2.0 * j_float))

    # ``2*j`` must be a non-negative integer for an allowed spin quantum
    # number. This validation occurs during ordinary Python execution, before
    # the operator is captured by any jitted Hamiltonian function.
    if two_j < 0 or abs(2.0 * j_float - two_j) > 1e-12:
        raise ValueError(
            "j must be a non-negative integer or half-integer; "
            f"received {j!r}."
        )

    dim = two_j + 1
    j_plus = jnp.zeros((dim, dim), dtype=jnp.complex128)

    # Column ``column`` corresponds to the ket with m = j - column. The
    # raising operator connects it to the preceding basis vector.
    for column in range(1, dim):
        m = j_float - column
        coefficient = jnp.sqrt(
            j_float * (j_float + 1.0) - m * (m + 1.0)
        )
        j_plus = j_plus.at[column - 1, column].set(coefficient)

    j_minus = jnp.conj(j_plus.T)

    if axis == "x":
        data = 0.5 * (j_plus + j_minus)
    elif axis == "y":
        data = (j_plus - j_minus) / (2.0j)
    elif axis == "z":
        m_values = j_float - jnp.arange(dim, dtype=jnp.float64)
        data = jnp.diag(m_values).astype(jnp.complex128)
    else:
        raise ValueError("axis must be one of 'x', 'y', or 'z'.")

    return jqt.Qarray.create(data, dims=(dim,))


def _dense_data(operator: jqt.Qarray) -> jax.Array:
    """Extract dense array data from a JAXQuantum object.

    Parameters
    ----------
    operator : jaxquantum.Qarray
        Operator, ket, or batched collection of operators/states.

    Returns
    -------
    jax.Array
        Dense array stored by ``operator``. No copy is intentionally
        requested; the exact behavior follows :meth:`Qarray.to_dense`.
    """
    return operator.to_dense().data


def _state_matrix(eigenstates: jqt.Qarray) -> jax.Array:
    """Convert batched ket eigenstates to a row-oriented state matrix.

    Parameters
    ----------
    eigenstates : jaxquantum.Qarray
        Batched ket object returned by :func:`jaxquantum.eigenstates`. Its
        dense representation is expected to have shape
        ``(num_states, dim, 1)``.

    Returns
    -------
    jax.Array, shape (num_states, dim)
        Dense state matrix. Row ``s`` stores the ket ``|psi_s>``.
    """
    return _dense_data(eigenstates)[..., :, 0]


def _eigenvector_columns(eigenstates: jqt.Qarray) -> jax.Array:
    """Convert batched ket eigenstates to a column-oriented matrix.

    Parameters
    ----------
    eigenstates : jaxquantum.Qarray
        Batched ket object returned by :func:`jaxquantum.eigenstates`.

    Returns
    -------
    jax.Array, shape (dim, num_states)
        Eigenvector matrix in the conventional linear-algebra orientation.
        Column ``s`` stores ``|psi_s>``.
    """
    return jnp.swapaxes(_state_matrix(eigenstates), -1, -2)


def _expect_batched(
    operator: jqt.Qarray,
    eigenstates: jqt.Qarray,
) -> jax.Array:
    """Evaluate an expectation value for every state in a ket batch.

    Parameters
    ----------
    operator : jaxquantum.Qarray
        Operator with dense shape ``(dim, dim)``.
    eigenstates : jaxquantum.Qarray
        Batched kets with dense shape ``(num_states, dim, 1)``.

    Returns
    -------
    jax.Array, shape (num_states,)
        Real parts of ``<psi_s|operator|psi_s>`` for all states.

    Notes
    -----
    The real part is returned because every use in this module involves a
    Hermitian observable. Taking the real part also removes roundoff-scale
    imaginary residuals from eigendecomposition.
    """
    states = _state_matrix(eigenstates)
    operator_data = _dense_data(operator)
    values = jnp.einsum(
        "si,ij,sj->s",
        jnp.conj(states),
        operator_data,
        states,
    )
    return jnp.real(values)


def _operator_expectations(
    eigenvectors: jax.Array,
    operator: jqt.Qarray,
) -> jax.Array:
    """Evaluate one operator in a column-oriented eigenvector basis.

    Parameters
    ----------
    eigenvectors : jax.Array, shape (dim, num_states)
        Eigenvectors arranged as columns.
    operator : jaxquantum.Qarray
        Operator with dense shape ``(dim, dim)``.

    Returns
    -------
    jax.Array, shape (num_states,)
        Complex expectation values ``<psi_i|operator|psi_i>``.
    """
    return jnp.einsum(
        "di,de,ei->i",
        jnp.conj(eigenvectors),
        _dense_data(operator),
        eigenvectors,
    )


def _dipole_matrix_elements(
    ground_eigenvectors: jax.Array,
    excited_eigenvectors: jax.Array,
    dipoles: Sequence[jqt.Qarray],
) -> jax.Array:
    """Evaluate all Cartesian dipole matrix elements.

    Parameters
    ----------
    ground_eigenvectors : jax.Array, shape (dim, num_ground)
        Ground-manifold eigenvectors arranged as columns.
    excited_eigenvectors : jax.Array, shape (dim, num_excited)
        Excited-manifold eigenvectors arranged as columns.
    dipoles : sequence of jaxquantum.Qarray
        Cartesian dipole operators, conventionally ``[p_x, p_y, p_z]``.

    Returns
    -------
    jax.Array, shape (3, num_excited, num_ground)
        Matrix elements ``M[j, l, k] = <exc_l|p_j|gnd_k>``.
    """
    dipole_data = jnp.stack(
        [_dense_data(operator) for operator in dipoles],
        axis=0,
    )
    return jnp.einsum(
        "dl,jde,ek->jlk",
        jnp.conj(excited_eigenvectors),
        dipole_data,
        ground_eigenvectors,
    )


def _reduced_spin_density_matrices(
    eigenvectors: jax.Array,
    orbital_dim: int,
    electron_dim: int,
    nuclear_dim: int,
) -> jax.Array:
    """Trace the orbital subsystem from pure-state density matrices.

    Parameters
    ----------
    eigenvectors : jax.Array, shape (dim, num_states)
        Full eigenvectors arranged as columns in the basis
        ``orbital x electron x nuclear``.
    orbital_dim : int
        Orbital Hilbert-space dimension.
    electron_dim : int
        Electron-spin Hilbert-space dimension.
    nuclear_dim : int
        Nuclear-spin Hilbert-space dimension.

    Returns
    -------
    jax.Array, shape (num_states, spin_dim, spin_dim)
        Reduced electron-nuclear density matrices, where
        ``spin_dim = electron_dim * nuclear_dim``.

    Raises
    ------
    ValueError
        If the dense state dimension does not equal
        ``orbital_dim * electron_dim * nuclear_dim``.

    Notes
    -----
    For ``psi[k, o, s]``, where ``o`` is orbital and ``s`` is the combined
    electron-nuclear index, the reduced state is

    ``rho[k, s, t] = sum_o psi[k, o, s] * conj(psi[k, o, t])``.
    """
    dim, num_states = eigenvectors.shape
    spin_dim = electron_dim * nuclear_dim

    if dim != orbital_dim * spin_dim:
        raise ValueError(
            "The eigenvector dimension is inconsistent with the orbital, "
            "electron, and nuclear subsystem dimensions."
        )

    # Convert from columns ``(dim, states)`` to explicit subsystem axes
    # ``(states, orbital, combined_spin)``.
    psi = jnp.swapaxes(eigenvectors, 0, 1).reshape(
        num_states,
        orbital_dim,
        spin_dim,
    )

    return jnp.einsum("kos,kot->kst", psi, jnp.conj(psi))


def _field_components(
    B: Any,
    theta: Any,
    phi: Any,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Convert spherical field coordinates to Cartesian components.

    Parameters
    ----------
    B : scalar
        Magnetic-field magnitude in Hamiltonian units.
    theta : scalar
        Polar angle measured from the local positive z axis, in radians.
    phi : scalar
        Azimuthal angle measured in the local x-y plane, in radians.

    Returns
    -------
    bx, by, bz : tuple of jax.Array
        Cartesian field components with the same broadcast-compatible scalar
        dtype as the inputs.
    """
    bx = B * jnp.sin(theta) * jnp.cos(phi)
    by = B * jnp.sin(theta) * jnp.sin(phi)
    bz = B * jnp.cos(theta)
    return bx, by, bz


# -----------------------------------------------------------------------------
# Dynamic-Jahn--Teller hyperfine conversion and validation
# -----------------------------------------------------------------------------


@jax.jit
def djt_hyperfine_tensors(
    Aperp: Any,
    Apar: Any,
    A1: Any,
    A2: Any,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    r"""Convert four DJT scalar couplings to Cartesian hyperfine tensors.

    The returned tensors reproduce the raising/lowering-operator DJT
    Hamiltonian when contracted as

    .. math::

        H_\mathrm{hf} =
        \sum_{ij} A_{ij} I_\mathrm{orb}\otimes S_i\otimes I_j
        + \sum_{ij} (A_x)_{ij}(-2X_\mathrm{orb})\otimes S_i\otimes I_j
        + \sum_{ij} (A_y)_{ij}(-2Y_\mathrm{orb})\otimes S_i\otimes I_j.

    Parameters
    ----------
    Aperp : scalar
        Orbital-independent transverse hyperfine coupling.
    Apar : scalar
        Orbital-independent longitudinal hyperfine coupling. The historical
        convention produces a ``2 * Apar * Sz * Iz`` contribution.
    A1 : scalar
        First orbital-modulated DJT hyperfine coupling.
    A2 : scalar
        Second orbital-modulated DJT hyperfine coupling.

    Returns
    -------
    A : jax.Array, shape (3, 3)
        Orbital-independent Cartesian hyperfine tensor.
    Ax : jax.Array, shape (3, 3)
        Cartesian tensor multiplying ``-2 * X_orbital``.
    Ay : jax.Array, shape (3, 3)
        Cartesian tensor multiplying ``-2 * Y_orbital``.

    Notes
    -----
    The nonzero entries are

    ``A_xx = A_yy = Aperp`` and ``A_zz = 2 * Apar``;

    ``Ax_xx = -A2/2``, ``Ax_yy = A2/2``, and
    ``Ax_xz = Ax_zx = A1``;

    ``Ay_xy = Ay_yx = -A2/2`` and ``Ay_yz = Ay_zy = -A1``.
    """
    dtype = jnp.result_type(Aperp, Apar, A1, A2, jnp.complex128)
    Aperp = jnp.asarray(Aperp, dtype=dtype)
    Apar = jnp.asarray(Apar, dtype=dtype)
    A1 = jnp.asarray(A1, dtype=dtype)
    A2 = jnp.asarray(A2, dtype=dtype)

    A = jnp.zeros((3, 3), dtype=dtype)
    A = A.at[0, 0].set(Aperp)
    A = A.at[1, 1].set(Aperp)
    A = A.at[2, 2].set(2.0 * Apar)

    Ax = jnp.zeros((3, 3), dtype=dtype)
    Ax = Ax.at[0, 0].set(-0.5 * A2)
    Ax = Ax.at[1, 1].set(0.5 * A2)
    Ax = Ax.at[0, 2].set(A1)
    Ax = Ax.at[2, 0].set(A1)

    Ay = jnp.zeros((3, 3), dtype=dtype)
    Ay = Ay.at[0, 1].set(-0.5 * A2)
    Ay = Ay.at[1, 0].set(-0.5 * A2)
    Ay = Ay.at[1, 2].set(-A1)
    Ay = Ay.at[2, 1].set(-A1)

    return A, Ax, Ay


def _validate_hyperfine_tensor(value: Any, name: str) -> jax.Array:
    """Convert and validate one explicit hyperfine tensor.

    Parameters
    ----------
    value : array_like
        Tensor candidate.
    name : str
        Human-readable argument name used in error messages.

    Returns
    -------
    jax.Array, shape (3, 3)
        Complex-valued tensor suitable for Hamiltonian contraction.

    Raises
    ------
    ValueError
        If ``value`` does not have shape ``(3, 3)``.
    """
    tensor = jnp.asarray(value, dtype=jnp.complex128)
    if tensor.shape != (3, 3):
        raise ValueError(
            f"{name} must have shape (3, 3); received {tensor.shape}."
        )
    return tensor


def _resolve_hyperfine_tensors(
    A: Optional[Any],
    Ax: Optional[Any],
    Ay: Optional[Any],
    default_Aperp: Any,
    default_Apar: Any,
    default_A1: Any,
    default_A2: Any,
    manifold_name: str,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Resolve optional tensor inputs into a complete hyperfine triplet.

    Parameters
    ----------
    A, Ax, Ay : array_like, shape (3, 3), or None
        Cartesian hyperfine tensors. ``A`` selects the defaulting mode.
    default_Aperp, default_Apar, default_A1, default_A2 : scalar
        Manifold-specific DJT scalar couplings used when ``A is None``.
    manifold_name : str
        Name used to make validation errors more informative.

    Returns
    -------
    A_resolved, Ax_resolved, Ay_resolved : tuple of jax.Array
        Complete complex-valued hyperfine tensors, each with shape ``(3, 3)``.

    Raises
    ------
    ValueError
        If ``A is None`` while ``Ax`` or ``Ay`` is supplied, or if a supplied
        tensor does not have shape ``(3, 3)``.

    Notes
    -----
    The supported modes are

    ``A=None, Ax=None, Ay=None``
        Construct the complete default DJT tensor triplet for the selected
        manifold with :func:`djt_hyperfine_tensors`.

    ``A=<array>, Ax=None, Ay=None``
        Use a custom orbital-independent tensor with zero orbital-dependent
        tensor blocks.

    ``A=<array>, Ax=<array>, Ay=<array>``
        Use a fully custom tensor triplet. Either dynamic tensor may also be
        omitted independently, in which case only that tensor is replaced by
        zero.

    The Python-level ``A is None`` branch is compatible with :func:`jax.jit`.
    Calls using defaults and calls using custom arrays have different pytree
    structures and are compiled separately.
    """
    if A is None:
        if Ax is not None or Ay is not None:
            raise ValueError(
                f"For the {manifold_name} manifold, Ax and Ay cannot be "
                "supplied when A is None. Supply A as well, or omit all "
                "three tensors to use the default DJT triplet."
            )
        return djt_hyperfine_tensors(
            default_Aperp,
            default_Apar,
            default_A1,
            default_A2,
        )

    A_resolved = _validate_hyperfine_tensor(A, f"A_{manifold_name}")
    zero = jnp.zeros((3, 3), dtype=jnp.complex128)
    Ax_resolved = (
        zero
        if Ax is None
        else _validate_hyperfine_tensor(Ax, f"Ax_{manifold_name}")
    )
    Ay_resolved = (
        zero
        if Ay is None
        else _validate_hyperfine_tensor(Ay, f"Ay_{manifold_name}")
    )
    return A_resolved, Ax_resolved, Ay_resolved


# -----------------------------------------------------------------------------
# Hamiltonian construction
# -----------------------------------------------------------------------------


def create_hamiltonian_nuclear() -> Tuple[
    Callable,
    Callable,
    List[jqt.Qarray],
    jqt.Qarray,
]:
    """Create reusable one-point DJT Hamiltonian builders and observables.

    Returns
    -------
    H : callable
        Full single-manifold Hamiltonian builder with signature
        ``H(bx, by, bz, rg, q, A, Ax, Ay, L, alpha, beta, upsilon, delta_f)``.
        ``A=None`` selects the ground DJT defaults for this generic builder.
        With custom ``A``, omitted ``Ax`` or ``Ay`` tensors are zero.
    Href : callable
        Reference Hamiltonian builder with signature
        ``Href(L, alpha, beta)``. It includes spin-orbit coupling and strain
        only.
    p : list of jaxquantum.Qarray
        Cartesian dipole operators ``[p_x, p_y, p_z]`` in the complete
        single-manifold Hilbert space.
    J2 : jaxquantum.Qarray
        Total electron-plus-nuclear angular-momentum-squared operator.

    Notes
    -----
    The tensor-product order is ``orbital x electron x nuclear``. Operators
    are created once at module import and captured by jitted functions, rather
    than rebuilt for every parameter point.
    """
    electron_spin = params.S
    nuclear_spin = params.Sn

    # The orbital pseudospin and electron spin use the same spin quantum
    # number in this model. For SnV both are two-dimensional (spin 1/2).
    X = _spin_operator(electron_spin, "x")
    Y = _spin_operator(electron_spin, "y")
    Z = _spin_operator(electron_spin, "z")
    I = jqt.identity(int(round(2.0 * electron_spin + 1.0)))

    Xn = _spin_operator(nuclear_spin, "x")
    Yn = _spin_operator(nuclear_spin, "y")
    Zn = _spin_operator(nuclear_spin, "z")
    In = jqt.identity(int(round(2.0 * nuclear_spin + 1.0)))

    tensor = jqt.tensor
    identity_full = tensor(I, I, In)

    # Full-space electron-spin operators: I_orb x S_i x I_nuc.
    Sx = tensor(I, X, In)
    Sy = tensor(I, Y, In)
    Sz = tensor(I, Z, In)

    # Full-space nuclear-spin operators: I_orb x I_e x I_i.
    Ix = tensor(I, I, Xn)
    Iy = tensor(I, I, Yn)
    Iz = tensor(I, I, Zn)

    # Full-space orbital pseudospin operators: L_i x I_e x I_nuc.
    Lx = tensor(X, I, In)
    Ly = tensor(Y, I, In)
    Lz = tensor(Z, I, In)

    # Products used by the spin-orbit and iso-orbital interactions.
    LzSz = tensor(Z, Z, In)
    LzIz = tensor(Z, I, Zn)

    # J^2 = S^2 + I^2 + 2 S dot I. The orbital factor is the identity because
    # J here refers only to electron plus nuclear angular momentum.
    J2 = (
        (
            electron_spin * (electron_spin + 1.0)
            + nuclear_spin * (nuclear_spin + 1.0)
        )
        * identity_full
        + 2.0
        * (
            tensor(I, X, Xn)
            + tensor(I, Y, Yn)
            + tensor(I, Z, Zn)
        )
    )

    # Precompute the 27 operator products needed for the three 3x3 hyperfine
    # tensors. The orbital prefactors preserve the validated convention:
    # A -> I_orb, Ax -> -2 X_orb, and Ay -> -2 Y_orb.
    spin_operators = (X, Y, Z)
    nuclear_operators = (Xn, Yn, Zn)
    orbital_prefactors = (I, -2.0 * X, -2.0 * Y)

    hyperfine_operator_blocks = tuple(
        tuple(
            tuple(
                tensor(
                    orbital_operator,
                    spin_operators[i],
                    nuclear_operators[j],
                )
                for j in range(3)
            )
            for i in range(3)
        )
        for orbital_operator in orbital_prefactors
    )

    def hyperfine_block(
        coupling_tensor: jax.Array,
        block_index: int,
    ) -> jqt.Qarray:
        """Contract one Cartesian tensor with one orbital operator block.

        Parameters
        ----------
        coupling_tensor : jax.Array, shape (3, 3)
            Electron-nuclear Cartesian coupling tensor.
        block_index : int
            Orbital-prefactor block: ``0`` for ``A``, ``1`` for ``Ax``, and
            ``2`` for ``Ay``.

        Returns
        -------
        jaxquantum.Qarray
            Operator ``sum_ij coupling_tensor[i, j] O x S_i x I_j``.
        """
        result = 0.0 * identity_full
        operator_block = hyperfine_operator_blocks[block_index]

        # These loops have fixed trip counts and are unrolled during tracing.
        for i in range(3):
            for j in range(3):
                result = result + coupling_tensor[i, j] * operator_block[i][j]

        return result

    def Hhf(
        A: jax.Array,
        Ax: jax.Array,
        Ay: jax.Array,
    ) -> jqt.Qarray:
        """Build the complete Cartesian-tensor hyperfine Hamiltonian.

        Parameters
        ----------
        A, Ax, Ay : jax.Array, shape (3, 3)
            Orbital-independent and orbital-modulated hyperfine tensors.

        Returns
        -------
        jaxquantum.Qarray
            Sum of all three hyperfine tensor contractions.
        """
        return (
            hyperfine_block(A, 0)
            + hyperfine_block(Ax, 1)
            + hyperfine_block(Ay, 2)
        )

    def Hsoc(L: Any) -> jqt.Qarray:
        """Build the spin-orbit interaction.

        Parameters
        ----------
        L : scalar
            Spin-orbit coupling strength.

        Returns
        -------
        jaxquantum.Qarray
            Operator ``-2 * L * Lz * Sz``.

        Notes
        -----
        The minus sign preserves the convention of the validated DJT source,
        which differs from the sign in some older non-DJT implementations.
        """
        return -2.0 * L * LzSz

    def Hioc(upsilon: Any) -> jqt.Qarray:
        """Build the iso-orbital electron-nuclear interaction.

        Parameters
        ----------
        upsilon : scalar
            Iso-orbital coupling strength.

        Returns
        -------
        jaxquantum.Qarray
            Operator ``2 * upsilon * Lz * Iz``.
        """
        return 2.0 * upsilon * LzIz

    def Hegx(alpha: Any) -> jqt.Qarray:
        """Build the x-like strain/Jahn--Teller interaction.

        Parameters
        ----------
        alpha : scalar
            X-like strain parameter.

        Returns
        -------
        jaxquantum.Qarray
            Operator ``-2 * alpha * Lx``.
        """
        return -2.0 * alpha * Lx

    def Hegy(beta: Any) -> jqt.Qarray:
        """Build the y-like strain/Jahn--Teller interaction.

        Parameters
        ----------
        beta : scalar
            Y-like strain parameter.

        Returns
        -------
        jaxquantum.Qarray
            Operator ``2 * beta * Ly``.
        """
        return 2.0 * beta * Ly

    # Dipole operators in the e+/e- orbital basis. p_z is proportional to the
    # identity in the complete single-manifold space.
    p = [2.0 * Lx, 2.0 * Ly, 2.0 * identity_full]

    def Href(L: Any, alpha: Any, beta: Any) -> jqt.Qarray:
        """Build the spin-orbit-plus-strain reference Hamiltonian.

        Parameters
        ----------
        L : scalar
            Spin-orbit coupling strength.
        alpha, beta : scalar
            X-like and y-like strain/Jahn--Teller parameters.

        Returns
        -------
        jaxquantum.Qarray
            Reference Hamiltonian used to define optical transition frequency
            offsets.
        """
        return Hsoc(L) + Hegx(alpha) + Hegy(beta)

    def H(
        bx: Any,
        by: Any,
        bz: Any,
        rg: Any,
        q: Any,
        A: Optional[Any],
        Ax: Optional[Any],
        Ay: Optional[Any],
        L: Any,
        alpha: Any,
        beta: Any,
        upsilon: Any,
        delta_f: Any,
    ) -> jqt.Qarray:
        """Evaluate the complete one-point single-manifold Hamiltonian.

        Parameters
        ----------
        bx, by, bz : scalar
            Cartesian magnetic-field components in Hamiltonian units.
        rg : scalar
            Nuclear-to-electron Zeeman scaling ratio.
        q : scalar
            Orbital magnetic-field susceptibility.
        A, Ax, Ay : array_like, shape (3, 3), or None
            Cartesian hyperfine tensors. ``A=None`` selects the ground DJT
            defaults for this generic low-level builder. With custom ``A``,
            omitted ``Ax`` or ``Ay`` tensors are zero.
        L : scalar
            Spin-orbit coupling strength.
        alpha, beta : scalar
            X-like and y-like strain/Jahn--Teller parameters.
        upsilon : scalar
            Iso-orbital coupling strength.
        delta_f : scalar
            Asymmetric-Ham correction to the electron z-Zeeman term.

        Returns
        -------
        jaxquantum.Qarray
            Total Hamiltonian in the basis
            ``orbital x electron x nuclear``.
        """
        A, Ax, Ay = _resolve_hyperfine_tensors(
            A,
            Ax,
            Ay,
            _DEFAULT_APERP_GND,
            _DEFAULT_APAR_GND,
            _DEFAULT_A1_GND,
            _DEFAULT_A2_GND,
            "generic-ground",
        )

        electron_zeeman = bx * Sx + by * Sy + bz * Sz
        nuclear_zeeman = rg * (bx * Ix + by * Iy + bz * Iz)

        # ``bz`` already uses the electron-Zeeman frequency convention. The
        # factor 2/gS converts the fitted orbital and asymmetric-Ham parameters
        # into that convention.
        orbital_zeeman = (2.0 * q / _G_S) * bz * Lz
        asymmetric_ham = (2.0 * delta_f / _G_S) * bz * Sz

        return (
            Href(L, alpha, beta)
            + electron_zeeman
            + nuclear_zeeman
            + orbital_zeeman
            + asymmetric_ham
            + Hhf(A, Ax, Ay)
            + Hioc(upsilon)
        )

    return H, Href, p, J2


def create_B_hamiltonian() -> Callable:
    """Create a one-point magnetic-field-only Hamiltonian builder.

    Returns
    -------
    Hb : callable
        Function with signature ``Hb(bx, by, bz, rg, q, delta_f)``. It
        returns the electron, nuclear, orbital, and asymmetric-Ham magnetic
        terms in one electronic manifold.

    Notes
    -----
    This builder intentionally excludes spin-orbit, strain, hyperfine, and
    iso-orbital terms. It is used for the time-dependent microwave-drive
    operator, whose amplitude is supplied separately from the static field.
    """
    electron_spin = params.S
    nuclear_spin = params.Sn

    X = _spin_operator(electron_spin, "x")
    Y = _spin_operator(electron_spin, "y")
    Z = _spin_operator(electron_spin, "z")
    I = jqt.identity(int(round(2.0 * electron_spin + 1.0)))

    Xn = _spin_operator(nuclear_spin, "x")
    Yn = _spin_operator(nuclear_spin, "y")
    Zn = _spin_operator(nuclear_spin, "z")
    In = jqt.identity(int(round(2.0 * nuclear_spin + 1.0)))

    tensor = jqt.tensor

    Sx = tensor(I, X, In)
    Sy = tensor(I, Y, In)
    Sz = tensor(I, Z, In)

    Ix = tensor(I, I, Xn)
    Iy = tensor(I, I, Yn)
    Iz = tensor(I, I, Zn)

    Lz = tensor(Z, I, In)

    def Hb(
        bx: Any,
        by: Any,
        bz: Any,
        rg: Any,
        q: Any,
        delta_f: Any,
    ) -> jqt.Qarray:
        """Evaluate all magnetic-field-dependent Hamiltonian terms.

        Parameters
        ----------
        bx, by, bz : scalar
            Cartesian drive-field components in Hamiltonian units.
        rg : scalar
            Nuclear-to-electron Zeeman scaling ratio.
        q : scalar
            Orbital magnetic-field susceptibility.
        delta_f : scalar
            Asymmetric-Ham electron z-Zeeman correction.

        Returns
        -------
        jaxquantum.Qarray
            Magnetic-field Hamiltonian in one electronic manifold.
        """
        electron_zeeman = bx * Sx + by * Sy + bz * Sz
        nuclear_zeeman = rg * (bx * Ix + by * Iy + bz * Iz)
        orbital_zeeman = (2.0 * q / _G_S) * bz * Lz
        asymmetric_ham = (2.0 * delta_f / _G_S) * bz * Sz
        return (
            electron_zeeman
            + nuclear_zeeman
            + orbital_zeeman
            + asymmetric_ham
        )

    return Hb


# Construct immutable operator trees once. JAXQuantum Qarrays are pytrees, so
# jitted functions can safely capture these objects as closed-over constants.
_H_SINGLE, _HREF, _DIPOLES, _J2 = create_hamiltonian_nuclear()
_H_FIELD = create_B_hamiltonian()


@jax.jit
def build_single_manifold_hamiltonian(
    B: Any,
    theta: Any,
    phi: Any,
    rg: Any,
    q: Any,
    A: Optional[Any],
    Ax: Optional[Any],
    Ay: Optional[Any],
    L: Any,
    alpha: Any,
    beta: Any,
    upsilon: Any,
    delta_f: Any,
) -> jqt.Qarray:
    """Build a generic single-manifold Hamiltonian at one parameter point.

    Parameters
    ----------
    B : scalar
        Static magnetic-field magnitude in Hamiltonian units.
    theta, phi : scalar
        Polar and azimuthal static-field angles in radians.
    rg : scalar
        Nuclear-to-electron Zeeman scaling ratio.
    q : scalar
        Orbital magnetic-field susceptibility.
    A, Ax, Ay : array_like, shape (3, 3), or None
        Cartesian hyperfine tensors. Because this generic helper has no
        manifold label, ``A=None`` selects the ground DJT defaults. Use
        :func:`build_excited_hamiltonian` for excited-manifold defaults. With
        custom ``A``, omitted ``Ax`` or ``Ay`` tensors are zero.
    L : scalar
        Spin-orbit coupling strength.
    alpha, beta : scalar
        X-like and y-like strain/Jahn--Teller parameters.
    upsilon : scalar
        Iso-orbital coupling strength.
    delta_f : scalar
        Asymmetric-Ham electron z-Zeeman correction.

    Returns
    -------
    jaxquantum.Qarray
        Total single-manifold Hamiltonian.

    Raises
    ------
    ValueError
        If any hyperfine tensor does not have shape ``(3, 3)``.
    """
    A, Ax, Ay = _resolve_hyperfine_tensors(
        A,
        Ax,
        Ay,
        _DEFAULT_APERP_GND,
        _DEFAULT_APAR_GND,
        _DEFAULT_A1_GND,
        _DEFAULT_A2_GND,
        "generic-ground",
    )

    bx, by, bz = _field_components(B, theta, phi)
    return _H_SINGLE(
        bx,
        by,
        bz,
        rg,
        q,
        A,
        Ax,
        Ay,
        L,
        alpha,
        beta,
        upsilon,
        delta_f,
    )


@jax.jit
def build_ground_hamiltonian(
    B: Any,
    theta: Any,
    phi: Any,
    alpha: Any = 0.0,
    beta: Any = 0.0,
    rg: Any = params.rg_117,
    q: Any = params.q,
    A: Optional[Any] = None,
    Ax: Optional[Any] = None,
    Ay: Optional[Any] = None,
    L: Any = params.L,
    upsilon: Any = 0.0,
    delta_f: Any = _DEFAULT_DELTA_F_GND,
) -> jqt.Qarray:
    """Build the ground-manifold Hamiltonian at one parameter point.

    Parameters
    ----------
    B : scalar
        Static magnetic-field magnitude in Hamiltonian units.
    theta, phi : scalar
        Polar and azimuthal static-field angles in radians.
    alpha, beta : scalar, optional
        Ground-manifold strain/Jahn--Teller parameters.
    rg : scalar, optional
        Nuclear-to-electron Zeeman scaling ratio.
    q : scalar, optional
        Ground-manifold orbital magnetic-field susceptibility.
    A, Ax, Ay : array_like, shape (3, 3), or None, optional
        Ground-manifold hyperfine tensors. When ``A`` is ``None``, the
        tensors are generated by
        ``djt_hyperfine_tensors(_DEFAULT_APERP_GND, _DEFAULT_APAR_GND,
        _DEFAULT_A1_GND, _DEFAULT_A2_GND)``. With custom ``A``, omitted ``Ax``
        or ``Ay`` tensors are zero.
    L : scalar, optional
        Ground-manifold spin-orbit coupling.
    upsilon : scalar, optional
        Ground-manifold iso-orbital coupling.
    delta_f : scalar, optional
        Ground-manifold asymmetric-Ham correction.

    Returns
    -------
    jaxquantum.Qarray
        Ground-manifold Hamiltonian in the basis
        ``orbital x electron x nuclear``.
    """
    A, Ax, Ay = _resolve_hyperfine_tensors(
        A,
        Ax,
        Ay,
        _DEFAULT_APERP_GND,
        _DEFAULT_APAR_GND,
        _DEFAULT_A1_GND,
        _DEFAULT_A2_GND,
        "ground",
    )
    return build_single_manifold_hamiltonian(
        B,
        theta,
        phi,
        rg,
        q,
        A,
        Ax,
        Ay,
        L,
        alpha,
        beta,
        upsilon,
        delta_f,
    )


@jax.jit
def build_excited_hamiltonian(
    B: Any,
    theta: Any,
    phi: Any,
    alpha: Any = 0.0,
    beta: Any = 0.0,
    rg: Any = params.rg_117,
    q: Any = params.q_exc,
    A: Optional[Any] = None,
    Ax: Optional[Any] = None,
    Ay: Optional[Any] = None,
    L: Any = params.L_exc,
    upsilon: Any = 0.0,
    delta_f: Any = _DEFAULT_DELTA_F_EXC,
) -> jqt.Qarray:
    """Build the excited-manifold Hamiltonian at one parameter point.

    Parameters
    ----------
    B : scalar
        Static magnetic-field magnitude in Hamiltonian units.
    theta, phi : scalar
        Polar and azimuthal static-field angles in radians.
    alpha, beta : scalar, optional
        Excited-manifold strain/Jahn--Teller parameters.
    rg : scalar, optional
        Nuclear-to-electron Zeeman scaling ratio.
    q : scalar, optional
        Excited-manifold orbital magnetic-field susceptibility.
    A, Ax, Ay : array_like, shape (3, 3), or None, optional
        Excited-manifold hyperfine tensors. When ``A`` is ``None``, the
        tensors are generated by
        ``djt_hyperfine_tensors(_DEFAULT_APERP_EXC, _DEFAULT_APAR_EXC,
        _DEFAULT_A1_EXC, _DEFAULT_A2_EXC)``. With custom ``A``, omitted ``Ax``
        or ``Ay`` tensors are zero.
    L : scalar, optional
        Excited-manifold spin-orbit coupling.
    upsilon : scalar, optional
        Excited-manifold iso-orbital coupling.
    delta_f : scalar, optional
        Excited-manifold asymmetric-Ham correction.

    Returns
    -------
    jaxquantum.Qarray
        Excited-manifold Hamiltonian in the basis
        ``orbital x electron x nuclear``.
    """
    A, Ax, Ay = _resolve_hyperfine_tensors(
        A,
        Ax,
        Ay,
        _DEFAULT_APERP_EXC,
        _DEFAULT_APAR_EXC,
        _DEFAULT_A1_EXC,
        _DEFAULT_A2_EXC,
        "excited",
    )
    return build_single_manifold_hamiltonian(
        B,
        theta,
        phi,
        rg,
        q,
        A,
        Ax,
        Ay,
        L,
        alpha,
        beta,
        upsilon,
        delta_f,
    )


# -----------------------------------------------------------------------------
# Eigensystem solvers
# -----------------------------------------------------------------------------


@jax.jit
def solve_hamiltonian(
    B: Any,
    theta: Any,
    phi: Any,
    rg: Any,
    q: Any,
    A: Optional[Any],
    Ax: Optional[Any],
    Ay: Optional[Any],
    L: Any,
    alpha: Any,
    beta: Any,
    upsilon: Any,
    delta_f: Any,
) -> Tuple[jax.Array, jax.Array, jax.Array, jqt.Qarray, jax.Array]:
    """Diagonalize a generic single-manifold Hamiltonian.

    Parameters
    ----------
    B : scalar
        Static magnetic-field magnitude in Hamiltonian units.
    theta, phi : scalar
        Polar and azimuthal static-field angles in radians.
    rg : scalar
        Nuclear-to-electron Zeeman scaling ratio.
    q : scalar
        Orbital magnetic-field susceptibility.
    A, Ax, Ay : array_like, shape (3, 3), or None
        Cartesian hyperfine tensors. ``A=None`` selects ground DJT defaults in
        this generic solver; with custom ``A``, omitted ``Ax`` or ``Ay``
        tensors are zero.
    L : scalar
        Spin-orbit coupling strength.
    alpha, beta : scalar
        X-like and y-like strain/Jahn--Teller parameters.
    upsilon : scalar
        Iso-orbital coupling strength.
    delta_f : scalar
        Asymmetric-Ham electron z-Zeeman correction.

    Returns
    -------
    E : jax.Array, shape (num_states,)
        Full-Hamiltonian eigenvalues in ascending order.
    Eref : jax.Array, shape (num_states,)
        Eigenvalues of the spin-orbit-plus-strain reference Hamiltonian.
    U : jax.Array, shape (dim, num_states)
        Full-Hamiltonian eigenvectors arranged as columns.
    U_states : jaxquantum.Qarray
        Batched ket representation returned by
        :func:`jaxquantum.eigenstates`.
    alignment : jax.Array, shape (num_states,)
        Expectation values of electron-plus-nuclear ``J2``.
    """
    hamiltonian = build_single_manifold_hamiltonian(
        B,
        theta,
        phi,
        rg,
        q,
        A,
        Ax,
        Ay,
        L,
        alpha,
        beta,
        upsilon,
        delta_f,
    )
    E, U_states = jqt.eigenstates(hamiltonian)
    U = _eigenvector_columns(U_states)
    alignment = _expect_batched(_J2, U_states)

    # The reference Hamiltonian uses the same L, alpha, and beta values but
    # excludes field, hyperfine, and iso-orbital interactions.
    Eref, _ = jqt.eigenstates(_HREF(L, alpha, beta))
    return E, Eref, U, U_states, alignment


@jax.jit
def solve_ground_hamiltonian(
    B: Any,
    theta: Any,
    phi: Any,
    alpha: Any = 0.0,
    beta: Any = 0.0,
    rg: Any = params.rg_117,
    q: Any = params.q,
    A: Optional[Any] = None,
    Ax: Optional[Any] = None,
    Ay: Optional[Any] = None,
    L: Any = params.L,
    upsilon: Any = 0.0,
    delta_f: Any = _DEFAULT_DELTA_F_GND,
) -> Tuple[jax.Array, jax.Array, jax.Array, jqt.Qarray, jax.Array]:
    """Diagonalize the ground-manifold Hamiltonian.

    Parameters
    ----------
    B : scalar
        Static magnetic-field magnitude in Hamiltonian units.
    theta, phi : scalar
        Polar and azimuthal static-field angles in radians.
    alpha, beta : scalar, optional
        Ground-manifold strain/Jahn--Teller parameters.
    rg, q : scalar, optional
        Nuclear Zeeman ratio and ground orbital susceptibility.
    A, Ax, Ay : array_like, shape (3, 3), or None, optional
        Ground hyperfine tensor triplet. Passing ``A=None`` with ``Ax`` and
        ``Ay`` also omitted inserts the default ground DJT tensors. With a
        custom ``A``, omitted ``Ax`` or ``Ay`` tensors are zero.
    L : scalar, optional
        Ground spin-orbit coupling.
    upsilon : scalar, optional
        Ground iso-orbital coupling.
    delta_f : scalar, optional
        Ground asymmetric-Ham correction.

    Returns
    -------
    E, Eref, U, U_states, alignment
        Eigensystem quantities described by :func:`solve_hamiltonian`.
    """
    A, Ax, Ay = _resolve_hyperfine_tensors(
        A,
        Ax,
        Ay,
        _DEFAULT_APERP_GND,
        _DEFAULT_APAR_GND,
        _DEFAULT_A1_GND,
        _DEFAULT_A2_GND,
        "ground",
    )
    return solve_hamiltonian(
        B,
        theta,
        phi,
        rg,
        q,
        A,
        Ax,
        Ay,
        L,
        alpha,
        beta,
        upsilon,
        delta_f,
    )


@jax.jit
def solve_excited_hamiltonian(
    B: Any,
    theta: Any,
    phi: Any,
    alpha: Any = 0.0,
    beta: Any = 0.0,
    rg: Any = params.rg_117,
    q: Any = params.q_exc,
    A: Optional[Any] = None,
    Ax: Optional[Any] = None,
    Ay: Optional[Any] = None,
    L: Any = params.L_exc,
    upsilon: Any = 0.0,
    delta_f: Any = _DEFAULT_DELTA_F_EXC,
) -> Tuple[jax.Array, jax.Array, jax.Array, jqt.Qarray, jax.Array]:
    """Diagonalize the excited-manifold Hamiltonian.

    Parameters
    ----------
    B : scalar
        Static magnetic-field magnitude in Hamiltonian units.
    theta, phi : scalar
        Polar and azimuthal static-field angles in radians.
    alpha, beta : scalar, optional
        Excited-manifold strain/Jahn--Teller parameters.
    rg, q : scalar, optional
        Nuclear Zeeman ratio and excited orbital susceptibility.
    A, Ax, Ay : array_like, shape (3, 3), or None, optional
        Excited hyperfine tensor triplet. Passing ``A=None`` with ``Ax`` and
        ``Ay`` also omitted inserts the default excited DJT tensors. With a
        custom ``A``, omitted ``Ax`` or ``Ay`` tensors are zero.
    L : scalar, optional
        Excited spin-orbit coupling.
    upsilon : scalar, optional
        Excited iso-orbital coupling.
    delta_f : scalar, optional
        Excited asymmetric-Ham correction.

    Returns
    -------
    E, Eref, U, U_states, alignment
        Eigensystem quantities described by :func:`solve_hamiltonian`.
    """
    A, Ax, Ay = _resolve_hyperfine_tensors(
        A,
        Ax,
        Ay,
        _DEFAULT_APERP_EXC,
        _DEFAULT_APAR_EXC,
        _DEFAULT_A1_EXC,
        _DEFAULT_A2_EXC,
        "excited",
    )
    return solve_hamiltonian(
        B,
        theta,
        phi,
        rg,
        q,
        A,
        Ax,
        Ay,
        L,
        alpha,
        beta,
        upsilon,
        delta_f,
    )


def _solve_ground_and_excited(
    B: Any,
    theta: Any,
    phi: Any,
    alpha: Any,
    beta: Any,
    alpha_exc: Any,
    beta_exc: Any,
    rg: Any,
    q_gnd: Any,
    A_gnd: Optional[Any],
    Ax_gnd: Optional[Any],
    Ay_gnd: Optional[Any],
    L_gnd: Any,
    upsilon_gnd: Any,
    delta_f_gnd: Any,
    q_exc: Any,
    A_exc: Optional[Any],
    Ax_exc: Optional[Any],
    Ay_exc: Optional[Any],
    L_exc: Any,
    upsilon_exc: Any,
    delta_f_exc: Any,
) -> Tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jqt.Qarray,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jqt.Qarray,
    jax.Array,
]:
    """Solve the ground and excited manifolds at one common field point.

    Parameters
    ----------
    B, theta, phi : scalar
        Common magnetic-field magnitude and orientation.
    alpha, beta : scalar
        Ground-manifold strain parameters.
    alpha_exc, beta_exc : scalar
        Excited-manifold strain parameters.
    rg : scalar
        Common nuclear-to-electron Zeeman scaling ratio.
    q_gnd, q_exc : scalar
        Ground- and excited-manifold orbital susceptibilities.
    A_gnd, Ax_gnd, Ay_gnd : array_like, shape (3, 3), or None
        Ground hyperfine tensors. Omitting the triplet selects ground DJT
        defaults.
    A_exc, Ax_exc, Ay_exc : array_like, shape (3, 3), or None
        Excited hyperfine tensors. Omitting the triplet selects excited DJT
        defaults.
    L_gnd, L_exc : scalar
        Ground- and excited-manifold spin-orbit couplings.
    upsilon_gnd, upsilon_exc : scalar
        Ground- and excited-manifold iso-orbital couplings.
    delta_f_gnd, delta_f_exc : scalar
        Ground- and excited-manifold asymmetric-Ham corrections.

    Returns
    -------
    E, Eref, U, U_states, alignment
        Ground-manifold eigensystem quantities.
    E_exc, Eref_exc, U_exc, U_exc_states, alignment_exc
        Excited-manifold eigensystem quantities.
    """
    ground_result = solve_ground_hamiltonian(
        B,
        theta,
        phi,
        alpha,
        beta,
        rg,
        q_gnd,
        A_gnd,
        Ax_gnd,
        Ay_gnd,
        L_gnd,
        upsilon_gnd,
        delta_f_gnd,
    )
    excited_result = solve_excited_hamiltonian(
        B,
        theta,
        phi,
        alpha_exc,
        beta_exc,
        rg,
        q_exc,
        A_exc,
        Ax_exc,
        Ay_exc,
        L_exc,
        upsilon_exc,
        delta_f_exc,
    )
    return (*ground_result, *excited_result)


# -----------------------------------------------------------------------------
# Optical rates and cyclicity
# -----------------------------------------------------------------------------


@jax.jit
def calculate_cyclicity(transition: Any) -> jax.Array:
    """Normalize transition rates into excited-state branching ratios.

    Parameters
    ----------
    transition : array_like, shape (..., num_excited, num_ground)
        Non-negative transition or spontaneous-emission rates. The final axis
        enumerates destination ground states.

    Returns
    -------
    jax.Array
        Row-normalized branching ratios with the same shape as ``transition``.
        Rows whose total rate is zero are returned as zeros.

    Raises
    ------
    ValueError
        If ``transition`` has fewer than two dimensions.
    """
    transition = jnp.asarray(transition)
    if transition.ndim < 2:
        raise ValueError(
            "transition must have at least two dimensions: "
            "(..., num_excited, num_ground)."
        )

    total_rate = jnp.sum(transition, axis=-1, keepdims=True)
    safe_total = jnp.where(total_rate > 0.0, total_rate, 1.0)
    return jnp.where(total_rate > 0.0, transition / safe_total, 0.0)


def _spontaneous_emission_from_eigenvectors(
    U_gnd: jax.Array,
    U_exc: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Calculate polarization-resolved and polarization-summed emission.

    Parameters
    ----------
    U_gnd : jax.Array, shape (dim, num_ground)
        Ground eigenvectors arranged as columns.
    U_exc : jax.Array, shape (dim, num_excited)
        Excited eigenvectors arranged as columns.

    Returns
    -------
    matrix_elements : jax.Array, shape (3, num_excited, num_ground)
        Cartesian dipole amplitudes ``<exc_l|p_j|gnd_k>``.
    emission : jax.Array, shape (num_excited, num_ground)
        Incoherent polarization sum
        ``sum_j |<exc_l|p_j|gnd_k>|**2``.
    """
    matrix_elements = _dipole_matrix_elements(U_gnd, U_exc, _DIPOLES)
    emission = jnp.sum(jnp.abs(matrix_elements) ** 2, axis=0)
    return matrix_elements, emission


def _validate_included_states(
    included_states: Optional[Tuple[int, ...]],
    num_ground_states: int,
    num_excited_states: int,
) -> Optional[Tuple[int, ...]]:
    """Validate matched ground/excited state indices for reduced models.

    Parameters
    ----------
    included_states : tuple of int or None
        Matched state indices. An index ``i`` selects both ground state ``i``
        and excited state ``i``.
    num_ground_states : int
        Number of available ground eigenstates.
    num_excited_states : int
        Number of available excited eigenstates.

    Returns
    -------
    tuple of int or None
        Validated immutable index tuple, or ``None`` for the full model.

    Raises
    ------
    TypeError
        If ``included_states`` is not a tuple or contains non-integer values.
    ValueError
        If the tuple is empty, contains duplicates, or contains an index
        outside either manifold.

    Notes
    -----
    The tuple controls output dimensions and is therefore a static argument in
    every jitted public function that accepts it.
    """
    if included_states is None:
        return None

    if not isinstance(included_states, tuple):
        raise TypeError(
            "included_states must be None or a tuple of ints, "
            "for example included_states=(0, 1, 2)."
        )
    if len(included_states) == 0:
        raise ValueError(
            "included_states must be None or a non-empty tuple of ints."
        )

    indices = []
    for value in included_states:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                "included_states must contain only Python int values."
            )
        indices.append(int(value))

    if len(set(indices)) != len(indices):
        raise ValueError("included_states contains duplicate state indices.")

    bad_ground = [
        index
        for index in indices
        if index < 0 or index >= num_ground_states
    ]
    bad_excited = [
        index
        for index in indices
        if index < 0 or index >= num_excited_states
    ]
    if bad_ground:
        raise ValueError(
            f"Ground-state indices out of range [0, {num_ground_states}): "
            f"{bad_ground}."
        )
    if bad_excited:
        raise ValueError(
            f"Excited-state indices out of range [0, {num_excited_states}): "
            f"{bad_excited}."
        )

    return tuple(indices)


@partial(jax.jit, static_argnames=("included_states",))
def calculate_spontaneous_cyclicity(
    B: Any,
    theta: Any,
    phi: Any,
    alpha: Any = 0.0,
    beta: Any = 0.0,
    alpha_exc: Any = 0.0,
    beta_exc: Any = 0.0,
    rg: Any = params.rg_117,
    q_gnd: Any = params.q,
    A_gnd: Optional[Any] = None,
    Ax_gnd: Optional[Any] = None,
    Ay_gnd: Optional[Any] = None,
    L_gnd: Any = params.L,
    upsilon_gnd: Any = 0.0,
    delta_f_gnd: Any = _DEFAULT_DELTA_F_GND,
    q_exc: Any = params.q_exc,
    A_exc: Optional[Any] = None,
    Ax_exc: Optional[Any] = None,
    Ay_exc: Optional[Any] = None,
    L_exc: Any = params.L_exc,
    upsilon_exc: Any = 0.0,
    delta_f_exc: Any = _DEFAULT_DELTA_F_EXC,
    included_states: Optional[Tuple[int, ...]] = None,
) -> jax.Array:
    """Calculate spontaneous-emission branching ratios.

    Parameters
    ----------
    B : scalar
        Static magnetic-field magnitude in Hamiltonian units.
    theta, phi : scalar
        Polar and azimuthal field angles in radians.
    alpha, beta : scalar, optional
        Ground-manifold strain parameters.
    alpha_exc, beta_exc : scalar, optional
        Excited-manifold strain parameters.
    rg : scalar, optional
        Nuclear-to-electron Zeeman scaling ratio.
    q_gnd, q_exc : scalar, optional
        Ground- and excited-manifold orbital susceptibilities.
    A_gnd, Ax_gnd, Ay_gnd : array_like, shape (3, 3), or None, optional
        Ground hyperfine tensor triplet. Passing ``A_gnd=None`` with
        ``Ax_gnd`` and ``Ay_gnd`` also omitted inserts the default ground DJT
        tensors. With a custom ``A_gnd``, omitted ``Ax_gnd`` or ``Ay_gnd``
        tensors are zero.
    A_exc, Ax_exc, Ay_exc : array_like, shape (3, 3), or None, optional
        Excited hyperfine tensor triplet. Passing ``A_exc=None`` with
        ``Ax_exc`` and ``Ay_exc`` also omitted inserts the default excited DJT
        tensors. With a custom ``A_exc``, omitted ``Ax_exc`` or ``Ay_exc``
        tensors are zero.
    L_gnd, L_exc : scalar, optional
        Ground- and excited-manifold spin-orbit couplings.
    upsilon_gnd, upsilon_exc : scalar, optional
        Ground- and excited-manifold iso-orbital couplings.
    delta_f_gnd, delta_f_exc : scalar, optional
        Ground- and excited-manifold asymmetric-Ham corrections.
    included_states : tuple of int or None, optional
        Matched ground/excited state indices to retain. The tuple is static
        under :func:`jax.jit` because it changes the output shape.

    Returns
    -------
    cyclicity : jax.Array, shape (num_excited, num_ground)
        Row-normalized polarization-summed spontaneous-emission branching
        ratios. If ``included_states`` is provided, both axes have length
        ``len(included_states)`` and the retained submatrix is renormalized.

    Notes
    -----
    Spontaneous emission is an incoherent sum over Cartesian polarizations:

    ``Gamma[l, k] = sum_j |<exc_l|p_j|gnd_k>|**2``.
    """
    (
        _,
        _,
        U,
        _,
        _,
        _,
        _,
        U_exc,
        _,
        _,
    ) = _solve_ground_and_excited(
        B,
        theta,
        phi,
        alpha,
        beta,
        alpha_exc,
        beta_exc,
        rg,
        q_gnd,
        A_gnd,
        Ax_gnd,
        Ay_gnd,
        L_gnd,
        upsilon_gnd,
        delta_f_gnd,
        q_exc,
        A_exc,
        Ax_exc,
        Ay_exc,
        L_exc,
        upsilon_exc,
        delta_f_exc,
    )

    _, spontaneous_rates = _spontaneous_emission_from_eigenvectors(U, U_exc)

    pair_indices = _validate_included_states(
        included_states,
        int(spontaneous_rates.shape[1]),
        int(spontaneous_rates.shape[0]),
    )
    if pair_indices is not None:
        indices = jnp.asarray(pair_indices, dtype=jnp.int32)
        spontaneous_rates = spontaneous_rates[
            indices[:, None],
            indices[None, :],
        ]

    return calculate_cyclicity(spontaneous_rates)


def _spinflip_cyclicity_from_eigenvectors(
    U_gnd: jax.Array,
    U_exc: jax.Array,
    theta: Any,
    phi: Any,
    cap: Any,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Calculate folded orbital-relaxation cyclicity from solved states.

    Parameters
    ----------
    U_gnd : jax.Array, shape (dim, num_ground)
        Ground eigenvectors arranged as columns and ordered by energy.
    U_exc : jax.Array, shape (dim, num_excited)
        Excited eigenvectors arranged as columns and ordered by energy.
    theta, phi : scalar
        Static-field orientation used to define electron-spin projection.
    cap : scalar
        Maximum effective cyclicity used to regularize vanishing pump-out
        rates.

    Returns
    -------
    emission : jax.Array, shape (num_excited, num_ground)
        Polarization-summed direct spontaneous-emission strengths.
    cyclicity : jax.Array, shape (num_excited, num_ground)
        Photons-before-pump-out metric. Values are populated only for the lower
        orbital ground branch; upper-branch entries are zero.
    spin_g : jax.Array, shape (num_ground,)
        Signs of ground electron-spin projections along the field.
    spin_e : jax.Array, shape (num_excited,)
        Signs of excited electron-spin projections along the field.
    emission_folded : jax.Array, shape (num_excited, num_lower)
        Emission into the lower ground branch after folding upper-branch decay
        through spin-preserving orbital relaxation.

    Notes
    -----
    The model assumes that the lower half of the energy-ordered ground states
    form the lower orbital branch and the upper half form the upper branch.
    Upper-branch population relaxes to the lower branch according to overlaps
    of reduced electron-nuclear density matrices.
    """
    num_states = int(U_gnd.shape[1])
    orbital_dim = int(round(2.0 * float(params.S) + 1.0))
    electron_dim = orbital_dim
    nuclear_dim = int(round(2.0 * float(params.Sn) + 1.0))
    num_lower = num_states // 2

    # Unit vector parallel to the static magnetic field.
    nx = jnp.sin(theta) * jnp.cos(phi)
    ny = jnp.sin(theta) * jnp.sin(phi)
    nz = jnp.cos(theta)

    X = _spin_operator(params.S, "x")
    Y = _spin_operator(params.S, "y")
    Z = _spin_operator(params.S, "z")
    I = jqt.identity(orbital_dim)
    In = jqt.identity(nuclear_dim)

    # The factor of two maps spin-1/2 operators to Pauli-normalized
    # projections with eigenvalues approximately +/-1.
    spin_along_field = 2.0 * (
        nx * jqt.tensor(I, X, In)
        + ny * jqt.tensor(I, Y, In)
        + nz * jqt.tensor(I, Z, In)
    )

    spin_g = jnp.sign(
        jnp.real(_operator_expectations(U_gnd, spin_along_field))
    )
    spin_e = jnp.sign(
        jnp.real(_operator_expectations(U_exc, spin_along_field))
    )

    _, emission = _spontaneous_emission_from_eigenvectors(U_gnd, U_exc)

    # Trace out the orbital subsystem. The resulting density matrices retain
    # all electron-nuclear spin coherences needed to model spin-preserving
    # orbital relaxation.
    rho_spin = _reduced_spin_density_matrices(
        U_gnd,
        orbital_dim=orbital_dim,
        electron_dim=electron_dim,
        nuclear_dim=nuclear_dim,
    )
    rho_upper = rho_spin[num_lower:, :, :]
    rho_lower = rho_spin[:num_lower, :, :]

    # relaxation[u, k] = Tr(rho_upper[u] rho_lower[k]).
    relaxation = jnp.real(
        jnp.einsum("ust,kts->uk", rho_upper, rho_lower)
    )
    relaxation = jnp.maximum(relaxation, 0.0)

    # Normalize each upper-state relaxation row. A uniform distribution is a
    # numerically safe fallback for a zero-overlap row.
    row_total = jnp.sum(relaxation, axis=-1, keepdims=True)
    uniform = jnp.full_like(relaxation, 1.0 / num_lower)
    relaxation = jnp.where(
        row_total > 0.0,
        relaxation / jnp.where(row_total > 0.0, row_total, 1.0),
        uniform,
    )

    # Direct lower-branch emission is retained. Emission into each upper state
    # is redistributed over lower states through the relaxation matrix.
    emission_folded = (
        emission[:, :num_lower]
        + jnp.einsum(
            "lu,uk->lk",
            emission[:, num_lower:],
            relaxation,
        )
    )

    total_emission = jnp.sum(emission, axis=-1)
    safe_cap = jnp.maximum(jnp.asarray(cap), jnp.finfo(jnp.float64).tiny)
    floor = jnp.where(
        total_emission > 0.0,
        total_emission / safe_cap,
        1.0,
    )
    pumpout = total_emission[:, None] - emission_folded

    # The numerator remains the direct resonant decay rate into the addressed
    # lower state, while the denominator includes every route that ultimately
    # pumps population out of that state.
    cyclicity_lower = (
        emission[:, :num_lower]
        / jnp.maximum(pumpout, floor[:, None])
    )
    cyclicity = jnp.zeros_like(emission)
    cyclicity = cyclicity.at[:, :num_lower].set(cyclicity_lower)

    return emission, cyclicity, spin_g, spin_e, emission_folded


@jax.jit
def calculate_cyclicity_spinflip(
    B: Any,
    theta: Any,
    phi: Any,
    alpha: Any = 0.0,
    beta: Any = 0.0,
    alpha_exc: Any = 0.0,
    beta_exc: Any = 0.0,
    rg: Any = params.rg_117,
    q_gnd: Any = params.q,
    A_gnd: Optional[Any] = None,
    Ax_gnd: Optional[Any] = None,
    Ay_gnd: Optional[Any] = None,
    L_gnd: Any = params.L,
    upsilon_gnd: Any = 0.0,
    delta_f_gnd: Any = _DEFAULT_DELTA_F_GND,
    q_exc: Any = params.q_exc,
    A_exc: Optional[Any] = None,
    Ax_exc: Optional[Any] = None,
    Ay_exc: Optional[Any] = None,
    L_exc: Any = params.L_exc,
    upsilon_exc: Any = 0.0,
    delta_f_exc: Any = _DEFAULT_DELTA_F_EXC,
    cap: Any = 1e6,
) -> Tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Calculate spin-aware photons-before-pump-out cyclicity.

    Parameters
    ----------
    B : scalar
        Static magnetic-field magnitude in Hamiltonian units.
    theta, phi : scalar
        Polar and azimuthal field angles in radians.
    alpha, beta : scalar, optional
        Ground-manifold strain parameters.
    alpha_exc, beta_exc : scalar, optional
        Excited-manifold strain parameters.
    rg : scalar, optional
        Nuclear-to-electron Zeeman scaling ratio.
    q_gnd, q_exc : scalar, optional
        Ground- and excited-manifold orbital susceptibilities.
    A_gnd, Ax_gnd, Ay_gnd : array_like, shape (3, 3), or None, optional
        Ground hyperfine tensors. Passing ``A_gnd=None`` with ``Ax_gnd`` and
        ``Ay_gnd`` also omitted inserts the default ground DJT triplet.
    A_exc, Ax_exc, Ay_exc : array_like, shape (3, 3), or None, optional
        Excited hyperfine tensors. Passing ``A_exc=None`` with ``Ax_exc`` and
        ``Ay_exc`` also omitted inserts the default excited DJT triplet.
    L_gnd, L_exc : scalar, optional
        Ground- and excited-manifold spin-orbit couplings.
    upsilon_gnd, upsilon_exc : scalar, optional
        Ground- and excited-manifold iso-orbital couplings.
    delta_f_gnd, delta_f_exc : scalar, optional
        Ground- and excited-manifold asymmetric-Ham corrections.
    cap : scalar, optional
        Maximum effective cyclicity used to regularize a zero pump-out rate.
        The implementation uses a minimum denominator of
        ``total_emission / cap``.

    Returns
    -------
    E : jax.Array, shape (num_ground,)
        Ground-manifold eigenvalues.
    E_exc : jax.Array, shape (num_excited,)
        Excited-manifold eigenvalues.
    emission : jax.Array, shape (num_excited, num_ground)
        Polarization-summed direct emission strengths.
    cyclicity : jax.Array, shape (num_excited, num_ground)
        Spin-aware photons-before-pump-out metric. Only lower-branch ground
        columns are populated.
    spin_g : jax.Array, shape (num_ground,)
        Signs of ground electron-spin projections along the field.
    spin_e : jax.Array, shape (num_excited,)
        Signs of excited electron-spin projections along the field.
    emission_folded : jax.Array, shape (num_excited, num_lower)
        Emission after upper-to-lower orbital relaxation is folded in.

    Notes
    -----
    This function evaluates one field point. Apply :func:`jax.vmap` outside the
    function for magnetic-field or strain sweeps.
    """
    (
        E,
        _,
        U,
        _,
        _,
        E_exc,
        _,
        U_exc,
        _,
        _,
    ) = _solve_ground_and_excited(
        B,
        theta,
        phi,
        alpha,
        beta,
        alpha_exc,
        beta_exc,
        rg,
        q_gnd,
        A_gnd,
        Ax_gnd,
        Ay_gnd,
        L_gnd,
        upsilon_gnd,
        delta_f_gnd,
        q_exc,
        A_exc,
        Ax_exc,
        Ay_exc,
        L_exc,
        upsilon_exc,
        delta_f_exc,
    )

    emission, cyclicity, spin_g, spin_e, emission_folded = (
        _spinflip_cyclicity_from_eigenvectors(
            U,
            U_exc,
            theta,
            phi,
            cap,
        )
    )

    return (
        E,
        E_exc,
        emission,
        cyclicity,
        spin_g,
        spin_e,
        emission_folded,
    )


@jax.jit
def PLE_transitions(
    B: Any,
    theta: Any,
    phi: Any,
    eta_x: Any,
    eta_y: Any,
    eta_z: Any,
    alpha: Any = 0.0,
    beta: Any = 0.0,
    alpha_exc: Any = 0.0,
    beta_exc: Any = 0.0,
    rg: Any = params.rg_117,
    q_gnd: Any = params.q,
    A_gnd: Optional[Any] = None,
    Ax_gnd: Optional[Any] = None,
    Ay_gnd: Optional[Any] = None,
    L_gnd: Any = params.L,
    upsilon_gnd: Any = 0.0,
    delta_f_gnd: Any = _DEFAULT_DELTA_F_GND,
    q_exc: Any = params.q_exc,
    A_exc: Optional[Any] = None,
    Ax_exc: Optional[Any] = None,
    Ay_exc: Optional[Any] = None,
    L_exc: Any = params.L_exc,
    upsilon_exc: Any = 0.0,
    delta_f_exc: Any = _DEFAULT_DELTA_F_EXC,
) -> Tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Calculate polarization-resolved PLE transition intensities.

    Parameters
    ----------
    B : scalar
        Static magnetic-field magnitude in Hamiltonian units.
    theta, phi : scalar
        Polar and azimuthal field angles in radians.
    eta_x, eta_y, eta_z : scalar
        Complex Cartesian components of the excitation polarization. Keeping
        these as separate scalar arguments makes polarization sweeps convenient
        with :func:`jax.vmap`.
    alpha, beta : scalar, optional
        Ground-manifold strain parameters.
    alpha_exc, beta_exc : scalar, optional
        Excited-manifold strain parameters.
    rg : scalar, optional
        Nuclear-to-electron Zeeman scaling ratio.
    q_gnd, q_exc : scalar, optional
        Ground- and excited-manifold orbital susceptibilities.
    A_gnd, Ax_gnd, Ay_gnd : array_like, shape (3, 3), or None, optional
        Ground hyperfine tensors. Passing ``A_gnd=None`` with ``Ax_gnd`` and
        ``Ay_gnd`` also omitted inserts the default ground DJT triplet.
    A_exc, Ax_exc, Ay_exc : array_like, shape (3, 3), or None, optional
        Excited hyperfine tensors. Passing ``A_exc=None`` with ``Ax_exc`` and
        ``Ay_exc`` also omitted inserts the default excited DJT triplet.
    L_gnd, L_exc : scalar, optional
        Ground- and excited-manifold spin-orbit couplings.
    upsilon_gnd, upsilon_exc : scalar, optional
        Ground- and excited-manifold iso-orbital couplings.
    delta_f_gnd, delta_f_exc : scalar, optional
        Ground- and excited-manifold asymmetric-Ham corrections.

    Returns
    -------
    E : jax.Array, shape (num_ground,)
        Ground eigenvalues.
    Eref : jax.Array, shape (num_ground,)
        Ground reference-Hamiltonian eigenvalues.
    U : jax.Array, shape (dim, num_ground)
        Ground eigenvectors arranged as columns.
    alignment : jax.Array, shape (num_ground,)
        Ground ``J2`` expectation values.
    E_exc : jax.Array, shape (num_excited,)
        Excited eigenvalues.
    Eref_exc : jax.Array, shape (num_excited,)
        Excited reference-Hamiltonian eigenvalues.
    U_exc : jax.Array, shape (dim, num_excited)
        Excited eigenvectors arranged as columns.
    alignment_exc : jax.Array, shape (num_excited,)
        Excited ``J2`` expectation values.
    transition : jax.Array, shape (num_excited, num_ground)
        Coherent polarization-projected excitation strengths
        ``|sum_j eta_j <exc_l|p_j|gnd_k>|**2``.
    cyclicity : jax.Array, shape (num_excited, num_ground)
        Polarization-summed spontaneous-emission branching ratios.

    Notes
    -----
    Excitation is coherent across polarization components, whereas spontaneous
    emission is summed incoherently over ``p_x``, ``p_y``, and ``p_z``.
    """
    (
        E,
        Eref,
        U,
        _,
        alignment,
        E_exc,
        Eref_exc,
        U_exc,
        _,
        alignment_exc,
    ) = _solve_ground_and_excited(
        B,
        theta,
        phi,
        alpha,
        beta,
        alpha_exc,
        beta_exc,
        rg,
        q_gnd,
        A_gnd,
        Ax_gnd,
        Ay_gnd,
        L_gnd,
        upsilon_gnd,
        delta_f_gnd,
        q_exc,
        A_exc,
        Ax_exc,
        Ay_exc,
        L_exc,
        upsilon_exc,
        delta_f_exc,
    )

    matrix_elements, emission = _spontaneous_emission_from_eigenvectors(
        U,
        U_exc,
    )

    # Build the three-component complex polarization vector without requiring
    # the caller to create a dynamically shaped array.
    eta = jnp.stack(
        [
            jnp.asarray(eta_x, dtype=jnp.complex128),
            jnp.asarray(eta_y, dtype=jnp.complex128),
            jnp.asarray(eta_z, dtype=jnp.complex128),
        ]
    )
    amplitude = jnp.einsum("j,jlk->lk", eta, matrix_elements)
    transition = jnp.abs(amplitude) ** 2
    cyclicity = calculate_cyclicity(emission)

    return (
        E,
        Eref,
        U,
        alignment,
        E_exc,
        Eref_exc,
        U_exc,
        alignment_exc,
        transition,
        cyclicity,
    )


def _folded_cyclicity_weight(
    cyclicity: Any,
    cyclicity_min: Any,
    cyclicity_weight: Any,
    cyclicity_half: Any,
    cyclicity_softness: Any,
) -> Tuple[jax.Array, jax.Array]:
    """Convert folded cyclicity into a full PLE brightness matrix.

    Parameters
    ----------
    cyclicity : array_like, shape (num_excited, num_ground)
        Spin-aware cyclicity matrix from
        :func:`calculate_cyclicity_spinflip`.
    cyclicity_min : scalar
        Soft lower cyclicity threshold. A value of zero disables thresholding.
    cyclicity_weight : scalar
        Interpolation amount for smooth cyclicity weighting. Zero leaves line
        brightness unchanged; one multiplies lower-branch lines by
        ``C / (C + cyclicity_half)``.
    cyclicity_half : scalar
        Cyclicity at which the smooth weighting factor equals one half.
    cyclicity_softness : scalar
        Hill exponent used for the soft threshold at ``cyclicity_min``.

    Returns
    -------
    weight : jax.Array, shape (num_excited, num_ground)
        Multiplicative PLE line-brightness matrix. Upper-branch ground columns
        remain one.
    C : jax.Array, shape (num_excited, num_lower)
        Non-negative lower-branch cyclicity values.
    """
    cyclicity = jnp.asarray(cyclicity)
    num_excited, num_ground = cyclicity.shape
    num_lower = num_ground // 2
    C = jnp.maximum(jnp.real(cyclicity[:, :num_lower]), 0.0)

    # Smooth saturation factor C/(C+C_half), protected against a non-positive
    # half-saturation value.
    half = jnp.maximum(
        jnp.asarray(cyclicity_half, dtype=C.dtype),
        jnp.finfo(C.dtype).tiny,
    )
    smooth = C / (C + half)
    weight_amount = jnp.asarray(cyclicity_weight, dtype=C.dtype)
    gate = 1.0 + weight_amount * (smooth - 1.0)

    # Hill-function threshold. ``jnp.where`` keeps this branch traceable and
    # makes threshold=0 exactly equivalent to no threshold.
    threshold = jnp.maximum(jnp.asarray(cyclicity_min, dtype=C.dtype), 0.0)
    exponent = jnp.asarray(cyclicity_softness, dtype=C.dtype)
    C_power = C**exponent
    threshold_power = threshold**exponent
    denominator = C_power + threshold_power
    hill = jnp.where(denominator > 0.0, C_power / denominator, 0.0)
    gate = gate * jnp.where(threshold > 0.0, hill, 1.0)

    weight = jnp.ones((num_excited, num_ground), dtype=C.dtype)
    weight = weight.at[:, :num_lower].set(gate)
    return weight, C


def _ple_spectrum_core(
    f_meas: Any,
    B: Any,
    theta: Any,
    phi: Any,
    eta_x: Any,
    eta_y: Any,
    eta_z: Any,
    intensity: Any,
    lw: Any,
    alpha: Any,
    beta: Any,
    alpha_exc: Any,
    beta_exc: Any,
    rg: Any,
    q_gnd: Any,
    A_gnd: Optional[Any],
    Ax_gnd: Optional[Any],
    Ay_gnd: Optional[Any],
    L_gnd: Any,
    upsilon_gnd: Any,
    delta_f_gnd: Any,
    q_exc: Any,
    A_exc: Optional[Any],
    Ax_exc: Optional[Any],
    Ay_exc: Optional[Any],
    L_exc: Any,
    upsilon_exc: Any,
    delta_f_exc: Any,
    cyclicity_min: Any,
    cyclicity_weight: Any,
    cyclicity_half: Any,
    cyclicity_softness: Any,
    cyclicity_cap: Any,
) -> Tuple[jax.Array, jax.Array]:
    """Evaluate one PLE frequency point and lower-branch cyclicity.

    Parameters
    ----------
    f_meas : scalar
        Frequency at which to evaluate the spectrum.
    B, theta, phi : scalar
        Static magnetic-field magnitude and orientation.
    eta_x, eta_y, eta_z : scalar
        Complex Cartesian excitation-polarization components.
    intensity : scalar
        Overall spectrum scale.
    lw : scalar
        Lorentzian full width at half maximum.
    alpha, beta, alpha_exc, beta_exc : scalar
        Ground and excited strain parameters.
    rg, q_gnd, q_exc : scalar
        Zeeman ratio and orbital susceptibilities.
    A_gnd, Ax_gnd, Ay_gnd : array_like, shape (3, 3), or None
        Ground hyperfine tensors or the all-``None`` default selector.
    A_exc, Ax_exc, Ay_exc : array_like, shape (3, 3), or None
        Excited hyperfine tensors or the all-``None`` default selector.
    L_gnd, L_exc : scalar
        Ground and excited spin-orbit couplings.
    upsilon_gnd, upsilon_exc : scalar
        Ground and excited iso-orbital couplings.
    delta_f_gnd, delta_f_exc : scalar
        Ground and excited asymmetric-Ham corrections.
    cyclicity_min, cyclicity_weight, cyclicity_half, cyclicity_softness : scalar
        Cyclicity-based brightness controls passed to
        :func:`_folded_cyclicity_weight`.
    cyclicity_cap : scalar
        Pump-out regularization cap.

    Returns
    -------
    spectrum : jax.Array, scalar
        Lorentzian-broadened PLE intensity at ``f_meas``.
    cyclicity_lower : jax.Array, shape (num_excited, num_lower)
        Lower-branch spin-aware cyclicity.
    """
    (
        E,
        Eref,
        U,
        _,
        _,
        E_exc,
        Eref_exc,
        U_exc,
        _,
        _,
    ) = _solve_ground_and_excited(
        B,
        theta,
        phi,
        alpha,
        beta,
        alpha_exc,
        beta_exc,
        rg,
        q_gnd,
        A_gnd,
        Ax_gnd,
        Ay_gnd,
        L_gnd,
        upsilon_gnd,
        delta_f_gnd,
        q_exc,
        A_exc,
        Ax_exc,
        Ay_exc,
        L_exc,
        upsilon_exc,
        delta_f_exc,
    )

    matrix_elements, _ = _spontaneous_emission_from_eigenvectors(U, U_exc)
    eta = jnp.stack(
        [
            jnp.asarray(eta_x, dtype=jnp.complex128),
            jnp.asarray(eta_y, dtype=jnp.complex128),
            jnp.asarray(eta_z, dtype=jnp.complex128),
        ]
    )
    amplitude = jnp.einsum("j,jlk->lk", eta, matrix_elements)
    transition = jnp.abs(amplitude) ** 2

    _, cyclicity, _, _, _ = _spinflip_cyclicity_from_eigenvectors(
        U,
        U_exc,
        theta,
        phi,
        cyclicity_cap,
    )
    weight, cyclicity_lower = _folded_cyclicity_weight(
        cyclicity,
        cyclicity_min,
        cyclicity_weight,
        cyclicity_half,
        cyclicity_softness,
    )

    # Reference all optical lines to the lowest eigenvalue of each strained
    # spin-orbit reference Hamiltonian, preserving the original DJT convention.
    transition_frequencies = (
        (E_exc - Eref_exc[0])[:, None]
        - (E - Eref[0])[None, :]
    )

    half_linewidth_squared = (jnp.asarray(lw) / 2.0) ** 2
    detuning = jnp.asarray(f_meas) - transition_frequencies
    lorentzians = (
        half_linewidth_squared
        / (detuning**2 + half_linewidth_squared)
    )
    spectrum = jnp.asarray(intensity) * jnp.sum(
        transition * weight * lorentzians
    )
    return spectrum, cyclicity_lower


@jax.jit
def PLE_spectrum(
    f_meas: Any,
    B: Any,
    theta: Any,
    phi: Any,
    eta_x: Any,
    eta_y: Any,
    eta_z: Any,
    intensity: Any = 1.0,
    lw: Any = 0.080,
    alpha: Any = 0.0,
    beta: Any = 0.0,
    alpha_exc: Any = 0.0,
    beta_exc: Any = 0.0,
    rg: Any = params.rg_117,
    q_gnd: Any = params.q,
    A_gnd: Optional[Any] = None,
    Ax_gnd: Optional[Any] = None,
    Ay_gnd: Optional[Any] = None,
    L_gnd: Any = params.L,
    upsilon_gnd: Any = 0.0,
    delta_f_gnd: Any = _DEFAULT_DELTA_F_GND,
    q_exc: Any = params.q_exc,
    A_exc: Optional[Any] = None,
    Ax_exc: Optional[Any] = None,
    Ay_exc: Optional[Any] = None,
    L_exc: Any = params.L_exc,
    upsilon_exc: Any = 0.0,
    delta_f_exc: Any = _DEFAULT_DELTA_F_EXC,
    cyclicity_min: Any = 0.0,
    cyclicity_weight: Any = 0.0,
    cyclicity_half: Any = 1.0,
    cyclicity_softness: Any = 4.0,
    cyclicity_cap: Any = 1e6,
) -> jax.Array:
    """Calculate PLE intensity at one frequency and one field point.

    Parameters
    ----------
    f_meas : scalar
        Frequency at which the spectrum is evaluated. Use :func:`jax.vmap` to
        evaluate a frequency grid.
    B : scalar
        Static magnetic-field magnitude in Hamiltonian units.
    theta, phi : scalar
        Polar and azimuthal field angles in radians.
    eta_x, eta_y, eta_z : scalar
        Complex Cartesian excitation-polarization components.
    intensity : scalar, optional
        Overall multiplicative spectrum scale.
    lw : scalar, optional
        Lorentzian full width at half maximum.
    alpha, beta : scalar, optional
        Ground-manifold strain parameters.
    alpha_exc, beta_exc : scalar, optional
        Excited-manifold strain parameters.
    rg : scalar, optional
        Nuclear-to-electron Zeeman scaling ratio.
    q_gnd, q_exc : scalar, optional
        Ground- and excited-manifold orbital susceptibilities.
    A_gnd, Ax_gnd, Ay_gnd : array_like, shape (3, 3), or None, optional
        Ground hyperfine tensors. Omitting the triplet inserts ground DJT
        defaults.
    A_exc, Ax_exc, Ay_exc : array_like, shape (3, 3), or None, optional
        Excited hyperfine tensors. Omitting the triplet inserts excited DJT
        defaults.
    L_gnd, L_exc : scalar, optional
        Ground- and excited-manifold spin-orbit couplings.
    upsilon_gnd, upsilon_exc : scalar, optional
        Ground- and excited-manifold iso-orbital couplings.
    delta_f_gnd, delta_f_exc : scalar, optional
        Ground- and excited-manifold asymmetric-Ham corrections.
    cyclicity_min : scalar, optional
        Soft lower threshold for spin-aware cyclicity. Zero disables it.
    cyclicity_weight : scalar, optional
        Interpolation amount for smooth cyclicity-dependent line brightness.
        Zero gives the unweighted spectrum; one applies the full smooth factor.
    cyclicity_half : scalar, optional
        Half-saturation cyclicity for smooth brightness weighting.
    cyclicity_softness : scalar, optional
        Hill exponent for the soft cyclicity threshold.
    cyclicity_cap : scalar, optional
        Pump-out regularization cap used in folded cyclicity.

    Returns
    -------
    jax.Array, scalar
        Sum of all Lorentzian-broadened transition contributions at
        ``f_meas``.

    Examples
    --------
    Evaluate a frequency grid while keeping all other inputs fixed::

        spectrum = jax.vmap(
            lambda f: PLE_spectrum(
                f, B, theta, phi, eta_x, eta_y, eta_z
            )
        )(frequency_grid)
    """
    spectrum, _ = _ple_spectrum_core(
        f_meas,
        B,
        theta,
        phi,
        eta_x,
        eta_y,
        eta_z,
        intensity,
        lw,
        alpha,
        beta,
        alpha_exc,
        beta_exc,
        rg,
        q_gnd,
        A_gnd,
        Ax_gnd,
        Ay_gnd,
        L_gnd,
        upsilon_gnd,
        delta_f_gnd,
        q_exc,
        A_exc,
        Ax_exc,
        Ay_exc,
        L_exc,
        upsilon_exc,
        delta_f_exc,
        cyclicity_min,
        cyclicity_weight,
        cyclicity_half,
        cyclicity_softness,
        cyclicity_cap,
    )
    return spectrum


@jax.jit
def PLE_spectrum_with_cyclicity(
    f_meas: Any,
    B: Any,
    theta: Any,
    phi: Any,
    eta_x: Any,
    eta_y: Any,
    eta_z: Any,
    intensity: Any = 1.0,
    lw: Any = 0.080,
    alpha: Any = 0.0,
    beta: Any = 0.0,
    alpha_exc: Any = 0.0,
    beta_exc: Any = 0.0,
    rg: Any = params.rg_117,
    q_gnd: Any = params.q,
    A_gnd: Optional[Any] = None,
    Ax_gnd: Optional[Any] = None,
    Ay_gnd: Optional[Any] = None,
    L_gnd: Any = params.L,
    upsilon_gnd: Any = 0.0,
    delta_f_gnd: Any = _DEFAULT_DELTA_F_GND,
    q_exc: Any = params.q_exc,
    A_exc: Optional[Any] = None,
    Ax_exc: Optional[Any] = None,
    Ay_exc: Optional[Any] = None,
    L_exc: Any = params.L_exc,
    upsilon_exc: Any = 0.0,
    delta_f_exc: Any = _DEFAULT_DELTA_F_EXC,
    cyclicity_min: Any = 0.0,
    cyclicity_weight: Any = 0.0,
    cyclicity_half: Any = 1.0,
    cyclicity_softness: Any = 4.0,
    cyclicity_cap: Any = 1e6,
) -> Tuple[jax.Array, jax.Array]:
    """Return one PLE intensity and the lower-branch folded cyclicity.

    Parameters
    ----------
    f_meas, B, theta, phi, eta_x, eta_y, eta_z
        Frequency, magnetic-field, and polarization inputs described by
        :func:`PLE_spectrum`.
    intensity, lw : scalar, optional
        Overall scale and Lorentzian full width at half maximum.
    alpha, beta, alpha_exc, beta_exc : scalar, optional
        Ground- and excited-manifold strain parameters.
    rg, q_gnd, q_exc : scalar, optional
        Zeeman ratio and orbital susceptibilities.
    A_gnd, Ax_gnd, Ay_gnd : array_like, shape (3, 3), or None, optional
        Ground hyperfine tensor triplet or all-``None`` DJT defaults.
    A_exc, Ax_exc, Ay_exc : array_like, shape (3, 3), or None, optional
        Excited hyperfine tensor triplet or all-``None`` DJT defaults.
    L_gnd, L_exc, upsilon_gnd, upsilon_exc : scalar, optional
        Spin-orbit and iso-orbital couplings.
    delta_f_gnd, delta_f_exc : scalar, optional
        Asymmetric-Ham corrections.
    cyclicity_min, cyclicity_weight, cyclicity_half, cyclicity_softness : scalar
        Cyclicity brightness controls described by :func:`PLE_spectrum`.
    cyclicity_cap : scalar, optional
        Pump-out regularization cap.

    Returns
    -------
    spectrum : jax.Array, scalar
        PLE intensity at ``f_meas``.
    cyclicity_lower : jax.Array, shape (num_excited, num_lower)
        Folded cyclicity for transitions ending in the lower orbital ground
        branch.
    """
    return _ple_spectrum_core(
        f_meas,
        B,
        theta,
        phi,
        eta_x,
        eta_y,
        eta_z,
        intensity,
        lw,
        alpha,
        beta,
        alpha_exc,
        beta_exc,
        rg,
        q_gnd,
        A_gnd,
        Ax_gnd,
        Ay_gnd,
        L_gnd,
        upsilon_gnd,
        delta_f_gnd,
        q_exc,
        A_exc,
        Ax_exc,
        Ay_exc,
        L_exc,
        upsilon_exc,
        delta_f_exc,
        cyclicity_min,
        cyclicity_weight,
        cyclicity_half,
        cyclicity_softness,
        cyclicity_cap,
    )


# -----------------------------------------------------------------------------
# Expanded optical-dynamics Hamiltonians
# -----------------------------------------------------------------------------


def _project_operator_data(
    operator_data: jax.Array,
    row_states: jax.Array,
    col_states: jax.Array,
) -> jax.Array:
    """Project a dense operator between two state bases.

    Parameters
    ----------
    operator_data : jax.Array, shape (dim, dim)
        Dense operator in the original basis.
    row_states : jax.Array, shape (num_rows, dim)
        Row-basis kets. Output row ``a`` corresponds to ``row_states[a]``.
    col_states : jax.Array, shape (num_cols, dim)
        Column-basis kets. Output column ``b`` corresponds to
        ``col_states[b]``.

    Returns
    -------
    jax.Array, shape (num_rows, num_cols)
        Projected matrix with entries
        ``<row_states[a]|operator|col_states[b]>``.
    """
    return jnp.einsum(
        "ai,ij,bj->ab",
        jnp.conj(row_states),
        operator_data,
        col_states,
    )


def _block_data(
    ground_data: jax.Array,
    excited_data: jax.Array,
    excited_shift: Any,
    subtract_ground_mean: bool,
) -> jax.Array:
    """Build a two-manifold block-diagonal dense matrix.

    Parameters
    ----------
    ground_data : jax.Array, shape (num_ground, num_ground)
        Ground-manifold operator.
    excited_data : jax.Array, shape (num_excited, num_excited)
        Excited-manifold operator.
    excited_shift : scalar
        Scalar identity shift added to the excited block.
    subtract_ground_mean : bool
        If ``True``, subtract the mean diagonal ground energy from both blocks.

    Returns
    -------
    jax.Array, shape (num_ground + num_excited, num_ground + num_excited)
        Matrix ``diag(ground_data, excited_data + excited_shift)`` after the
        optional common energy subtraction.
    """
    ground_dim = int(ground_data.shape[0])
    excited_dim = int(excited_data.shape[0])
    dtype = jnp.result_type(ground_data, excited_data, excited_shift)

    ground_data = ground_data.astype(dtype)
    excited_data = excited_data.astype(dtype)
    eye_ground = jnp.eye(ground_dim, dtype=dtype)
    eye_excited = jnp.eye(excited_dim, dtype=dtype)

    excited_data = excited_data + excited_shift * eye_excited
    if subtract_ground_mean:
        average_ground_energy = jnp.mean(jnp.diag(ground_data))
        ground_data = ground_data - average_ground_energy * eye_ground
        excited_data = excited_data - average_ground_energy * eye_excited

    zero_ge = jnp.zeros((ground_dim, excited_dim), dtype=dtype)
    zero_eg = jnp.zeros((excited_dim, ground_dim), dtype=dtype)
    top = jnp.concatenate([ground_data, zero_ge], axis=1)
    bottom = jnp.concatenate([zero_eg, excited_data], axis=1)
    return jnp.concatenate([top, bottom], axis=0)


def _three_block_data(
    ground_data: jax.Array,
    excited_data: jax.Array,
    dark_energy: Any,
    excited_shift: Any,
    subtract_ground_mean: bool,
) -> jax.Array:
    """Build ground, excited, and dark-state diagonal blocks.

    Parameters
    ----------
    ground_data : jax.Array, shape (num_ground, num_ground)
        Ground-manifold operator.
    excited_data : jax.Array, shape (num_excited, num_excited)
        Excited-manifold operator.
    dark_energy : scalar
        Energy assigned to the appended one-dimensional dark state.
    excited_shift : scalar
        Scalar identity shift added to the excited block.
    subtract_ground_mean : bool
        Subtract the mean ground diagonal energy from the ground and excited
        blocks before appending the dark state.

    Returns
    -------
    jax.Array
        Dense matrix ``diag(H_ground, H_excited + shift, dark_energy)``.
    """
    two_block = _block_data(
        ground_data,
        excited_data,
        excited_shift,
        subtract_ground_mean,
    )
    dtype = jnp.result_type(two_block, dark_energy)
    two_block = two_block.astype(dtype)
    total_dim = int(two_block.shape[0]) + 1
    result = jnp.zeros((total_dim, total_dim), dtype=dtype)
    result = result.at[:-1, :-1].set(two_block)
    result = result.at[-1, -1].set(jnp.asarray(dark_energy, dtype=dtype))
    return result


def _offdiagonal_data(
    lower_left: jax.Array,
    upper_right: jax.Array,
) -> jax.Array:
    """Build a two-manifold matrix containing only off-diagonal blocks.

    Parameters
    ----------
    lower_left : jax.Array, shape (num_excited, num_ground)
        Block mapping the ground sector to the excited sector.
    upper_right : jax.Array, shape (num_ground, num_excited)
        Block mapping the excited sector to the ground sector.

    Returns
    -------
    jax.Array, shape (num_ground + num_excited, num_ground + num_excited)
        Dense matrix ``[[0, upper_right], [lower_left, 0]]``.
    """
    excited_dim = int(lower_left.shape[0])
    ground_dim = int(upper_right.shape[0])
    dtype = jnp.result_type(lower_left, upper_right)

    zero_ground = jnp.zeros((ground_dim, ground_dim), dtype=dtype)
    zero_excited = jnp.zeros((excited_dim, excited_dim), dtype=dtype)
    top = jnp.concatenate(
        [zero_ground, upper_right.astype(dtype)],
        axis=1,
    )
    bottom = jnp.concatenate(
        [lower_left.astype(dtype), zero_excited],
        axis=1,
    )
    return jnp.concatenate([top, bottom], axis=0)


@partial(jax.jit, static_argnames=("included_states",))
def get_dynamic_hamiltonian(
    B: Any,
    theta: Any,
    phi: Any,
    excited_ground_split: Any,
    excited_state_lifetime: Any,
    pump_eta_x: Any,
    pump_eta_y: Any,
    pump_eta_z: Any,
    B_drive_strength: Any,
    B_drive_theta: Any,
    B_drive_phi: Any,
    alpha: Any = 0.0,
    beta: Any = 0.0,
    alpha_exc: Any = 0.0,
    beta_exc: Any = 0.0,
    rg: Any = params.rg_117,
    q_gnd: Any = params.q,
    A_gnd: Optional[Any] = None,
    Ax_gnd: Optional[Any] = None,
    Ay_gnd: Optional[Any] = None,
    L_gnd: Any = params.L,
    upsilon_gnd: Any = 0.0,
    delta_f_gnd: Any = _DEFAULT_DELTA_F_GND,
    q_exc: Any = params.q_exc,
    A_exc: Optional[Any] = None,
    Ax_exc: Optional[Any] = None,
    Ay_exc: Optional[Any] = None,
    L_exc: Any = params.L_exc,
    upsilon_exc: Any = 0.0,
    delta_f_exc: Any = _DEFAULT_DELTA_F_EXC,
    included_states: Optional[Tuple[int, ...]] = None,
    dark_state: Any = 0.0,
) -> Tuple[jqt.Qarray, jqt.Qarray, List[jqt.Qarray], jqt.Qarray]:
    """Build static, microwave, optical, and spontaneous-decay operators.

    Parameters
    ----------
    B : scalar
        Static magnetic-field magnitude in Hamiltonian units.
    theta, phi : scalar
        Polar and azimuthal static-field angles in radians.
    excited_ground_split : scalar
        Common energy offset added to the excited manifold.
    excited_state_lifetime : scalar
        Excited-state lifetime. The total decay rate is its reciprocal.
    pump_eta_x, pump_eta_y, pump_eta_z : scalar
        Complex Cartesian polarization components of the resonant optical
        drive.
    B_drive_strength : scalar
        Microwave-drive field magnitude in Hamiltonian units.
    B_drive_theta, B_drive_phi : scalar
        Polar and azimuthal microwave-drive angles in radians.
    alpha, beta : scalar, optional
        Ground-manifold strain parameters.
    alpha_exc, beta_exc : scalar, optional
        Excited-manifold strain parameters.
    rg : scalar, optional
        Nuclear-to-electron Zeeman scaling ratio.
    q_gnd, q_exc : scalar, optional
        Ground- and excited-manifold orbital susceptibilities.
    A_gnd, Ax_gnd, Ay_gnd : array_like, shape (3, 3), or None, optional
        Ground hyperfine tensors. Passing ``A_gnd=None`` with ``Ax_gnd`` and
        ``Ay_gnd`` also omitted inserts the default ground DJT triplet.
    A_exc, Ax_exc, Ay_exc : array_like, shape (3, 3), or None, optional
        Excited hyperfine tensors. Passing ``A_exc=None`` with ``Ax_exc`` and
        ``Ay_exc`` also omitted inserts the default excited DJT triplet.
    L_gnd, L_exc : scalar, optional
        Ground- and excited-manifold spin-orbit couplings.
    upsilon_gnd, upsilon_exc : scalar, optional
        Ground- and excited-manifold iso-orbital couplings.
    delta_f_gnd, delta_f_exc : scalar, optional
        Ground- and excited-manifold asymmetric-Ham corrections.
    included_states : tuple of int or None, optional
        Matched ground/excited eigenstate indices to retain. ``None`` keeps the
        full Hilbert space. A tuple produces the ordered direct-sum basis
        ``kept ground + kept excited + dark`` and is static under
        :func:`jax.jit`.
    dark_state : scalar, optional
        Energy assigned to the appended dark state in the reduced model.

    Returns
    -------
    H0 : jaxquantum.Qarray
        Static expanded Hamiltonian. In the full model its basis is
        ``manifold x orbital x electron x nuclear``. In the reduced model its
        basis is ``kept ground + kept excited + dark``.
    HB : jaxquantum.Qarray
        Expanded microwave-drive Hamiltonian in the same basis as ``H0``.
    Hs_optical : list of jaxquantum.Qarray
        Two fixed optical operators: excited-to-ground and ground-to-excited
        blocks, respectively.
    c_ops : jaxquantum.Qarray
        Batched spontaneous-emission collapse operators.

    Notes
    -----
    In a reduced model, every decay channel from a retained excited state is
    preserved. Decay to a retained ground state targets that state; decay to an
    omitted ground state is redirected to the single dark state with unchanged
    rate.
    """
    H_gnd_static = build_ground_hamiltonian(
        B,
        theta,
        phi,
        alpha,
        beta,
        rg,
        q_gnd,
        A_gnd,
        Ax_gnd,
        Ay_gnd,
        L_gnd,
        upsilon_gnd,
        delta_f_gnd,
    )
    H_exc_static = build_excited_hamiltonian(
        B,
        theta,
        phi,
        alpha_exc,
        beta_exc,
        rg,
        q_exc,
        A_exc,
        Ax_exc,
        Ay_exc,
        L_exc,
        upsilon_exc,
        delta_f_exc,
    )

    # The drive Hamiltonian contains only terms linear in the independently
    # specified microwave field.
    bx_drive, by_drive, bz_drive = _field_components(
        B_drive_strength,
        B_drive_theta,
        B_drive_phi,
    )
    H_gnd_drive = _H_FIELD(
        bx_drive,
        by_drive,
        bz_drive,
        rg,
        q_gnd,
        delta_f_gnd,
    )
    H_exc_drive = _H_FIELD(
        bx_drive,
        by_drive,
        bz_drive,
        rg,
        q_exc,
        delta_f_exc,
    )

    # Coherent optical operator p.eta and its adjoint. Keeping the polarization
    # components scalar avoids a dynamically shaped polarization argument.
    p_eta = (
        pump_eta_x * _DIPOLES[0]
        + pump_eta_y * _DIPOLES[1]
        + pump_eta_z * _DIPOLES[2]
    )
    p_eta_data = _dense_data(p_eta)
    p_eta_dag_data = jnp.conj(p_eta_data.T)

    # Collapse operators are defined in the static eigenbases so that each
    # excited-state row can be assigned physical branching fractions.
    _, U_gnd_states = jqt.eigenstates(H_gnd_static)
    _, U_exc_states = jqt.eigenstates(H_exc_static)
    gnd_states = _state_matrix(U_gnd_states)
    exc_states = _state_matrix(U_exc_states)

    _, spontaneous_rates = _spontaneous_emission_from_eigenvectors(
        _eigenvector_columns(U_gnd_states),
        _eigenvector_columns(U_exc_states),
    )
    cyclicity = calculate_cyclicity(spontaneous_rates)

    num_ground_states = int(gnd_states.shape[0])
    num_excited_states = int(exc_states.shape[0])
    pair_indices = _validate_included_states(
        included_states,
        num_ground_states,
        num_excited_states,
    )

    orbital_dim = int(round(2.0 * float(params.S) + 1.0))
    electron_dim = orbital_dim
    nuclear_dim = int(round(2.0 * float(params.Sn) + 1.0))
    base_dims = (orbital_dim, electron_dim, nuclear_dim)
    expanded_dims = (2,) + base_dims
    base_dim = orbital_dim * electron_dim * nuclear_dim

    if pair_indices is None:
        # Full direct-product representation. The leading dimension of size two
        # labels the ground and excited electronic manifolds.
        H0_data = _block_data(
            _dense_data(H_gnd_static),
            _dense_data(H_exc_static),
            excited_ground_split,
            False,
        )
        HB_data = _block_data(
            _dense_data(H_gnd_drive),
            _dense_data(H_exc_drive),
            0.0,
            False,
        )
        H0 = jqt.Qarray.create(H0_data, dims=expanded_dims)
        HB = jqt.Qarray.create(HB_data, dims=expanded_dims)

        zero_base = jnp.zeros(
            (base_dim, base_dim),
            dtype=jnp.result_type(p_eta_data),
        )
        Hs_optical = [
            jqt.Qarray.create(
                _offdiagonal_data(zero_base, p_eta_dag_data),
                dims=expanded_dims,
            ),
            jqt.Qarray.create(
                _offdiagonal_data(p_eta_data, zero_base),
                dims=expanded_dims,
            ),
        ]

        total_decay_rate = 1.0 / excited_state_lifetime
        c_ops_list = []
        for excited_index in range(num_excited_states):
            for ground_index in range(num_ground_states):
                # |g_k><e_l| embedded in the upper-right block. The square
                # root converts a rate into a Lindblad jump amplitude.
                jump_base = jnp.outer(
                    gnd_states[ground_index],
                    jnp.conj(exc_states[excited_index]),
                )
                jump_rate = (
                    total_decay_rate
                    * cyclicity[excited_index, ground_index]
                )
                jump_ge = jnp.sqrt(jump_rate) * jump_base
                c_ops_list.append(
                    jqt.Qarray.create(
                        _offdiagonal_data(zero_base, jump_ge),
                        dims=expanded_dims,
                    )
                )
    else:
        # Reduced direct-sum representation in the requested eigenstate order.
        pair_indices_array = jnp.asarray(pair_indices, dtype=jnp.int32)
        kept_gnd_states = gnd_states[pair_indices_array]
        kept_exc_states = exc_states[pair_indices_array]

        reduced_dim = len(pair_indices)
        total_reduced_dim = 2 * reduced_dim + 1
        dark_index = total_reduced_dim - 1
        reduced_dims = (total_reduced_dim,)

        H_gnd_static_reduced = _project_operator_data(
            _dense_data(H_gnd_static),
            kept_gnd_states,
            kept_gnd_states,
        )
        H_exc_static_reduced = _project_operator_data(
            _dense_data(H_exc_static),
            kept_exc_states,
            kept_exc_states,
        )
        H0 = jqt.Qarray.create(
            _three_block_data(
                H_gnd_static_reduced,
                H_exc_static_reduced,
                dark_state,
                excited_ground_split,
                True,
            ),
            dims=reduced_dims,
        )

        H_gnd_drive_reduced = _project_operator_data(
            _dense_data(H_gnd_drive),
            kept_gnd_states,
            kept_gnd_states,
        )
        H_exc_drive_reduced = _project_operator_data(
            _dense_data(H_exc_drive),
            kept_exc_states,
            kept_exc_states,
        )
        HB = jqt.Qarray.create(
            _three_block_data(
                H_gnd_drive_reduced,
                H_exc_drive_reduced,
                0.0,
                0.0,
                False,
            ),
            dims=reduced_dims,
        )

        # Project p.eta between the retained ground and excited eigenstates,
        # then append a zero row and column for the dark state.
        p_ge_reduced = _project_operator_data(
            p_eta_data,
            kept_exc_states,
            kept_gnd_states,
        )
        p_eg_reduced = _project_operator_data(
            p_eta_dag_data,
            kept_gnd_states,
            kept_exc_states,
        )
        zero_reduced = jnp.zeros(
            (reduced_dim, reduced_dim),
            dtype=jnp.result_type(p_ge_reduced, p_eg_reduced),
        )
        optical_lowering = jnp.pad(
            _offdiagonal_data(zero_reduced, p_eg_reduced),
            ((0, 1), (0, 1)),
        )
        optical_raising = jnp.pad(
            _offdiagonal_data(p_ge_reduced, zero_reduced),
            ((0, 1), (0, 1)),
        )
        Hs_optical = [
            jqt.Qarray.create(optical_lowering, dims=reduced_dims),
            jqt.Qarray.create(optical_raising, dims=reduced_dims),
        ]

        total_decay_rate = 1.0 / excited_state_lifetime
        c_ops_list = []
        zero_reduced_full = jnp.zeros(
            (total_reduced_dim, total_reduced_dim),
            dtype=cyclicity.dtype,
        )

        # This mapping is static Python data because ``included_states`` is a
        # static JIT argument and determines the compiled output structure.
        kept_ground_positions = {
            original_index: reduced_index
            for reduced_index, original_index in enumerate(pair_indices)
        }

        for excited_reduced, excited_full in enumerate(pair_indices):
            source_index = reduced_dim + excited_reduced
            for ground_full in range(num_ground_states):
                target_index = kept_ground_positions.get(
                    ground_full,
                    dark_index,
                )
                jump_rate = (
                    total_decay_rate
                    * cyclicity[excited_full, ground_full]
                )
                jump_operator = zero_reduced_full.at[
                    target_index,
                    source_index,
                ].set(jnp.sqrt(jump_rate))
                c_ops_list.append(
                    jqt.Qarray.create(jump_operator, dims=reduced_dims)
                )

    c_ops = jqt.Qarray.from_list(c_ops_list)
    return H0, HB, Hs_optical, c_ops


def _project_single_manifold(
    H0: jqt.Qarray,
    HB: jqt.Qarray,
    included_states: Optional[Tuple[int, ...]],
) -> Tuple[jqt.Qarray, jqt.Qarray]:
    """Project static and drive operators into selected static eigenstates.

    Parameters
    ----------
    H0 : jaxquantum.Qarray
        Full static single-manifold Hamiltonian.
    HB : jaxquantum.Qarray
        Full single-manifold microwave-drive Hamiltonian.
    included_states : tuple of int or None
        Static-Hamiltonian eigenstate indices to retain. ``None`` returns the
        input operators unchanged.

    Returns
    -------
    H0_projected : jaxquantum.Qarray
        Full or reduced static Hamiltonian. In the reduced case, the mean
        retained static energy is subtracted.
    HB_projected : jaxquantum.Qarray
        Full or reduced drive Hamiltonian. No energy shift is applied to the
        drive operator.
    """
    if included_states is None:
        return H0, HB

    num_states = int(_dense_data(H0).shape[0])
    indices = _validate_included_states(
        included_states,
        num_states,
        num_states,
    )
    assert indices is not None

    _, eigenstates = jqt.eigenstates(H0)
    state_matrix = _state_matrix(eigenstates)
    state_indices = jnp.asarray(indices, dtype=jnp.int32)
    kept_states = state_matrix[state_indices]

    H0_reduced = _project_operator_data(
        _dense_data(H0),
        kept_states,
        kept_states,
    )
    HB_reduced = _project_operator_data(
        _dense_data(HB),
        kept_states,
        kept_states,
    )

    reduced_dim = len(indices)
    reduced_dtype = jnp.result_type(H0_reduced, HB_reduced)
    identity_reduced = jnp.eye(reduced_dim, dtype=reduced_dtype)
    H0_reduced = H0_reduced.astype(reduced_dtype)
    HB_reduced = HB_reduced.astype(reduced_dtype)

    # Center the retained static energies, matching the reduced ground block of
    # ``get_dynamic_hamiltonian``.
    average_energy = jnp.mean(jnp.diag(H0_reduced))
    H0_reduced = H0_reduced - average_energy * identity_reduced

    return (
        jqt.Qarray.create(H0_reduced, dims=(reduced_dim,)),
        jqt.Qarray.create(HB_reduced, dims=(reduced_dim,)),
    )


@partial(jax.jit, static_argnames=("included_states",))
def get_ground_hamiltonian(
    B: Any,
    theta: Any,
    phi: Any,
    B_drive_strength: Any,
    B_drive_theta: Any,
    B_drive_phi: Any,
    alpha: Any = 0.0,
    beta: Any = 0.0,
    rg: Any = params.rg_117,
    q: Any = params.q,
    A: Optional[Any] = None,
    Ax: Optional[Any] = None,
    Ay: Optional[Any] = None,
    L: Any = params.L,
    upsilon: Any = 0.0,
    delta_f: Any = _DEFAULT_DELTA_F_GND,
    included_states: Optional[Tuple[int, ...]] = None,
) -> Tuple[jqt.Qarray, jqt.Qarray]:
    """Build static and microwave-drive ground-manifold Hamiltonians.

    Parameters
    ----------
    B : scalar
        Static magnetic-field magnitude in Hamiltonian units.
    theta, phi : scalar
        Polar and azimuthal static-field angles in radians.
    B_drive_strength : scalar
        Microwave-drive field magnitude in Hamiltonian units.
    B_drive_theta, B_drive_phi : scalar
        Polar and azimuthal microwave-drive angles in radians.
    alpha, beta : scalar, optional
        Ground-manifold strain parameters.
    rg : scalar, optional
        Nuclear-to-electron Zeeman scaling ratio.
    q : scalar, optional
        Ground orbital magnetic-field susceptibility.
    A, Ax, Ay : array_like, shape (3, 3), or None, optional
        Ground hyperfine tensors. Passing ``A=None`` inserts the default ground
        DJT tensor triplet.
    L : scalar, optional
        Ground spin-orbit coupling.
    upsilon : scalar, optional
        Ground iso-orbital coupling.
    delta_f : scalar, optional
        Ground asymmetric-Ham correction.
    included_states : tuple of int or None, optional
        Ground eigenstate indices to retain. The tuple is static under
        :func:`jax.jit`. ``None`` returns full-space operators.

    Returns
    -------
    H0 : jaxquantum.Qarray
        Full or projected static ground Hamiltonian.
    HB : jaxquantum.Qarray
        Full or projected microwave-drive Hamiltonian.
    """
    H0 = build_ground_hamiltonian(
        B,
        theta,
        phi,
        alpha,
        beta,
        rg,
        q,
        A,
        Ax,
        Ay,
        L,
        upsilon,
        delta_f,
    )
    bx_drive, by_drive, bz_drive = _field_components(
        B_drive_strength,
        B_drive_theta,
        B_drive_phi,
    )
    HB = _H_FIELD(bx_drive, by_drive, bz_drive, rg, q, delta_f)
    return _project_single_manifold(H0, HB, included_states)


@partial(jax.jit, static_argnames=("included_states",))
def get_excited_hamiltonian(
    B: Any,
    theta: Any,
    phi: Any,
    B_drive_strength: Any,
    B_drive_theta: Any,
    B_drive_phi: Any,
    alpha: Any = 0.0,
    beta: Any = 0.0,
    rg: Any = params.rg_117,
    q: Any = params.q_exc,
    A: Optional[Any] = None,
    Ax: Optional[Any] = None,
    Ay: Optional[Any] = None,
    L: Any = params.L_exc,
    upsilon: Any = 0.0,
    delta_f: Any = _DEFAULT_DELTA_F_EXC,
    included_states: Optional[Tuple[int, ...]] = None,
) -> Tuple[jqt.Qarray, jqt.Qarray]:
    """Build static and microwave-drive excited-manifold Hamiltonians.

    Parameters
    ----------
    B : scalar
        Static magnetic-field magnitude in Hamiltonian units.
    theta, phi : scalar
        Polar and azimuthal static-field angles in radians.
    B_drive_strength : scalar
        Microwave-drive field magnitude in Hamiltonian units.
    B_drive_theta, B_drive_phi : scalar
        Polar and azimuthal microwave-drive angles in radians.
    alpha, beta : scalar, optional
        Excited-manifold strain parameters.
    rg : scalar, optional
        Nuclear-to-electron Zeeman scaling ratio.
    q : scalar, optional
        Excited orbital magnetic-field susceptibility.
    A, Ax, Ay : array_like, shape (3, 3), or None, optional
        Excited hyperfine tensors. Passing ``A=None`` inserts the default
        excited DJT tensor triplet.
    L : scalar, optional
        Excited spin-orbit coupling.
    upsilon : scalar, optional
        Excited iso-orbital coupling.
    delta_f : scalar, optional
        Excited asymmetric-Ham correction.
    included_states : tuple of int or None, optional
        Excited eigenstate indices to retain. The tuple is static under
        :func:`jax.jit`. ``None`` returns full-space operators.

    Returns
    -------
    H0 : jaxquantum.Qarray
        Full or projected static excited Hamiltonian.
    HB : jaxquantum.Qarray
        Full or projected microwave-drive Hamiltonian.
    """
    H0 = build_excited_hamiltonian(
        B,
        theta,
        phi,
        alpha,
        beta,
        rg,
        q,
        A,
        Ax,
        Ay,
        L,
        upsilon,
        delta_f,
    )
    bx_drive, by_drive, bz_drive = _field_components(
        B_drive_strength,
        B_drive_theta,
        B_drive_phi,
    )
    HB = _H_FIELD(bx_drive, by_drive, bz_drive, rg, q, delta_f)
    return _project_single_manifold(H0, HB, included_states)
