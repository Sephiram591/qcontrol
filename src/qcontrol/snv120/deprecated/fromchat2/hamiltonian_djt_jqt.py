"""JAXQuantum implementation of the SnV117 dynamic-Jahn--Teller model.

The Hamiltonian in this module follows the physics conventions of
``hamiltonian_DJT(1).py`` while expressing the hyperfine interaction with the
three Cartesian tensors ``A``, ``Ax``, and ``Ay`` used by
``hamiltonian_jqt(2).py``.

The single-manifold Hilbert-space order is

``orbital x electron x nuclear``.

The magnetic-field components supplied to the Hamiltonian are in the original
frequency-like code units, where ``bz = gS * mu_B * B_z``. All coupling
parameters are in GHz.

Notes
-----
JAX 64-bit mode is enabled before importing :mod:`jax.numpy`. This is important
because the model combines THz-scale spin-orbit energies with sub-MHz dynamic
hyperfine terms.

The preferred hyperfine interface is the tensor triplet ``A``, ``Ax``, and
``Ay``. The legacy scalar parameters ``Aperp``, ``Apar``, ``A1``, and ``A2``
remain available and are converted to an algebraically equivalent tensor
triplet by :func:`djt_hyperfine_tensors`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from jax import config

# Configure precision before importing jax.numpy.
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jaxquantum as jqt

import parameters_DJT as params


__all__ = [
    "djt_hyperfine_tensors",
    "create_hamiltonian_nuclear",
    "solve_hamiltonian",
    "calculate_cyclicity",
    "calculate_cyclicity_spinflip",
    "PLE_transitions",
    "PLE_spectrum",
]


# -----------------------------------------------------------------------------
# Dense-array and angular-momentum helpers
# -----------------------------------------------------------------------------


def _spin_operator(j: float, axis: str) -> jqt.Qarray:
    """Construct one angular-momentum operator.

    Parameters
    ----------
    j : float
        Spin quantum number. It must be a non-negative integer or
        half-integer.
    axis : {'x', 'y', 'z'}
        Cartesian component to construct.

    Returns
    -------
    jaxquantum.Qarray
        Dense angular-momentum operator in the basis
        ``|j, j>, |j, j-1>, ..., |j, -j>``.

    Raises
    ------
    ValueError
        If ``j`` is not a non-negative integer or half-integer, or if ``axis``
        is invalid.
    """
    j_float = float(j)
    two_j = int(round(2.0 * j_float))

    if two_j < 0 or abs(2.0 * j_float - two_j) > 1e-12:
        raise ValueError(
            "j must be a non-negative integer or half-integer; "
            f"received {j!r}."
        )

    dim = two_j + 1
    j_plus = jnp.zeros((dim, dim), dtype=jnp.complex128)

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


def _dense_data(operator: jqt.Qarray) -> jnp.ndarray:
    """Return dense array data from a JAXQuantum object.

    Parameters
    ----------
    operator : jaxquantum.Qarray
        Operator or state to convert.

    Returns
    -------
    jax.Array
        Dense array stored by ``operator``.
    """
    return operator.to_dense().data


def _state_matrix(eigenstates: jqt.Qarray) -> jnp.ndarray:
    """Convert JAXQuantum eigenstate kets to a row-oriented state matrix.

    Parameters
    ----------
    eigenstates : jaxquantum.Qarray
        Batched kets returned by :func:`jaxquantum.eigenstates`. Dense data is
        expected to have shape ``(num_states, dim, 1)``.

    Returns
    -------
    jax.Array
        State matrix with shape ``(num_states, dim)``. Row ``s`` contains
        eigenstate ``|psi_s>``.
    """
    return _dense_data(eigenstates)[..., :, 0]


def _eigenvector_columns(eigenstates: jqt.Qarray) -> jnp.ndarray:
    """Convert JAXQuantum eigenstates to the original column convention.

    Parameters
    ----------
    eigenstates : jaxquantum.Qarray
        Batched ket eigenstates returned by
        :func:`jaxquantum.eigenstates`.

    Returns
    -------
    jax.Array
        Matrix with shape ``(dim, num_states)``. Column ``s`` contains
        eigenstate ``|psi_s>``.
    """
    return jnp.swapaxes(_state_matrix(eigenstates), -1, -2)


def _expect_batched(
    operator: jqt.Qarray,
    eigenstates: jqt.Qarray,
) -> jnp.ndarray:
    """Evaluate one expectation value for every eigenstate.

    Parameters
    ----------
    operator : jaxquantum.Qarray
        Operator with shape ``(dim, dim)``.
    eigenstates : jaxquantum.Qarray
        Batched ket eigenstates with dense shape
        ``(num_states, dim, 1)``.

    Returns
    -------
    jax.Array
        Real expectation values with shape ``(num_states,)``.
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
    eigenvectors: jnp.ndarray,
    operator: jqt.Qarray,
) -> jnp.ndarray:
    """Evaluate an operator for column eigenvectors over a field sweep.

    Parameters
    ----------
    eigenvectors : jax.Array, shape (num_fields, dim, num_states)
        Eigenvectors arranged as columns.
    operator : jaxquantum.Qarray
        Operator with shape ``(dim, dim)``.

    Returns
    -------
    jax.Array, shape (num_fields, num_states)
        Expectation values for all field points and eigenstates.
    """
    return jnp.einsum(
        "bdi,de,bei->bi",
        jnp.conj(eigenvectors),
        _dense_data(operator),
        eigenvectors,
    )


def _dipole_matrix_elements(
    ground_eigenvectors: jnp.ndarray,
    excited_eigenvectors: jnp.ndarray,
    dipoles: Sequence[jqt.Qarray],
) -> jnp.ndarray:
    """Evaluate all ground-to-excited dipole matrix elements.

    Parameters
    ----------
    ground_eigenvectors : jax.Array, shape (num_fields, dim, num_ground)
        Ground-manifold eigenvectors arranged as columns.
    excited_eigenvectors : jax.Array, shape (num_fields, dim, num_excited)
        Excited-manifold eigenvectors arranged as columns.
    dipoles : sequence of jaxquantum.Qarray
        Dipole operators ``[p_x, p_y, p_z]``.

    Returns
    -------
    jax.Array, shape (num_fields, 3, num_excited, num_ground)
        Matrix elements ``<exc_l|p_j|gnd_k>``.
    """
    dipole_data = jnp.stack(
        [_dense_data(operator) for operator in dipoles],
        axis=0,
    )
    return jnp.einsum(
        "bdl,jde,bek->bjlk",
        jnp.conj(excited_eigenvectors),
        dipole_data,
        ground_eigenvectors,
    )


def _reduced_spin_density_matrices(
    eigenvectors: jnp.ndarray,
    orbital_dim: int,
    electron_dim: int,
    nuclear_dim: int,
) -> jnp.ndarray:
    """Trace the orbital subsystem from pure eigenstate density matrices.

    Parameters
    ----------
    eigenvectors : jax.Array, shape (num_fields, dim, num_states)
        Eigenvectors arranged as columns.
    orbital_dim : int
        Orbital Hilbert-space dimension.
    electron_dim : int
        Electron-spin Hilbert-space dimension.
    nuclear_dim : int
        Nuclear-spin Hilbert-space dimension.

    Returns
    -------
    jax.Array
        Reduced electron-nuclear density matrices with shape
        ``(num_fields, num_states, spin_dim, spin_dim)``, where
        ``spin_dim = electron_dim * nuclear_dim``.

    Raises
    ------
    ValueError
        If the dense state dimension is inconsistent with the supplied
        subsystem dimensions.
    """
    num_fields, dim, num_states = eigenvectors.shape
    spin_dim = electron_dim * nuclear_dim

    if dim != orbital_dim * spin_dim:
        raise ValueError(
            "The eigenvector dimension is inconsistent with the orbital, "
            "electron, and nuclear subsystem dimensions."
        )

    # psi[b, state, orbital, combined_spin]
    psi = jnp.swapaxes(eigenvectors, 1, 2).reshape(
        num_fields,
        num_states,
        orbital_dim,
        spin_dim,
    )

    # rho_spin[s, t] = sum_o psi[o, s] psi*[o, t].
    return jnp.einsum("bkos,bkot->bkst", psi, jnp.conj(psi))


# -----------------------------------------------------------------------------
# Hyperfine tensor conversion
# -----------------------------------------------------------------------------


def djt_hyperfine_tensors(
    Aperp: Any,
    Apar: Any,
    A1: Any,
    A2: Any,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""Convert the original DJT scalar couplings to Cartesian tensors.

    The returned tensors reproduce the original raising/lowering-operator
    hyperfine Hamiltonian exactly when used as

    .. math::

        H_\mathrm{hf} =
        \sum_{ij} A_{ij}\,\mathbb{I}_\mathrm{orb}\otimes S_i\otimes I_j
        + \sum_{ij} (A_x)_{ij}(-2X_\mathrm{orb})\otimes S_i\otimes I_j
        + \sum_{ij} (A_y)_{ij}(-2Y_\mathrm{orb})\otimes S_i\otimes I_j.

    Parameters
    ----------
    Aperp : scalar
        Perpendicular DJT hyperfine coupling.
    Apar : scalar
        Parallel DJT hyperfine coupling. The original convention contributes
        ``2 * Apar * Sz * Iz``.
    A1 : scalar
        First orbital-off-diagonal DJT hyperfine coupling.
    A2 : scalar
        Second orbital-off-diagonal DJT hyperfine coupling.

    Returns
    -------
    A : jax.Array, shape (3, 3)
        Orbital-independent hyperfine tensor.
    Ax : jax.Array, shape (3, 3)
        Tensor multiplying ``-2 * X_orbital``.
    Ay : jax.Array, shape (3, 3)
        Tensor multiplying ``-2 * Y_orbital``.

    Notes
    -----
    The conversion is

    ``A = diag(Aperp, Aperp, 2*Apar)``

    with the nonzero dynamic terms

    ``Ax_xx=-A2/2``, ``Ax_yy=A2/2``, ``Ax_xz=Ax_zx=A1``

    and

    ``Ay_xy=Ay_yx=-A2/2``, ``Ay_yz=Ay_zy=-A1``.
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


def _as_hyperfine_tensor(value: Any, name: str) -> jnp.ndarray:
    """Convert and validate one explicit hyperfine tensor.

    Parameters
    ----------
    value : array_like
        Tensor candidate.
    name : str
        Parameter name used in an error message.

    Returns
    -------
    jax.Array, shape (3, 3)
        Complex-valued hyperfine tensor.

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
    *,
    A: Any,
    Ax: Any,
    Ay: Any,
    Aperp: Any,
    Apar: Any,
    A1: Any,
    A2: Any,
    default_Aperp: Any,
    default_Apar: Any,
    default_A1: Any,
    default_A2: Any,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Resolve explicit tensors or backward-compatible DJT scalar inputs.

    Parameters
    ----------
    A, Ax, Ay : array_like or None
        Explicit hyperfine tensors. Supplying any one selects tensor mode;
        omitted tensors are replaced by zero matrices.
    Aperp, Apar, A1, A2 : scalar or None
        Legacy DJT scalar overrides.
    default_Aperp, default_Apar, default_A1, default_A2 : scalar
        Manifold-specific scalar defaults.

    Returns
    -------
    A, Ax, Ay : tuple of jax.Array
        Resolved hyperfine tensor triplet.

    Raises
    ------
    ValueError
        If explicit tensors and legacy scalar overrides are mixed.
    """
    tensor_mode = any(value is not None for value in (A, Ax, Ay))
    scalar_override_mode = any(
        value is not None for value in (Aperp, Apar, A1, A2)
    )

    if tensor_mode and scalar_override_mode:
        raise ValueError(
            "Use either hyperfine tensors (A, Ax, Ay) or legacy DJT "
            "scalars (Aperp, Apar, A1, A2), not both."
        )

    if tensor_mode:
        zero = jnp.zeros((3, 3), dtype=jnp.complex128)
        return (
            _as_hyperfine_tensor(zero if A is None else A, "A"),
            _as_hyperfine_tensor(zero if Ax is None else Ax, "Ax"),
            _as_hyperfine_tensor(zero if Ay is None else Ay, "Ay"),
        )

    return djt_hyperfine_tensors(
        default_Aperp if Aperp is None else Aperp,
        default_Apar if Apar is None else Apar,
        default_A1 if A1 is None else A1,
        default_A2 if A2 is None else A2,
    )


def _resolve_manifold(manifold: str) -> Dict[str, float]:
    """Return manifold-specific defaults from :mod:`parameters_DJT`.

    Parameters
    ----------
    manifold : {'ground', 'excited'}
        Electronic manifold to resolve.

    Returns
    -------
    dict
        Dictionary containing ``rg``, ``q``, ``L``, ``Aperp``, ``Apar``,
        ``A1``, ``A2``, and ``delta_f``.

    Raises
    ------
    ValueError
        If ``manifold`` is invalid.
    """
    if manifold == "ground":
        return {
            "rg": params.rg,
            "q": params.q,
            "L": params.L,
            "Aperp": params.Aperp_gnd,
            "Apar": params.Apar_gnd,
            "A1": params.A1_gnd,
            "A2": params.A2_gnd,
            "delta_f": params.delta_f_gnd,
        }

    if manifold == "excited":
        return {
            "rg": params.rg,
            "q": params.q_exc,
            "L": params.L_exc,
            "Aperp": params.Aperp_exc,
            "Apar": params.Apar_exc,
            "A1": params.A1_exc,
            "A2": params.A2_exc,
            "delta_f": params.delta_f_exc,
        }

    raise ValueError(
        "manifold must be 'ground' or 'excited'; "
        f"received {manifold!r}."
    )


# -----------------------------------------------------------------------------
# Hamiltonian construction
# -----------------------------------------------------------------------------


def create_hamiltonian_nuclear(
    manifold: str = "ground",
) -> Tuple[Any, Any, List[jqt.Qarray], jqt.Qarray]:
    """Create the SnV117 single-manifold Hamiltonian builders.

    The returned full Hamiltonian follows ``hamiltonian_DJT(1).py`` for every
    non-hyperfine term. The hyperfine interaction is evaluated using the
    ``A/Ax/Ay`` Cartesian tensor contraction from ``hamiltonian_jqt(2).py``.

    Parameters
    ----------
    manifold : {'ground', 'excited'}, optional
        Manifold whose physical parameters are used as defaults.

    Returns
    -------
    H : callable
        Full Hamiltonian builder. ``H(bx, by, bz, ...)`` returns one
        :class:`jaxquantum.Qarray` operator.
    Href : callable
        Reference Hamiltonian builder containing spin-orbit coupling and
        strain only.
    p : list of jaxquantum.Qarray
        Dipole operators ``[p_x, p_y, p_z]``.
    J2 : jaxquantum.Qarray
        Total electron-plus-nuclear angular-momentum-squared operator.

    Notes
    -----
    The tensor-product order is ``orbital x electron x nuclear``.
    """
    defaults = _resolve_manifold(manifold)

    default_rg = defaults["rg"]
    default_q = defaults["q"]
    default_L = defaults["L"]
    default_Aperp = defaults["Aperp"]
    default_Apar = defaults["Apar"]
    default_A1 = defaults["A1"]
    default_A2 = defaults["A2"]
    default_delta_f = defaults["delta_f"]

    electron_spin = params.S
    nuclear_spin = params.Sn

    # Orbital and electron spaces are both two-dimensional for S = 1/2.
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

    # Reusable full-space operators.
    Sx = tensor(I, X, In)
    Sy = tensor(I, Y, In)
    Sz = tensor(I, Z, In)

    Ix = tensor(I, I, Xn)
    Iy = tensor(I, I, Yn)
    Iz = tensor(I, I, Zn)

    Lx = tensor(X, I, In)
    Ly = tensor(Y, I, In)
    Lz = tensor(Z, I, In)

    LzSz = tensor(Z, Z, In)
    LzIz = tensor(Z, I, Zn)

    # J^2 = S(S+1) + I(I+1) + 2 S dot I.
    J2 = (
        (electron_spin * (electron_spin + 1.0)
         + nuclear_spin * (nuclear_spin + 1.0))
        * identity_full
        + 2.0
        * (
            tensor(I, X, Xn)
            + tensor(I, Y, Yn)
            + tensor(I, Z, Zn)
        )
    )

    # Hyperfine tensor blocks. The orbital prefactors match hamiltonian_jqt:
    # A -> I_orb, Ax -> -2 X_orb, Ay -> -2 Y_orb.
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
        coupling_tensor: jnp.ndarray,
        block_index: int,
    ) -> jqt.Qarray:
        """Contract one 3-by-3 tensor with one orbital hyperfine block.

        Parameters
        ----------
        coupling_tensor : jax.Array, shape (3, 3)
            Cartesian electron-nuclear coupling tensor.
        block_index : int
            Orbital-prefactor block: ``0`` for ``A``, ``1`` for ``Ax``, and
            ``2`` for ``Ay``.

        Returns
        -------
        jaxquantum.Qarray
            Contracted hyperfine operator.
        """
        result = 0.0 * identity_full
        operator_block = hyperfine_operator_blocks[block_index]

        for i in range(3):
            for j in range(3):
                result = result + coupling_tensor[i, j] * operator_block[i][j]

        return result

    def Hhf(
        A: jnp.ndarray,
        Ax: jnp.ndarray,
        Ay: jnp.ndarray,
    ) -> jqt.Qarray:
        """Build the full Cartesian-tensor hyperfine Hamiltonian.

        Parameters
        ----------
        A, Ax, Ay : jax.Array, shape (3, 3)
            Orbital-independent and orbital-modulated hyperfine tensors.

        Returns
        -------
        jaxquantum.Qarray
            Hyperfine Hamiltonian.
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
            Spin-orbit coupling.

        Returns
        -------
        jaxquantum.Qarray
            Operator ``-2 * L * Lz * Sz``.
        """
        return -2.0 * L * LzSz

    def Hioc(upsilon: Any) -> jqt.Qarray:
        """Build the iso-orbital interaction.

        Parameters
        ----------
        upsilon : scalar
            Iso-orbital coupling.

        Returns
        -------
        jaxquantum.Qarray
            Operator ``2 * upsilon * Lz * Iz``.
        """
        return 2.0 * upsilon * LzIz

    def Hegx(alpha: Any) -> jqt.Qarray:
        """Build the x-like strain interaction.

        Parameters
        ----------
        alpha : scalar
            X-like strain/Jahn--Teller parameter.

        Returns
        -------
        jaxquantum.Qarray
            Operator ``-2 * alpha * Lx``.
        """
        return -2.0 * alpha * Lx

    def Hegy(beta: Any) -> jqt.Qarray:
        """Build the y-like strain interaction.

        Parameters
        ----------
        beta : scalar
            Y-like strain/Jahn--Teller parameter.

        Returns
        -------
        jaxquantum.Qarray
            Operator ``2 * beta * Ly``.
        """
        return 2.0 * beta * Ly

    # Dipole operators in the e+/e- basis.
    p = [2.0 * Lx, 2.0 * Ly, 2.0 * identity_full]

    def Href(
        alpha: Any = 0.0,
        beta: Any = 0.0,
        L: Any = default_L,
    ) -> jqt.Qarray:
        """Build the spin-orbit-plus-strain reference Hamiltonian.

        Parameters
        ----------
        alpha, beta : scalar, optional
            Strain/Jahn--Teller parameters.
        L : scalar, optional
            Spin-orbit coupling.

        Returns
        -------
        jaxquantum.Qarray
            Reference Hamiltonian.
        """
        return Hsoc(L) + Hegx(alpha) + Hegy(beta)

    def H(
        bx: Any,
        by: Any,
        bz: Any,
        alpha: Any = 0.0,
        beta: Any = 0.0,
        rg: Any = default_rg,
        q: Any = default_q,
        Aperp: Any = None,
        Apar: Any = None,
        L: Any = default_L,
        upsilon: Any = 0.0,
        A1: Any = None,
        A2: Any = None,
        delta_f: Any = default_delta_f,
        A: Any = None,
        Ax: Any = None,
        Ay: Any = None,
    ) -> jqt.Qarray:
        """Evaluate the complete single-manifold Hamiltonian.

        Parameters
        ----------
        bx, by, bz : scalar
            Magnetic-field components in the original frequency-like code
            units.
        alpha, beta : scalar, optional
            Strain/Jahn--Teller parameters.
        rg : scalar, optional
            Nuclear-to-electron Zeeman ratio.
        q : scalar, optional
            Orbital magnetic-field susceptibility.
        Aperp, Apar, A1, A2 : scalar or None, optional
            Legacy DJT hyperfine parameters. They are converted to ``A``,
            ``Ax``, and ``Ay``. Do not mix these arguments with explicit
            tensors.
        L : scalar, optional
            Spin-orbit coupling.
        upsilon : scalar, optional
            Iso-orbital coupling.
        delta_f : scalar, optional
            Asymmetric-Ham electron-Zeeman correction.
        A, Ax, Ay : array_like, shape (3, 3), optional
            Preferred hyperfine tensors. Supplying any tensor selects tensor
            mode; omitted tensors are replaced with zeros.

        Returns
        -------
        jaxquantum.Qarray
            Total Hamiltonian operator.

        Raises
        ------
        ValueError
            If tensor and legacy scalar hyperfine inputs are mixed, or if an
            explicit tensor has the wrong shape.
        """
        A_tensor, Ax_tensor, Ay_tensor = _resolve_hyperfine_tensors(
            A=A,
            Ax=Ax,
            Ay=Ay,
            Aperp=Aperp,
            Apar=Apar,
            A1=A1,
            A2=A2,
            default_Aperp=default_Aperp,
            default_Apar=default_Apar,
            default_A1=default_A1,
            default_A2=default_A2,
        )

        electron_zeeman = bx * Sx + by * Sy + bz * Sz
        nuclear_zeeman = rg * (bx * Ix + by * Iy + bz * Iz)
        orbital_zeeman = (2.0 * q / params.gS) * bz * Lz
        asymmetric_ham = (2.0 * delta_f / params.gS) * bz * Sz

        return (
            Href(alpha=alpha, beta=beta, L=L)
            + electron_zeeman
            + nuclear_zeeman
            + orbital_zeeman
            + asymmetric_ham
            + Hhf(A_tensor, Ax_tensor, Ay_tensor)
            + Hioc(upsilon)
        )

    return H, Href, p, J2


# -----------------------------------------------------------------------------
# Eigensystem solvers
# -----------------------------------------------------------------------------


def _solve_hamiltonian_core(
    B: Any,
    theta: Any,
    phi: Any,
    manifold: str = "ground",
    alpha: Any = 0.0,
    beta: Any = 0.0,
    **kwargs: Any,
) -> Tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    List[jqt.Qarray],
    jnp.ndarray,
    List[jqt.Qarray],
]:
    """Solve a magnetic-field sweep and retain batched JAXQuantum states.

    Parameters
    ----------
    B : scalar or array_like
        Magnetic-field magnitude or one-dimensional sweep.
    theta, phi : scalar
        Polar and azimuthal field angles in radians.
    manifold : {'ground', 'excited'}, optional
        Electronic manifold.
    alpha, beta : scalar, optional
        Strain parameters.
    **kwargs
        Overrides forwarded to the full Hamiltonian builder.

    Returns
    -------
    E : jax.Array, shape (num_fields, num_states)
        Eigenvalues.
    Eref : jax.Array, shape (num_states,)
        Reference-Hamiltonian eigenvalues.
    U : jax.Array, shape (num_fields, dim, num_states)
        Eigenvectors arranged as columns.
    state_batches : list of jaxquantum.Qarray
        One batched eigenstate object per field point.
    alignment : jax.Array, shape (num_fields, num_states)
        ``J2`` expectation values.
    p : list of jaxquantum.Qarray
        Dipole operators.
    """
    H, Href, p, J2 = create_hamiltonian_nuclear(manifold)

    B_values = jnp.ravel(jnp.atleast_1d(jnp.asarray(B, dtype=jnp.float64)))
    theta = jnp.asarray(theta, dtype=jnp.float64)
    phi = jnp.asarray(phi, dtype=jnp.float64)

    bz_values = B_values * jnp.cos(theta)
    bx_values = B_values * jnp.sin(theta) * jnp.cos(phi)
    by_values = B_values * jnp.sin(theta) * jnp.sin(phi)

    energies = []
    eigenvector_matrices = []
    state_batches: List[jqt.Qarray] = []
    alignments = []

    for field_index in range(int(B_values.shape[0])):
        hamiltonian = H(
            bx_values[field_index],
            by_values[field_index],
            bz_values[field_index],
            alpha=alpha,
            beta=beta,
            **kwargs,
        )
        eigenvalues, eigenstates = jqt.eigenstates(hamiltonian)

        energies.append(eigenvalues)
        eigenvector_matrices.append(_eigenvector_columns(eigenstates))
        state_batches.append(eigenstates)
        alignments.append(_expect_batched(J2, eigenstates))

    E = jnp.stack(energies, axis=0)
    U = jnp.stack(eigenvector_matrices, axis=0)
    alignment = jnp.stack(alignments, axis=0)

    reference_kwargs = {
        key: kwargs[key]
        for key in ("L",)
        if key in kwargs
    }
    Eref, _ = jqt.eigenstates(
        Href(alpha=alpha, beta=beta, **reference_kwargs)
    )

    return E, Eref, U, state_batches, alignment, p


def solve_hamiltonian(
    B: Any,
    theta: Any,
    phi: Any,
    manifold: str = "ground",
    alpha: Any = 0.0,
    beta: Any = 0.0,
    **kwargs: Any,
):
    """Diagonalize the Hamiltonian over a magnetic-field sweep.

    Parameters
    ----------
    B : scalar or array_like
        Magnetic-field magnitude or one-dimensional sweep.
    theta : scalar
        Polar field angle in radians.
    phi : scalar
        Azimuthal field angle in radians.
    manifold : {'ground', 'excited'}, optional
        Electronic manifold.
    alpha, beta : scalar, optional
        Strain parameters.
    **kwargs
        Hamiltonian overrides. Hyperfine tensors may be supplied as ``A``,
        ``Ax``, and ``Ay``.

    Returns
    -------
    E : jax.Array, shape (num_fields, num_states)
        Eigenvalues.
    Eref : jax.Array, shape (num_states,)
        Reference-Hamiltonian eigenvalues.
    U : jax.Array, shape (num_fields, dim, num_states)
        Eigenvectors arranged as columns.
    U_states : list of list of jaxquantum.Qarray
        ``U_states[b][s]`` is state ``s`` at field index ``b``.
    alignment : jax.Array, shape (num_fields, num_states)
        ``J2`` expectation values.
    p : list of jaxquantum.Qarray
        Dipole operators ``[p_x, p_y, p_z]``.
    """
    E, Eref, U, state_batches, alignment, p = _solve_hamiltonian_core(
        B,
        theta,
        phi,
        manifold=manifold,
        alpha=alpha,
        beta=beta,
        **kwargs,
    )

    num_states = int(E.shape[1])
    U_states = [
        [state_batch[state_index] for state_index in range(num_states)]
        for state_batch in state_batches
    ]

    return E, Eref, U, U_states, alignment, p


# -----------------------------------------------------------------------------
# Optical rates and cyclicity
# -----------------------------------------------------------------------------


def calculate_cyclicity(transition: Any) -> jnp.ndarray:
    """Normalize transition rates into branching ratios.

    Parameters
    ----------
    transition : array_like, shape (..., num_excited, num_ground)
        Transition or emission rates. The final axis enumerates destination
        ground states.

    Returns
    -------
    jax.Array
        Row-normalized branching ratios with the same shape as ``transition``.
        Rows with zero total rate are returned as zeros.

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


def calculate_cyclicity_spinflip(
    B: Any,
    theta: Any,
    phi: Any,
    alpha: Any = 0.0,
    beta: Any = 0.0,
    alpha_exc: Any = 0.0,
    beta_exc: Any = 0.0,
    gnd_kwargs: Optional[Dict[str, Any]] = None,
    exc_kwargs: Optional[Dict[str, Any]] = None,
    cap: float = 1e6,
):
    """Calculate eigenstate-keyed optical-cycling cyclicity.

    For a driven transition from ground state ``k`` to excited state ``l``,
    the returned cyclicity is

    ``emission[l, k] / (total_l - emission_folded[l, k])``.

    Decay into the upper orbital ground branch is folded onto the lower branch
    using overlaps of the reduced electron-nuclear density matrices. This
    models fast orbital relaxation that acts as the identity on spin.

    Parameters
    ----------
    B : scalar or array_like
        Magnetic-field magnitude or sweep.
    theta, phi : scalar
        Field direction in radians.
    alpha, beta : scalar, optional
        Ground-manifold strain.
    alpha_exc, beta_exc : scalar, optional
        Excited-manifold strain.
    gnd_kwargs, exc_kwargs : dict or None, optional
        Ground- and excited-manifold Hamiltonian overrides.
    cap : float, optional
        Soft upper bound used when the pump-out rate underflows.

    Returns
    -------
    E : jax.Array
        Ground-manifold eigenvalues.
    E_exc : jax.Array
        Excited-manifold eigenvalues.
    emission : jax.Array
        Polarization-summed spontaneous-emission rates.
    cyclicity : jax.Array
        Eigenstate-keyed photons-before-pump-out metric.
    spin_g : jax.Array
        Sign of the ground-state electron-spin projection along the field.
    spin_e : jax.Array
        Sign of the excited-state electron-spin projection along the field.
    emission_folded : jax.Array
        Decay rates after folding the upper orbital branch onto the lower
        branch.

    Raises
    ------
    ValueError
        If ``cap`` is not positive.
    """
    if cap <= 0.0:
        raise ValueError("cap must be positive.")

    gnd_kwargs = {} if gnd_kwargs is None else dict(gnd_kwargs)
    exc_kwargs = {} if exc_kwargs is None else dict(exc_kwargs)

    E, _, U_gnd, _, _, p = _solve_hamiltonian_core(
        B,
        theta,
        phi,
        manifold="ground",
        alpha=alpha,
        beta=beta,
        **gnd_kwargs,
    )
    E_exc, _, U_exc, _, _, _ = _solve_hamiltonian_core(
        B,
        theta,
        phi,
        manifold="excited",
        alpha=alpha_exc,
        beta=beta_exc,
        **exc_kwargs,
    )

    _, _, num_states = U_gnd.shape
    orbital_dim = int(round(2.0 * float(params.S) + 1.0))
    electron_dim = orbital_dim
    nuclear_dim = int(round(2.0 * float(params.Sn) + 1.0))
    num_lower = num_states // 2

    nx = jnp.sin(theta) * jnp.cos(phi)
    ny = jnp.sin(theta) * jnp.sin(phi)
    nz = jnp.cos(theta)

    X = _spin_operator(params.S, "x")
    Y = _spin_operator(params.S, "y")
    Z = _spin_operator(params.S, "z")
    I = jqt.identity(orbital_dim)
    In = jqt.identity(nuclear_dim)

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

    matrix_elements = _dipole_matrix_elements(U_gnd, U_exc, p)
    emission = jnp.sum(jnp.abs(matrix_elements) ** 2, axis=1)

    rho_spin = _reduced_spin_density_matrices(
        U_gnd,
        orbital_dim=orbital_dim,
        electron_dim=electron_dim,
        nuclear_dim=nuclear_dim,
    )
    rho_upper = rho_spin[:, num_lower:, :, :]
    rho_lower = rho_spin[:, :num_lower, :, :]

    # T[u, k] = Tr(rho_upper[u] rho_lower[k]).
    relaxation = jnp.real(
        jnp.einsum("bust,bkts->buk", rho_upper, rho_lower)
    )
    relaxation = jnp.maximum(relaxation, 0.0)

    row_total = jnp.sum(relaxation, axis=-1, keepdims=True)
    uniform = jnp.full_like(relaxation, 1.0 / num_lower)
    relaxation = jnp.where(
        row_total > 0.0,
        relaxation / row_total,
        uniform,
    )

    emission_folded = (
        emission[:, :, :num_lower]
        + jnp.einsum(
            "blu,buk->blk",
            emission[:, :, num_lower:],
            relaxation,
        )
    )

    total_emission = jnp.sum(emission, axis=-1)
    floor = jnp.where(
        total_emission > 0.0,
        total_emission / cap,
        1.0,
    )
    pumpout = total_emission[:, :, None] - emission_folded

    cyclicity_lower = (
        emission[:, :, :num_lower]
        / jnp.maximum(pumpout, floor[:, :, None])
    )
    cyclicity = jnp.zeros_like(emission)
    cyclicity = cyclicity.at[:, :, :num_lower].set(cyclicity_lower)

    return (
        E,
        E_exc,
        emission,
        cyclicity,
        spin_g,
        spin_e,
        emission_folded,
    )


def PLE_transitions(
    B: Any,
    theta: Any,
    phi: Any,
    eta: Any,
    alpha: Any = 0.0,
    beta: Any = 0.0,
    alpha_exc: Any = 0.0,
    beta_exc: Any = 0.0,
    gnd_kwargs: Optional[Dict[str, Any]] = None,
    exc_kwargs: Optional[Dict[str, Any]] = None,
):
    """Calculate polarization-resolved PLE transition intensities.

    Excitation is a coherent projection onto the laser polarization,

    ``|<exc_l|eta_x p_x + eta_y p_y + eta_z p_z|gnd_k>|^2``.

    Spontaneous emission is an incoherent sum over the three dipole channels
    and is used to calculate the returned branching ratios.

    Parameters
    ----------
    B : scalar or array_like
        Magnetic-field magnitude or sweep.
    theta, phi : scalar
        Field direction in radians.
    eta : array_like, shape (3,)
        Optical polarization coefficients.
    alpha, beta : scalar, optional
        Ground-manifold strain.
    alpha_exc, beta_exc : scalar, optional
        Excited-manifold strain.
    gnd_kwargs, exc_kwargs : dict or None, optional
        Ground- and excited-manifold Hamiltonian overrides.

    Returns
    -------
    E, Eref, U, alignment : jax.Array
        Ground-manifold eigensystem quantities.
    E_exc, Eref_exc, U_exc, alignment_exc : jax.Array
        Excited-manifold eigensystem quantities.
    transition : jax.Array
        Coherent polarization-projected excitation rates.
    cyclicity : jax.Array
        Polarization-summed spontaneous-emission branching ratios.
    """
    gnd_kwargs = {} if gnd_kwargs is None else dict(gnd_kwargs)
    exc_kwargs = {} if exc_kwargs is None else dict(exc_kwargs)

    E, Eref, U, _, alignment, p = _solve_hamiltonian_core(
        B,
        theta,
        phi,
        manifold="ground",
        alpha=alpha,
        beta=beta,
        **gnd_kwargs,
    )
    E_exc, Eref_exc, U_exc, _, alignment_exc, _ = _solve_hamiltonian_core(
        B,
        theta,
        phi,
        manifold="excited",
        alpha=alpha_exc,
        beta=beta_exc,
        **exc_kwargs,
    )

    eta = jnp.asarray(eta, dtype=jnp.complex128)
    if eta.shape != (3,):
        raise ValueError(f"eta must have shape (3,); received {eta.shape}.")

    matrix_elements = _dipole_matrix_elements(U, U_exc, p)
    amplitude = jnp.einsum("j,bjlk->blk", eta, matrix_elements)
    transition = jnp.abs(amplitude) ** 2

    emission = jnp.sum(jnp.abs(matrix_elements) ** 2, axis=1)
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
    cyclicity_min: Optional[float] = None,
    cyclicity_weight: bool = False,
    cyclicity_half: float = 1.0,
    cyclicity_softness: float = 4.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert folded cyclicity into a per-line PLE brightness weight.

    Parameters
    ----------
    cyclicity : array_like, shape (num_fields, num_excited, num_ground)
        Eigenstate-keyed cyclicity from
        :func:`calculate_cyclicity_spinflip`.
    cyclicity_min : float or None, optional
        Soft visibility threshold. ``None`` disables the threshold.
    cyclicity_weight : bool, optional
        Apply the smooth factor ``C / (C + cyclicity_half)``.
    cyclicity_half : float, optional
        Cyclicity at which the smooth factor is one half.
    cyclicity_softness : float, optional
        Hill exponent used by ``cyclicity_min``.

    Returns
    -------
    weight : jax.Array
        Multiplicative brightness factor with the full transition-matrix
        shape.
    cyclicity_lower : jax.Array
        Lower-branch cyclicity values.

    Raises
    ------
    ValueError
        If ``cyclicity`` is not three-dimensional.
    """
    cyclicity = jnp.asarray(cyclicity)

    if cyclicity.ndim != 3:
        raise ValueError(
            "cyclicity must have shape "
            "(num_fields, num_excited, num_ground)."
        )

    num_fields, num_states, _ = cyclicity.shape
    num_lower = num_states // 2
    C = cyclicity[:, :, :num_lower]
    gate = jnp.ones((num_fields, num_states, num_lower), dtype=C.dtype)

    if cyclicity_weight:
        gate = gate * C / (C + cyclicity_half)

    if cyclicity_min is not None:
        exponent = cyclicity_softness
        gate = gate * (
            C**exponent
            / (C**exponent + cyclicity_min**exponent)
        )

    weight = jnp.ones(
        (num_fields, num_states, num_states),
        dtype=C.dtype,
    )
    weight = weight.at[:, :, :num_lower].set(gate)
    return weight, C


def PLE_spectrum(
    f_meas: Any,
    B: Any,
    theta: Any,
    phi: Any,
    eta: Any,
    intensity: float = 1.0,
    lw: float = 0.080,
    alpha: Any = 0.0,
    beta: Any = 0.0,
    alpha_exc: Any = 0.0,
    beta_exc: Any = 0.0,
    gnd_kwargs: Optional[Dict[str, Any]] = None,
    exc_kwargs: Optional[Dict[str, Any]] = None,
    cyclicity_min: Optional[float] = None,
    cyclicity_weight: bool = False,
    cyclicity_half: float = 1.0,
    cyclicity_softness: float = 4.0,
    cyclicity_cap: float = 1e6,
    return_cyclicity: bool = False,
):
    """Calculate a Lorentzian-broadened PLE spectrum.

    Frequencies are referenced to the strained, hyperfine-free C transition,
    matching the original DJT implementation. Optical-cycling brightness is
    disabled unless one of the cyclicity controls is requested.

    Parameters
    ----------
    f_meas : array_like
        Frequency grid.
    B : scalar or array_like
        Magnetic-field magnitude or sweep.
    theta, phi : scalar
        Field direction in radians.
    eta : array_like, shape (3,)
        Optical polarization coefficients.
    intensity : float, optional
        Overall spectrum scale.
    lw : float, optional
        Lorentzian full width at half maximum.
    alpha, beta : scalar, optional
        Ground-manifold strain.
    alpha_exc, beta_exc : scalar, optional
        Excited-manifold strain.
    gnd_kwargs, exc_kwargs : dict or None, optional
        Ground- and excited-manifold Hamiltonian overrides.
    cyclicity_min : float or None, optional
        Soft lower visibility threshold for folded cyclicity.
    cyclicity_weight : bool, optional
        Apply ``C / (C + cyclicity_half)`` brightness weighting.
    cyclicity_half : float, optional
        Half-brightness cyclicity.
    cyclicity_softness : float, optional
        Hill exponent for ``cyclicity_min``.
    cyclicity_cap : float, optional
        Underflow cap used by :func:`calculate_cyclicity_spinflip`.
    return_cyclicity : bool, optional
        Return ``(spectrum, cyclicity)`` instead of only the spectrum.

    Returns
    -------
    spectrum : jax.Array
        Spectrum with shape ``(num_fields, num_frequencies)``. A one-field
        input is squeezed to ``(num_frequencies,)``.
    cyclicity : jax.Array, optional
        Lower-branch folded cyclicity, returned only when
        ``return_cyclicity=True``.
    """
    f_meas = jnp.ravel(
        jnp.atleast_1d(jnp.asarray(f_meas, dtype=jnp.float64))
    )

    (
        E,
        Eref,
        _,
        _,
        E_exc,
        Eref_exc,
        _,
        _,
        transition,
        _,
    ) = PLE_transitions(
        B,
        theta,
        phi,
        eta,
        alpha=alpha,
        beta=beta,
        alpha_exc=alpha_exc,
        beta_exc=beta_exc,
        gnd_kwargs=gnd_kwargs,
        exc_kwargs=exc_kwargs,
    )

    weight = None
    cyclicity = None

    if cyclicity_min is not None or cyclicity_weight or return_cyclicity:
        _, _, _, cyclicity_full, _, _, _ = calculate_cyclicity_spinflip(
            B,
            theta,
            phi,
            alpha=alpha,
            beta=beta,
            alpha_exc=alpha_exc,
            beta_exc=beta_exc,
            gnd_kwargs=gnd_kwargs,
            exc_kwargs=exc_kwargs,
            cap=cyclicity_cap,
        )
        weight, cyclicity = _folded_cyclicity_weight(
            cyclicity_full,
            cyclicity_min=cyclicity_min,
            cyclicity_weight=cyclicity_weight,
            cyclicity_half=cyclicity_half,
            cyclicity_softness=cyclicity_softness,
        )

    transition_frequencies = (
        (E_exc - Eref_exc[0])[:, :, None]
        - (E - Eref[0])[:, None, :]
    )
    amplitudes = transition if weight is None else transition * weight

    half_linewidth_squared = (jnp.asarray(lw, dtype=jnp.float64) / 2.0) ** 2
    detuning = (
        f_meas[None, None, None, :]
        - transition_frequencies[:, :, :, None]
    )
    lorentzians = (
        half_linewidth_squared
        / (detuning**2 + half_linewidth_squared)
    )

    spectrum = intensity * jnp.sum(
        amplitudes[:, :, :, None] * lorentzians,
        axis=(1, 2),
    )

    if int(E.shape[0]) == 1:
        spectrum = spectrum[0]
        if cyclicity is not None:
            cyclicity = cyclicity[0]

    if return_cyclicity:
        return spectrum, cyclicity

    return spectrum
