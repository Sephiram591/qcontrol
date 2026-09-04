"""Solvers"""

from diffrax import (
    diffeqsolve,
    ODETerm,
    SaveAt,
    PIDController,
    TqdmProgressMeter,
    NoProgressMeter,
)
from flax import struct
from jax import config
config.update("jax_enable_x64", True)
from jax import Array
from typing import Any, Callable, Optional, Sequence, Union
from jax.tree_util import tree_map
import diffrax
import jax.numpy as jnp
import warnings
import tqdm
import logging


from jaxquantum.core.qarray import Qarray, Qtypes, dag_data
from jaxquantum.core.conversions import jnp2jqt
from jaxquantum.core.operators import identity_like, multi_mode_basis_set
from jaxquantum.core.solvers import SolverOptions, solve
from jaxquantum.utils.utils import robust_isscalar

def sesolve_components(
    hamiltonians: Sequence[Union[Qarray, Array]],
    coefficients: Sequence[Any],
    psi0: Qarray,
    tlist: Array,
    *,
    filters: Optional[Sequence[Optional[Callable]]] = None,
    filter_y0s: Optional[Sequence[Any]] = None,
    saveat_tlist: Optional[Array] = None,
    solver_options: Optional[SolverOptions] = None,
    return_filter_states: bool = False,
):
    """Solve a Schrödinger equation with independently filtered terms.

    This function evolves an initial ket under a Hamiltonian of the form

    .. math::

        H(t) = \\sum_i c_i(t) H_i,

    where each coefficient :math:`c_i(t)` may optionally pass through an
    independent dynamical filter before being applied to its corresponding
    Hamiltonian term.

    A filter is represented by a callable with the signature::

        filter_fn(t, filter_state, input_value)
            -> (filter_state_derivative, output_value)

    The filter input is the raw coefficient for a Hamiltonian term, and the
    filter output is the coefficient actually applied to that term. Filter
    states are integrated simultaneously with the quantum state.

    Hamiltonian terms without filters do not introduce auxiliary ODE states.
    Filter states may be arbitrary PyTrees containing JAX-compatible arrays.

    Args:
        hamiltonians: Sequence of Hamiltonian operators. Each element must be
            either a :class:`Qarray` or an array with trailing matrix dimensions
            ``(..., n, n)``. Batched Hamiltonians are supported.
        coefficients: Sequence of coefficients corresponding one-to-one with
            ``hamiltonians``. Each coefficient may be either a scalar,
            a JAX array, or a callable with signature ``coefficient(t)`` that
            returns a scalar or batched scalar array.
        psi0: Initial quantum state. The state must be a ket; density matrices
            and general operators are not supported.
        tlist: One-dimensional array containing the integration interval. The
            first and last entries define the initial and final integration
            times. At least two entries are required.
        filters: Optional sequence of filter functions corresponding one-to-one
            with ``hamiltonians``. An entry of ``None`` applies the associated
            coefficient directly. When omitted, no Hamiltonian terms are
            filtered.
        filter_y0s: Optional sequence containing the initial internal state of
            each filter. An entry is required wherever the corresponding entry
            in ``filters`` is not ``None``. Entries corresponding to unfiltered
            terms may be ``None``. Each filter state may be an arbitrary PyTree
            of JAX-compatible values.
        saveat_tlist: Optional one-dimensional array of times at which to save
            the solution. When omitted, states are saved at every time in
            ``tlist``.
        solver_options: Optional configuration passed to the underlying ODE
            solver.
        return_filter_states: Whether to return the saved internal filter
            states in addition to the quantum states.

    Returns:
        If ``return_filter_states`` is ``False``, returns a :class:`Qarray`
        containing the evolved ket at each requested save time.

        If ``return_filter_states`` is ``True``, returns a tuple
        ``(states, filter_states)``, where:

        * ``states`` is a :class:`Qarray` containing the evolved ket.
        * ``filter_states`` is a tuple aligned with ``hamiltonians``. Entries
          corresponding to unfiltered terms are ``None``. Entries
          corresponding to filtered terms contain the saved trajectory of that
          filter's internal state.

    Raises:
        ValueError: If no Hamiltonian terms are provided.
        ValueError: If ``hamiltonians``, ``coefficients``, ``filters``, and
            ``filter_y0s`` do not have matching lengths.
        ValueError: If an active filter does not have an initial state.
        ValueError: If ``psi0`` is an operator or density matrix rather than a
            ket.
        ValueError: If ``tlist`` contains fewer than two times.

    Notes:
        The returned state may contain leading batch dimensions when a
        Hamiltonian or coefficient is batched. The initial ket is broadcast to
        the batch shape inferred from the right-hand side of the Schrödinger
        equation.

        The Hamiltonians are converted to dense arrays before integration to
        avoid performing :class:`Qarray` operations inside the ODE solver's
        inner loop.
    """
    hamiltonians = tuple(hamiltonians)
    coefficients = tuple(coefficients)
    n_terms = len(hamiltonians)

    if n_terms == 0:
        raise ValueError("At least one Hamiltonian term is required.")

    if len(coefficients) != n_terms:
        raise ValueError(
            "hamiltonians and coefficients must have the same length."
        )

    # A missing filter sequence means that no Hamiltonian terms are filtered.
    filters = (None,) * n_terms if filters is None else tuple(filters)

    # None is the natural placeholder for Hamiltonian terms without filters.
    filter_y0s = (
        (None,) * n_terms if filter_y0s is None else tuple(filter_y0s)
    )

    if len(filters) != n_terms:
        raise ValueError(
            "hamiltonians and filters must have the same length."
        )

    if len(filter_y0s) != n_terms:
        raise ValueError(
            "hamiltonians and filter_y0s must have the same length."
        )

    def as_coefficient_fn(coefficient):
        """Convert a constant or callable coefficient into a callable."""
        if callable(coefficient):
            return coefficient

        coefficient = jnp.asarray(coefficient)
        return lambda _t, coefficient=coefficient: coefficient

    # Normalize all coefficients to callables so that the ODE right-hand side
    # can handle constant and time-dependent coefficients identically.
    coefficient_fns = tuple(
        as_coefficient_fn(coefficient)
        for coefficient in coefficients
    )

    # Only active filters require auxiliary ODE states. Excluding unfiltered
    # terms avoids placing None values inside the Diffrax state PyTree.
    filtered_term_indices = tuple(
        i
        for i, filter_fn in enumerate(filters)
        if filter_fn is not None
    )

    # Map each Hamiltonian-term index to its position in the compact tuple of
    # active filter states.
    state_slot_for_term = {
        term_index: state_index
        for state_index, term_index in enumerate(filtered_term_indices)
    }

    for i in filtered_term_indices:
        if filter_y0s[i] is None:
            raise ValueError(
                f"filter_y0s[{i}] is required because filters[{i}] "
                "is not None."
            )

    # Each filter may use a differently structured PyTree state.
    filter_states0 = tuple(
        tree_map(jnp.asarray, filter_y0s[i])
        for i in filtered_term_indices
    )

    # Convert all Hamiltonians to dense arrays before entering the ODE solver's
    # hot loop.
    hamiltonian_data = tuple(
        H.to_dense().data if isinstance(H, Qarray) else jnp.asarray(H)
        for H in hamiltonians
    )

    if psi0.qtype == Qtypes.oper:
        raise ValueError(
            "sesolve_components requires a ket. "
            "Use a master-equation version for density matrices."
        )

    psi0_dense = psi0.to_ket().to_dense()
    dims = psi0_dense.dims
    psi0_data = psi0_dense.data

    tlist = jnp.asarray(tlist)

    if tlist.shape[0] < 2:
        raise ValueError("tlist must contain at least two times.")

    saveat_tlist = (
        tlist
        if saveat_tlist is None
        else jnp.atleast_1d(saveat_tlist)
    )

    # Use a quantum-state-only ODE when there are no active filters. This
    # avoids placing an empty auxiliary tuple in the Diffrax state.
    if len(filtered_term_indices) == 0:

        def rhs_unfiltered(t, psi_t, _args):
            """Evaluate the unfiltered Schrödinger-equation derivative."""
            H_t = jnp.zeros_like(hamiltonian_data[0])

            for H_i, coefficient_fn in zip(
                hamiltonian_data,
                coefficient_fns,
            ):
                coefficient = jnp.asarray(coefficient_fn(t))

                # The trailing singleton dimensions support both scalar and
                # batched coefficients.
                H_t = H_t + coefficient[..., None, None] * H_i

            return -1j * (H_t @ psi_t)

        # Infer any batch dimensions from the derivative and broadcast the
        # initial ket to match them.
        dpsi0 = rhs_unfiltered(tlist[0], psi0_data, None)
        psi0_data = jnp.broadcast_to(psi0_data, dpsi0.shape)

        sol = solve(
            rhs_unfiltered,
            psi0_data,
            tlist,
            saveat_tlist,
            args=None,
            solver_options=solver_options,
        )

        states = jnp2jqt(sol.ys, dims=dims)

        if return_filter_states:
            return states, (None,) * n_terms

        return states

    def rhs(t, state, _args):
        """Evaluate the coupled quantum-state and filter-state derivatives."""
        psi_t, filter_states_t = state

        H_t = jnp.zeros_like(hamiltonian_data[0])
        d_filter_states = []

        for i, (H_i, coefficient_fn, filter_fn) in enumerate(
            zip(hamiltonian_data, coefficient_fns, filters)
        ):
            # Evaluate the raw coefficient entering this Hamiltonian channel.
            raw_coefficient = jnp.asarray(coefficient_fn(t))

            if filter_fn is None:
                # Apply unfiltered coefficients directly without introducing
                # an auxiliary state.
                applied_coefficient = raw_coefficient
            else:
                state_slot = state_slot_for_term[i]
                filter_state = filter_states_t[state_slot]

                d_filter_state, applied_coefficient = filter_fn(
                    t,
                    filter_state,
                    raw_coefficient,
                )

                # Hamiltonian-term iteration order matches the order used to
                # construct filter_states0.
                d_filter_states.append(d_filter_state)

            # The trailing singleton dimensions support scalar and batched
            # coefficient values.
            coefficient_matrix = (
                jnp.asarray(applied_coefficient)[..., None, None]
            )
            H_t = H_t + coefficient_matrix * H_i

        dpsi_t = -1j * (H_t @ psi_t)

        return dpsi_t, tuple(d_filter_states)

    initial_state = (
        psi0_data,
        filter_states0,
    )

    # Match the existing sesolve broadcasting behavior by inferring the
    # quantum-state batch shape from the initial right-hand-side evaluation.
    dpsi0, _ = rhs(tlist[0], initial_state, None)
    psi0_data = jnp.broadcast_to(psi0_data, dpsi0.shape)

    initial_state = (
        psi0_data,
        filter_states0,
    )

    sol = solve(
        rhs,
        initial_state,
        tlist,
        saveat_tlist,
        args=None,
        solver_options=solver_options,
    )

    psi_values, saved_filter_states = sol.ys
    states = jnp2jqt(psi_values, dims=dims)

    if not return_filter_states:
        return states

    # Restore alignment with the original Hamiltonian sequence. Unfiltered
    # terms receive None, while filtered terms receive their saved trajectories.
    aligned_filter_states = tuple(
        None
        if filters[i] is None
        else saved_filter_states[state_slot_for_term[i]]
        for i in range(n_terms)
    )

    return states, aligned_filter_states

def mesolve_components(
    hamiltonians: Sequence[Union[Qarray, Array]],
    coefficients: Sequence[Any],
    rho0: Qarray,
    tlist: Array,
    *,
    collapse_operators: Optional[Sequence[Union[Qarray, Array]]] = None,
    filters: Optional[Sequence[Optional[Callable]]] = None,
    filter_y0s: Optional[Sequence[Any]] = None,
    saveat_tlist: Optional[Array] = None,
    solver_options: Optional[SolverOptions] = None,
    return_filter_states: bool = False,
):
    """Solve a Lindblad master equation with independently filtered H terms.

    This function evolves an initial density matrix according to

        d rho / dt =
            -i [H(t), rho]
            + sum_k (
                C_k rho C_k^dag
                - 1/2 C_k^dag C_k rho
                - 1/2 rho C_k^dag C_k
            ),

    with a Hamiltonian

        H(t) = sum_i c_i(t) H_i,

    where each coefficient c_i(t) may optionally pass through an independent
    dynamical filter before being applied to its corresponding Hamiltonian
    term.

    A filter must have the signature

        filter_fn(t, filter_state, input_value)
            -> (filter_state_derivative, output_value)

    and its internal state is integrated simultaneously with the density
    matrix.

    Args:
        hamiltonians:
            Sequence of Hamiltonian operators. Each element must be either a
            Qarray or an array with trailing matrix dimensions (..., n, n).

        coefficients:
            Sequence of coefficients corresponding one-to-one with
            ``hamiltonians``. Each coefficient may be a scalar, JAX array,
            or callable

                coefficient(t)

            returning a scalar or batched scalar array.

        rho0:
            Initial quantum state. May be either a ket or density matrix.

            If a ket is supplied, it is automatically converted to

                rho0 = |psi0><psi0|.

        tlist:
            One-dimensional array defining the integration interval. The first
            and last entries are used as the initial and final integration
            times.

        collapse_operators:
            Optional sequence of Lindblad collapse operators C_k.

            These are currently treated as time-independent operators.

        filters:
            Optional sequence of filter functions corresponding one-to-one
            with ``hamiltonians``.

            ``None`` means that the associated Hamiltonian coefficient is
            applied directly.

        filter_y0s:
            Initial internal states of the filters. An initial state is
            required for every active filter.

        saveat_tlist:
            Times at which to save the solution. Defaults to ``tlist``.

        solver_options:
            Options passed to the underlying ``solve`` function.

        return_filter_states:
            If True, also return the saved trajectories of all active filter
            states.

    Returns:
        If ``return_filter_states`` is False:

            states

        where ``states`` is a Qarray containing the density matrix at each
        requested save time.

        If ``return_filter_states`` is True:

            states, filter_states

        where ``filter_states`` is aligned with ``hamiltonians``. Entries for
        unfiltered Hamiltonian terms are ``None``.

    Notes:
        The Hamiltonian filters act only on Hamiltonian coefficients. Collapse
        operators are not filtered by this function.

        All Hamiltonians and collapse operators are converted to dense arrays
        before entering the Diffrax integration loop.
    """

    # ------------------------------------------------------------------
    # Normalize inputs
    # ------------------------------------------------------------------

    hamiltonians = tuple(hamiltonians)
    coefficients = tuple(coefficients)

    n_terms = len(hamiltonians)

    if n_terms == 0:
        raise ValueError("At least one Hamiltonian term is required.")

    if len(coefficients) != n_terms:
        raise ValueError(
            "hamiltonians and coefficients must have the same length."
        )

    # ``get_dynamic_hamiltonian`` returns its collapse operators as one
    # batched Qarray.  Do not call ``tuple(qarray)`` here: Qarray does not
    # implement ``__iter__``, so Python falls back to repeatedly calling
    # ``__getitem__`` until IndexError.  JAX array indexing does not raise
    # IndexError for an out-of-bounds traced index, which can make tracing
    # continue indefinitely.  Use the Qarray's static leading batch size to
    # perform a finite, explicit unstack instead.
    if collapse_operators is None:
        collapse_operators = ()
    elif isinstance(collapse_operators, Qarray):
        if len(collapse_operators.bdims) == 0:
            collapse_operators = (collapse_operators,)
        else:
            collapse_operators = tuple(
                collapse_operators[i]
                for i in range(len(collapse_operators))
            )
    else:
        collapse_operators = tuple(collapse_operators)

    filters = (
        (None,) * n_terms
        if filters is None
        else tuple(filters)
    )

    filter_y0s = (
        (None,) * n_terms
        if filter_y0s is None
        else tuple(filter_y0s)
    )

    if len(filters) != n_terms:
        raise ValueError(
            "hamiltonians and filters must have the same length."
        )

    if len(filter_y0s) != n_terms:
        raise ValueError(
            "hamiltonians and filter_y0s must have the same length."
        )

    # ------------------------------------------------------------------
    # Normalize coefficient objects into functions
    # ------------------------------------------------------------------

    def as_coefficient_fn(coefficient):
        """Convert a constant or callable coefficient into a callable."""

        if callable(coefficient):
            return coefficient

        coefficient = jnp.asarray(coefficient)

        return lambda _t, coefficient=coefficient: coefficient

    coefficient_fns = tuple(
        as_coefficient_fn(coefficient)
        for coefficient in coefficients
    )

    # ------------------------------------------------------------------
    # Determine which Hamiltonian terms have dynamical filters
    # ------------------------------------------------------------------

    filtered_term_indices = tuple(
        i
        for i, filter_fn in enumerate(filters)
        if filter_fn is not None
    )

    state_slot_for_term = {
        term_index: state_index
        for state_index, term_index
        in enumerate(filtered_term_indices)
    }

    for i in filtered_term_indices:
        if filter_y0s[i] is None:
            raise ValueError(
                f"filter_y0s[{i}] is required because "
                f"filters[{i}] is not None."
            )

    filter_states0 = tuple(
        tree_map(jnp.asarray, filter_y0s[i])
        for i in filtered_term_indices
    )

    # ------------------------------------------------------------------
    # Convert Hamiltonians to dense arrays outside the ODE loop
    # ------------------------------------------------------------------

    hamiltonian_data = tuple(
        H.to_dense().data
        if isinstance(H, Qarray)
        else jnp.asarray(H)
        for H in hamiltonians
    )

    # ------------------------------------------------------------------
    # Convert collapse operators to dense arrays outside the ODE loop
    # ------------------------------------------------------------------

    collapse_data = tuple(
        C.to_dense().data
        if isinstance(C, Qarray)
        else jnp.asarray(C)
        for C in collapse_operators
    )

    # Precompute C^dag and C^dag C because collapse operators are static.
    collapse_dag_data = tuple(
        jnp.swapaxes(jnp.conj(C), -1, -2)
        for C in collapse_data
    )

    collapse_dag_c_data = tuple(
        C_dag @ C
        for C_dag, C in zip(
            collapse_dag_data,
            collapse_data,
        )
    )

    # ------------------------------------------------------------------
    # Convert initial state to density matrix
    # ------------------------------------------------------------------

    if rho0.qtype == Qtypes.oper:

        rho0_dense = rho0.to_dense()

        dims = rho0_dense.dims
        rho0_data = rho0_dense.data

    else:

        psi0_dense = rho0.to_ket().to_dense()

        dims = psi0_dense.dims
        psi0_data = psi0_dense.data

        psi0_dag = jnp.swapaxes(
            jnp.conj(psi0_data),
            -1,
            -2,
        )

        rho0_data = psi0_data @ psi0_dag

    # ------------------------------------------------------------------
    # Times
    # ------------------------------------------------------------------

    tlist = jnp.asarray(tlist)

    if tlist.shape[0] < 2:
        raise ValueError(
            "tlist must contain at least two times."
        )

    saveat_tlist = (
        tlist
        if saveat_tlist is None
        else jnp.atleast_1d(saveat_tlist)
    )

    # ------------------------------------------------------------------
    # Lindblad RHS helper
    # ------------------------------------------------------------------

    def master_equation_rhs(H_t, rho_t):
        """Evaluate the Hamiltonian + Lindblad derivative."""

        # Coherent evolution:
        #
        #     -i [H, rho]
        #
        drho_t = -1j * (
            H_t @ rho_t
            - rho_t @ H_t
        )

        # Dissipative evolution:
        #
        # C rho C^dag
        # - 1/2 C^dag C rho
        # - 1/2 rho C^dag C
        #
        for C, C_dag, C_dag_C in zip(
            collapse_data,
            collapse_dag_data,
            collapse_dag_c_data,
        ):

            drho_t = drho_t + (
                C @ rho_t @ C_dag
                - 0.5 * (C_dag_C @ rho_t)
                - 0.5 * (rho_t @ C_dag_C)
            )

        return drho_t

    # ==================================================================
    # No active filters
    # ==================================================================

    if len(filtered_term_indices) == 0:

        def rhs_unfiltered(t, rho_t, _args):
            """Master-equation derivative without auxiliary filters."""

            H_t = jnp.zeros_like(
                hamiltonian_data[0]
            )

            for H_i, coefficient_fn in zip(
                hamiltonian_data,
                coefficient_fns,
            ):

                coefficient = jnp.asarray(
                    coefficient_fn(t)
                )

                H_t = H_t + (
                    coefficient[..., None, None]
                    * H_i
                )

            return master_equation_rhs(
                H_t,
                rho_t,
            )

        # --------------------------------------------------------------
        # Infer possible batch dimensions
        # --------------------------------------------------------------

        drho0 = rhs_unfiltered(
            tlist[0],
            rho0_data,
            None,
        )

        rho0_data = jnp.broadcast_to(
            rho0_data,
            drho0.shape,
        )

        # --------------------------------------------------------------
        # Solve
        # --------------------------------------------------------------

        sol = solve(
            rhs_unfiltered,
            rho0_data,
            tlist,
            saveat_tlist,
            args=None,
            solver_options=solver_options,
        )

        states = jnp2jqt(
            sol.ys,
            dims=dims,
        )

        if return_filter_states:
            return states, (None,) * n_terms

        return states

    # ==================================================================
    # Hamiltonian + dynamical filters
    # ==================================================================

    def rhs(t, state, _args):
        """Evaluate coupled density-matrix and filter-state derivatives."""

        rho_t, filter_states_t = state

        H_t = jnp.zeros_like(
            hamiltonian_data[0]
        )

        d_filter_states = []

        # --------------------------------------------------------------
        # Construct filtered Hamiltonian
        # --------------------------------------------------------------

        for i, (
            H_i,
            coefficient_fn,
            filter_fn,
        ) in enumerate(
            zip(
                hamiltonian_data,
                coefficient_fns,
                filters,
            )
        ):

            raw_coefficient = jnp.asarray(
                coefficient_fn(t)
            )

            if filter_fn is None:

                applied_coefficient = (
                    raw_coefficient
                )

            else:

                state_slot = (
                    state_slot_for_term[i]
                )

                filter_state = (
                    filter_states_t[state_slot]
                )

                (
                    d_filter_state,
                    applied_coefficient,
                ) = filter_fn(
                    t,
                    filter_state,
                    raw_coefficient,
                )

                d_filter_states.append(
                    d_filter_state
                )

            coefficient_matrix = (
                jnp.asarray(
                    applied_coefficient
                )[..., None, None]
            )

            H_t = (
                H_t
                + coefficient_matrix * H_i
            )

        # --------------------------------------------------------------
        # Master equation
        # --------------------------------------------------------------

        drho_t = master_equation_rhs(
            H_t,
            rho_t,
        )

        return (
            drho_t,
            tuple(d_filter_states),
        )

    # ------------------------------------------------------------------
    # Initial combined ODE state
    # ------------------------------------------------------------------

    initial_state = (
        rho0_data,
        filter_states0,
    )

    # Infer batch dimensions using the initial RHS, matching the behavior
    # of sesolve_components.
    drho0, _ = rhs(
        tlist[0],
        initial_state,
        None,
    )

    rho0_data = jnp.broadcast_to(
        rho0_data,
        drho0.shape,
    )

    initial_state = (
        rho0_data,
        filter_states0,
    )

    # ------------------------------------------------------------------
    # Solve coupled master equation + filter dynamics
    # ------------------------------------------------------------------

    sol = solve(
        rhs,
        initial_state,
        tlist,
        saveat_tlist,
        args=None,
        solver_options=solver_options,
    )

    rho_values, saved_filter_states = (
        sol.ys
    )

    states = jnp2jqt(
        rho_values,
        dims=dims,
    )

    if not return_filter_states:
        return states

    # ------------------------------------------------------------------
    # Restore filter-state alignment with Hamiltonian terms
    # ------------------------------------------------------------------

    aligned_filter_states = tuple(
        None
        if filters[i] is None
        else saved_filter_states[
            state_slot_for_term[i]
        ]
        for i in range(n_terms)
    )

    return states, aligned_filter_states