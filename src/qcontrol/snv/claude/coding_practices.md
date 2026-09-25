# Coding Practices

Distilled from projects widely regarded as well organized: scikit-learn, SQLAlchemy, Django, attrs/Pydantic, and in the JAX ecosystem, Equinox, Diffrax, Optax, NumPyro, and dynamiqs.

## 1. API design

- **One consistent interface per concept.** scikit-learn's `fit`/`predict`/`transform` works because *every* estimator follows it. Pick the verbs for your domain (`hamiltonian(params)`, `evolve(state, t)`, `steady_state(L)`) and use them everywhere.
- **Keep configuration separate from data.** Settings go in the constructor or a frozen dataclass; arrays go through method arguments. A method shouldn't secretly change `self`.
- **Small public surface.** Export only what users need via `__init__.py` / `__all__`, and prefix internals with `_`. A smaller API is easier to keep stable.
- **Explicit over implicit.** No hidden globals and no behavior that depends on import order. Pass what you need.

## 2. Structure

- **Layer the code.** SQLAlchemy separates Core (low level) from ORM (high level). Here, that would be physics primitives (operators, tensors) → model construction (Hamiltonians, Liouvillians) → solvers → experiment/inference.
- **Dependencies point downward.** Low-level modules never import high-level ones. Circular imports usually mean a layer boundary is wrong.
- **Group modules by concept, not by type.** Prefer `hamiltonian.py`, `distribution.py` over `utils.py`, `helpers.py`. A growing `*_helpers.py` is a sign that some concept doesn't have a proper home yet.
- **Notebooks are for exploration, not a source of truth.** Once logic stabilizes, move it into a module and import it from the notebook.

## 3. JAX-specific (Equinox / Diffrax / Optax style)

- **Pure functions.** No side effects, no mutation, no reads of global state inside anything you `jit`, `grad`, or `vmap`.
- **Models are PyTrees.** Use `equinox.Module` (or a frozen dataclass registered as a PyTree) so parameters flow through transforms automatically. Mark non-array fields as static.
- **Composable transforms.** Optax chains gradient transforms (`optax.chain(clip, adam)`). Design pieces so they compose instead of adding flags to one big function.
- **Abstract the solver interface.** Diffrax separates `Term` (what's being solved) from `Solver` (how) and `StepSizeController` (when). Similarly, the Hamiltonian/Liouvillian construction shouldn't know which integrator consumes it.
- **Keep Python control flow out of traced code.** Use `jax.lax.cond`/`scan`/`while_loop`, or make branches static. Keep shapes static where you can to avoid recompiles.
- **Explicit PRNG keys.** Pass keys in and split them. Never reuse one.
- **`vmap` rather than loops** over parameter sweeps, ensembles, and tones.

## 4. Types and validation

- **Type hints on all public functions.** Use `jaxtyping` for array shapes/dtypes (`Float[Array, "n n"]`). This is what Equinox/Diffrax use, and it doubles as documentation.
- **Validate at boundaries only.** Check user inputs once at the entry point (Pydantic/attrs style). Internal functions trust their callers.
- **Frozen dataclasses for parameter bundles.** They're immutable, hashable, and self-documenting.

## 5. Testing

- **Test physics invariants, not just outputs.** Hermiticity, trace preservation, positivity, known analytic limits (two-level Rabi, CPT dark state), and symmetry under relabeling.
- **Regression tests with fixed seeds and tolerances.** Use `np.testing.assert_allclose` with an explicit `rtol`/`atol`.
- **Test transforms.** Check that `jit(f)(x) == f(x)`, that `grad` matches finite differences, and that `vmap` matches a loop.
- **Tests live in `tests/`, mirroring the package layout,** and run with `pytest`.

## 6. Documentation

- **Docstrings on the public API**, NumPy style (the scientific-Python standard): Parameters, Returns, units, and conventions (ħ = 1? angular vs. ordinary frequency?).
- **State conventions once and centrally.** Units, sign conventions, frame (lab vs. rotating), and basis ordering. Mismatches here cause most physics bugs.
- **Comments explain *why*, not *what*.** Names should cover the "what".

## 7. Evolution and hygiene

- **Deprecate, don't break** (Django/SQLAlchemy). Emit a `DeprecationWarning` for a release before removing anything that other code depends on.
- **Automated formatting/linting.** Ruff (lint + format) plus a type checker (pyright/mypy), run in pre-commit. It ends style debates.
- **Small, focused commits** with messages that explain why the change was made.
- **Pin dependencies in `pyproject.toml`**, especially JAX, where minor versions can change behavior.

## Quick checklist for new code

- [ ] Pure function / PyTree module, with no hidden state?
- [ ] Fits an existing interface convention?
- [ ] Lives in the right layer, with no upward imports?
- [ ] Type-hinted with shapes?
- [ ] Units and conventions documented?
- [ ] A test covering at least one physical invariant?
- [ ] Works under `jit` and `vmap`?
