
# ==============================================================================
# Validation of `hamiltonian_djt_jqt.py` Required files
# ==============================================================================

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from jax import config

# Enable x64 before importing jax.numpy so THz and sub-MHz terms coexist safely.
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import qutip
import jaxquantum

print("JAX version:", getattr(jax, "__version__", "unknown"))
print("JAXQuantum version:", getattr(jaxquantum, "__version__", "unknown"))
print("QuTiP version:", getattr(qutip, "__version__", "unknown"))
print("JAX x64 enabled:", jax.config.read("jax_enable_x64"))


# ==============================================================================
# Load the original and new modules
# ==============================================================================

REQUIRED_FILENAMES = (
    "hamiltonian_DJT(1).py",
    "parameters_DJT(1).py",
    "hamiltonian_djt_jqt.py",
)


def locate_workspace() -> Path:
    """Find a directory containing all validation inputs."""
    candidates = (Path.cwd(), Path("/mnt/data"))
    for candidate in candidates:
        if all((candidate / name).exists() for name in REQUIRED_FILENAMES):
            return candidate.resolve()
    missing_report = {
        str(candidate): [
            name for name in REQUIRED_FILENAMES
            if not (candidate / name).exists()
        ]
        for candidate in candidates
    }
    raise FileNotFoundError(
        "Could not find all validation files. Missing by directory: "
        f"{missing_report}"
    )


def load_module(module_name: str, path: Path):
    """Load one Python source file under a chosen module name."""
    sys.modules.pop(module_name, None)
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not create an import specification for {path}.")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


ROOT = locate_workspace()
print("Validation directory:", ROOT)

# Both Hamiltonian modules import this exact module name.
params = load_module("parameters_DJT", ROOT / "parameters_DJT(1).py")
original = load_module(
    "hamiltonian_DJT_original",
    ROOT / "hamiltonian_DJT(1).py",
)
new = load_module(
    "hamiltonian_djt_jqt",
    ROOT / "hamiltonian_djt_jqt.py",
)

print("Loaded original:", ROOT / "hamiltonian_DJT(1).py")
print("Loaded new:", ROOT / "hamiltonian_djt_jqt.py")


# ==============================================================================
# Comparison helpers
# ==============================================================================

RTOL = 1e-5
ATOL = 1e-8
results: list[dict[str, object]] = []


def qutip_dense(operator) -> jnp.ndarray:
    """Return one QuTiP operator as a JAX array."""
    return jnp.asarray(operator.full())


def jaxquantum_dense(operator) -> jnp.ndarray:
    """Return one JAXQuantum operator as a JAX array."""
    return jnp.asarray(operator.to_dense().data)


def state_projectors(eigenvectors) -> jnp.ndarray:
    """Build phase-invariant projectors from column eigenvectors."""
    eigenvectors = jnp.asarray(eigenvectors)
    return jnp.einsum(
        "bik,bjk->bkij",
        eigenvectors,
        jnp.conj(eigenvectors),
    )


def record_allclose(name: str, expected, actual) -> bool:
    """Run jnp.allclose, record diagnostics, and print one result line."""
    expected = jnp.asarray(expected)
    actual = jnp.asarray(actual)

    if expected.shape != actual.shape:
        passed = False
        max_abs = float("inf")
    else:
        passed = bool(jnp.allclose(expected, actual, rtol=RTOL, atol=ATOL))
        max_abs = (
            0.0
            if expected.size == 0
            else float(jnp.max(jnp.abs(expected - actual)))
        )

    results.append(
        {
            "name": name,
            "passed": passed,
            "max_abs_difference": max_abs,
            "expected_shape": tuple(expected.shape),
            "actual_shape": tuple(actual.shape),
        }
    )
    status = "PASS" if passed else "FAIL"
    print(f"{status:4s}  {name:58s}  max |Δ| = {max_abs:.6e}")
    return passed


def independent_djt_tensors(Aperp, Apar, A1, A2):
    """Construct the DJT tensor mapping independently of the new module."""
    dtype = jnp.complex128
    A = jnp.array(
        [
            [Aperp, 0.0, 0.0],
            [0.0, Aperp, 0.0],
            [0.0, 0.0, 2.0 * Apar],
        ],
        dtype=dtype,
    )
    Ax = jnp.array(
        [
            [-0.5 * A2, 0.0, A1],
            [0.0, 0.5 * A2, 0.0],
            [A1, 0.0, 0.0],
        ],
        dtype=dtype,
    )
    Ay = jnp.array(
        [
            [0.0, -0.5 * A2, 0.0],
            [-0.5 * A2, 0.0, -A1],
            [0.0, -A1, 0.0],
        ],
        dtype=dtype,
    )
    return A, Ax, Ay


def manifold_scalars(manifold: str) -> dict[str, float]:
    """Return the original DJT scalar hyperfine values for one manifold."""
    suffix = "gnd" if manifold == "ground" else "exc"
    return {
        "Aperp": getattr(params, f"Aperp_{suffix}"),
        "Apar": getattr(params, f"Apar_{suffix}"),
        "A1": getattr(params, f"A1_{suffix}"),
        "A2": getattr(params, f"A2_{suffix}"),
    }


def manifold_tensors(manifold: str):
    """Return the independent tensor representation for one manifold."""
    values = manifold_scalars(manifold)
    return independent_djt_tensors(**values)


# ==============================================================================
# 1. Validate the scalar-to-tensor mapping
# ==============================================================================

for manifold in ("ground", "excited"):
    scalar_values = manifold_scalars(manifold)
    expected_tensors = independent_djt_tensors(**scalar_values)
    actual_tensors = new.djt_hyperfine_tensors(**scalar_values)

    for tensor_name, expected, actual in zip(
        ("A", "Ax", "Ay"),
        expected_tensors,
        actual_tensors,
    ):
        record_allclose(
            f"{manifold}: independent mapping for {tensor_name}",
            expected,
            actual,
        )

    print(f"\n{manifold.capitalize()} tensors")
    print("A =\n", expected_tensors[0])
    print("Ax =\n", expected_tensors[1])
    print("Ay =\n", expected_tensors[2], "\n")


# ==============================================================================
# 2. Compare isolated hyperfine terms and Hamiltonian building blocks
# ==============================================================================

full_hamiltonian_cases = (
    {
        "label": "zero field and zero strain",
        "bx": 0.0,
        "by": 0.0,
        "bz": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "upsilon": 0.0,
    },
    {
        "label": "mixed field, strain, and positive IOC",
        "bx": 0.125,
        "by": -0.087,
        "bz": 0.431,
        "alpha": 2.3,
        "beta": -1.7,
        "upsilon": 0.004,
    },
    {
        "label": "custom non-hyperfine parameters",
        "bx": -0.93,
        "by": 0.52,
        "bz": -0.28,
        "alpha": -3.5,
        "beta": 4.2,
        "upsilon": -0.003,
        "q": 0.123,
        "L": 777.7,
        "delta_f": 0.019,
        "rg": 1.1 * params.rg,
    },
)

for manifold in ("ground", "excited"):
    H_original, Href_original, p_original, J2_original = (
        original.create_hamiltonian_nuclear(manifold)
    )
    H_new, Href_new, p_new, J2_new = new.create_hamiltonian_nuclear(manifold)

    scalar_values = manifold_scalars(manifold)
    A, Ax, Ay = manifold_tensors(manifold)

    # Isolate H_hf in both implementations.
    hyperfine_original = H_original(
        0.0,
        0.0,
        0.0,
        alpha=0.0,
        beta=0.0,
        rg=0.0,
        q=0.0,
        L=0.0,
        upsilon=0.0,
        delta_f=0.0,
        **scalar_values,
    )
    hyperfine_new = H_new(
        0.0,
        0.0,
        0.0,
        alpha=0.0,
        beta=0.0,
        rg=0.0,
        q=0.0,
        L=0.0,
        upsilon=0.0,
        delta_f=0.0,
        A=A,
        Ax=Ax,
        Ay=Ay,
    )
    record_allclose(
        f"{manifold}: isolated hyperfine Hamiltonian",
        qutip_dense(hyperfine_original),
        jaxquantum_dense(hyperfine_new),
    )

    record_allclose(
        f"{manifold}: J2",
        qutip_dense(J2_original),
        jaxquantum_dense(J2_new),
    )

    for dipole_index, (old_dipole, new_dipole) in enumerate(
        zip(p_original, p_new)
    ):
        record_allclose(
            f"{manifold}: dipole p[{dipole_index}]",
            qutip_dense(old_dipole),
            jaxquantum_dense(new_dipole),
        )

    for case in full_hamiltonian_cases:
        case_kwargs = {key: value for key, value in case.items() if key != "label"}

        old_matrix = H_original(**case_kwargs, **scalar_values)
        new_matrix = H_new(**case_kwargs, A=A, Ax=Ax, Ay=Ay)
        record_allclose(
            f"{manifold}: full H -- {case['label']}",
            qutip_dense(old_matrix),
            jaxquantum_dense(new_matrix),
        )

        reference_kwargs = {
            "alpha": case_kwargs["alpha"],
            "beta": case_kwargs["beta"],
        }
        if "L" in case_kwargs:
            reference_kwargs["L"] = case_kwargs["L"]

        record_allclose(
            f"{manifold}: Href -- {case['label']}",
            qutip_dense(Href_original(**reference_kwargs)),
            jaxquantum_dense(Href_new(**reference_kwargs)),
        )


# ==============================================================================
# 3. Compare `solve_hamiltonian`
# ==============================================================================

B_VALUES = np.array([0.17, 0.63, 1.11])
THETA = 0.71
PHI = -0.43
ALPHA = 2.1
BETA = -1.3
UPSILON = 0.002

for manifold in ("ground", "excited"):
    A, Ax, Ay = manifold_tensors(manifold)

    old_solution = original.solve_hamiltonian(
        B_VALUES,
        THETA,
        PHI,
        manifold=manifold,
        alpha=ALPHA,
        beta=BETA,
        upsilon=UPSILON,
    )
    new_solution = new.solve_hamiltonian(
        B_VALUES,
        THETA,
        PHI,
        manifold=manifold,
        alpha=ALPHA,
        beta=BETA,
        upsilon=UPSILON,
        A=A,
        Ax=Ax,
        Ay=Ay,
    )

    record_allclose(
        f"{manifold}: solve_hamiltonian eigenvalues",
        old_solution[0],
        new_solution[0],
    )
    record_allclose(
        f"{manifold}: solve_hamiltonian reference eigenvalues",
        old_solution[1],
        new_solution[1],
    )
    record_allclose(
        f"{manifold}: solve_hamiltonian eigenstate projectors",
        state_projectors(old_solution[2]),
        state_projectors(new_solution[2]),
    )
    record_allclose(
        f"{manifold}: solve_hamiltonian J2 alignment",
        old_solution[4],
        new_solution[4],
    )


# ==============================================================================
# 4. Compare cyclicity and optical functions
# ==============================================================================

# Direct unit test of calculate_cyclicity, including a zero row.
example_rates = jnp.array(
    [
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 0.0, 0.0, 0.0],
        [5.0, 1.0, 0.5, 2.5],
    ],
    dtype=jnp.float64,
)
record_allclose(
    "calculate_cyclicity",
    original.calculate_cyclicity(np.asarray(example_rates)),
    new.calculate_cyclicity(example_rates),
)

A_gnd, Ax_gnd, Ay_gnd = manifold_tensors("ground")
A_exc, Ax_exc, Ay_exc = manifold_tensors("excited")

old_gnd_kwargs = {"upsilon": 0.0015}
old_exc_kwargs = {"upsilon": -0.0007}
new_gnd_kwargs = {
    "upsilon": 0.0015,
    "A": A_gnd,
    "Ax": Ax_gnd,
    "Ay": Ay_gnd,
}
new_exc_kwargs = {
    "upsilon": -0.0007,
    "A": A_exc,
    "Ax": Ax_exc,
    "Ay": Ay_exc,
}

ETA = np.array([0.31, -0.44, 0.77], dtype=np.complex128)
ALPHA_EXC = -1.8
BETA_EXC = 0.9

old_ple = original.PLE_transitions(
    B_VALUES,
    THETA,
    PHI,
    ETA,
    alpha=ALPHA,
    beta=BETA,
    alpha_exc=ALPHA_EXC,
    beta_exc=BETA_EXC,
    gnd_kwargs=old_gnd_kwargs,
    exc_kwargs=old_exc_kwargs,
)
new_ple = new.PLE_transitions(
    B_VALUES,
    THETA,
    PHI,
    ETA,
    alpha=ALPHA,
    beta=BETA,
    alpha_exc=ALPHA_EXC,
    beta_exc=BETA_EXC,
    gnd_kwargs=new_gnd_kwargs,
    exc_kwargs=new_exc_kwargs,
)

for index, label in (
    (0, "ground energies"),
    (1, "ground reference energies"),
    (3, "ground alignment"),
    (4, "excited energies"),
    (5, "excited reference energies"),
    (7, "excited alignment"),
    (8, "transition intensities"),
    (9, "emission branching ratios"),
):
    record_allclose(
        f"PLE_transitions: {label}",
        old_ple[index],
        new_ple[index],
    )

record_allclose(
    "PLE_transitions: ground eigenstate projectors",
    state_projectors(old_ple[2]),
    state_projectors(new_ple[2]),
)
record_allclose(
    "PLE_transitions: excited eigenstate projectors",
    state_projectors(old_ple[6]),
    state_projectors(new_ple[6]),
)


# ==============================================================================
# Folded optical-cycling cyclicity
# ==============================================================================

old_spinflip = original.calculate_cyclicity_spinflip(
    B_VALUES,
    THETA,
    PHI,
    alpha=ALPHA,
    beta=BETA,
    alpha_exc=ALPHA_EXC,
    beta_exc=BETA_EXC,
    gnd_kwargs=old_gnd_kwargs,
    exc_kwargs=old_exc_kwargs,
    cap=1e6,
)
new_spinflip = new.calculate_cyclicity_spinflip(
    B_VALUES,
    THETA,
    PHI,
    alpha=ALPHA,
    beta=BETA,
    alpha_exc=ALPHA_EXC,
    beta_exc=BETA_EXC,
    gnd_kwargs=new_gnd_kwargs,
    exc_kwargs=new_exc_kwargs,
    cap=1e6,
)

for index, label in enumerate(
    (
        "ground energies",
        "excited energies",
        "spontaneous-emission rates",
        "folded per-line cyclicity",
        "ground spin signs",
        "excited spin signs",
        "folded emission rates",
    )
):
    record_allclose(
        f"calculate_cyclicity_spinflip: {label}",
        old_spinflip[index],
        new_spinflip[index],
    )


# ==============================================================================
# Bare and cyclicity-weighted PLE spectra
# ==============================================================================

FREQUENCIES = np.linspace(-5.0, 5.0, 201)

common_spectrum_kwargs = dict(
    f_meas=FREQUENCIES,
    B=B_VALUES,
    theta=THETA,
    phi=PHI,
    eta=ETA,
    intensity=0.7,
    lw=0.09,
    alpha=ALPHA,
    beta=BETA,
    alpha_exc=ALPHA_EXC,
    beta_exc=BETA_EXC,
)

old_bare_spectrum = original.PLE_spectrum(
    **common_spectrum_kwargs,
    gnd_kwargs=old_gnd_kwargs,
    exc_kwargs=old_exc_kwargs,
)
new_bare_spectrum = new.PLE_spectrum(
    **common_spectrum_kwargs,
    gnd_kwargs=new_gnd_kwargs,
    exc_kwargs=new_exc_kwargs,
)
record_allclose(
    "PLE_spectrum: bare spectrum",
    old_bare_spectrum,
    new_bare_spectrum,
)

old_weighted_spectrum, old_returned_cyclicity = original.PLE_spectrum(
    **common_spectrum_kwargs,
    gnd_kwargs=old_gnd_kwargs,
    exc_kwargs=old_exc_kwargs,
    cyclicity_min=0.5,
    cyclicity_weight=True,
    cyclicity_half=1.0,
    cyclicity_softness=4.0,
    cyclicity_cap=1e6,
    return_cyclicity=True,
)
new_weighted_spectrum, new_returned_cyclicity = new.PLE_spectrum(
    **common_spectrum_kwargs,
    gnd_kwargs=new_gnd_kwargs,
    exc_kwargs=new_exc_kwargs,
    cyclicity_min=0.5,
    cyclicity_weight=True,
    cyclicity_half=1.0,
    cyclicity_softness=4.0,
    cyclicity_cap=1e6,
    return_cyclicity=True,
)
record_allclose(
    "PLE_spectrum: folded-cyclicity-weighted spectrum",
    old_weighted_spectrum,
    new_weighted_spectrum,
)
record_allclose(
    "PLE_spectrum: returned folded cyclicity",
    old_returned_cyclicity,
    new_returned_cyclicity,
)


# ==============================================================================
# 5. Final result
# ==============================================================================

print(f"\n{'Result':6s}  {'Maximum |difference|':>22s}  Check")
print("-" * 100)
for result in results:
    status = "PASS" if result["passed"] else "FAIL"
    print(
        f"{status:6s}  "
        f"{result['max_abs_difference']:22.6e}  "
        f"{result['name']}"
    )

failures = [result for result in results if not result["passed"]]
print("-" * 100)
print(f"Passed {len(results) - len(failures)} of {len(results)} checks.")

assert not failures, (
    "Validation failed for: "
    + ", ".join(result["name"] for result in failures)
)

print("ALL VALIDATION CHECKS PASSED")
