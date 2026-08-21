"""
Parameters for SnV117 center in diamond (for use with DJT Hamiltonian).

References:
  - Ham reduction & orbital parameters (gL, p, delta_p):
    Thiering & Gali 2018, Table IV, PhysRevX.8.021063
  - Spin-orbit coupling (L, L_exc):
    Thiering & Gali 2018, PhysRevX.8.021063
  - DJT hyperfine parameters (A1, A2, A_parallel, A_perpendicular):
    Thiering & Gali 2025, Phys. Rev. B, DOI: 10.1103/fq19-lfmv
    "Magneto-optical properties of group-IV vacancy centers in diamond
     upon hydrostatic pressure"
    NOTE: These are NOT the same as the Harris et al. 2023 (PRX Quantum 4, 040301)
    values. Harris uses Afc/Add (orbital-averaged form), while the DJT decomposition
    separates the hyperfine into orbital-independent (A_par, A_perp) and
    orbital-nuclear coupling (A1, A2) components. The numerical values differ
    because they represent different decompositions of the hyperfine tensor.

Hyperfine parameters use the DJT form with A1, A2, A_parallel, A_perpendicular.
"""
import math

import jax.numpy as jnp
from typing import Tuple, Dict
from enum import IntEnum

# Spin values
S = 1/2  # Electron spin
Sn = 1/2  # Nuclear spin (117Sn)

# Electron g-factor
gS = 2.0023

# Ham reduction & orbital parameters (Table IV, Thiering & Gali 2018, PhysRevX.8.021063)
# Ground state SnV(2Eg)
gL_gnd = 0.328       # Stevens orbital reduction factor
p_32_gnd = 0.513     # Ham reduction factor for E_{3/2} state
p_12_gnd = 0.429     # Ham reduction factor for E_{1/2} state
p_gnd = 0.471        # average: (p_{3/2} + p_{1/2}) / 2
delta_p_gnd = 0.042  # asymmetry: (p_{3/2} - p_{1/2}) / 2
# Zeeman zz correction δf = δp * gL (paper: 0.014). Physical, keep enabled;
# the "abnormal spectrum" it used to cause came from a sign error in Hbze_corr
# (now fixed: the term is +2δf·μB·Bz·Sz, giving g_zz(E_3/2) = gS + 2·gL·p_{3/2}).
delta_f_gnd = delta_p_gnd * gL_gnd  # = 0.0138

# Excited state SnV(2Eu)
# gL_exc = 0.782
gL_exc = 0.88 # Asher revised for accuracy
p_32_exc = 0.429
p_12_exc = -0.178    # negative: E_{1/2} vibronic state dominated by SOC-favored components
p_exc = 0.125
delta_p_exc = 0.303
delta_f_exc = delta_p_exc * gL_exc  # = 0.237 (paper: 0.238) — large, dominates ES g_zz

# Orbital magnetic field susceptibility: f = p * gL
q = gL_gnd * p_gnd       # ground state
q_exc = gL_exc * p_exc   # excited state

# Spin-orbit coupling
L = 830.0  # [GHz] spin-orbit coupling ground state
L_exc = 3000.0  # [GHz] spin-orbit coupling excited state

LEVEL_OFFSET = 483796.026775 + L_exc/2
GAMMA_FREQ = 483796.026775 + L/2

# Hyperfine Properties
# Ratio of electron to proton mass (mu_N/mu_B)
rmep = 5.44617021e-4
# Nuclear/electron Zeeman ratio in code units (bz = gS*mu_B*B):
# H_n = -g_n*mu_N*B·I with g_n(117Sn) = -2.00208 (mu = -1.00104 mu_N, I = 1/2),
# so the coefficient is -g_n*mu_N/(gS*mu_B) = +2.00208*rmep/gS — positive, i.e.
# the same sign as the electron term, because the 117Sn moment is negative.
rg_117 = 2.00208 * rmep / gS
rg_c13 = -2*0.702369*rmep/gS

# Hyperfine parameters for 117Sn (S_n = 1/2), DJT form.
# Values and SIGNS verbatim from Table III of Tóth, Gali & Thiering, PRB 112,
# 155201 (2025), DOI 10.1103/fq19-lfmv (= arXiv:2408.10407v3, published version;
# note the arXiv v1 used a different parametrization: A∥SzIz with A∥ = 976 MHz —
# same net Hamiltonian). Used with their eq. (2) implemented verbatim in
# hamiltonian_DJT.Hhf (note its 2A∥SzIz: A∥ is defined as ½A_zz, their eq. 4).
# Observable check: A_PLE = A∥_exc - A∥_gnd = -473 MHz vs measured -484(8) MHz
# (Harris et al. 2023).

# Ground state (2Eg)
A1_gnd = 1.1 / 1000.0        # [GHz] dynamic (orbital-off-diagonal) hyperfine A1
A2_gnd = 1.9 / 1000.0        # [GHz] dynamic (orbital-off-diagonal) hyperfine A2
Apar_gnd = 488.0 / 1000.0    # [GHz] parallel hyperfine coupling A∥ (= ½A_zz)
Aperp_gnd = 1029.7 / 1000.0  # [GHz] perpendicular hyperfine coupling A⊥

# Excited state (2Eu)
A1_exc = 0.1 / 1000.0        # [GHz] dynamic hyperfine A1
A2_exc = -0.43 / 1000.0      # [GHz] dynamic hyperfine A2
Apar_exc = 15.0 / 1000.0     # [GHz] parallel hyperfine coupling A∥ (= ½A_zz)
Aperp_exc = 32.3 / 1000.0    # [GHz] perpendicular hyperfine coupling A⊥


class HyperfineNeighbor(IntEnum):
    """Nuclear species and lattice site encoded in one integer ID.

    The six first-neighbor and six second-neighbor carbon positions are stored
    as distinct enum values, so ``SnV120Distribution`` does not need a separate
    carbon-site field.  Sites 1/4, 2/5, and 3/6 are inversion partners.  The
    present D3d hyperfine model therefore gives each inversion pair identical
    tensors, while retaining distinct IDs for future symmetry-breaking models.

    The values 0--3 preserve the numeric meanings of the previous enum:
    no nucleus, the Table-XII reference first-neighbor site, the Table-XII
    reference second-neighbor site, and central 117Sn, respectively.
    """

    NONEIGHBOR = 0
    SNV117 = 1

    C13_FIRST_SITE_1_4 = 2
    C13_FIRST_SITE_2_5 = 3
    C13_FIRST_SITE_3_6 = 4

    C13_SECOND_SITE_1_4 = 5
    C13_SECOND_SITE_2_5 = 6
    C13_SECOND_SITE_3_6 = 7


_ORDERED_HYPERFINE_NEIGHBORS = tuple(
    sorted(HyperfineNeighbor, key=int)
)

_expected_hyperfine_values = list(
    range(len(_ORDERED_HYPERFINE_NEIGHBORS))
)
_actual_hyperfine_values = [
    int(neighbor) for neighbor in _ORDERED_HYPERFINE_NEIGHBORS
]
if _actual_hyperfine_values != _expected_hyperfine_values:
    raise ValueError(
        "HyperfineNeighbor values must be contiguous integers starting at "
        f"zero. Got {_actual_hyperfine_values}."
    )


# Site angles are active rotations about the local defect Z axis relative to
# the selected Table-XII carbon site.  The paper's Appendix D gives the C3
# transformations for the three azimuthal orientations.  Inversion partners
# have the same tensor because electron and nuclear spins are axial vectors and
# the hyperfine interaction is inversion even.
_CARBON_SITE_INFO = {
    HyperfineNeighbor.C13_FIRST_SITE_1_4: ("first", 1, 0.0),
    HyperfineNeighbor.C13_FIRST_SITE_2_5: ("first", 2, +2.0 * math.pi / 3.0),
    HyperfineNeighbor.C13_FIRST_SITE_3_6: ("first", 3, -2.0 * math.pi / 3.0),
    # HyperfineNeighbor.C13_FIRST_SITE_4: ("first", 4, 0.0),
    # HyperfineNeighbor.C13_FIRST_SITE_5: ("first", 5, +2.0 * math.pi / 3.0),
    # HyperfineNeighbor.C13_FIRST_SITE_6: ("first", 6, -2.0 * math.pi / 3.0),
    HyperfineNeighbor.C13_SECOND_SITE_1_4: ("second", 1, 0.0),
    HyperfineNeighbor.C13_SECOND_SITE_2_5: ("second", 2, +2.0 * math.pi / 3.0),
    HyperfineNeighbor.C13_SECOND_SITE_3_6: ("second", 3, -2.0 * math.pi / 3.0),
    # HyperfineNeighbor.C13_SECOND_SITE_4: ("second", 4, 0.0),
    # HyperfineNeighbor.C13_SECOND_SITE_5: ("second", 5, +2.0 * math.pi / 3.0),
    # HyperfineNeighbor.C13_SECOND_SITE_6: ("second", 6, -2.0 * math.pi / 3.0),
}


# Nuclear/electron Zeeman ratio indexed directly by HyperfineNeighbor.
rg = jnp.asarray(
    [
        (
            0.0
            if neighbor == HyperfineNeighbor.NONEIGHBOR
            else rg_117
            if neighbor == HyperfineNeighbor.SNV117
            else rg_c13
        )
        for neighbor in _ORDERED_HYPERFINE_NEIGHBORS
    ]
)


# Reference Table-XII tensors use the Figure-8(e) defect frame:
#
#   X = [2, -1, -1] / sqrt(6)
#   Y = [0,  1, -1] / sqrt(2)
#   Z = [1,  1,  1] / sqrt(3)
#
# Component order is Axx, Ayy, Azz, Axy, Axz, Ayz.  Carbon values are reported
# in MHz and converted here to GHz.  The excited-state second-neighbor coupling
# is approximated as zero because the paper reports it as almost negligible and
# does not tabulate a separate tensor triplet.
_ZERO6 = jnp.zeros((6,))

_C13_FIRST_GND_REFERENCE = (
    jnp.asarray([70.1, 28.3, 33.4, 0.0, 14.6, 0.0]) * 1e-3,
    jnp.asarray([-21.4, -9.3, -10.8, 1.3, -4.3, 0.4]) * 1e-3,
    jnp.asarray([-49.3, -21.3, -24.9, -1.0, -9.8, -0.3]) * 1e-3,
)
_C13_SECOND_GND_REFERENCE = (
    jnp.asarray([-4.1, -5.3, -4.9, -0.8, -0.2, 0.5]) * 1e-3,
    jnp.asarray([1.9, 2.1, 2.1, 0.3, 0.4, -0.3]) * 1e-3,
    jnp.asarray([2.1, 2.4, 3.2, 0.2, -0.1, 0.2]) * 1e-3,
)
_C13_FIRST_EXC_REFERENCE = (
    jnp.asarray([54.96, 33.86, 28.43, 0.03, 8.83, -0.02]) * 1e-3,
    jnp.asarray([-18.26, -3.71, -7.19, -2.77, -4.75, 2.38]) * 1e-3,
    jnp.asarray([-42.01, -8.31, -16.44, 2.15, -10.99, -1.85]) * 1e-3,
)
_C13_SECOND_EXC_REFERENCE = (
    _ZERO6,
    _ZERO6,
    _ZERO6,
)

_SN117_GND = (
    jnp.asarray([Aperp_gnd, Aperp_gnd, 2.0 * Apar_gnd, 0.0, 0.0, 0.0]),
    jnp.asarray([-A2_gnd / 2.0, A2_gnd / 2.0, 0.0, 0.0, A1_gnd, 0.0]),
    jnp.asarray([0.0, 0.0, 0.0, -A2_gnd / 2.0, 0.0, -A1_gnd]),
)
_SN117_EXC = (
    jnp.asarray([Aperp_exc, Aperp_exc, 2.0 * Apar_exc, 0.0, 0.0, 0.0]),
    jnp.asarray([-A2_exc / 2.0, A2_exc / 2.0, 0.0, 0.0, A1_exc, 0.0]),
    jnp.asarray([0.0, 0.0, 0.0, -A2_exc / 2.0, 0.0, -A1_exc]),
)


def _tensor6_to_symmetric_matrix(values: jnp.ndarray) -> jnp.ndarray:
    """Expand ``[xx, yy, zz, xy, xz, yz]`` to a symmetric matrix."""
    values = jnp.asarray(values)
    if values.shape != (6,):
        raise ValueError(
            "A six-component hyperfine tensor must have shape (6,), "
            f"got {values.shape}."
        )

    xx, yy, zz, xy, xz, yz = values
    return jnp.asarray(
        [
            [xx, xy, xz],
            [xy, yy, yz],
            [xz, yz, zz],
        ]
    )


def _symmetric_matrix_to_tensor6(tensor: jnp.ndarray) -> jnp.ndarray:
    """Compress a symmetric 3x3 matrix to ``[xx, yy, zz, xy, xz, yz]``."""
    tensor = jnp.asarray(tensor)
    if tensor.shape != (3, 3):
        raise ValueError(
            "A Cartesian hyperfine tensor must have shape (3, 3), "
            f"got {tensor.shape}."
        )

    # Remove only roundoff-level antisymmetric components introduced by the
    # matrix products below.
    tensor = 0.5 * (tensor + tensor.T)
    return jnp.asarray(
        [
            tensor[0, 0],
            tensor[1, 1],
            tensor[2, 2],
            tensor[0, 1],
            tensor[0, 2],
            tensor[1, 2],
        ]
    )


def _rotate_carbon_hyperfine_triplet(
    triplet: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    angle: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Rotate a selected-site ``(A, Ax, Ay)`` triplet to another C3 site.

    ``angle`` is an active rotation of the physical carbon site about the local
    defect Z axis.  The Cartesian tensor indices rotate by ``angle``.  The
    orbital pair multiplying ``(sigma_z, sigma_x)`` rotates by ``2 * angle``:

        Ax(phi) = R(phi) [cos(2phi) Ax - sin(2phi) Ay] R(phi)^T
        Ay(phi) = R(phi) [sin(2phi) Ax + cos(2phi) Ay] R(phi)^T

    The double angle is required because ``sigma_z`` and ``sigma_x`` are
    quadratic bilinears of the real ``(e_x, e_y)`` orbital amplitudes.  This is
    the site transformation underlying Appendix-D Eqs. (D5a)--(D5b).
    """
    A, Ax, Ay = (
        _tensor6_to_symmetric_matrix(values)
        for values in triplet
    )

    cos_phi = math.cos(angle)
    sin_phi = math.sin(angle)
    rotation = jnp.asarray(
        [
            [cos_phi, -sin_phi, 0.0],
            [sin_phi, cos_phi, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    cos_2phi = math.cos(2.0 * angle)
    sin_2phi = math.sin(2.0 * angle)
    Ax_orbital_rotated = cos_2phi * Ax - sin_2phi * Ay
    Ay_orbital_rotated = sin_2phi * Ax + cos_2phi * Ay

    def rotate_cartesian(tensor: jnp.ndarray) -> jnp.ndarray:
        return rotation @ tensor @ rotation.T

    return (
        _symmetric_matrix_to_tensor6(rotate_cartesian(A)),
        _symmetric_matrix_to_tensor6(
            rotate_cartesian(Ax_orbital_rotated)
        ),
        _symmetric_matrix_to_tensor6(
            rotate_cartesian(Ay_orbital_rotated)
        ),
    )


def _build_hyperfine_vector_dictionaries() -> Tuple[
    Dict[HyperfineNeighbor, jnp.ndarray],
    Dict[HyperfineNeighbor, jnp.ndarray],
    Dict[HyperfineNeighbor, jnp.ndarray],
    Dict[HyperfineNeighbor, jnp.ndarray],
    Dict[HyperfineNeighbor, jnp.ndarray],
    Dict[HyperfineNeighbor, jnp.ndarray],
]:
    """Build six-component tensor dictionaries for every enum member."""
    A_gnd: Dict[HyperfineNeighbor, jnp.ndarray] = {}
    Ax_gnd: Dict[HyperfineNeighbor, jnp.ndarray] = {}
    Ay_gnd: Dict[HyperfineNeighbor, jnp.ndarray] = {}
    A_exc: Dict[HyperfineNeighbor, jnp.ndarray] = {}
    Ax_exc: Dict[HyperfineNeighbor, jnp.ndarray] = {}
    Ay_exc: Dict[HyperfineNeighbor, jnp.ndarray] = {}

    for neighbor in _ORDERED_HYPERFINE_NEIGHBORS:
        if neighbor == HyperfineNeighbor.NONEIGHBOR:
            ground_triplet = (_ZERO6, _ZERO6, _ZERO6)
            excited_triplet = (_ZERO6, _ZERO6, _ZERO6)
        elif neighbor == HyperfineNeighbor.SNV117:
            ground_triplet = _SN117_GND
            excited_triplet = _SN117_EXC
        else:
            shell, _, angle = _CARBON_SITE_INFO[neighbor]
            if shell == "first":
                ground_reference = _C13_FIRST_GND_REFERENCE
                excited_reference = _C13_FIRST_EXC_REFERENCE
            else:
                ground_reference = _C13_SECOND_GND_REFERENCE
                excited_reference = _C13_SECOND_EXC_REFERENCE

            ground_triplet = _rotate_carbon_hyperfine_triplet(
                ground_reference,
                angle,
            )
            excited_triplet = _rotate_carbon_hyperfine_triplet(
                excited_reference,
                angle,
            )

        A_gnd[neighbor], Ax_gnd[neighbor], Ay_gnd[neighbor] = ground_triplet
        A_exc[neighbor], Ax_exc[neighbor], Ay_exc[neighbor] = excited_triplet

    return A_gnd, Ax_gnd, Ay_gnd, A_exc, Ax_exc, Ay_exc


(
    A_gnd,
    Ax_gnd,
    Ay_gnd,
    A_exc,
    Ax_exc,
    Ay_exc,
) = _build_hyperfine_vector_dictionaries()


def build_hyperfine_tensors(
    A_gnd: Dict[HyperfineNeighbor, jnp.ndarray],
    Ax_gnd: Dict[HyperfineNeighbor, jnp.ndarray],
    Ay_gnd: Dict[HyperfineNeighbor, jnp.ndarray],
    A_exc: Dict[HyperfineNeighbor, jnp.ndarray],
    Ax_exc: Dict[HyperfineNeighbor, jnp.ndarray],
    Ay_exc: Dict[HyperfineNeighbor, jnp.ndarray],
) -> Tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Convert six hyperfine dictionaries into dense symmetric tensors.

    Each dictionary maps every unique ``HyperfineNeighbor`` ID to a vector
    ordered as ``[Axx, Ayy, Azz, Axy, Axz, Ayz]``.

    Returns
    -------
    A_gnd, Ax_gnd, Ay_gnd, A_exc, Ax_exc, Ay_exc
        Arrays of shape ``(n_neighbors, 3, 3)`` indexed directly by the enum's
        integer value.
    """

    def build_one(
        source: Dict[HyperfineNeighbor, jnp.ndarray],
    ) -> jnp.ndarray:
        tensors = []

        for neighbor in _ORDERED_HYPERFINE_NEIGHBORS:
            if neighbor not in source:
                raise ValueError(
                    "Missing hyperfine tensor for "
                    f"HyperfineNeighbor.{neighbor.name}."
                )

            tensors.append(
                _tensor6_to_symmetric_matrix(source[neighbor])
            )

        return jnp.stack(tensors, axis=0)

    return (
        build_one(A_gnd),
        build_one(Ax_gnd),
        build_one(Ay_gnd),
        build_one(A_exc),
        build_one(Ax_exc),
        build_one(Ay_exc),
    )


(
    A_GND_TENSORS,
    AX_GND_TENSORS,
    AY_GND_TENSORS,
    A_EXC_TENSORS,
    AX_EXC_TENSORS,
    AY_EXC_TENSORS,
) = build_hyperfine_tensors(
    A_gnd,
    Ax_gnd,
    Ay_gnd,
    A_exc,
    Ax_exc,
    Ay_exc,
)
