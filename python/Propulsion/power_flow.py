"""Matrix power flow of Mokotoff & Cinar (2026), Eq. (27), solved in one step.

P holds each component's output power; lambda^d and Pi^d have consumer rows, as
in the paper. ``architecture`` is the n x n binary matrix A, A[i, j] = 1 when
component i supplies component j, as in ``ArchitectureRecord.adjacency`` from ADORE.
"""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from Propulsion.adore_minimal import _is_acyclic

Direction = Literal["upstream", "downstream"]


def efficiency_matrices(
    architecture: ArrayLike, component: ArrayLike, path_efficiency: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Upstream and downstream efficiency matrices, Eqs. (21)-(22).

    ``component`` holds one efficiency per component. ``path_efficiency`` is
    indexed like the architecture; unconnected entries are 1.
    """

    connected = np.asarray(architecture) != 0
    component = np.asarray(component, dtype=float)
    path_efficiency = np.asarray(path_efficiency, dtype=float)
    upstream = np.where(connected, component[np.newaxis, :] * path_efficiency, 1.0)
    downstream = np.where(connected.T, component[:, np.newaxis] * path_efficiency.T, 1.0)
    return upstream, downstream


def step_matrix(
    architecture: ArrayLike,
    split: ArrayLike,
    efficiency: ArrayLike,
    direction: Direction = "downstream",
) -> np.ndarray:
    """The one-connection propagation matrix M of Eq. (27).

    ``split`` and ``efficiency`` are lambda and Pi for ``direction``.
    """

    architecture = np.asarray(architecture, dtype=float)
    connected = architecture + architecture.T
    # Not cast to float, so complex-step derivatives pass through.
    split = np.asarray(split)
    efficiency = np.asarray(efficiency, dtype=float)
    # A vector split would otherwise broadcast silently.
    if not split.shape == efficiency.shape == architecture.shape:
        raise ValueError("split and efficiency must match the architecture")
    if direction == "upstream":
        return (split * connected * efficiency).T
    if direction == "downstream":
        return (split * connected / efficiency).T
    raise ValueError("direction must be 'upstream' or 'downstream'")


def power_flow(
    architecture: ArrayLike,
    split: ArrayLike,
    efficiency: ArrayLike,
    power: ArrayLike,
    direction: Direction = "downstream",
) -> np.ndarray:
    """Output power of every component once Eq. (27) has fully propagated.

    ``power`` is P0: demands at sinks (downstream) or supplies at sources
    (upstream), which Algorithm 1 would overwrite if they also received power.
    """

    step = step_matrix(architecture, split, efficiency, direction)
    receives = step != 0
    if not _is_acyclic(range(len(step)), [tuple(pair) for pair in np.argwhere(receives)]):
        raise ValueError("power flow contains a cycle; Algorithm 1 requires an acyclic flow")
    if np.any((np.asarray(power) != 0) & receives.any(axis=1)):
        raise ValueError("prescribed power at a component that also receives power")
    # On an acyclic flow, Algorithm 1's repeated P <- M P converges to this solution.
    return np.linalg.solve(np.eye(len(step)) - step, power)
