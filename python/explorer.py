"""The page's solve: Propulsion/power_flow.py, copied from the research code."""

import json

import numpy as np

from Propulsion.power_flow import efficiency_matrices, power_flow, step_matrix


def solve(problem: str) -> str:
    """Component powers and the power on every connection, as JSON.

    ``problem`` holds A (supplier rows), lambda^d (consumer rows), one efficiency
    per component, the path efficiencies (indexed like A), and P0.
    """

    data = json.loads(problem)
    architecture = np.array(data["architecture"])
    split = np.array(data["split"], dtype=float)
    _, pi_d = efficiency_matrices(architecture, data["efficiency"], data["path"])
    try:
        power = power_flow(architecture, split, pi_d, data["power"])
    except (ValueError, np.linalg.LinAlgError) as error:
        return json.dumps({"error": str(error)})
    # The M P terms: what each supplier sends each consumer.
    flow = step_matrix(architecture, split, pi_d) * power * (architecture != 0)
    return json.dumps({"power": power.tolist(), "flow": flow.tolist()})
