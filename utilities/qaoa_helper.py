# SPDX-License-Identifier: Apache-2.0 AND CC-BY-NC-4.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small, self-contained QAOA utilities used by the hybrid-workflows lesson.

Cost-layer convention: each edge contributes ``CNOT, rz(2 * gamma * weight),
CNOT``, which implements ``exp(-i * gamma * weight * Z_u Z_v)``. The mixer is
``rx(2 * beta)`` on every qubit. Any kernel written in the notebook that reuses
parameters returned by :func:`optimize_maxcut_qaoa` must build its cost layer
the same way, otherwise the parameters describe a different circuit.
"""

from collections.abc import Sequence

import cudaq
from cudaq import spin
import networkx as nx
import numpy as np
from scipy.optimize import minimize


def get_maxcut_hamiltonian(graph: nx.Graph) -> cudaq.SpinOperator:
    """Return a weighted Max-Cut cost Hamiltonian for a zero-indexed graph."""
    if graph.number_of_edges() == 0:
        raise ValueError("graph must contain at least one edge.")

    hamiltonian = 0
    for source, target, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        hamiltonian += 0.5 * weight * (
            spin.z(source) * spin.z(target)
            - spin.i(source) * spin.i(target)
        )
    return hamiltonian


def maxcut_problem(graph: nx.Graph):
    """Return the qubit count, edge arrays, and cost Hamiltonian for a graph.

    Nodes are mapped to qubit indices in sorted order, so the edge arrays and
    the Hamiltonian always share one qubit numbering. Pass the edge arrays to a
    QAOA kernel and the Hamiltonian to ``cudaq.observe``.
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("graph must contain at least one node.")
    if graph.number_of_edges() == 0:
        raise ValueError("graph must contain at least one edge.")

    ordered_nodes = sorted(graph.nodes())
    node_to_qubit = {node: index for index, node in enumerate(ordered_nodes)}
    mapped_graph = nx.relabel_nodes(graph, node_to_qubit, copy=True)

    edge_sources = []
    edge_targets = []
    edge_weights = []
    for source, target, data in mapped_graph.edges(data=True):
        edge_sources.append(source)
        edge_targets.append(target)
        edge_weights.append(float(data.get("weight", 1.0)))

    return (
        mapped_graph.number_of_nodes(),
        edge_sources,
        edge_targets,
        edge_weights,
        get_maxcut_hamiltonian(mapped_graph),
    )


@cudaq.kernel
def _qaoa_kernel(
    qubit_count: int,
    layer_count: int,
    edge_sources: list[int],
    edge_targets: list[int],
    edge_weights: list[float],
    parameters: list[float],
):
    qubits = cudaq.qvector(qubit_count)
    h(qubits)

    for layer in range(layer_count):
        gamma = parameters[layer]
        beta = parameters[layer + layer_count]

        for edge in range(len(edge_sources)):
            source = edge_sources[edge]
            target = edge_targets[edge]
            x.ctrl(qubits[source], qubits[target])
            rz(2.0 * gamma * edge_weights[edge], qubits[target])
            x.ctrl(qubits[source], qubits[target])

        for qubit in range(qubit_count):
            rx(2.0 * beta, qubits[qubit])


def optimize_maxcut_qaoa(
    graph: nx.Graph,
    layer_count: int,
    initial_parameters: Sequence[float],
    *,
    optimizer: str = "cobyla",
    shots_count: int = 10_000,
):
    """Optimize and sample a weighted Max-Cut QAOA circuit.

    ``initial_parameters`` holds all gamma values followed by all beta values,
    so a depth-p circuit takes ``[gamma_0, ..., gamma_p-1, beta_0, ..., beta_p-1]``.

    Returns the optimal expectation value, optimal parameters, and sample
    counts used by the lesson.
    """
    if optimizer.lower() != "cobyla":
        raise ValueError("This lesson helper currently supports only COBYLA.")
    if layer_count < 1:
        raise ValueError("layer_count must be at least 1.")

    expected_parameters = 2 * layer_count
    parameters = np.asarray(initial_parameters, dtype=float)
    if parameters.size != expected_parameters:
        raise ValueError(
            f"Expected {expected_parameters} parameters, got {parameters.size}."
        )

    qubit_count, edge_sources, edge_targets, edge_weights, hamiltonian = (
        maxcut_problem(graph)
    )
    kernel_arguments = (
        qubit_count,
        layer_count,
        edge_sources,
        edge_targets,
        edge_weights,
    )

    def objective(candidate_parameters):
        return cudaq.observe(
            _qaoa_kernel,
            hamiltonian,
            *kernel_arguments,
            candidate_parameters.tolist(),
        ).expectation()

    result = minimize(objective, parameters, method="COBYLA")
    optimal_parameters = np.asarray(result.x)
    counts = cudaq.sample(
        _qaoa_kernel,
        *kernel_arguments,
        optimal_parameters.tolist(),
        shots_count=shots_count,
    )

    return float(result.fun), optimal_parameters, counts
