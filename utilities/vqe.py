# SPDX-License-Identifier: Apache-2.0

"""Small Python replacement for :func:`cudaq_solvers.vqe`.

The public signature and three-part return value intentionally match
``cudaq_solvers.vqe`` so existing teaching material only needs an import
change.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence

import cudaq
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime


class ObserveExecutionType(Enum):
    """Why an expectation value was evaluated during optimization."""

    function = 0
    gradient = 1


@dataclass(frozen=True)
class ObserveIteration:
    """One expectation-value evaluation made by :func:`vqe`."""

    parameters: list[float]
    result: Any
    type: ObserveExecutionType


_OPTIMIZERS = {
    "cobyla": cudaq.optimizers.COBYLA,
    "lbfgs": cudaq.optimizers.LBFGS,
}

_GRADIENTS = {
    "parameter_shift": cudaq.gradients.ParameterShift,
    "central_difference": cudaq.gradients.CentralDifference,
    "forward_difference": cudaq.gradients.ForwardDifference,
}


def _observe(kernel: Callable, spin_op: Any, parameters: list[float], shots: int):
    # Calling the ansatz inside an observe execution context is what lets this
    # accept the same lightweight wrappers as CUDA-Q Solvers, for example
    # ``lambda values: ansatz(values[0])``. ``cudaq.observe`` itself only
    # accepts decorated kernels and cannot accept that wrapper directly.
    operator = spin_op.copy().canonicalize()
    context = cudaq_runtime.ExecutionContext(
        "observe", shots if shots > 0 else 0
    )
    context.setSpinOperator(operator)
    context.allowJitEngineCaching = True

    # CUDA-Q 0.15 routes observe executions through an ObservePolicy. Merely
    # entering the ExecutionContext (the 0.14 mechanism below) launches the
    # circuit but leaves the expectation value unset on 0.15, which matters
    # for the lightweight lambda wrappers used throughout the QAOA lessons.
    if hasattr(cudaq_runtime, "ObservePolicy"):
        policy = cudaq_runtime.ObservePolicy(context, "", operator)
        return cudaq_runtime.launch_observe(
            policy, context, lambda: kernel(parameters)
        )

    with context:
        kernel(parameters)

    sample_result = context.result
    expectation = context.getExpectationValue()
    if expectation is None:
        expectation = 0.0
        for term in operator:
            coefficient = term.evaluate_coefficient().real
            if term.is_identity():
                expectation += coefficient
            else:
                expectation += sample_result.expectation(term.term_id) * coefficient
    return cudaq_runtime.ObserveResult(expectation, operator, sample_result)


def _builtin_optimize(
    objective: Callable[[list[float]], float],
    initial_parameters: list[float],
    optimizer_name: str,
    gradient_name: str,
    max_iterations: int,
    tolerance: float,
    function_at: Callable[[list[float]], float],
) -> tuple[float, list[float]]:
    optimizer_type = _OPTIMIZERS.get(optimizer_name)
    if optimizer_type is None:
        raise RuntimeError(f"Invalid optimizer '{optimizer_name}'.")

    if optimizer_name == "cobyla":
        # CUDA-Q Solvers uses PRIMA COBYLA with these exact defaults and
        # bounds. SciPy's COBYLA is also PRIMA-based, so this follows the old
        # solver much more closely than CUDA-Q core's NLopt COBYLA wrapper.
        from scipy.optimize import minimize

        max_evaluations = (
            max_iterations
            if max_iterations >= 0
            else len(initial_parameters) * 200
        )
        result = minimize(
            objective,
            initial_parameters,
            method="COBYLA",
            bounds=[(-math.pi, math.pi)] * len(initial_parameters),
            options={
                "maxiter": max_evaluations,
                "rhobeg": 1.0,
                "tol": 1e-4,
            },
        )
        return float(result.fun), [float(value) for value in result.x]

    optimizer = optimizer_type.from_json(json.dumps({"f_tol": tolerance}))
    optimizer.initial_parameters = initial_parameters
    if max_iterations >= 0:
        optimizer.max_iterations = max_iterations

    if not optimizer.requires_gradients():
        value, parameters = optimizer.optimize(len(initial_parameters), objective)
        return float(value), list(parameters)

    gradient_type = _GRADIENTS.get(gradient_name)
    if gradient_type is None:
        raise RuntimeError(f"Invalid gradient method '{gradient_name}'.")
    gradient = gradient_type()

    def objective_with_gradient(parameters: list[float]):
        parameters = list(parameters)
        value = objective(parameters)
        derivatives = gradient.compute(parameters, function_at, value)
        return value, derivatives

    value, parameters = optimizer.optimize(
        len(initial_parameters), objective_with_gradient
    )
    return float(value), list(parameters)


def _scipy_optimize(
    optimizer: Callable,
    objective: Callable[[list[float]], float],
    initial_parameters: list[float],
    options: dict[str, Any],
) -> tuple[float, list[float]]:
    if getattr(optimizer, "__name__", "") != "minimize":
        raise RuntimeError(
            "Invalid functional optimizer provided (only "
            "scipy.optimize.minimize supported)."
        )

    scipy_options = dict(options.pop("options", {}) or {})
    max_iterations = options.pop("max_iterations", -1)
    if max_iterations >= 0:
        scipy_options.setdefault("maxiter", max_iterations)

    result = optimizer(
        objective,
        initial_parameters,
        options=scipy_options or None,
        **options,
    )
    return float(result.fun), [float(value) for value in result.x]


def vqe(
    kernel: Callable,
    spin_op: Any,
    initial_parameters: Sequence[float],
    **kwargs: Any,
) -> tuple[float, list[float], list[ObserveIteration]]:
    """Execute the Variational Quantum Eigensolver.

    Parameters and return values match ``cudaq_solvers.vqe``. Supported
    keyword arguments are ``shots``, ``max_iterations``, ``verbose``,
    ``optimizer``, ``gradient`` and ``tol``. A SciPy ``minimize`` callable is
    also accepted as the optimizer, with its remaining keyword arguments
    forwarded unchanged.
    """

    parameters = [float(value) for value in initial_parameters]
    shots = int(kwargs.pop("shots", -1))
    verbose = bool(kwargs.pop("verbose", False))
    optimizer = kwargs.pop("optimizer", "cobyla")
    gradient = str(kwargs.pop("gradient", "parameter_shift"))
    max_iterations = int(kwargs.pop("max_iterations", -1))
    has_explicit_tol = "tol" in kwargs
    tol = float(kwargs.pop("tol", 1e-12))
    iteration_data: list[ObserveIteration] = []

    def evaluate(values: list[float], execution_type: ObserveExecutionType):
        values = [float(value) for value in values]
        result = _observe(kernel, spin_op, values, shots)
        iteration_data.append(ObserveIteration(values, result, execution_type))
        value = float(result.expectation())
        if verbose and execution_type is ObserveExecutionType.function:
            print(f"<H> = {value:.12f}")
        return value

    def objective(values: list[float]):
        return evaluate(values, ObserveExecutionType.function)

    if callable(optimizer):
        if getattr(optimizer, "__name__", "") != "minimize":
            raise RuntimeError(
                "Invalid functional optimizer provided (only "
                "scipy.optimize.minimize supported)."
            )
    elif str(optimizer) not in _OPTIMIZERS:
        raise RuntimeError(f"Invalid optimizer '{optimizer}'.")
    elif str(optimizer) == "lbfgs" and gradient not in _GRADIENTS:
        raise RuntimeError(f"Invalid gradient method '{gradient}'.")

    if not parameters:
        return objective([]), [], iteration_data

    if callable(optimizer):
        scipy_kwargs = dict(kwargs)
        if has_explicit_tol:
            scipy_kwargs.setdefault("tol", tol)
        energy, optimal_parameters = _scipy_optimize(
            optimizer, objective, parameters, scipy_kwargs
        )
    else:
        def gradient_evaluation(values: list[float]):
            return evaluate(values, ObserveExecutionType.gradient)

        energy, optimal_parameters = _builtin_optimize(
            objective,
            parameters,
            str(optimizer),
            gradient,
            max_iterations,
            tol,
            gradient_evaluation,
        )

    return energy, optimal_parameters, iteration_data


__all__ = [
    "ObserveExecutionType",
    "ObserveIteration",
    "vqe",
]
