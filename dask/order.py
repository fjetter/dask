from __future__ import annotations

from collections import namedtuple

import dask
from dask.core import get_deps, reverse_dict


def order(dsk, dependencies=None):
    version = dask.config.get("order.version", "current")
    version = version.replace(".", "_")
    import importlib

    mod = importlib.import_module(f"dask.order_{version}")
    return mod.order(dsk, dependencies)


OrderInfo = namedtuple(
    "OrderInfo",
    (
        "order",
        "age",
        "num_data_when_run",
        "num_data_when_released",
        "num_dependencies_freed",
    ),
)


def diagnostics(dsk, o=None, dependencies=None):
    """Simulate runtime metrics as though running tasks one at a time in order.

    These diagnostics can help reveal behaviors of and issues with ``order``.

    Returns a dict of `namedtuple("OrderInfo")` and a list of the number of outputs held over time.

    OrderInfo fields:
    - order : the order in which the node is run.
    - age : how long the output of a node is held.
    - num_data_when_run : the number of outputs held in memory when a node is run.
    - num_data_when_released : the number of outputs held in memory when the output is released.
    - num_dependencies_freed : the number of dependencies freed by running the node.
    """
    if dependencies is None:
        dependencies, dependents = get_deps(dsk)
    else:
        dependents = reverse_dict(dependencies)
    if o is None:
        o = order(dsk, dependencies=dependencies)

    pressure = []
    num_in_memory = 0
    age = {}
    runpressure = {}
    releasepressure = {}
    freed = {}
    num_needed = {key: len(val) for key, val in dependents.items()}
    for i, key in enumerate(sorted(dsk, key=o.__getitem__)):
        pressure.append(num_in_memory)
        runpressure[key] = num_in_memory
        released = 0
        for dep in dependencies[key]:
            num_needed[dep] -= 1
            if num_needed[dep] == 0:
                age[dep] = i - o[dep]
                releasepressure[dep] = num_in_memory
                released += 1
        freed[key] = released
        if dependents[key]:
            num_in_memory -= released - 1
        else:
            age[key] = 0
            releasepressure[key] = num_in_memory
            num_in_memory -= released

    rv = {
        key: OrderInfo(
            val, age[key], runpressure[key], releasepressure[key], freed[key]
        )
        for key, val in o.items()
    }
    return rv, pressure


def ndependencies(dependencies, dependents):
    """Number of total data elements on which this key depends

    For each key we return the number of tasks that must be run for us to run
    this task.

    Examples
    --------
    >>> dsk = {'a': 1, 'b': (inc, 'a'), 'c': (inc, 'b')}
    >>> dependencies, dependents = get_deps(dsk)
    >>> num_dependencies, total_dependencies = ndependencies(dependencies, dependents)
    >>> sorted(total_dependencies.items())
    [('a', 1), ('b', 2), ('c', 3)]

    Returns
    -------
    num_dependencies: Dict[key, int]
    total_dependencies: Dict[key, int]
    """
    num_needed = {k: len(v) for k, v in dependencies.items()}
    num_dependencies = num_needed.copy()
    current = []
    current_pop = current.pop
    current_append = current.append
    result = {k: 1 for k, v in dependencies.items() if not v}
    for key in result:
        for parent in dependents[key]:
            num_needed[parent] -= 1
            if not num_needed[parent]:
                current_append(parent)
    while current:
        key = current_pop()
        result[key] = 1 + sum(result[child] for child in dependencies[key])
        for parent in dependents[key]:
            num_needed[parent] -= 1
            if not num_needed[parent]:
                current_append(parent)
    return num_dependencies, result
