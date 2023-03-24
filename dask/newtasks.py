from __future__ import annotations

from typing import Any, Callable

from distributed.utils_comm import WrappedKey

from dask.core import reverse_dict


def toposort(dsk, known=None):
    """_summary_

    Parameters
    ----------
    dsk :
        The dask graph
    known : _type_, optional
        Keys external to dsk (e.g. dependencies)

    Returns
    -------
    _type_
        _description_
    """
    if not known:
        known = set()
    known = set(known)

    ordered = []
    dependencies = dict()
    stack = []
    for ts in dsk.values():
        d = set(ts.dependencies) - known
        if not d:
            stack.append(ts.key)
        else:
            dependencies[ts.key] = d
    dependents = reverse_dict(dependencies)
    while stack:
        key = stack.pop()
        ts = dsk[key]
        ordered.append(ts)
        for dts in dependents.pop(ts.key):
            dependencies[dts].remove(ts.key)
            if not (dependencies[dts] - known):
                stack.append(dts)
                del dependencies[dts]
    assert not dependencies
    assert not dependents
    return ordered


def get_from_tasks(dsk, out, cache=None):
    if cache is None:
        cache = {}
    for task in toposort(dsk, cache):
        cache[task.key] = task(*(cache[d] for d in task.dependencies))
    if isinstance(out, (list, tuple)):
        return type(out)([cache[k] for k in out])
    else:
        return cache[out]


from dask.optimization import SubgraphCallable


class NewSubgraphCallable(SubgraphCallable):
    @property
    def dependencies(self):
        return self.inkeys

    @property
    def key(self):
        return self.outkey

    def __call__(self, *args):
        if not len(args) == len(self.inkeys):
            raise ValueError("Expected %d args, got %d" % (len(self.inkeys), len(args)))
        return get_from_tasks(self.dsk, self.outkey, dict(zip(self.inkeys, args)))


class Key(str):
    @property
    def key(self):
        return str(self)


_RemotePlaceholder = object()


class Task:
    func: Callable
    key: Key
    dependencies: list[Key | WrappedKey]
    dependency_ix: list[int]
    argspec: list[Any]

    __slots__ = tuple(__annotations__)

    def __eq__(self, other):
        if not type(other) == type(self):
            return False
        return (
            other.func == self.func
            and self.key == other.key
            and self.dependencies == other.dependencies
            and self.argspec == other.argspec
        )

    def __init__(self, key: str | Key, func, args: list[Any]):
        self.key = Key(key)
        self.func = func
        self.dependencies = []
        self.dependency_ix = []
        parsed_args: list[Any] = [None] * len(args)
        for ix, arg in enumerate(args):
            if isinstance(arg, (Key, WrappedKey)):
                self.dependency_ix.append(ix)
                self.dependencies.append(arg.key)
                parsed_args[ix] = _RemotePlaceholder
            else:
                parsed_args[ix] = arg
        self.argspec = parsed_args

    def __call__(self, *args):
        # resolve the input and match it to the dependencies
        argspec = self.argspec.copy()
        if args:
            assert len(args) == len(self.dependencies)
            matched_keys = dict(zip(self.dependency_ix, args))
            for ix, value in matched_keys.items():
                argspec[ix] = value
        return self.func(*argspec)


counter = 0
_name = "_anon"

from distributed.worker import istask

from dask.core import get_deps
from dask.optimization import SubgraphCallable


def _convert_tuple_to_task(key, task, dependencies, dependents):
    if isinstance(task, Task):
        return task
    assert istask(task)
    dsk = {}
    args = []
    func = task[0]
    args_orig = task[1:]
    for it in args_orig:
        if istask(it):
            global counter
            anon_key = _name + "-" + str(counter)
            counter += 1
            dsk[anon_key] = _convert_tuple_to_task(
                anon_key, it, dependencies, dependents
            )
            args.append(Key(anon_key))
        else:
            if it in dependencies:
                it = Key(it)
            args.append(it)
    task = Task(key, func, args)
    if dsk:
        dsk[key] = task
        return NewSubgraphCallable(
            dsk=dsk,
            outkey=key,
            inkeys=dependencies[key],
        )
    else:
        return task


def convert_graph(dsk):
    new_dsk = {}
    dependencies, dependents = get_deps(dsk)
    for key, task in dsk.items():
        new_dsk[key] = _convert_tuple_to_task(
            key, task, dependencies=dependencies, dependents=dependents
        )
    return new_dsk
