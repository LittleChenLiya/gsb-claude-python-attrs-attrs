# SPDX-License-Identifier: MIT

"""
Explicit copy plans for attrs instances.

:class:`CopyPlan` lets you control how each field of an attrs instance is
copied: shared, shallow-copied, or recursively deep-copied.  Internal sharing
and cycles are preserved by the deep copier.
"""

from __future__ import annotations

import copy as _copy

from typing import Any

from ._funcs import has
from ._make import _OBJ_SETATTR, fields
from .exceptions import CopyError, NotAnAttrsClassError


# Field / type strategies.
SHARE = "share"
SHALLOW = "shallow"
DEEP = "deep"
REJECT = "reject"

_TYPE_STRATEGIES = frozenset({SHARE, SHALLOW, DEEP, REJECT})

# Types that the deep copier treats as immutable leaves (shared, not copied).
_IMMUTABLE_TYPES = frozenset(
    {
        type(None),
        bool,
        int,
        float,
        str,
        complex,
        bytes,
        type(...),
        type,
        range,
        property,
    }
)


def _read_field(inst: Any, name: str) -> Any:
    """
    Read a field value without invoking user getters.

    Reads directly from ``__dict__`` for non-slotted classes (bypassing
    properties and ``__getattribute__`` overrides), and falls back to
    ``object.__getattribute__`` for slotted fields (whose descriptors are
    C-level, not user getters).
    """
    try:
        d = object.__getattribute__(inst, "__dict__")
    except AttributeError:
        d = None
    if d is not None and name in d:
        return d[name]
    return object.__getattribute__(inst, name)


def _format_path(path: list[str]) -> str:
    """
    Format a path list like ``["a", "b", "[0]", "c"]`` as ``"a.b[0].c"``.
    """
    if not path:
        return "<root>"
    result = path[0]
    for part in path[1:]:
        if part.startswith("["):
            result += part
        else:
            result += "." + part
    return result


class CopyPlan:
    """
    An explicit copy plan for attrs instances.

    Fields are copied according to the strategy registered for them:

    - ``"share"``: the new field shares the same object as the source (the
      default).
    - ``"shallow"``: the new field is a shallow copy (`copy.copy`).
    - ``"deep"``: the new field is a recursive deep copy that preserves
      internal sharing and handles cycles.

    Types can be registered with a strategy or a custom copier callable via
    :meth:`register_type`.  When the deep copier encounters an object whose
    type has a registered strategy, that strategy is applied.  Registering a
    type with ``"reject"`` causes the deep copier to raise
    :class:`~attrs.exceptions.CopyError` when it encounters an object of that
    type.

    Args:
        default: The default strategy for fields that have no registered
            strategy.  One of ``"share"``, ``"shallow"``, ``"deep"``.

    .. versionadded:: 26.1.0
    """

    __slots__ = ("_default", "_field_strategies", "_type_strategies")

    def __init__(self, default: str = SHARE) -> None:
        if default not in (SHARE, SHALLOW, DEEP):
            msg = (
                f"default must be one of {SHARE!r}, {SHALLOW!r}, {DEEP!r}, "
                f"not {default!r}"
            )
            raise ValueError(msg)
        self._field_strategies: dict[str, str] = {}
        self._type_strategies: dict[type, Any] = {}
        self._default = default

    def _set_field_strategy(self, field: str, strategy: str) -> None:
        if strategy not in (SHARE, SHALLOW, DEEP):
            msg = (
                f"strategy must be one of {SHARE!r}, {SHALLOW!r}, {DEEP!r}, "
                f"not {strategy!r}"
            )
            raise ValueError(msg)
        self._field_strategies[field] = strategy

    def share(self, *fields: str) -> CopyPlan:
        """
        Register *fields* to be shared.

        The new field references the same object as the source field.
        """
        for f in fields:
            self._set_field_strategy(f, SHARE)
        return self

    def shallow(self, *fields: str) -> CopyPlan:
        """
        Register *fields* to be shallow-copied (`copy.copy`).
        """
        for f in fields:
            self._set_field_strategy(f, SHALLOW)
        return self

    def deep(self, *fields: str) -> CopyPlan:
        """
        Register *fields* to be recursively deep-copied.

        The deep copier preserves internal sharing (two fields pointing to
        the same object still point to the same object after copying) and
        handles cycles without infinite recursion.
        """
        for f in fields:
            self._set_field_strategy(f, DEEP)
        return self

    def register_type(self, tp: type, strategy: Any) -> CopyPlan:
        """
        Register a strategy for type *tp*.

        *strategy* is one of ``"share"``, ``"shallow"``, ``"deep"``,
        ``"reject"``, or a callable ``(obj, copy, path) -> new_obj``.

        A ``"reject"`` strategy causes the deep copier to raise
        :class:`~attrs.exceptions.CopyError` when it encounters an object of
        type *tp*.

        The callable receives the object, a *copy* function that deep-copies
        an object using this plan, and the current path.
        """
        if isinstance(strategy, str):
            if strategy not in _TYPE_STRATEGIES:
                msg = (
                    f"strategy must be one of "
                    f"{', '.join(sorted(_TYPE_STRATEGIES))}, not {strategy!r}"
                )
                raise ValueError(msg)
        elif not callable(strategy):
            msg = "strategy must be a string or callable"
            raise TypeError(msg)
        self._type_strategies[tp] = strategy
        return self

    def copy(self, inst: Any) -> Any:
        """
        Return a copy of *inst* according to this plan.

        Raises:
            attrs.exceptions.NotAnAttrsClassError: If *inst* is not an attrs
                instance.
            attrs.exceptions.CopyError: If the deep copier encounters an
                unsupported or rejected object.
        """
        return copy(inst, self)


def copy(inst: Any, plan: CopyPlan | None = None) -> Any:
    """
    Return a copy of *inst* according to *plan*.

    If *plan* is ``None``, a default :class:`CopyPlan` is used (all fields
    shared).

    Raises:
        attrs.exceptions.NotAnAttrsClassError: If *inst* is not an attrs
            instance.
        attrs.exceptions.CopyError: If the deep copier encounters an
            unsupported or rejected object.

    .. versionadded:: 26.1.0
    """
    if not has(type(inst)):
        msg = f"{inst!r} is not an attrs-decorated class."
        raise NotAnAttrsClassError(msg)
    if plan is None:
        plan = CopyPlan()
    return _Copier(plan).copy(inst)


class _Copier:
    """
    Two-phase copier: validate the graph first, then copy.

    The validation phase walks the whole graph and raises `CopyError` before
    any copying takes place, so a failure never delivers a half-built object.
    The copy phase walks the graph again with a memo that preserves internal
    sharing and handles cycles.
    """

    __slots__ = ("_memo", "_plan", "_validated")

    def __init__(self, plan: CopyPlan) -> None:
        self._plan = plan
        self._memo: dict[int, Any] = {}
        self._validated: set[int] = set()

    def copy(self, inst: Any) -> Any:
        self._validate(inst, [])
        return self._copy(inst, [])

    # -- validation -------------------------------------------------------

    def _validate(self, obj: Any, path: list[str]) -> None:
        obj_id = id(obj)
        if obj_id in self._validated:
            return
        self._validated.add(obj_id)

        strategy = self._lookup_type_strategy(type(obj))
        if strategy == REJECT:
            msg = (
                f"Copying of {type(obj).__name__!r} is rejected by type "
                f"strategy at {_format_path(path)}."
            )
            raise CopyError(msg)
        if callable(strategy) or strategy in (SHARE, SHALLOW):
            return

        # strategy == DEEP or None -> deep-copy behavior
        self._validate_deep(obj, path)

    def _validate_deep(self, obj: Any, path: list[str]) -> None:
        obj_type = type(obj)
        if has(obj_type):
            for a in fields(obj_type):
                value = _read_field(obj, a.name)
                self._validate(value, [*path, a.name])
        elif isinstance(obj, (list, tuple, set, frozenset)):
            for i, item in enumerate(obj):
                self._validate(item, [*path, f"[{i}]"])
        elif isinstance(obj, dict):
            for k, v in obj.items():
                key_path = [*path, f"[{k!r}]"]
                self._validate(k, key_path)
                self._validate(v, key_path)
        elif obj_type in _IMMUTABLE_TYPES:
            return
        else:
            msg = (
                f"Cannot copy object of type {obj_type.__name__!r} at "
                f"{_format_path(path)}: no deep-copy strategy registered "
                f"for this type."
            )
            raise CopyError(msg)

    # -- copying ----------------------------------------------------------

    def _copy(self, obj: Any, path: list[str]) -> Any:
        obj_id = id(obj)
        if obj_id in self._memo:
            return self._memo[obj_id]

        strategy = self._lookup_type_strategy(type(obj))
        if strategy == REJECT:
            msg = (
                f"Copying of {type(obj).__name__!r} is rejected by type "
                f"strategy at {_format_path(path)}."
            )
            raise CopyError(msg)
        if callable(strategy):

            def copy_func(x: Any, subpath: list[str] | None = None) -> Any:
                return self._copy(x, subpath if subpath is not None else path)

            new_obj = strategy(obj, copy_func, path)
            self._memo[obj_id] = new_obj
            return new_obj
        if strategy == SHARE:
            return obj
        if strategy == SHALLOW:
            new_obj = _copy.copy(obj)
            self._memo[obj_id] = new_obj
            return new_obj

        # strategy == DEEP or None -> deep-copy behavior
        return self._copy_deep(obj, path, obj_id)

    def _copy_deep(self, obj: Any, path: list[str], obj_id: int) -> Any:  # noqa: PLR0911
        obj_type = type(obj)
        if has(obj_type):
            new_obj = object.__new__(obj_type)
            self._memo[obj_id] = new_obj  # before fields, to handle cycles
            for a in fields(obj_type):
                value = _read_field(obj, a.name)
                field_strategy = self._plan._field_strategies.get(
                    a.name, self._plan._default
                )
                if field_strategy == SHARE:
                    new_value = value
                elif field_strategy == SHALLOW:
                    new_value = _copy.copy(value)
                else:  # DEEP
                    new_value = self._copy(value, [*path, a.name])
                _OBJ_SETATTR(new_obj, a.name, new_value)
            return new_obj
        if isinstance(obj, list):
            new_obj = []
            self._memo[obj_id] = new_obj
            for i, item in enumerate(obj):
                new_obj.append(self._copy(item, [*path, f"[{i}]"]))
            return new_obj
        if isinstance(obj, tuple):
            new_items = [
                self._copy(item, [*path, f"[{i}]"])
                for i, item in enumerate(obj)
            ]
            new_obj = tuple(new_items)
            self._memo[obj_id] = new_obj
            return new_obj
        if isinstance(obj, set):
            new_obj = set()
            self._memo[obj_id] = new_obj
            for item in obj:
                new_obj.add(self._copy(item, [*path, "[<set element>]"]))
            return new_obj
        if isinstance(obj, frozenset):
            new_items = [
                self._copy(item, [*path, "[<set element>]"]) for item in obj
            ]
            new_obj = frozenset(new_items)
            self._memo[obj_id] = new_obj
            return new_obj
        if isinstance(obj, dict):
            new_obj = {}
            self._memo[obj_id] = new_obj
            for k, v in obj.items():
                key_path = [*path, f"[{k!r}]"]
                new_k = self._copy(k, key_path)
                new_v = self._copy(v, key_path)
                new_obj[new_k] = new_v
            return new_obj
        if obj_type in _IMMUTABLE_TYPES:
            return obj
        msg = (
            f"Cannot copy object of type {obj_type.__name__!r} at "
            f"{_format_path(path)}: no deep-copy strategy registered "
            f"for this type."
        )
        raise CopyError(msg)

    # -- helpers ----------------------------------------------------------

    def _lookup_type_strategy(self, tp: type) -> Any:
        """
        Look up the registered strategy for *tp*, walking the MRO.

        Returns ``None`` if no strategy is registered (meaning "use the
        default deep-copy behavior").
        """
        for base in tp.__mro__:
            strategy = self._plan._type_strategies.get(base)
            if strategy is not None:
                return strategy
        return None
