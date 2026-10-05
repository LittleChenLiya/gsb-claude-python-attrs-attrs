# SPDX-License-Identifier: MIT

"""
Explicit copy plans for *attrs* classes.

A `CopyPlan` describes, per field and per type, how values are copied when
duplicating an instance:

- ``CopyStrategy.SHARED``: the value is reused as-is (aliased).
- ``CopyStrategy.SHALLOW``: the value is copied with `copy.copy`.
- ``CopyStrategy.RECURSIVE``: the value is copied recursively.

Recursive copying keeps a memo of already-copied objects, so shared
sub-objects stay shared in the copy and cyclic object graphs are copied
without infinite loops.

Run the dedicated tests with::

    python -m pytest tests/test_copy_plan.py -v
"""

import copy as _stdlib_copy

from enum import Enum

from ._funcs import _ATOMIC_TYPES, has
from ._make import _OBJ_SETATTR, fields
from .exceptions import CopyPlanError, UnsupportedCopyTypeError


__all__ = [
    "REJECT",
    "CopyPlan",
    "CopyStrategy",
    "copy_with_plan",
]


class CopyStrategy(Enum):
    """
    How a field (or a type) is copied by a `CopyPlan`.

    .. versionadded:: 25.4.0
    """

    SHARED = "shared"
    SHALLOW = "shallow"
    RECURSIVE = "recursive"


class _Reject:
    """
    Sentinel for `CopyPlan` policies that explicitly reject a type or field.
    """

    __slots__ = ()

    def __repr__(self):
        return "REJECT"


#: Policy sentinel: explicitly refuse to copy matching values.
REJECT = _Reject()


def _read_field(inst, name):
    """
    Read the stored value of field *name* without invoking user-defined
    getters (e.g. properties shadowing the attribute on the class).
    """
    inst_dict = getattr(inst, "__dict__", None)
    if inst_dict is not None:
        try:
            return inst_dict[name]
        except KeyError:
            pass
    # Slotted classes store values in slot descriptors, which are storage
    # accessors, not user getters.
    return getattr(inst, name)


def _write_field(inst, name, value):
    """
    Store *value* for field *name* without invoking user-defined setters,
    ``__setattr__`` hooks, or frozen guards.
    """
    # If any class in the MRO stores *name* in a slot, write through
    # object.__setattr__ so the value lands in the slot. Otherwise write
    # the instance dict directly, bypassing data descriptors (properties).
    for klass in type(inst).__mro__:
        slots = getattr(klass, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        if name in slots:
            _OBJ_SETATTR(inst, name, value)
            return
    inst_dict = getattr(inst, "__dict__", None)
    if inst_dict is not None:
        inst_dict[name] = value
    else:
        _OBJ_SETATTR(inst, name, value)


class CopyPlan:
    """
    An explicit plan for copying *attrs* instances.

    Args:
        default (CopyStrategy):
            Strategy applied to fields and values without a more specific
            registration. Defaults to `CopyStrategy.RECURSIVE`.

        fields (dict[str, CopyStrategy] | None):
            Mapping of field names to strategies. Applies to matching field
            names on every *attrs* class encountered while copying.

        types (dict[type, CopyStrategy | REJECT | Callable] | None):
            Mapping of types to policies. A policy is a `CopyStrategy`,
            `REJECT` to explicitly refuse copying values of that type, or a
            callable taking the value and returning its copy.

    .. versionadded:: 25.4.0
    """

    #: Policy sentinel: explicitly refuse to copy matching values.
    REJECT = REJECT

    __slots__ = ("default", "field_strategies", "type_policies")

    def __init__(
        self, default=CopyStrategy.RECURSIVE, *, fields=None, types=None
    ):
        if not isinstance(default, CopyStrategy):
            msg = f"default must be a CopyStrategy, got {default!r}."
            raise TypeError(msg)
        self.default = default
        self.field_strategies = dict(fields) if fields else {}
        self.type_policies = dict(types) if types else {}

    def field(self, name, strategy):
        """
        Register *strategy* for the field called *name*. Returns ``self`` so
        calls can be chained.
        """
        if not isinstance(strategy, CopyStrategy) and strategy is not REJECT:
            msg = (
                f"strategy must be a CopyStrategy or REJECT, got {strategy!r}."
            )
            raise TypeError(msg)
        self.field_strategies[name] = strategy
        return self

    def on_type(self, type_, policy):
        """
        Register *policy* for values of type *type_*. *policy* is a
        `CopyStrategy`, `REJECT`, or a callable ``copier(value) -> copy``.
        Returns ``self`` so calls can be chained.
        """
        if (
            not isinstance(policy, CopyStrategy)
            and policy is not REJECT
            and not callable(policy)
        ):
            msg = (
                "policy must be a CopyStrategy, REJECT, or a callable, "
                f"got {policy!r}."
            )
            raise TypeError(msg)
        self.type_policies[type_] = policy
        return self

    def copy(self, inst, **changes):
        """
        Copy *inst* according to this plan, applying *changes* (keyed by
        field alias or name) on top. See `copy_with_plan` for details.
        """
        return copy_with_plan(inst, self, **changes)

    def _policy_for(self, value):
        """
        Return the type policy registered for the type of *value*, or None.
        """
        # Exact-type lookup first, then registered base classes.
        policy = self.type_policies.get(type(value))
        if policy is not None:
            return policy
        for type_, policy in self.type_policies.items():
            if isinstance(value, type_):
                return policy
        return None


def copy_with_plan(inst, plan=None, **changes):
    """
    Create a copy of *inst* following an explicit `CopyPlan`.

    Unlike `attrs.evolve`, the new instance is assembled field-by-field from
    the stored values of *inst*:

    - ``__init__`` is not called, so `Factory` defaults do not run again,
      validators/converters are not re-applied, and ``init=False`` fields
      are copied too;
    - stored values are read directly, so user-defined getters (properties)
      are not called;
    - frozen classes stay frozen;
    - internal sharing is preserved: if two fields point at the same
      object, the copied fields point at the same (single) copy;
    - cyclic object graphs are copied without infinite loops.

    Before anything is copied, the reachable object graph is validated:
    values whose type is not supported by the plan (and not covered by a
    type policy) raise `attrs.exceptions.UnsupportedCopyTypeError` naming
    the full field path. If anything fails, no (partial) copy is produced
    and *inst* is left untouched.

    Args:
        inst: Instance of an *attrs*-decorated class.

        plan (CopyPlan | None):
            The plan to follow. Defaults to a fully recursive plan.

        changes:
            New values, keyed by field alias (like `attrs.evolve`) or field
            name. Used as-is (shared).

    Returns:
        A new instance of the class of *inst*.

    Raises:
        attrs.exceptions.NotAnAttrsClassError:
            If *inst* is not an *attrs* instance.

        attrs.exceptions.UnsupportedCopyTypeError:
            If a value cannot be copied under *plan*. Carries the field
            ``path``.

    .. versionadded:: 25.4.0
    """
    if plan is None:
        plan = CopyPlan()

    cls = inst.__class__
    attrs = fields(cls)  # raises NotAnAttrsClassError for non-attrs classes

    # Resolve changes against aliases (like evolve) and plain field names.
    overrides = {}
    for a in attrs:
        if a.alias in changes:
            overrides[a.name] = changes.pop(a.alias)
        elif a.name in changes:
            overrides[a.name] = changes.pop(a.name)
    if changes:
        msg = f"Unexpected changes for {cls.__qualname__}: {sorted(changes)}"
        raise CopyPlanError(msg)

    # Phase 1: validate the whole reachable graph before copying anything,
    # so unsupported values are reported (with their field path) before
    # execution and no half-built object is ever produced.
    _validate_value(plan, inst, plan.default, cls.__name__, set())

    # Phase 2: execute. The memo preserves internal sharing and makes
    # cycles terminate.
    memo = {}
    return _execute_value(
        plan, inst, plan.default, memo, cls.__name__, overrides
    )


def _validate_value(plan, value, strategy, path, seen):
    """
    Ensure *value* can be copied under *strategy*; raise before any copying
    happens otherwise. *seen* guards against infinite loops on cycles.
    """
    policy = plan._policy_for(value)
    if policy is REJECT:
        raise UnsupportedCopyTypeError(path, type(value), rejected=True)
    if callable(policy):
        return  # A custom copier takes responsibility for the value.
    if policy is not None:
        strategy = policy

    if strategy is REJECT:
        raise UnsupportedCopyTypeError(path, type(value), rejected=True)
    if strategy in (CopyStrategy.SHARED, CopyStrategy.SHALLOW):
        return

    # Recursive strategy from here on.
    if type(value) in _ATOMIC_TYPES:
        return

    oid = id(value)
    if oid in seen:
        return
    seen.add(oid)

    value_type = type(value)
    if has(value_type):
        for a in fields(value_type):
            field_strategy = plan.field_strategies.get(a.name, plan.default)
            _validate_value(
                plan,
                _read_field(value, a.name),
                field_strategy,
                f"{path}.{a.name}",
                seen,
            )
    elif value_type in (list, set, frozenset, tuple):
        for i, item in enumerate(value):
            _validate_value(plan, item, plan.default, f"{path}[{i}]", seen)
    elif value_type is dict:
        for k, v in value.items():
            _validate_value(plan, k, plan.default, f"{path}.keys()", seen)
            _validate_value(plan, v, plan.default, f"{path}[{k!r}]", seen)
    else:
        raise UnsupportedCopyTypeError(path, value_type)


def _execute_value(plan, value, strategy, memo, path, overrides=None):  # noqa: PLR0911
    """
    Copy *value* under *strategy*, using *memo* to preserve sharing and to
    terminate cycles. Only called after validation succeeded.
    """
    policy = plan._policy_for(value)
    if callable(policy):
        return policy(value)
    if policy is not None and policy is not REJECT:
        strategy = policy

    if strategy is CopyStrategy.SHARED:
        return value
    if strategy is CopyStrategy.SHALLOW:
        return _stdlib_copy.copy(value)

    # Recursive strategy from here on.
    if type(value) in _ATOMIC_TYPES:
        return value

    oid = id(value)
    if oid in memo:
        return memo[oid]

    value_type = type(value)
    if has(value_type):
        # Build the instance without calling __init__: factories must not
        # run again, and frozen/slotted classes are handled uniformly by
        # writing storage directly.
        new = value_type.__new__(value_type)
        memo[oid] = new  # pre-register so cycles back to us terminate
        for a in fields(value_type):
            if overrides is not None and a.name in overrides:
                copied = overrides[a.name]
            else:
                field_strategy = plan.field_strategies.get(
                    a.name, plan.default
                )
                copied = _execute_value(
                    plan,
                    _read_field(value, a.name),
                    field_strategy,
                    memo,
                    f"{path}.{a.name}",
                )
            _write_field(new, a.name, copied)
        return new
    if value_type is list:
        new = []
        memo[oid] = new
        new.extend(
            _execute_value(plan, item, plan.default, memo, f"{path}[{i}]")
            for i, item in enumerate(value)
        )
        return new
    if value_type is set:
        new = set()
        memo[oid] = new
        for item in value:
            new.add(_execute_value(plan, item, plan.default, memo, path))
        return new
    if value_type is dict:
        new = {}
        memo[oid] = new
        for k, v in value.items():
            new_key = _execute_value(plan, k, plan.default, memo, path)
            new[new_key] = _execute_value(
                plan, v, plan.default, memo, f"{path}[{k!r}]"
            )
        return new
    if value_type is frozenset:
        new = frozenset(
            _execute_value(plan, item, plan.default, memo, path)
            for item in value
        )
        memo[oid] = new
        return new
    if value_type is tuple:
        new = tuple(
            _execute_value(plan, item, plan.default, memo, f"{path}[{i}]")
            for i, item in enumerate(value)
        )
        memo[oid] = new
        return new

    # Unreachable after validation, but never deliver a half-built object.
    raise UnsupportedCopyTypeError(path, value_type)  # pragma: no cover
