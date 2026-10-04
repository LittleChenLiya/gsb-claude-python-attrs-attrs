# SPDX-License-Identifier: MIT

"""
Transaction-style in-place batch updates of *attrs* instances.
"""

import copy

from . import setters
from ._compat import _lazy_is_generator
from ._funcs import has
from ._make import _OBJ_SETATTR, _frozen_setattrs, fields, validate
from .exceptions import FrozenInstanceError, NotAnAttrsClassError


def batch_update(inst, **changes):
    """
    Update multiple attributes of *inst* in one transaction-style operation.

    Unlike a series of plain assignments, the new values are first applied to
    an isolated candidate copy of *inst*: each changed attribute's converter
    runs exactly once on the candidate and afterwards **all** validators run
    against the fully updated candidate. Validators therefore never observe a
    half-updated ("intermediate") state, which allows multiple interdependent
    fields to change together.

    Only if conversion and validation pass, the original instance is updated.
    If anything fails, *inst* keeps its original field values -- including its
    ``__slots__`` state -- and the exception propagates.

    In the commit phase, only attributes whose values actually changed are
    written, in field declaration order, and each changed attribute's
    ``on_setattr`` hook runs exactly once. `attrs.setters.convert` and
    `attrs.setters.validate` are *not* re-run at commit time, because they
    already ran on the candidate.

    Args:
        inst:
            Instance of a *mutable* *attrs* class. It is updated in place.

        changes:
            Keyword changes. Following the same public update policy as
            `attrs.evolve`, init-able attributes are addressed by their
            ``__init__`` *alias* (e.g. a private ``_x`` is passed as ``x``).
            Attributes with ``init=False`` -- which `attrs.evolve` cannot
            update at all -- are addressed by their attribute name.

    Returns:
        The same *inst*, updated in place.

    Raises:
        attrs.exceptions.NotAnAttrsClassError:
            If *inst* is not an instance of an *attrs* class.

        attrs.exceptions.FrozenInstanceError:
            If *inst* is frozen.

        TypeError:
            If an unknown attribute name is passed, or if the class of *inst*
            has a custom ``__setattr__``.

    .. note::
       **Transaction limitations**

       - The commit phase is *not* atomic: if an ``on_setattr`` hook raises,
         attributes that were already committed keep their new values and the
         remaining attributes keep their old ones. The exception propagates.
       - No locking is performed; mutating *inst* concurrently while
         ``batch_update()`` runs is unsupported.
       - Whether a value "actually changed" is decided by ``!=`` between the
         converted candidate value and the current value.
       - The candidate is a shallow copy (`copy.copy`); validators running on
         it see the same shared mutable sub-objects as the original.

    For example:

    >>> import attrs
    >>> @attrs.define
    ... class Range:
    ...     start: int
    ...     end: int = attrs.field()
    ...     @end.validator
    ...     def _ordered(self, attribute, value):
    ...         if value < self.start:
    ...             raise ValueError("end must not be before start")
    >>> r = Range(1, 2)
    >>> attrs.batch_update(r, start=3, end=4)
    Range(start=3, end=4)
    >>> attrs.batch_update(r, start=5)  # would leave end < start midway
    Traceback (most recent call last):
    ...
    ValueError: end must not be before start
    >>> r  # the failed batch left the instance untouched
    Range(start=3, end=4)

    .. versionadded:: 26.2.0
    """
    cls = inst.__class__

    if not has(cls):
        msg = "batch_update() can only be used on instances of attrs classes."
        raise NotAnAttrsClassError(msg)

    if cls.__setattr__ is _frozen_setattrs:
        raise FrozenInstanceError

    if (
        not getattr(cls, "__attrs_own_setattr__", False)
        and cls.__setattr__ is not _OBJ_SETATTR
    ):
        msg = (
            "batch_update() does not support classes with a custom "
            "__setattr__."
        )
        raise TypeError(msg)

    attrs = fields(cls)

    # Resolve the public change keys to attributes, following attrs' public
    # update policy (same as evolve()): init-able attributes are addressed by
    # their __init__ alias; init=False attributes -- which evolve() can't
    # update at all -- are addressed by their attribute name.
    by_name = {a.name: a for a in attrs}
    by_alias = {}
    for a in attrs:
        if a.init:
            by_alias.setdefault(a.alias, a)

    updates = {}
    for key, value in changes.items():
        a = by_name.get(key) or by_alias.get(key)
        if a is None:
            msg = f"batch_update() got an unexpected attribute {key!r}."
            raise TypeError(msg)
        updates[a.name] = value

    if not updates:
        return inst

    # -- Prepare phase: run converters and validators on an isolated
    # candidate. The original instance is never touched here, so any failure
    # leaves it -- including its slots state -- completely unchanged.
    candidate = copy.copy(inst)
    new_values = {}
    for a in attrs:  # declaration order, independent of kwargs order
        if a.name not in updates:
            continue
        value = setters.convert(candidate, a, updates[a.name])
        _OBJ_SETATTR(candidate, a.name, value)
        new_values[a.name] = value

    # Validate the candidate as a whole, so validators only ever see the
    # final state, never a mix of old and new values.
    validate(candidate)

    # -- Commit phase: decide upfront (in declaration order) which attributes
    # actually changed, then write them back, running each attribute's
    # on_setattr hook(s) exactly once.
    to_commit = [
        a
        for a in attrs
        if a.name in new_values and new_values[a.name] != getattr(inst, a.name)
    ]
    if not to_commit:
        return inst

    if getattr(cls, "__attrs_own_setattr__", False):
        props = getattr(cls, "__attrs_props__", None)
        cls_hook = props.on_setattr_hook if props is not None else None
        for a in to_commit:
            _run_commit_hooks(
                inst, a, _commit_hooks(a, cls_hook), new_values[a.name]
            )
    else:
        for a in to_commit:
            _OBJ_SETATTR(inst, a.name, new_values[a.name])

    return inst


def _commit_hooks(attribute, cls_hook):
    """
    Return the tuple of on_setattr hooks to run for *attribute* at commit
    time.

    Mirrors how the generated ``__setattr__`` resolves hooks (per-attribute
    hook, falling back to the class-level hook), except that
    `attrs.setters.convert` and `attrs.setters.validate` are left out: they
    already ran on the isolated candidate in the prepare phase.
    """
    hook = attribute.on_setattr or cls_hook
    if hook is None or hook is setters.NO_OP:
        return ()

    return tuple(
        h
        for h in _flatten_hook(hook)
        if h is not setters.convert and h is not setters.validate
    )


def _flatten_hook(hook):
    """
    Yield the individual callables that make up *hook*, unwrapping
    `attrs.setters.pipe` chains and lists/tuples.
    """
    if isinstance(hook, (list, tuple)):
        parts = hook
    else:
        parts = getattr(hook, "__attrs_piped__", None) or (hook,)

    for part in parts:
        if isinstance(part, (list, tuple)) or hasattr(part, "__attrs_piped__"):
            yield from _flatten_hook(part)
        else:
            yield part


def _run_commit_hooks(inst, attribute, hooks, value):
    """
    Run *hooks* for *attribute* on *inst* and set the resulting value.
    """
    if not hooks:
        _OBJ_SETATTR(inst, attribute.name, value)

        return

    if len(hooks) == 1 and _lazy_is_generator(hooks[0])():
        # Same protocol as the __setattr__ that attrs generates.
        gen = hooks[0](inst, attribute, value)
        nval = next(gen)
        _OBJ_SETATTR(inst, attribute.name, nval)
        try:
            next(gen)
        except StopIteration:
            return

        gen.close()
        msg = "Generator on_setattr hook yielded more than once."
        raise RuntimeError(msg)

    for hook in hooks:
        value = hook(inst, attribute, value)

    _OBJ_SETATTR(inst, attribute.name, value)
