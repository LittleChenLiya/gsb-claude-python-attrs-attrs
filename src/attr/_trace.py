# SPDX-License-Identifier: MIT

"""
Explicit, opt-in tracing of attrs instance creation.

`trace_init` records the *actual* order in which default factories,
converters, validators, and ``__attrs_post_init__`` run while an instance of
an attrs class is being created.  It never re-creates or re-runs an object
for tracing purposes: it temporarily wraps the callables that the generated
``__init__`` already invokes, and restores them afterwards.  Construction
outside of the context manager is completely untouched and carries no
tracing overhead.

The trace deliberately stores *stages and identities only* -- attribute
names, the stage, the failure cause, and (for ``takes_self`` factories) the
names of the fields visible at call time.  It never stores field values.
"""

from __future__ import annotations

import contextlib

from collections.abc import Iterator

from ._make import _INIT_FACTORY_PAT, Factory, fields


__all__ = ["InitTrace", "TraceEvent", "trace_init"]


#: The stages that can appear in a trace, in pipeline order.
STAGES = ("factory", "converter", "validator", "post_init")


class TraceEvent:
    """
    A single recorded step of an attrs instance creation.

    .. attribute:: stage

        One of ``"factory"``, ``"converter"``, ``"validator"``, or
        ``"post_init"``.

    .. attribute:: attr

        The name of the attribute the stage belongs to, or `None` for
        ``"post_init"`` (which is instance-wide).

    .. attribute:: outcome

        ``"ok"`` if the stage completed, ``"failed"`` if it raised.

    .. attribute:: cause

        The exception that made the stage fail, or `None`.  Never a value.

    .. attribute:: visible_fields

        For ``takes_self`` factories only: the names of the fields that were
        already initialized on the instance when the factory ran
        (identities, not values).  `None` for all other events.

    .. versionadded:: 25.4.0
    """

    __slots__ = ("attr", "cause", "outcome", "stage", "visible_fields")

    def __init__(
        self,
        stage: str,
        attr: str | None,
        outcome: str,
        cause: BaseException | None = None,
        visible_fields: tuple[str, ...] | None = None,
    ):
        self.stage = stage
        self.attr = attr
        self.outcome = outcome
        self.cause = cause
        self.visible_fields = visible_fields

    @property
    def failed(self) -> bool:
        """Whether this event is the failure of its stage."""
        return self.outcome == "failed"

    def __repr__(self) -> str:
        parts = [f"stage={self.stage!r}", f"attr={self.attr!r}"]
        if self.failed:
            parts.append(f"cause={self.cause!r}")
        if self.visible_fields is not None:
            parts.append(f"visible_fields={self.visible_fields!r}")
        return f"<TraceEvent({', '.join(parts)})>"


class InitTrace:
    """
    An ordered record of the stages an attrs instance creation went through.

    Instances are handed out by `trace_init` and filled while the traced
    class is being instantiated.  Only stage metadata is stored -- never
    field values.

    .. versionadded:: 25.4.0
    """

    __slots__ = ("events",)

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def _record(self, event: TraceEvent) -> None:
        self.events.append(event)

    @property
    def stages(self) -> list[str]:
        """The stages of all events, in the order they actually ran."""
        return [e.stage for e in self.events]

    @property
    def failed(self) -> bool:
        """Whether any recorded stage failed."""
        return any(e.failed for e in self.events)

    def events_for(self, attr: str) -> list[TraceEvent]:
        """All events belonging to the attribute named *attr*."""
        return [e for e in self.events if e.attr == attr]

    def count(self, stage: str, attr: str | None = None) -> int:
        """How often *stage* ran (optionally limited to *attr*)."""
        return sum(
            1
            for e in self.events
            if e.stage == stage and (attr is None or e.attr == attr)
        )

    def __iter__(self) -> Iterator[TraceEvent]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def __repr__(self) -> str:
        return f"<InitTrace({self.events!r})>"


def _visible_field_names(
    self_, field_names: tuple[str, ...]
) -> tuple[str, ...]:
    """
    Names of the fields already initialized on *self_* -- identities only.
    """
    return tuple(n for n in field_names if hasattr(self_, n))


def _wrap_stage(
    recorder: InitTrace,
    stage: str,
    attr_name: str,
    fn,
    *,
    takes_self: bool,
    field_names: tuple[str, ...],
):
    def traced(*args, **kwargs):
        visible = None
        if takes_self and args:
            visible = _visible_field_names(args[0], field_names)
        try:
            rv = fn(*args, **kwargs)
        except Exception as e:
            recorder._record(
                TraceEvent(
                    stage, attr_name, "failed", cause=e, visible_fields=visible
                )
            )
            raise
        recorder._record(
            TraceEvent(stage, attr_name, "ok", visible_fields=visible)
        )
        return rv

    return traced


def _wrap_post_init(recorder: InitTrace, fn):
    def traced(self_):
        try:
            rv = fn(self_)
        except Exception as e:
            recorder._record(TraceEvent("post_init", None, "failed", cause=e))
            raise
        recorder._record(TraceEvent("post_init", None, "ok"))
        return rv

    return traced


def _patch_class(cls: type, recorder: InitTrace, patches: list) -> None:
    attrs = fields(cls)  # raises NotAnAttrsClassError for foreign classes
    init = cls.__dict__.get("__attrs_init__") or cls.__init__
    globs = init.__globals__
    field_names = tuple(a.name for a in attrs)

    for a in attrs:
        factory_key = _INIT_FACTORY_PAT % (a.name,)
        if factory_key in globs:
            takes_self = (
                isinstance(a.default, Factory) and a.default.takes_self
            )
            patches.append((globs, factory_key, globs[factory_key]))
            globs[factory_key] = _wrap_stage(
                recorder,
                "factory",
                a.name,
                globs[factory_key],
                takes_self=takes_self,
                field_names=field_names,
            )

        converter_key = f"__attr_converter_{a.name}"
        if converter_key in globs:
            patches.append((globs, converter_key, globs[converter_key]))
            globs[converter_key] = _wrap_stage(
                recorder,
                "converter",
                a.name,
                globs[converter_key],
                takes_self=False,
                field_names=field_names,
            )

        validator_key = f"__attr_validator_{a.name}"
        if validator_key in globs:
            patches.append((globs, validator_key, globs[validator_key]))
            globs[validator_key] = _wrap_stage(
                recorder,
                "validator",
                a.name,
                globs[validator_key],
                takes_self=False,
                field_names=field_names,
            )

    post_init = cls.__dict__.get("__attrs_post_init__")
    if post_init is not None:
        patches.append((cls, "__attrs_post_init__", post_init))
        cls.__attrs_post_init__ = _wrap_post_init(recorder, post_init)


@contextlib.contextmanager
def trace_init(cls: type, /, *more_classes: type) -> Iterator[InitTrace]:
    """
    Trace instance creation of the attrs class *cls* (and optionally
    *more_classes*) for the duration of the ``with`` block.

    Yields an `InitTrace` that records -- in actual execution order -- every
    default factory, converter, validator, and ``__attrs_post_init__`` call
    performed by the generated ``__init__`` of instances created inside the
    block.  The object under construction is *not* rebuilt or re-run for
    tracing; the real callables are observed exactly once each.

    - ``takes_self`` factories record the identities (names) of the fields
      visible at call time, never their values.
    - A failing stage is recorded with the attribute name and the exception
      as ``cause``; the exception propagates unchanged.
    - Arguments passed explicitly by the caller are never reported as
      factory results, because only the factory callable itself is observed.
    - ``__attrs_post_init__`` is only recorded when the construction is
      valid up to that point -- exactly when attrs would really call it.
    - Outside the ``with`` block, construction is completely untouched:
      nothing is traced and no overhead is added.

    Example (four stages, one per kind)::

        >>> import attrs
        >>> @attrs.define
        ... class C:
        ...     x: int = attrs.field(
        ...         factory=lambda: 1,
        ...         converter=int,
        ...         validator=attrs.validators.gt(0),
        ...     )
        ...     def __attrs_post_init__(self):
        ...         pass
        >>> with attrs.trace_init(C) as trace:
        ...     C()
        >>> trace.stages
        ['factory', 'converter', 'validator', 'post_init']

    .. versionadded:: 25.4.0
    """
    recorder = InitTrace()
    patches: list = []
    try:
        for c in (cls, *more_classes):
            _patch_class(c, recorder, patches)
        yield recorder
    finally:
        for target, key, original in reversed(patches):
            if isinstance(target, dict):
                target[key] = original
            else:
                setattr(target, key, original)
