# SPDX-License-Identifier: MIT

"""
Tracing for attrs instance creation.

This module provides an explicit entry point for tracing the creation of
attrs instances. When tracing is enabled, the generated ``__init__`` records
the actual order of default factory calls, converter calls, validator calls,
and the post-init hook.

The trace does not record values, and is not used for tracking reconstructed
objects (for example ``__attrs_init__``).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator


__all__ = ["CreationTrace", "Stage", "trace"]


class Stage:
    """
    A single stage in the creation of an attrs instance.

    Attributes:
        stage: The stage type. One of ``"factory"``, ``"converter"``,
            ``"validator"``, or ``"post_init"``.
        attribute: The name of the attribute this stage pertains to, or
            ``None`` for the post-init stage.
        visible_fields: For ``takes_self`` factories, the names of the
            fields that were visible (already set on ``self``) when the
            factory ran. ``None`` otherwise.
        failed: Whether this stage failed.
        cause: The exception that caused the failure, if any.
    """

    __slots__ = ("attribute", "cause", "failed", "stage", "visible_fields")

    def __init__(
        self,
        stage: str,
        attribute: str | None = None,
        visible_fields: tuple[str, ...] | None = None,
        failed: bool = False,
        cause: BaseException | None = None,
    ) -> None:
        self.stage = stage
        self.attribute = attribute
        self.visible_fields = visible_fields
        self.failed = failed
        self.cause = cause

    def __repr__(self) -> str:
        parts = [f"stage={self.stage!r}"]
        if self.attribute is not None:
            parts.append(f"attribute={self.attribute!r}")
        if self.visible_fields is not None:
            parts.append(f"visible_fields={self.visible_fields!r}")
        if self.failed:
            parts.append("failed=True")
        if self.cause is not None:
            parts.append(f"cause={self.cause!r}")
        return f"Stage({', '.join(parts)})"


class CreationTrace:
    """
    A trace of a single attrs instance creation.

    Attributes:
        cls: The fully qualified name of the class that was instantiated.
        stages: The stages in the order they were executed.
        succeeded: Whether the creation succeeded.
        error: The exception that caused the creation to fail, if any.
    """

    __slots__ = ("cls", "error", "stages", "succeeded")

    def __init__(self, cls: str) -> None:
        self.cls = cls
        self.stages: list[Stage] = []
        self.succeeded = True
        self.error: BaseException | None = None

    def record_stage(
        self,
        stage: str,
        attribute: str | None = None,
        visible_fields: tuple[str, ...] | None = None,
    ) -> None:
        self.stages.append(Stage(stage, attribute, visible_fields))

    def record_failure(
        self, stage: str, attribute: str | None, cause: BaseException
    ) -> None:
        self.succeeded = False
        self.error = cause
        self.stages.append(Stage(stage, attribute, failed=True, cause=cause))

    def __repr__(self) -> str:
        return (
            f"CreationTrace(cls={self.cls!r}, succeeded={self.succeeded}, "
            f"stages={self.stages!r})"
        )


class _TraceContext:
    """
    Collects traces for instance creations within its scope.
    """

    __slots__ = ("traces",)

    def __init__(self) -> None:
        self.traces: list[CreationTrace] = []

    def record(self, trace: CreationTrace) -> None:
        self.traces.append(trace)


# Thread-local storage for the active trace context.
_trace_state = threading.local()


def _get_trace() -> _TraceContext | None:
    """
    Return the active trace context, or ``None`` if tracing is not active.
    """
    return getattr(_trace_state, "context", None)


@contextmanager
def trace() -> Iterator[_TraceContext]:
    """
    Enable tracing for attrs instance creations within this scope.

    This is an explicit entry point for tracing. When active, the generated
    ``__init__`` of attrs classes records the actual order of default
    factories, converters, validators, and the post-init hook.

    The trace does not record values. Caller-supplied values are not reported
    as factory results. Normal construction (outside this scope) is not
    traced.

    Returns:
        A context object with a ``traces`` attribute containing the
        :class:`CreationTrace` objects for instances created within the scope.

    Example:
        >>> with attrs.trace() as ctx:
        ...     obj = SomeClass()
        ...     for stage in ctx.traces[0].stages:
        ...         print(stage.stage, stage.attribute)
    """
    ctx = _TraceContext()
    old = getattr(_trace_state, "context", None)
    _trace_state.context = ctx
    try:
        yield ctx
    finally:
        _trace_state.context = old


# Tracing helpers called from the generated __init__.  These are injected into
# the globals of the generated function.


def _trace_factory(
    trace: CreationTrace | None,
    self_: Any,
    name: str,
    factory: Any,
    takes_self: bool,
    field_names: tuple[str, ...],
) -> Any:
    """
    Call a default factory, recording the stage if tracing is active.

    When *takes_self* is true, the names of the fields that are visible
    (already set on *self_*) at the time of the call are recorded.
    """
    if trace is None:
        return factory(self_) if takes_self else factory()
    visible = (
        tuple(n for n in field_names if hasattr(self_, n))
        if takes_self
        else None
    )
    trace.record_stage("factory", name, visible_fields=visible)
    try:
        return factory(self_) if takes_self else factory()
    except BaseException as exc:
        trace.record_failure("factory", name, exc)
        raise


def _trace_converter(
    trace: CreationTrace | None,
    name: str,
    converter: Any,
    value: Any,
    self_: Any,
    attr: Any,
) -> Any:
    """
    Run a converter, recording the stage if tracing is active.
    """
    if trace is None:
        return converter(value, self_, attr)
    trace.record_stage("converter", name)
    try:
        return converter(value, self_, attr)
    except BaseException as exc:
        trace.record_failure("converter", name, exc)
        raise


def _trace_validator(
    trace: CreationTrace | None,
    self_: Any,
    name: str,
    validator: Any,
    attr: Any,
    value: Any,
) -> None:
    """
    Run a validator, recording the stage if tracing is active.
    """
    if trace is None:
        validator(self_, attr, value)
        return
    trace.record_stage("validator", name)
    try:
        validator(self_, attr, value)
    except BaseException as exc:
        trace.record_failure("validator", name, exc)
        raise


def _trace_post_init(
    trace: CreationTrace | None,
    post_init: Any,
) -> None:
    """
    Run the post-init hook, recording the stage if tracing is active.

    *post_init* is the bound method (``self.__attrs_post_init__``), so it is
    called without arguments.
    """
    if trace is None:
        post_init()
        return
    trace.record_stage("post_init")
    try:
        post_init()
    except BaseException as exc:
        trace.record_failure("post_init", None, exc)
        raise
