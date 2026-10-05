# SPDX-License-Identifier: MIT

"""
Tests for the explicit tracing entry point for attrs instance creation.

The trace records the actual order of default factories, converters,
validators, and the post-init hook.  It does not save values, and it is not
used for tracking reconstructed objects.
"""

from __future__ import annotations

import attr
import pytest


class TestTraceOrder:
    """
    The factory, conversion, and validation order of two fields is consistent
    with the original construction.
    """

    def test_two_fields_factory_converter_validator_order(self):
        """
        Two fields with factories, converters, and validators are traced in
        the order they actually execute: factory1, converter1, factory2,
        converter2, validator1, validator2, post_init.
        """
        calls = []

        def factory_x():
            calls.append("factory_x")
            return 1

        def factory_y():
            calls.append("factory_y")
            return 2

        def converter_x(v):
            calls.append("converter_x")
            return v * 10

        def converter_y(v):
            calls.append("converter_y")
            return v * 100

        def validator_x(self, attribute, value):
            calls.append("validator_x")

        def validator_y(self, attribute, value):
            calls.append("validator_y")

        @attr.s
        class TwoFields:
            x = attr.ib(
                default=attr.Factory(factory_x),
                converter=converter_x,
                validator=validator_x,
            )
            y = attr.ib(
                default=attr.Factory(factory_y),
                converter=converter_y,
                validator=validator_y,
            )

            def __attrs_post_init__(self):
                calls.append("post_init")

        with attr.trace() as ctx:
            obj = TwoFields()

        # The actual call order matches the trace order.
        assert calls == [
            "factory_x",
            "converter_x",
            "factory_y",
            "converter_y",
            "validator_x",
            "validator_y",
            "post_init",
        ]

        # The trace records the same order.
        trace = ctx.traces[0]
        assert trace.succeeded
        stages = [(s.stage, s.attribute) for s in trace.stages]
        assert stages == [
            ("factory", "x"),
            ("converter", "x"),
            ("factory", "y"),
            ("converter", "y"),
            ("validator", "x"),
            ("validator", "y"),
            ("post_init", None),
        ]

        # The object was created correctly.
        assert obj.x == 10
        assert obj.y == 200

    def test_explicit_parameters_not_reported_as_factory_results(self):
        """
        Caller-supplied explicit parameters are not falsely reported as
        factory results.
        """
        calls = []

        def factory_x():
            calls.append("factory_x")
            return 1

        def factory_y():
            calls.append("factory_y")
            return 2

        @attr.s
        class TwoFields:
            x = attr.ib(default=attr.Factory(factory_x))
            y = attr.ib(default=attr.Factory(factory_y))

        with attr.trace() as ctx:
            obj = TwoFields(x=10, y=20)

        # Factories were NOT called.
        assert calls == []

        # The trace does not contain factory stages.
        trace = ctx.traces[0]
        stages = [(s.stage, s.attribute) for s in trace.stages]
        assert stages == []

        assert obj.x == 10
        assert obj.y == 20

    def test_normal_construction_not_traced(self):
        """
        Normal construction (outside the trace context) does not add a trace.
        """
        @attr.s
        class C:
            x = attr.ib()

        # Outside the trace context, no trace is recorded.
        obj = C(1)
        assert obj.x == 1

        with attr.trace() as ctx:
            obj = C(2)
            assert obj.x == 2

        assert len(ctx.traces) == 1
        assert ctx.traces[0].cls == "C"


class TestTraceFailure:
    """
    The second field failure has a clear stage and each factory is called
    exactly once.
    """

    def test_second_field_factory_failure(self):
        """
        When the second field's factory fails, the trace records the failure
        with the stage and attribute name, and each factory is called exactly
        once.
        """
        calls = []

        def factory_x():
            calls.append("factory_x")
            return 1

        def factory_y():
            calls.append("factory_y")
            raise ValueError("factory y failed")

        @attr.s
        class TwoFields:
            x = attr.ib(default=attr.Factory(factory_x))
            y = attr.ib(default=attr.Factory(factory_y))

            def __attrs_post_init__(self):
                calls.append("post_init")

        with attr.trace() as ctx:
            with pytest.raises(ValueError, match="factory y failed"):
                TwoFields()

        # Each factory was called exactly once.
        assert calls == ["factory_x", "factory_y"]

        # post_init was NOT called (failure happened before it).
        assert "post_init" not in calls

        trace = ctx.traces[0]
        assert not trace.succeeded
        assert isinstance(trace.error, ValueError)

        stages = [(s.stage, s.attribute, s.failed) for s in trace.stages]
        assert stages == [
            ("factory", "x", False),
            ("factory", "y", False),
            ("factory", "y", True),
        ]

        # The failed stage carries the cause.
        failed_stage = trace.stages[-1]
        assert failed_stage.failed
        assert failed_stage.attribute == "y"
        assert isinstance(failed_stage.cause, ValueError)

    def test_second_field_converter_failure(self):
        """
        When the second field's converter fails, the trace records the failure
        with the stage and attribute name.
        """
        calls = []

        def converter_x(v):
            calls.append("converter_x")
            return v

        def converter_y(v):
            calls.append("converter_y")
            raise ValueError("converter y failed")

        @attr.s
        class TwoFields:
            x = attr.ib(converter=converter_x)
            y = attr.ib(converter=converter_y)

        with attr.trace() as ctx:
            with pytest.raises(ValueError, match="converter y failed"):
                TwoFields(x=1, y=2)

        assert calls == ["converter_x", "converter_y"]

        trace = ctx.traces[0]
        assert not trace.succeeded

        stages = [(s.stage, s.attribute, s.failed) for s in trace.stages]
        assert stages == [
            ("converter", "x", False),
            ("converter", "y", False),
            ("converter", "y", True),
        ]

    def test_second_field_validator_failure(self):
        """
        When the second field's validator fails, the trace records the failure
        with the stage and attribute name.
        """
        calls = []

        def validator_x(self, attribute, value):
            calls.append("validator_x")

        def validator_y(self, attribute, value):
            calls.append("validator_y")
            raise ValueError("validator y failed")

        @attr.s
        class TwoFields:
            x = attr.ib(validator=validator_x)
            y = attr.ib(validator=validator_y)

        with attr.trace() as ctx:
            with pytest.raises(ValueError, match="validator y failed"):
                TwoFields(x=1, y=2)

        assert calls == ["validator_x", "validator_y"]

        trace = ctx.traces[0]
        assert not trace.succeeded

        stages = [(s.stage, s.attribute, s.failed) for s in trace.stages]
        assert stages == [
            ("validator", "x", False),
            ("validator", "y", False),
            ("validator", "y", True),
        ]

    def test_post_init_failure(self):
        """
        When post_init fails, the trace records the failure with the
        post_init stage.
        """
        @attr.s
        class C:
            x = attr.ib()

            def __attrs_post_init__(self):
                raise ValueError("post_init failed")

        with attr.trace() as ctx:
            with pytest.raises(ValueError, match="post_init failed"):
                C(1)

        trace = ctx.traces[0]
        assert not trace.succeeded

        stages = [(s.stage, s.attribute, s.failed) for s in trace.stages]
        assert stages == [
            ("post_init", None, False),
            ("post_init", None, True),
        ]


class TestTakesSelfFactory:
    """
    takes_self factories preserve the identity of fields visible at the time
    of the call.
    """

    def test_takes_self_visible_fields(self):
        """
        A takes_self factory records the names of the fields that were
        visible (already set on self) when the factory ran.
        """
        @attr.s
        class TakesSelf:
            x = attr.ib(default=attr.Factory(lambda: 1))
            y = attr.ib(
                default=attr.Factory(lambda self: self.x + 10, takes_self=True)
            )
            z = attr.ib(
                default=attr.Factory(
                    lambda self: self.x + self.y + 100, takes_self=True
                )
            )

        with attr.trace() as ctx:
            obj = TakesSelf()

        assert obj.x == 1
        assert obj.y == 11
        assert obj.z == 112

        trace = ctx.traces[0]
        stages = [(s.stage, s.attribute, s.visible_fields) for s in trace.stages]
        assert stages == [
            ("factory", "x", None),
            ("factory", "y", ("x",)),
            ("factory", "z", ("x", "y")),
        ]

    def test_takes_self_not_visible_for_non_takes_self(self):
        """
        Non-takes_self factories do not record visible fields.
        """
        @attr.s
        class C:
            x = attr.ib(default=attr.Factory(lambda: 1))
            y = attr.ib(default=attr.Factory(lambda: 2))

        with attr.trace() as ctx:
            C()

        trace = ctx.traces[0]
        for stage in trace.stages:
            assert stage.visible_fields is None


class TestTraceNoValues:
    """
    The trace does not save values.
    """

    def test_trace_does_not_save_values(self):
        """
        The trace records only the stage, attribute name, and visible fields;
        it does not save the actual values.
        """
        @attr.s
        class C:
            x = attr.ib(default=attr.Factory(lambda: 42))
            y = attr.ib(default=0, converter=lambda v: v * 2)

        with attr.trace() as ctx:
            C(y=10)

        trace = ctx.traces[0]
        for stage in trace.stages:
            # The stage has no value attribute.
            assert not hasattr(stage, "value")
            # The stage has no result attribute.
            assert not hasattr(stage, "result")


class TestTraceNotForReconstruction:
    """
    The trace is not used for tracking reconstructed objects (__attrs_init__).
    """

    def test_attrs_init_not_traced(self):
        """
        __attrs_init__ (the reconstruction path) is not traced.
        """
        @attr.s(init=False)
        class C:
            x = attr.ib(default=attr.Factory(lambda: 1))
            y = attr.ib(default=attr.Factory(lambda: 2))

            def __init__(self):
                self.__attrs_init__()

        with attr.trace() as ctx:
            C()

        # __attrs_init__ is not traced, so no trace is recorded.
        assert len(ctx.traces) == 0


class TestTraceContext:
    """
    The trace context manager is the explicit entry point for tracing.
    """

    def test_trace_context_manager(self):
        """
        The trace context manager enables tracing for instance creations
        within its scope.
        """
        @attr.s
        class C:
            x = attr.ib()

        with attr.trace() as ctx:
            C(1)
            C(2)

        assert len(ctx.traces) == 2

    def test_trace_context_nested(self):
        """
        Nested trace contexts are supported.  The innermost context takes
        precedence, so creations within the inner context are recorded only
        by the inner context.
        """
        @attr.s
        class C:
            x = attr.ib()

        with attr.trace() as outer:
            C(1)
            with attr.trace() as inner:
                C(2)
            C(3)

        # The outer context records C(1) and C(3) (not C(2)).
        assert len(outer.traces) == 2
        # The inner context records C(2).
        assert len(inner.traces) == 1

    def test_trace_context_resets_on_exit(self):
        """
        The trace context is reset when exiting the scope.
        """
        @attr.s
        class C:
            x = attr.ib()

        with attr.trace() as ctx:
            C(1)

        # After exiting, no trace is recorded.
        C(2)

        assert len(ctx.traces) == 1


class TestFourStageExample:
    """
    A four-stage example showing factory, converter, validator, and post_init.
    """

    def test_four_stage_example(self):
        """
        A complete example with all four stages: factory, converter,
        validator, and post_init.
        """
        events = []

        def make_value():
            events.append("factory")
            return 10

        def convert(v):
            events.append("converter")
            return v + 1

        def validate(self, attribute, value):
            events.append("validator")
            if value < 0:
                raise ValueError("must be non-negative")

        @attr.s
        class Example:
            value = attr.ib(
                default=attr.Factory(make_value),
                converter=convert,
                validator=validate,
            )

            def __attrs_post_init__(self):
                events.append("post_init")

        with attr.trace() as ctx:
            obj = Example()

        # All four stages executed in order.
        assert events == ["factory", "converter", "validator", "post_init"]

        # The trace records all four stages.
        trace = ctx.traces[0]
        stages = [s.stage for s in trace.stages]
        assert stages == ["factory", "converter", "validator", "post_init"]

        # The object was created correctly.
        assert obj.value == 11