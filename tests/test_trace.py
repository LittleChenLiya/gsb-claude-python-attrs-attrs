# SPDX-License-Identifier: MIT

"""
Tests for `attrs.trace_init` -- the explicit, opt-in trace entry point for
attrs instance creation.
"""

import pytest

import attrs

from attrs import InitTrace, TraceEvent, trace_init


def _positive(inst, attr, value):
    if value <= 0:
        raise ValueError(f"{attr.name} must be positive")


@attrs.define
class TwoFields:
    """Two fields, each with factory, converter, and validator."""

    x: int = attrs.field(factory=lambda: 1, converter=int, validator=_positive)
    y: int = attrs.field(factory=lambda: 2, converter=int, validator=_positive)


@attrs.define
class FourStages:
    """One field exercising all four stages."""

    x: int = attrs.field(factory=lambda: 1, converter=int, validator=_positive)

    def __attrs_post_init__(self):
        pass


class TestFourStageExample:
    def test_all_four_stages_in_order(self):
        """
        factory -> converter -> validator -> post_init, exactly once each.
        """
        with trace_init(FourStages) as trace:
            FourStages()

        assert ["factory", "converter", "validator", "post_init"] == (
            trace.stages
        )
        assert not trace.failed
        assert [None, None, None, None] == [e.cause for e in trace]

    def test_doctest_example(self):
        """
        The example from the trace_init docstring works as documented.
        """
        with attrs.trace_init(FourStages) as trace:
            FourStages()

        assert trace.stages == [
            "factory",
            "converter",
            "validator",
            "post_init",
        ]


class TestOrderMatchesRealConstruction:
    def test_two_field_order(self, slots):
        """
        The traced order of factories, converters, and validators for two
        fields is identical to the order the real callables run in, and each
        callable runs exactly once.
        """
        real_order = []

        def make(name, value):
            def factory():
                real_order.append(("factory", name))
                return value

            return factory

        def convert(name):
            def converter(v):
                real_order.append(("converter", name))
                return v

            return converter

        def validate(name):
            def validator(inst, attr, value):
                real_order.append(("validator", name))

            return validator

        @attrs.define(slots=slots)
        class C:
            x: int = attrs.field(
                factory=make("x", 1),
                converter=convert("x"),
                validator=validate("x"),
            )
            y: int = attrs.field(
                factory=make("y", 2),
                converter=convert("y"),
                validator=validate("y"),
            )

            def __attrs_post_init__(self):
                real_order.append(("post_init", None))

        with trace_init(C) as trace:
            C()

        expected = [
            ("factory", "x"),
            ("converter", "x"),
            ("factory", "y"),
            ("converter", "y"),
            ("validator", "x"),
            ("validator", "y"),
            ("post_init", None),
        ]
        # The trace matches the real construction order exactly ...
        assert expected == real_order
        assert expected == [(e.stage, e.attr) for e in trace]
        # ... and every stage ran exactly once per field.
        for stage in ("factory", "converter", "validator"):
            for name in ("x", "y"):
                assert trace.count(stage, name) == 1
        assert trace.count("post_init") == 1

    def test_explicit_argument_is_not_a_factory_result(self):
        """
        A value passed explicitly by the caller is never misreported as a
        factory result: no factory event is recorded for that field.
        """
        with trace_init(TwoFields) as trace:
            TwoFields(10)

        assert trace.count("factory", "x") == 0
        assert trace.count("factory", "y") == 1
        # The converter and validator still ran for the explicit argument.
        assert trace.count("converter", "x") == 1
        assert trace.count("validator", "x") == 1


class TestFailureStages:
    def test_second_field_validator_failure(self):
        """
        When the second field's validator fails, the failing event carries
        the stage, the attribute name, and the cause; both factories ran
        exactly once; post_init never ran.
        """

        @attrs.define
        class C:
            x: int = attrs.field(
                factory=lambda: 1, converter=int, validator=_positive
            )
            y: int = attrs.field(
                factory=lambda: -2, converter=int, validator=_positive
            )

            def __attrs_post_init__(self):
                raise AssertionError("must not run on invalid flow")

        with (
            trace_init(C) as trace,
            pytest.raises(ValueError, match="y must be positive"),
        ):
            C()

        assert [
            ("factory", "x"),
            ("converter", "x"),
            ("factory", "y"),
            ("converter", "y"),
            ("validator", "x"),
            ("validator", "y"),
        ] == [(e.stage, e.attr) for e in trace]

        # Each factory ran exactly once.
        assert trace.count("factory", "x") == 1
        assert trace.count("factory", "y") == 1

        # The failure event has a clear stage, attribute name, and cause.
        last = trace.events[-1]
        assert last.failed
        assert last.stage == "validator"
        assert last.attr == "y"
        assert isinstance(last.cause, ValueError)
        assert "y must be positive" == str(last.cause)

        # post_init only runs on the valid flow -- not here.
        assert trace.count("post_init") == 0
        assert trace.failed

    def test_post_init_runs_on_valid_flow(self):
        """
        The same class records post_init exactly once when validation
        succeeds.
        """

        @attrs.define
        class C:
            x: int = attrs.field(factory=lambda: 1, validator=_positive)

            def __attrs_post_init__(self):
                pass

        with trace_init(C) as trace:
            C()

        assert trace.count("post_init") == 1
        assert trace.stages[-1] == "post_init"
        assert not trace.failed

    def test_factory_failure(self):
        """
        A failing factory is reported with stage 'factory', the attribute
        name, and the cause.
        """
        boom = RuntimeError("boom")

        def bad_factory():
            raise boom

        @attrs.define
        class C:
            x: int = attrs.field(factory=lambda: 1)
            y: int = attrs.field(factory=bad_factory)

        with trace_init(C) as trace, pytest.raises(RuntimeError):
            C()

        last = trace.events[-1]
        assert last.failed
        assert last.stage == "factory"
        assert last.attr == "y"
        assert last.cause is boom
        assert trace.count("factory", "x") == 1

    def test_converter_failure(self):
        """
        A failing converter is reported with stage 'converter'.
        """

        def bad_converter(v):
            msg = "cannot convert"
            raise TypeError(msg)

        @attrs.define
        class C:
            x: int = attrs.field(factory=lambda: 1, converter=bad_converter)

        with (
            trace_init(C) as trace,
            pytest.raises(TypeError, match="cannot convert"),
        ):
            C()

        last = trace.events[-1]
        assert last.failed
        assert last.stage == "converter"
        assert last.attr == "x"
        assert isinstance(last.cause, TypeError)


class TestTakesSelfFactory:
    def test_visible_field_identities(self, slots):
        """
        A takes_self factory records the identities (names) of the fields
        visible at call time -- not their values.
        """
        seen = {}

        def y_factory(self_):
            seen["x"] = self_.x
            return self_.x + 1

        @attrs.define(slots=slots)
        class C:
            x: int = attrs.field(factory=lambda: 41)
            y: int = attrs.field(
                default=attrs.Factory(y_factory, takes_self=True)
            )

        with trace_init(C) as trace:
            C()

        (event,) = trace.events_for("y")
        assert event.stage == "factory"
        # Only x was initialized when y's factory ran.
        assert ("x",) == event.visible_fields
        # The factory really saw the same instance state (sanity check).
        assert seen["x"] == 41

    def test_no_values_in_trace(self):
        """
        The trace never stores field values -- only stage metadata and field
        identities.
        """
        with trace_init(TwoFields) as trace:
            TwoFields()

        for event in trace:
            assert set(TraceEvent.__slots__) == {
                "stage",
                "attr",
                "outcome",
                "cause",
                "visible_fields",
            }
            assert event.cause is None
            assert event.visible_fields is None


class TestNoTraceWhenInactive:
    def test_normal_construction_is_untraced(self):
        """
        Construction outside of trace_init records nothing and the wrappers
        are fully restored afterwards.
        """
        TwoFields()  # plain construction: no trace, no error

        with trace_init(TwoFields) as trace:
            TwoFields()
        assert len(trace) == 6  # 2 x (factory, converter, validator)

        # After the context, nothing is recorded anymore.
        TwoFields()
        assert len(trace) == 6

    def test_trace_is_empty_without_construction(self):
        with trace_init(TwoFields) as trace:
            pass

        assert [] == trace.stages
        assert not trace.failed

    def test_not_an_attrs_class(self):
        with (
            pytest.raises(attrs.exceptions.NotAnAttrsClassError),
            trace_init(object),
        ):
            pass

    def test_helpers(self):
        """
        InitTrace/TraceEvent helper APIs behave as documented.
        """
        with trace_init(TwoFields) as trace:
            TwoFields()

        assert isinstance(trace, InitTrace)
        assert all(isinstance(e, TraceEvent) for e in trace)
        assert 3 == len(trace.events_for("x"))
        assert 3 == len(trace.events_for("y"))
        assert repr(trace.events[0]).startswith("<TraceEvent(stage='factory'")
