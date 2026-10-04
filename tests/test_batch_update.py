# SPDX-License-Identifier: MIT

"""
Tests for `attrs.batch_update`.
"""

import pytest

import attr
import attrs

from attrs import batch_update


@attrs.define
class Range:
    """Two interdependent fields: end must not lie before start."""

    start: int
    end: int = attrs.field()

    @end.validator
    def _ordered(self, attribute, value):
        if value < self.start:
            raise ValueError("end must not be before start")


class TestBatchUpdate:
    def test_batch_update_together_passes(self):
        """
        Changing interdependent fields together validates against the final
        state and passes.
        """
        r = Range(1, 2)

        rv = batch_update(r, start=3, end=4)

        assert rv is r
        assert (r.start, r.end) == (3, 4)

    def test_single_change_failure_leaves_instance_untouched(self):
        """
        A batch whose intermediate state would be invalid fails and the
        original instance keeps its old values.
        """
        r = Range(1, 2)

        with pytest.raises(ValueError, match="end must not be before start"):
            batch_update(r, start=3)

        assert (r.start, r.end) == (1, 2)

    def test_slots_rollback(self):
        """
        Failed batches on slots classes leave the slots state untouched.
        """
        r = Range(1, 2)

        assert not hasattr(r, "__dict__")

        with pytest.raises(ValueError):
            batch_update(r, start=3, end=1)

        assert (r.start, r.end) == (1, 2)
        assert not hasattr(r, "__dict__")

    def test_no_slots_class(self):
        """
        Plain (non-slots, no on_setattr hooks) classes are updated in place.
        """

        @attr.s
        class C:
            x = attr.ib()
            y = attr.ib()

        c = C(1, 2)
        batch_update(c, x=10, y=20)

        assert (c.x, c.y) == (10, 20)

    def test_converter_failure_rolls_back(self):
        """
        A converter raising on the candidate leaves the original untouched.
        """

        @attrs.define
        class C:
            x: int = attrs.field(converter=int)

        c = C(1)

        with pytest.raises(ValueError):
            batch_update(c, x="not a number")

        assert c.x == 1

    def test_converter_runs_once_per_changed_field(self):
        """
        Converters run exactly once per changed field -- on the candidate,
        not again at commit time.
        """
        calls = []

        @attrs.define
        class C:
            x: int = attrs.field(converter=lambda v: calls.append("x") or v)
            y: int = attrs.field(converter=lambda v: calls.append("y") or v)

        c = C(1, 2)
        assert calls == ["x", "y"]  # from __init__

        calls.clear()
        batch_update(c, x=10, y=20)
        assert calls == ["x", "y"]

        calls.clear()
        batch_update(c, y=30)
        assert calls == ["y"]

    def test_validator_runs_once_per_field_on_candidate(self):
        """
        Validators run once per field, against the fully updated candidate.
        """
        calls = []

        @attrs.define
        class C:
            x: int = attrs.field(validator=lambda *a: calls.append("x"))
            y: int = attrs.field(validator=lambda *a: calls.append("y"))

        c = C(1, 2)
        assert calls == ["x", "y"]  # from __init__

        calls.clear()
        batch_update(c, x=10, y=20)

        assert calls == ["x", "y"]

    def test_hook_runs_once_per_changed_field_in_declaration_order(self):
        """
        Setter hooks run exactly once per actually-changed field, in field
        declaration order -- regardless of the keyword order.
        """
        calls = []

        def hook(inst, attrib, value):
            calls.append(attrib.name)
            return value

        @attrs.define(on_setattr=hook)
        class C:
            x: int
            y: int
            z: int

        c = C(1, 2, 3)

        batch_update(c, z=30, x=10)  # kwargs reversed vs. declaration order

        assert calls == ["x", "z"]
        assert (c.x, c.y, c.z) == (10, 2, 30)

    def test_hook_not_run_for_unchanged_values(self):
        """
        Fields whose new value equals the old one don't trigger their hook.
        """
        calls = []

        def hook(inst, attrib, value):
            calls.append(attrib.name)
            return value

        @attrs.define(on_setattr=hook)
        class C:
            x: int
            y: int

        c = C(1, 2)

        batch_update(c, x=1, y=20)

        assert calls == ["y"]
        assert (c.x, c.y) == (1, 20)

    def test_hook_raising_has_deterministic_partial_commit(self):
        """
        If a commit-phase hook raises, fields committed before it keep their
        new values, the failing and later fields keep their old ones, and the
        exception propagates.
        """

        def hook(inst, attrib, value):
            if attrib.name == "y":
                raise RuntimeError("boom")
            return value

        @attrs.define(on_setattr=hook)
        class C:
            x: int
            y: int
            z: int

        c = C(1, 2, 3)

        with pytest.raises(RuntimeError, match="boom"):
            batch_update(c, x=10, y=20, z=30)

        assert (c.x, c.y, c.z) == (10, 2, 3)

    def test_generator_hook(self):
        """
        Generator on_setattr hooks run with the same protocol as the
        generated __setattr__.
        """
        log = []

        def hook(inst, attrib, value):
            log.append(("pre", attrib.name))
            yield value * 2
            log.append(("post", attrib.name))

        @attrs.define
        class C:
            x: int = attrs.field(on_setattr=hook)

        c = C(1)
        log.clear()

        batch_update(c, x=5)

        assert c.x == 10
        assert log == [("pre", "x"), ("post", "x")]

    def test_default_on_setattr_convert_validate_not_rerun_at_commit(self):
        """
        With @define's default on_setattr (convert + validate), the commit
        phase does not re-run converters or validators.
        """
        converted = []
        validated = []

        @attrs.define
        class C:
            x: int = attrs.field(
                converter=lambda v: converted.append(v) or int(v),
                validator=lambda *a: validated.append(a[1].name),
            )

        c = C("1")
        assert converted == ["1"]
        assert validated == ["x"]

        batch_update(c, x="2")

        assert c.x == 2
        assert converted == ["1", "2"]  # converted once, on the candidate
        assert validated == ["x", "x"]  # validated once, on the candidate

    def test_private_attribute_addressed_by_alias(self):
        """
        Private attributes can be addressed by their init alias (without the
        underscore) as well as by their attribute name.
        """

        @attrs.define
        class C:
            _x: int

        c = C(1)

        batch_update(c, x=2)
        assert c._x == 2

        batch_update(c, _x=3)
        assert c._x == 3

    def test_custom_alias(self):
        """
        Fields with a custom init alias are addressed by that alias.
        """

        @attrs.define
        class C:
            x: int = attrs.field(alias="renamed")

        c = C(1)

        batch_update(c, renamed=5)

        assert c.x == 5

    def test_init_false_field_updated_by_name(self):
        """
        init=False fields -- which evolve() cannot set -- are updated by
        their attribute name.
        """

        @attrs.define
        class C:
            x: int
            y: int = attrs.field(init=False, default=0)

        c = C(1)

        batch_update(c, x=10, y=42)

        assert (c.x, c.y) == (10, 42)

    def test_frozen_instance_rejected(self):
        """
        Frozen instances are explicitly rejected.
        """

        @attrs.frozen
        class C:
            x: int

        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            batch_update(C(1), x=2)

    def test_not_an_attrs_class(self):
        """
        Non-attrs instances are rejected.
        """
        with pytest.raises(attrs.exceptions.NotAnAttrsClassError):
            batch_update(object(), x=1)

    def test_unknown_attribute(self):
        """
        Unknown attribute names raise a TypeError.
        """

        @attrs.define
        class C:
            x: int

        with pytest.raises(TypeError, match="unexpected attribute 'y'"):
            batch_update(C(1), y=2)

    def test_custom_setattr_rejected(self):
        """
        Classes with a custom __setattr__ (and no attrs hooks) are rejected
        rather than silently bypassed.
        """

        @attrs.define(on_setattr=attrs.setters.NO_OP)
        class C:
            x: int

            def __setattr__(self, name, value):
                object.__setattr__(self, name, value)

        with pytest.raises(TypeError, match="custom"):
            batch_update(C(1), x=2)

    def test_no_changes_is_a_noop(self):
        """
        An empty batch neither validates nor touches the instance.
        """
        calls = []

        @attrs.define
        class C:
            x: int = attrs.field(validator=lambda *a: calls.append(a))

        c = C(1)
        assert len(calls) == 1  # from __init__

        assert batch_update(c) is c
        assert c.x == 1
        assert len(calls) == 1
