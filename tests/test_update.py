# SPDX-License-Identifier: MIT

"""
Tests for `attrs.update`.
"""

import pytest

import attr

from attr import setters
from attr.exceptions import (
    AttrsAttributeNotFoundError,
    FrozenInstanceError,
    NotAnAttrsClassError,
)


def check_range(inst, attribute, value):
    if inst.start >= inst.end:
        raise ValueError("start must be less than end")


@attr.s
class Range:
    start = attr.ib(validator=check_range)
    end = attr.ib()


@attr.s(slots=True)
class SlotRange:
    start = attr.ib(validator=check_range)
    end = attr.ib()


class TestUpdate:
    def test_interdependent_fields_pass(self):
        """
        Changing multiple interdependent fields at once works because
        validators see the final state.
        """
        r = Range(1, 2)

        attr.update(r, start=3, end=4)

        assert r.start == 3
        assert r.end == 4

    def test_independent_change_fails_and_rolls_back(self):
        """
        Changing only one side of an interdependent pair fails validation and
        leaves the original untouched.
        """
        r = Range(1, 2)

        with pytest.raises(ValueError, match="start must be less than end"):
            attr.update(r, start=3)

        assert r.start == 1
        assert r.end == 2

    def test_slots_rollback(self):
        """
        Rollback works for slotted classes too.
        """
        r = SlotRange(1, 2)

        with pytest.raises(ValueError, match="start must be less than end"):
            attr.update(r, start=3)

        assert r.start == 1
        assert r.end == 2

    def test_slots_commit(self):
        """
        Committing changes works for slotted classes.
        """
        r = SlotRange(1, 2)

        attr.update(r, start=3, end=4)

        assert r.start == 3
        assert r.end == 4

    def test_converter_runs_once(self):
        """
        A field's converter runs exactly once during the trial phase.
        """
        calls = []

        def converter(value):
            calls.append(value)
            return value * 2

        @attr.s
        class C:
            x = attr.ib(converter=converter)
            y = attr.ib()

        c = C(1, 2)
        calls.clear()

        attr.update(c, x=10, y=20)

        assert calls == [10]
        assert c.x == 20
        assert c.y == 20

    def test_hook_runs_once_per_changed_field(self):
        """
        Setter hooks run once per actually changed field, in declaration
        order, during the commit phase.
        """
        calls = []

        def hook(instance, attribute, value):
            calls.append((attribute.name, value))
            return value

        @attr.s
        class C:
            a = attr.ib(on_setattr=hook)
            b = attr.ib(on_setattr=hook)
            c = attr.ib(on_setattr=hook)

        obj = C(1, 2, 3)
        calls.clear()

        attr.update(obj, a=10, b=20, c=30)

        assert calls == [("a", 10), ("b", 20), ("c", 30)]

    def test_hook_skips_unchanged_fields(self):
        """
        Setter hooks do not run for fields whose values did not change.
        """
        calls = []

        def hook(instance, attribute, value):
            calls.append(attribute.name)
            return value

        @attr.s
        class C:
            a = attr.ib(on_setattr=hook)
            b = attr.ib(on_setattr=hook)

        obj = C(1, 2)
        calls.clear()

        attr.update(obj, a=10)  # b is not changed

        assert calls == ["a"]

    def test_hook_throws_rolls_back(self):
        """
        If a setter hook raises during commit, all already-committed fields
        are rolled back to their original values.
        """
        calls = []

        def bad_hook(instance, attribute, value):
            calls.append(attribute.name)
            if attribute.name == "b":
                raise ValueError("boom")
            return value

        @attr.s
        class C:
            a = attr.ib(on_setattr=bad_hook)
            b = attr.ib(on_setattr=bad_hook)
            c = attr.ib(on_setattr=bad_hook)

        obj = C(1, 2, 3)

        with pytest.raises(ValueError, match="boom"):
            attr.update(obj, a=10, b=20, c=30)

        # a was committed before b's hook raised; it must be rolled back.
        assert obj.a == 1
        assert obj.b == 2
        assert obj.c == 3
        # c's hook never ran.
        assert calls == ["a", "b"]

    def test_hook_throws_rolls_back_slots(self):
        """
        Rollback after a hook error works for slotted classes.
        """

        def bad_hook(instance, attribute, value):
            if attribute.name == "b":
                raise ValueError("boom")
            return value

        @attr.s(slots=True)
        class C:
            a = attr.ib(on_setattr=bad_hook)
            b = attr.ib(on_setattr=bad_hook)

        obj = C(1, 2)

        with pytest.raises(ValueError, match="boom"):
            attr.update(obj, a=10, b=20)

        assert obj.a == 1
        assert obj.b == 2

    def test_frozen_rejected(self):
        """
        Frozen instances are explicitly rejected.
        """

        @attr.s(frozen=True)
        class C:
            x = attr.ib()

        obj = C(1)

        with pytest.raises(FrozenInstanceError):
            attr.update(obj, x=2)

        assert obj.x == 1

    def test_init_false_field_updated(self):
        """
        Fields with ``init=False`` are updated through the same public path.
        """

        @attr.s
        class C:
            x = attr.ib()
            y = attr.ib(init=False, default=0)

        obj = C(1)
        assert obj.y == 0

        attr.update(obj, x=5, y=10)

        assert obj.x == 5
        assert obj.y == 10

    def test_alias_accepted(self):
        """
        Attribute aliases are accepted as change keys.
        """

        @attr.s
        class C:
            _x = attr.ib(alias="x")

        obj = C(x=1)

        attr.update(obj, x=5)

        assert obj._x == 5

    def test_no_changes_no_hooks(self):
        """
        Passing no changes does not run any hooks.
        """
        calls = []

        def hook(instance, attribute, value):
            calls.append(attribute.name)
            return value

        @attr.s
        class C:
            a = attr.ib(on_setattr=hook)

        obj = C(1)

        attr.update(obj)

        assert calls == []
        assert obj.a == 1

    def test_unknown_attribute_raises(self):
        """
        Unknown attribute names raise AttrsAttributeNotFoundError.
        """
        r = Range(1, 2)

        with pytest.raises(AttrsAttributeNotFoundError):
            attr.update(r, nope=3)

        assert r.start == 1
        assert r.end == 2

    def test_non_attrs_instance_raises(self):
        """
        Non-attrs instances raise NotAnAttrsClassError.
        """

        class Plain:
            pass

        with pytest.raises(NotAnAttrsClassError):
            attr.update(Plain(), x=1)

    def test_converter_failure_leaves_original_untouched(self):
        """
        If a converter raises during the trial, the original is untouched.
        """
        calls = []

        def bad_converter(value):
            calls.append(value)
            if len(calls) > 1:
                raise ValueError("converter boom")
            return value

        @attr.s
        class C:
            x = attr.ib(converter=bad_converter)

        obj = C(1)
        assert obj.x == 1

        with pytest.raises(ValueError, match="converter boom"):
            attr.update(obj, x=2)

        assert obj.x == 1

    def test_validator_failure_leaves_original_untouched(self):
        """
        If a validator fails during the trial, the original is untouched.
        """
        r = Range(1, 2)

        with pytest.raises(ValueError, match="start must be less than end"):
            attr.update(r, start=100)

        assert r.start == 1
        assert r.end == 2

    def test_class_level_on_setattr(self):
        """
        Class-level on_setattr hooks are used during commit.
        """
        calls = []

        def hook(instance, attribute, value):
            calls.append(attribute.name)
            return value

        @attr.s(on_setattr=hook)
        class C:
            a = attr.ib()
            b = attr.ib()

        obj = C(1, 2)
        calls.clear()

        attr.update(obj, a=10, b=20)

        assert calls == ["a", "b"]

    def test_setters_convert_and_validate(self):
        """
        The built-in setters.convert and setters.validate work as expected.
        """

        @attr.s(on_setattr=[setters.convert, setters.validate])
        class C:
            x = attr.ib(
                converter=int, validator=attr.validators.instance_of(int)
            )

        obj = C(1)

        attr.update(obj, x="42")

        assert obj.x == 42

    def test_returns_instance(self):
        """
        update returns the instance for chaining.
        """
        r = Range(1, 2)

        result = attr.update(r, start=3, end=4)

        assert result is r

    def test_converter_identity_skips_hook_if_unchanged(self):
        """
        If a converter returns a value equal to the original, the hook is not
        run for that field.
        """
        calls = []

        def hook(instance, attribute, value):
            calls.append(attribute.name)
            return value

        def converter(value):
            return value  # identity converter

        @attr.s
        class C:
            x = attr.ib(converter=converter, on_setattr=hook)

        obj = C(1)
        calls.clear()

        attr.update(obj, x=1)  # same value

        assert calls == []
        assert obj.x == 1

    def test_converter_takes_self_sees_updated_dependency(self):
        """
        A converter with ``takes_self=True`` sees the updated value of a
        dependency field when the dependency is declared first.
        """
        from attr import Converter

        def conv(value, instance):
            return value + instance.y

        @attr.s
        class C:
            y = attr.ib(default=100)
            x = attr.ib(default=1, converter=Converter(conv, takes_self=True))

        obj = C()
        assert obj.x == 101

        attr.update(obj, x=5, y=200)

        assert obj.x == 205
        assert obj.y == 200

    def test_generator_hook_runs_during_commit(self):
        """
        Generator on_setattr hooks run during commit, and their post-yield
        code executes after the value is set.
        """
        post_yield = []

        def gen_hook(instance, attribute, value):
            yield value
            post_yield.append(attribute.name)

        @attr.s
        class C:
            a = attr.ib(on_setattr=gen_hook)
            b = attr.ib(on_setattr=gen_hook)

        obj = C(1, 2)

        attr.update(obj, a=10, b=20)

        assert obj.a == 10
        assert obj.b == 20
        assert post_yield == ["a", "b"]

    def test_no_op_setattr_bypasses_hook(self):
        """
        ``on_setattr=setters.NO_OP`` bypasses the hook during commit.
        """
        calls = []

        def hook(instance, attribute, value):
            calls.append(attribute.name)
            return value

        @attr.s
        class C:
            a = attr.ib(on_setattr=setters.NO_OP)
            b = attr.ib(on_setattr=hook)

        obj = C(1, 2)
        calls.clear()

        attr.update(obj, a=10, b=20)

        assert obj.a == 10
        assert obj.b == 20
        assert calls == ["b"]
