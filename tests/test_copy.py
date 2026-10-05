# SPDX-License-Identifier: MIT

"""
Tests for the explicit copy plan API (`attrs.CopyPlan` and `attrs.copy`).
"""

from __future__ import annotations

import pytest

import attr

from attr import CopyPlan, copy
from attr.exceptions import (
    CopyError,
    FrozenInstanceError,
    NotAnAttrsClassError,
)


# -- helpers ---------------------------------------------------------------


@attr.s
class Node:
    value = attr.ib()
    children = attr.ib(factory=list)


@attr.s
class Container:
    a = attr.ib()
    b = attr.ib()


@attr.s
class WithInitFalse:
    x = attr.ib()
    y = attr.ib(init=False, default=42)


@attr.s
class WithAlias:
    _x = attr.ib(alias="x")


@attr.s
class Secret:
    data = attr.ib()


class Unknown:
    pass


# -- three strategies ------------------------------------------------------


class TestStrategies:
    """
    The three field strategies: ``share`` (default), ``shallow``, and
    ``deep``.
    """

    def test_share_is_default(self):
        """
        By default, fields are shared: the new field references the same
        object as the source field.
        """
        obj = Container([1, 2], [3, 4])
        plan = CopyPlan()
        copied = plan.copy(obj)

        assert copied is not obj
        assert copied.a is obj.a
        assert copied.b is obj.b

    def test_shallow_creates_shallow_copy(self):
        """
        The ``shallow`` strategy creates a shallow copy of the field
        (`copy.copy`).
        """
        obj = Container([1, 2], [3, 4])
        plan = CopyPlan().shallow("a")
        copied = plan.copy(obj)

        assert copied.a is not obj.a
        assert copied.a == obj.a
        assert copied.b is obj.b  # still shared

    def test_deep_creates_deep_copy(self):
        """
        The ``deep`` strategy creates a recursive deep copy of the field.
        """
        obj = Container([1, 2], [3, 4])
        plan = CopyPlan().deep("a")
        copied = plan.copy(obj)

        assert copied.a is not obj.a
        assert copied.a == obj.a
        assert copied.b is obj.b  # not deep-copied

    def test_default_strategy_via_constructor(self):
        """
        The default strategy for unregistered fields can be set in the
        constructor.
        """
        obj = Container([1, 2], [3, 4])
        plan = CopyPlan(default=attr.DEEP)
        copied = plan.copy(obj)

        assert copied.a is not obj.a
        assert copied.b is not obj.b

    def test_invalid_strategy_rejected(self):
        with pytest.raises(ValueError):
            CopyPlan()._set_field_strategy("a", "bogus")

    def test_invalid_default_rejected(self):
        with pytest.raises(ValueError):
            CopyPlan(default="bogus")


# -- sharing preservation --------------------------------------------------


class TestSharingPreservation:
    """
    Internal sharing is preserved by the deep copier: two fields pointing to
    the same object still point to the same object after copying, but the
    object is different from the source.
    """

    def test_two_fields_same_list(self):
        shared = [1, 2, 3]
        obj = Container(shared, shared)
        plan = CopyPlan().deep("a").deep("b")
        copied = plan.copy(obj)

        assert copied.a is copied.b  # still same identity
        assert copied.a is not obj.a  # different from source
        assert copied.a == obj.a

    def test_nested_sharing(self):
        shared = [1, 2]
        inner = Container(shared, shared)
        outer = Container(inner, inner)
        plan = CopyPlan(default=attr.DEEP)
        copied = plan.copy(outer)

        assert copied.a is copied.b
        assert copied.a.a is copied.a.b
        assert copied.a.a is not inner.a

    def test_memo_prevents_duplicate_copies(self):
        """
        The memo ensures an object is only copied once, even if reachable via
        multiple paths.
        """
        shared = [1, 2]
        obj = Container(shared, shared)
        plan = CopyPlan().deep("a").deep("b")
        copied = plan.copy(obj)

        assert copied.a is copied.b
        assert copied.a is not shared


# -- cycles ----------------------------------------------------------------


class TestCycles:
    """
    Circular objects can be copied (the deep copier handles cycles via its
    memo).
    """

    def test_self_referential_attrs_instance(self):
        node = Node(1)
        node.children.append(node)  # cycle
        plan = CopyPlan().deep("children")
        copied = plan.copy(node)

        assert copied is not node
        assert copied.children[0] is copied  # cycle preserved
        assert copied.children[0] is not node

    def test_mutual_cycles(self):
        @attr.s
        class A:
            b = attr.ib()

        @attr.s
        class B:
            a = attr.ib()

        a = A(None)
        b = B(a)
        a.b = b  # mutual cycle
        plan = CopyPlan(default=attr.DEEP)
        copied = plan.copy(a)

        assert copied is not a
        assert copied.b is not b
        assert copied.b.a is copied  # cycle preserved

    def test_cycle_with_shared_object(self):
        shared = [1]
        node = Node(shared)
        node.children.append(node)
        plan = CopyPlan().deep("value").deep("children")
        copied = plan.copy(node)

        assert copied is not node
        assert copied.value is not shared
        assert copied.children[0] is copied


# -- type strategies -------------------------------------------------------


class TestTypeStrategies:
    """
    Types can be registered with a strategy or a custom copier callable.
    """

    def test_reject_strategy(self):
        obj = Container(Secret("x"), "ok")
        plan = CopyPlan().deep("a").register_type(Secret, attr.REJECT)

        with pytest.raises(CopyError) as exc_info:
            plan.copy(obj)
        assert "Secret" in str(exc_info.value)
        assert "a" in str(exc_info.value)  # field path

    def test_shallow_strategy_for_type(self):
        obj = Container(Secret("x"), "ok")
        plan = CopyPlan().deep("a").register_type(Secret, attr.SHALLOW)
        copied = plan.copy(obj)

        assert copied.a is not obj.a  # shallow copy
        assert copied.a.data == obj.a.data

    def test_share_strategy_for_type(self):
        obj = Container(Secret("x"), "ok")
        plan = CopyPlan().deep("a").register_type(Secret, attr.SHARE)
        copied = plan.copy(obj)

        assert copied.a is obj.a  # shared, not copied

    def test_custom_copier_callable(self):
        @attr.s
        class Custom:
            x = attr.ib()

        def my_copier(obj, copy_func, path):
            return Custom(copy_func(obj.x))

        obj = Custom([1, 2])
        plan = CopyPlan().register_type(Custom, my_copier)
        copied = plan.copy(obj)

        assert copied is not obj
        assert copied.x is not obj.x

    def test_invalid_type_strategy_rejected(self):
        plan = CopyPlan()
        with pytest.raises(ValueError):
            plan.register_type(Secret, "bogus")
        with pytest.raises(TypeError):
            plan.register_type(Secret, 42)

    def test_mro_lookup(self):
        """
        Type strategies are looked up via the MRO, so registering a base
        class strategy applies to subclasses.
        """

        class Base:
            pass

        class Derived(Base):
            pass

        obj = Container(Derived(), "ok")
        plan = CopyPlan().deep("a").register_type(Base, attr.REJECT)

        with pytest.raises(CopyError):
            plan.copy(obj)


# -- unsupported objects ---------------------------------------------------


class TestUnsupportedObjects:
    """
    Unsupported objects are reported with their field path before any copying
    takes place.
    """

    def test_unsupported_object_reports_path(self):
        obj = Container(Unknown(), "ok")
        plan = CopyPlan().deep("a")

        with pytest.raises(CopyError) as exc_info:
            plan.copy(obj)
        assert "Unknown" in str(exc_info.value)
        assert "a" in str(exc_info.value)

    def test_nested_unsupported_object_reports_path(self):
        @attr.s
        class Inner:
            bad = attr.ib()

        obj = Container(Inner(Unknown()), "ok")
        plan = CopyPlan().deep("a")

        with pytest.raises(CopyError) as exc_info:
            plan.copy(obj)
        assert "a.bad" in str(exc_info.value)

    def test_failure_does_not_deliver_half_object(self):
        """
        A failure during validation raises before any copying takes place, so
        the source is untouched and no partial result is returned.
        """
        obj = Container([1, 2], Unknown())
        plan = CopyPlan().deep("a").deep("b")

        with pytest.raises(CopyError):
            plan.copy(obj)

        # Source untouched.
        assert obj.a == [1, 2]
        assert isinstance(obj.b, Unknown)

    def test_unsupported_in_list(self):
        obj = Container([Unknown()], "ok")
        plan = CopyPlan().deep("a")

        with pytest.raises(CopyError) as exc_info:
            plan.copy(obj)
        assert "a[0]" in str(exc_info.value)

    def test_unsupported_in_dict(self):
        obj = Container({"key": Unknown()}, "ok")
        plan = CopyPlan().deep("a")

        with pytest.raises(CopyError) as exc_info:
            plan.copy(obj)
        assert "a['key']" in str(exc_info.value)


# -- factory not called extra ---------------------------------------------


class TestFactoryNotCalled:
    """
    Factories are not called during copying (the copy does not invoke
    ``__init__``).
    """

    def test_init_true_factory_not_called(self):
        calls = []

        def make_list():
            calls.append(1)
            return []

        @attr.s
        class WithFactory:
            items = attr.ib(factory=make_list)

        obj = WithFactory()
        calls.clear()
        obj.items.append(1)
        plan = CopyPlan().deep("items")
        copied = plan.copy(obj)

        assert len(calls) == 0  # factory not called during copy
        assert copied.items is not obj.items

    def test_init_false_factory_not_called(self):
        calls = []

        def make_val():
            calls.append(1)
            return "default"

        @attr.s
        class WithInitFalseFactory:
            x = attr.ib()
            y = attr.ib(init=False, factory=make_val)

        obj = WithInitFalseFactory(1)
        calls.clear()
        obj.y = "custom"
        plan = CopyPlan().deep("x")
        copied = plan.copy(obj)

        assert len(calls) == 0  # factory not called during copy
        assert copied.y == "custom"  # copied, not re-defaulted

    def test_post_init_not_called(self):
        calls = []

        @attr.s
        class WithPostInit:
            x = attr.ib()

            def __attrs_post_init__(self):
                calls.append(1)

        obj = WithPostInit(1)
        calls.clear()
        plan = CopyPlan().deep("x")
        plan.copy(obj)

        assert len(calls) == 0  # post_init not called during copy


# -- slots, aliases, init=False -------------------------------------------


class TestAttrsMetadata:
    """
    Slots, aliases, and ``init=False`` are handled along with attrs metadata.
    """

    def test_init_false_field_copied(self):
        obj = WithInitFalse(1)
        obj.y = 100
        plan = CopyPlan().deep("x")
        copied = plan.copy(obj)

        assert copied.x == 1
        assert copied.y == 100  # copied, not default

    def test_alias_handled(self):
        obj = WithAlias(42)
        plan = CopyPlan().deep("_x")
        copied = plan.copy(obj)

        assert copied._x == 42

    def test_slots_handled(self, slots):
        @attr.s(slots=slots)
        class Slotted:
            x = attr.ib()

        obj = Slotted([1, 2])
        plan = CopyPlan().deep("x")
        copied = plan.copy(obj)

        assert copied.x is not obj.x
        assert copied.x == obj.x

    def test_inherited_fields_copied(self):
        @attr.s
        class Base:
            x = attr.ib()

        @attr.s
        class Derived(Base):
            y = attr.ib()

        obj = Derived([1], [2])
        plan = CopyPlan(default=attr.DEEP)
        copied = plan.copy(obj)

        assert copied.x is not obj.x
        assert copied.y is not obj.y


# -- don't call user getters ----------------------------------------------


class TestNoUserGetters:
    """
    The copy does not invoke user-defined getters (properties or
    ``__getattribute__`` overrides).
    """

    def test_property_getter_not_called(self):
        @attr.s
        class WithProperty:
            _x = attr.ib()

            @property
            def x(self):
                raise RuntimeError("getter called!")

        obj = WithProperty(42)
        plan = CopyPlan().deep("_x")
        copied = plan.copy(obj)

        assert copied._x == 42

    def test_getattribute_override_not_called(self):
        calls = []

        @attr.s
        class WithGetAttrOverride:
            x = attr.ib()

            def __getattribute__(self, name):
                calls.append(name)
                return object.__getattribute__(self, name)

        obj = WithGetAttrOverride(42)
        calls.clear()
        plan = CopyPlan().deep("x")
        copied = plan.copy(obj)

        assert "x" not in calls  # copy didn't go through __getattribute__
        assert object.__getattribute__(copied, "x") == 42


# -- frozen state ----------------------------------------------------------


class TestFrozenState:
    """
    Frozen state is maintained: the copy is frozen (modifications raise
    ``FrozenInstanceError``).
    """

    def test_frozen_copy_is_frozen(self):
        @attr.s(frozen=True)
        class Frozen:
            x = attr.ib()

        obj = Frozen([1, 2])
        plan = CopyPlan().deep("x")
        copied = plan.copy(obj)

        assert copied.x is not obj.x
        with pytest.raises(FrozenInstanceError):
            copied.x = [3]

    def test_frozen_slotted_copy_is_frozen(self):
        @attr.s(frozen=True, slots=True)
        class FrozenSlotted:
            x = attr.ib()

        obj = FrozenSlotted([1, 2])
        plan = CopyPlan().deep("x")
        copied = plan.copy(obj)

        assert copied.x is not obj.x
        with pytest.raises(FrozenInstanceError):
            copied.x = [3]


# -- containers ------------------------------------------------------------


class TestContainers:
    """
    Built-in containers are deep-copied recursively.
    """

    def test_list(self):
        obj = Container([[1], [2]], "ok")
        plan = CopyPlan().deep("a")
        copied = plan.copy(obj)

        assert copied.a is not obj.a
        assert copied.a[0] is not obj.a[0]
        assert copied.a == obj.a

    def test_dict(self):
        obj = Container({"k": [1]}, "ok")
        plan = CopyPlan().deep("a")
        copied = plan.copy(obj)

        assert copied.a is not obj.a
        assert copied.a["k"] is not obj.a["k"]

    def test_tuple(self):
        obj = Container(([1], [2]), "ok")
        plan = CopyPlan().deep("a")
        copied = plan.copy(obj)

        assert copied.a is not obj.a
        assert copied.a[0] is not obj.a[0]

    def test_set(self):
        obj = Container({1, 2}, "ok")
        plan = CopyPlan().deep("a")
        copied = plan.copy(obj)

        assert copied.a is not obj.a
        assert copied.a == obj.a

    def test_frozenset(self):
        obj = Container(frozenset({1, 2}), "ok")
        plan = CopyPlan().deep("a")
        copied = plan.copy(obj)

        assert copied.a is not obj.a
        assert copied.a == obj.a

    def test_immutable_leaves_shared(self):
        obj = Container(42, "ok")
        plan = CopyPlan(default=attr.DEEP)
        copied = plan.copy(obj)

        assert copied.a is obj.a  # immutable, shared
        assert copied.b is obj.b


# -- copy function ---------------------------------------------------------


class TestCopyFunction:
    def test_copy_with_plan(self):
        obj = Container([1], "ok")
        plan = CopyPlan().deep("a")
        copied = copy(obj, plan)

        assert copied.a is not obj.a

    def test_copy_without_plan_uses_default(self):
        obj = Container([1], "ok")
        copied = copy(obj)

        assert copied.a is obj.a  # default: share

    def test_copy_non_attrs_raises(self):
        with pytest.raises(NotAnAttrsClassError):
            copy(object())

    def test_copy_non_attrs_class_raises(self):
        with pytest.raises(NotAnAttrsClassError):
            copy(Container)


# -- evolve unchanged ------------------------------------------------------


class TestEvolveUnchanged:
    """
    ``evolve`` is unchanged (it skips ``init=False`` fields and calls
    ``__init__``).
    """

    def test_evolve_skips_init_false(self):
        obj = WithInitFalse(1)
        obj.y = 100
        evolved = attr.evolve(obj, x=2)

        assert evolved.x == 2
        assert evolved.y == 42  # init=False skipped by evolve

    def test_evolve_passes_fields_explicitly(self):
        """
        ``evolve`` passes all fields explicitly to ``cls(**changes)``, so
        factories are not invoked (unchanged behavior).
        """
        calls = []

        @attr.s
        class WithFactory:
            items = attr.ib(factory=lambda: calls.append(1) or [])

        obj = WithFactory()
        calls.clear()
        evolved = attr.evolve(obj)

        assert len(calls) == 0  # evolve passes fields explicitly, no factory
        assert evolved.items == []


# -- plan is reusable ------------------------------------------------------


class TestPlanReusable:
    def test_plan_can_copy_multiple_objects(self):
        plan = CopyPlan().deep("a")
        obj1 = Container([1], "ok")
        obj2 = Container([2], "ok")

        copied1 = plan.copy(obj1)
        copied2 = plan.copy(obj2)

        assert copied1.a is not obj1.a
        assert copied2.a is not obj2.a
        assert copied1.a == [1]
        assert copied2.a == [2]

    def test_plan_methods_return_self(self):
        plan = CopyPlan()
        assert plan.share("a") is plan
        assert plan.shallow("b") is plan
        assert plan.deep("c") is plan
        assert plan.register_type(Secret, attr.REJECT) is plan
