# SPDX-License-Identifier: MIT

"""
Tests for ``attrs.CopyPlan`` / ``attrs.copy_with_plan``: explicit copy plans
with per-field and per-type strategies (shared / shallow / recursive).

Run just these tests with:

    python -m pytest tests/test_copy_plan.py -v
"""

import pytest

import attr
import attrs

from attrs import CopyPlan, CopyStrategy, copy_with_plan
from attrs.exceptions import (
    CopyPlanError,
    FrozenInstanceError,
    NotAnAttrsClassError,
    UnsupportedCopyTypeError,
)


SHARED = CopyStrategy.SHARED
SHALLOW = CopyStrategy.SHALLOW
RECURSIVE = CopyStrategy.RECURSIVE
REJECT = CopyPlan.REJECT


class TestThreeStrategies:
    """
    The three registered strategies applied side by side.
    """

    def test_shared_shallow_recursive(self):
        @attrs.define
        class C:
            shared: list
            shallow: list
            recursive: list

        plan = (
            CopyPlan()
            .field("shared", SHARED)
            .field("shallow", SHALLOW)
            .field("recursive", RECURSIVE)
        )
        src = C(shared=[[1]], shallow=[[2]], recursive=[[3]])

        out = plan.copy(src)

        # SHARED: the very same object is reused.
        assert out.shared is src.shared

        # SHALLOW: new outer container, identical (aliased) elements.
        assert out.shallow is not src.shallow
        assert out.shallow[0] is src.shallow[0]

        # RECURSIVE: fully independent deep structure.
        assert out.recursive is not src.recursive
        assert out.recursive[0] is not src.recursive[0]
        assert out.recursive == src.recursive

    def test_registration_via_constructor(self):
        plan = CopyPlan(
            default=RECURSIVE,
            fields={"cache": SHARED},
            types={},
        )
        assert plan.default is RECURSIVE
        assert plan.field_strategies == {"cache": SHARED}

    def test_invalid_strategy_rejected(self):
        with pytest.raises(TypeError):
            CopyPlan(default="deep")
        with pytest.raises(TypeError):
            CopyPlan().field("x", "deep")
        with pytest.raises(TypeError):
            CopyPlan().on_type(int, 42)


class TestSharingAndCycles:
    def test_two_fields_same_list_stay_shared(self):
        """
        Two fields pointing at the same list still point at one single
        (new) list after a recursive copy -- and not at the source list.
        """

        @attrs.define
        class C:
            a: list
            b: list

        shared = [1, 2]
        src = C(shared, shared)

        out = copy_with_plan(src)

        assert out.a is out.b
        assert out.a is not shared
        assert out.a == [1, 2]

    def test_diamond_attrs_sharing(self):
        @attrs.define
        class Leaf:
            v: int

        @attrs.define
        class C:
            left: object
            right: object

        leaf = Leaf(1)
        out = copy_with_plan(C(leaf, leaf))

        assert out.left is out.right
        assert out.left is not leaf
        assert out.left == leaf

    def test_sharing_across_containers(self):
        @attrs.define
        class C:
            direct: list
            nested: list

        shared = [1]
        src = C(shared, [shared])
        out = copy_with_plan(src)

        assert out.direct is out.nested[0]
        assert out.direct is not shared

    def test_cyclic_attrs_instance_is_copied(self):
        @attrs.define
        class Node:
            name: str
            link: object = None

        n = Node("root")
        n.link = n

        out = copy_with_plan(n)

        assert out is not n
        assert out.name == "root"
        assert out.link is out

    def test_cyclic_container_is_copied(self):
        @attrs.define
        class C:
            items: list

        lst = []
        lst.append(lst)
        out = copy_with_plan(C(lst))

        assert out.items is not lst
        assert out.items[0] is out.items

    def test_cycle_rejected_by_explicit_type_policy(self):
        """
        Instead of copying, a plan may explicitly reject a type; the error
        names the field path.
        """

        class Handle:
            pass

        @attrs.define
        class Node:
            handle: object
            link: object = None

        n = Node(Handle())
        n.link = n  # the cycle is fine; the rejected type is not

        plan = CopyPlan().on_type(Handle, REJECT)
        with pytest.raises(UnsupportedCopyTypeError) as ei:
            plan.copy(n)

        assert ei.value.path == "Node.handle"
        assert "rejected" in str(ei.value)


class TestValidationBeforeExecution:
    def test_unsupported_type_reports_field_path(self):
        @attrs.define
        class Inner:
            weird: object

        @attrs.define
        class Outer:
            inner: Inner
            items: list

        src = Outer(Inner(object()), [1])

        with pytest.raises(UnsupportedCopyTypeError) as ei:
            copy_with_plan(src)

        assert ei.value.path == "Outer.inner.weird"
        assert "object" in str(ei.value)

    def test_unsupported_inside_container_reports_path(self):
        @attrs.define
        class C:
            items: list

        with pytest.raises(UnsupportedCopyTypeError) as ei:
            copy_with_plan(C([1, object()]))

        assert ei.value.path == "C.items[1]"

    def test_nothing_is_copied_when_validation_fails(self):
        """
        Validation runs before execution: a custom copier must not be
        invoked at all if another part of the graph is unsupported, and no
        half-built object is delivered.
        """
        copied = []

        class Token:
            pass

        @attrs.define
        class C:
            token: object
            bad: object

        plan = CopyPlan().on_type(Token, lambda v: copied.append(v) or v)

        with pytest.raises(UnsupportedCopyTypeError):
            plan.copy(C(Token(), object()))

        assert copied == []

    def test_source_is_untouched_on_failure(self):
        @attrs.define
        class C:
            items: list
            bad: object

        src = C([1], object())
        with pytest.raises(UnsupportedCopyTypeError):
            copy_with_plan(src)

        assert src.items == [1]


class TestFactoriesAndInitFalse:
    def test_factories_are_not_called_again(self):
        calls = []

        def factory():
            calls.append(1)
            return []

        @attrs.define
        class C:
            items: list = attrs.field(factory=factory)
            cache: list = attrs.field(factory=factory, init=False)

        src = C()  # the only two factory calls happen here
        assert len(calls) == 2
        src.items.append("a")
        src.cache.append("b")

        out = copy_with_plan(src)

        assert len(calls) == 2  # not a single extra call
        assert out.items == ["a"]
        assert out.items is not src.items
        assert out.cache == ["b"]
        assert out.cache is not src.cache

    def test_init_false_value_is_carried_over(self, slots, frozen):
        @attr.s(slots=slots, frozen=frozen)
        class C:
            x = attr.ib()
            derived = attr.ib(init=False, default="default")

        src = C(1)
        object.__setattr__(src, "derived", "custom")

        out = copy_with_plan(src)

        assert out.x == 1
        # The stored value is copied, not reset to the default.
        assert out.derived == "custom"


class TestClassShapes:
    def test_slots_and_frozen(self, slots, frozen):
        @attr.s(slots=slots, frozen=frozen)
        class C:
            x = attr.ib()
            y = attr.ib(default=2)

        src = C([1], y=[2])
        out = copy_with_plan(src)

        assert out is not src
        assert out.x == [1]
        assert out.x is not src.x
        assert out.y == [2]
        assert out.y is not src.y

        if frozen:
            with pytest.raises(FrozenInstanceError):
                out.x = 5

    def test_private_attribute_and_alias(self):
        @attrs.define
        class C:
            _x: int
            y: int = attrs.field(alias="why")

        src = C(1, 2)

        # Changes may use the alias (like evolve) ...
        out = copy_with_plan(src, x=10, why=20)
        assert (out._x, out.y) == (10, 20)

        # ... or the plain field name.
        out2 = copy_with_plan(src, _x=5, y=6)
        assert (out2._x, out2.y) == (5, 6)

    def test_user_getter_is_not_called(self):
        """
        A property shadowing the field on the class must not be invoked:
        the stored value is read directly.
        """

        @attr.s
        class C:
            x = attr.ib()

        src = C(1)

        def boom(self):
            raise AssertionError("user getter was called")

        C.x = property(boom)
        try:
            out = copy_with_plan(src)
        finally:
            del C.x

        assert out.__dict__["x"] == 1


class TestChanges:
    def test_changes_are_used_as_is(self):
        @attrs.define
        class C:
            x: list
            y: list

        new = [1]
        src = C([], [])
        out = copy_with_plan(src, x=new)

        assert out.x is new  # overrides are shared, not copied
        assert out.y == []
        assert out.y is not src.y

    def test_unknown_change_raises(self):
        @attrs.define
        class C:
            x: int

        with pytest.raises(CopyPlanError):
            copy_with_plan(C(1), nope=2)


class TestTypePolicies:
    def test_callable_policy(self):
        class Token:
            def __init__(self, v):
                self.v = v

        @attrs.define
        class C:
            t: object

        plan = CopyPlan().on_type(Token, lambda v: Token(v.v * 2))
        src = C(Token(21))
        out = plan.copy(src)

        assert out.t is not src.t
        assert out.t.v == 42

    def test_shared_type_policy(self):
        class Db:
            pass

        @attrs.define
        class C:
            db: object
            name: str

        db = Db()
        out = copy_with_plan(C(db, "x"), CopyPlan().on_type(Db, SHARED))

        assert out.db is db
        assert out.name == "x"

    def test_base_class_policy_applies_to_subclasses(self):
        class Base:
            pass

        class Child(Base):
            pass

        @attrs.define
        class C:
            obj: object

        child = Child()
        out = copy_with_plan(C(child), CopyPlan().on_type(Base, SHARED))

        assert out.obj is child


class TestDefaultsAndErrors:
    def test_default_plan_is_fully_recursive(self):
        @attrs.define
        class Inner:
            tags: list

        @attrs.define
        class Outer:
            inner: Inner
            mapping: dict

        src = Outer(Inner(["a"]), {"k": [1]})
        out = copy_with_plan(src)

        assert out == src
        assert out.inner is not src.inner
        assert out.inner.tags is not src.inner.tags
        assert out.mapping is not src.mapping
        assert out.mapping["k"] is not src.mapping["k"]

    def test_not_an_attrs_class(self):
        with pytest.raises(NotAnAttrsClassError):
            copy_with_plan(object())

    def test_plan_copy_method_and_chaining(self):
        plan = CopyPlan().field("x", SHARED)
        assert plan.field_strategies == {"x": SHARED}
        assert plan.on_type(int, SHARED) is plan

        @attrs.define
        class C:
            x: int

        assert plan.copy(C(1)) == C(1)
