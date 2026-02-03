"""
Test synchronous product with Alice & Bob protocols
Full pipeline: SoupSemantics → SynchronousProduct → LS2RG → BFS
"""

from synchronous_product import (
    SynchronousProduct,
    PropertySemantics,
    ProductConfiguration,
    MutualExclusionProperty,
    AcceptingStateChecker,
    AB1WithUniqueActions,
)
from alicebob_semantics import AB2Semantics, AB3Semantics, AliceBobState
from ls2rg import LS2RG
from BFS_definition import BFS


def test_ab1_with_mutex_property():
    """AB1 x MutualExclusion → Should find ERROR state"""
    print("\n=== AB1 x Mutual Exclusion Property ===")

    program = AB1WithUniqueActions()  # Use wrapper with unique actions
    prop = MutualExclusionProperty()
    product = SynchronousProduct(program, prop)

    # Convert to RootedGraph
    graph = LS2RG(product)

    # BFS with accepting state checker
    checker = AcceptingStateChecker(product)
    visited = BFS(graph, checker, None)

    print(f"States returned by BFS: {len(visited)}")
    print(f"All states seen by callback: {len(checker.all_states)}")
    print(f"Accepting states (OK): {len(checker.accepting_states)}")

    # Use all_states from checker instead of visited
    all_states = checker.all_states

    # Find ERROR states
    error_states = [
        state
        for state in all_states
        if isinstance(state, ProductConfiguration) and state.property == "ERROR"
    ]

    print(f"ERROR states (mutex violation): {len(error_states)}")

    if error_states:
        print(f"Example violation: {error_states[0]}")
        print(f"  Program state: {error_states[0].program}")
        print(f"  Property state: {error_states[0].property}")
    else:
        # Debug: Check for CS states
        print("\nDEBUG: Looking for states where alice or bob are in CS:")
        cs_states = [
            state
            for state in all_states
            if isinstance(state, ProductConfiguration)
            and hasattr(state.program, "alice")
            and hasattr(state.program, "bob")
            and (state.program.alice == "CS" or state.program.bob == "CS")
        ]
        print(f"Found {len(cs_states)} states with someone in CS:")
        for i, state in enumerate(cs_states[:10]):
            print(f"  {i}: {state.program} -> prop: {state.property}")

        # Check both in CS
        both_cs = [
            state
            for state in all_states
            if isinstance(state, ProductConfiguration)
            and hasattr(state.program, "alice")
            and hasattr(state.program, "bob")
            and state.program.alice == "CS"
            and state.program.bob == "CS"
        ]
        print(f"\nStates with BOTH in CS: {len(both_cs)}")
        for state in both_cs:
            print(f"  {state}")

    assert len(error_states) > 0, "AB1 should violate mutual exclusion"
    print("✅ AB1 violates mutual exclusion as expected!")


def test_ab3_with_mutex_property():
    """AB3 x MutualExclusion → Should NOT find ERROR state"""
    print("\n=== AB3 x Mutual Exclusion Property ===")

    program = AB3Semantics()
    prop = MutualExclusionProperty()
    product = SynchronousProduct(program, prop)

    graph = LS2RG(product)
    checker = AcceptingStateChecker(product)
    visited = BFS(graph, checker, None)

    print(f"States returned by BFS: {len(visited)}")
    print(f"All states seen by callback: {len(checker.all_states)}")
    print(f"Accepting states (OK): {len(checker.accepting_states)}")

    # Use all_states from checker
    all_states = checker.all_states

    error_states = [
        state
        for state in all_states
        if isinstance(state, ProductConfiguration) and state.property == "ERROR"
    ]

    print(f"ERROR states: {len(error_states)}")
    if error_states:
        print(f"Unexpected violation: {error_states[0]}")

    assert len(error_states) == 0, "AB3 should preserve mutual exclusion"
    print("✅ AB3 satisfies mutual exclusion!")


def test_ab2_with_mutex_property():
    """AB2 x MutualExclusion → Should NOT find ERROR state"""
    print("\n=== AB2 x Mutual Exclusion Property ===")

    program = AB2Semantics()
    prop = MutualExclusionProperty()
    product = SynchronousProduct(program, prop)

    graph = LS2RG(product)
    checker = AcceptingStateChecker(product)
    visited = BFS(graph, checker, None)

    print(f"States explored: {len(checker.all_states)}")

    error_states = [
        state
        for state in checker.all_states
        if isinstance(state, ProductConfiguration) and state.property == "ERROR"
    ]

    print(f"ERROR states: {len(error_states)}")
    assert len(error_states) == 0, "AB2 should preserve mutual exclusion"
    print("✅ AB2 satisfies mutual exclusion!")


def test_product_structure():
    """Test ProductConfiguration structure"""
    print("\n=== Product Configuration Test ===")

    prog_state = AliceBobState(alice="W", bob="I", flag_alice="DOWN", flag_bob="DOWN")
    prop_state = "OK"

    config = ProductConfiguration(prog_state, prop_state)
    print(f"Config: {config}")
    print(f"Program: {config.program}")
    print(f"Property: {config.property}")

    # Test equality and hashing
    config2 = ProductConfiguration(prog_state, prop_state)
    assert config == config2
    assert hash(config) == hash(config2)
    print("✅ ProductConfiguration works correctly")


if __name__ == "__main__":
    test_product_structure()
    test_ab1_with_mutex_property()
    test_ab2_with_mutex_property()
    test_ab3_with_mutex_property()
    print("\n✅ All synchronous product tests passed!")
