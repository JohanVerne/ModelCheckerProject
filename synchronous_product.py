"""
Synchronous Product Operator
Implements composition of LanguageSemantics with PropertySemantics
"""

from languagesemantics import LanguageSemantics
from typing import Tuple, Set, Any, Callable


class ProductConfiguration:
    """State in the product: (program_state, property_state)"""

    def __init__(self, program_state: Any, property_state: Any):
        self.program = program_state
        self.property = property_state

    def __eq__(self, other):
        if not isinstance(other, ProductConfiguration):
            return False
        return self.program == other.program and self.property == other.property

    def __hash__(self):
        return hash((self.program, self.property))

    def __repr__(self):
        return f"({self.program}, {self.property})"


class PropertySemantics:
    """Abstract property automaton interface"""

    def initial(self) -> Any:
        """Initial state of the property"""
        raise NotImplementedError

    def accepts(self, state: Any) -> bool:
        """True if state is accepting"""
        raise NotImplementedError

    def actions(self, property_state: Any, program_state: Any) -> Set[Any]:
        """Filter/enable actions based on property state and program state"""
        raise NotImplementedError

    def execute(self, property_state: Any, action: Any, next_program_state: Any) -> Any:
        """Transition function: compute next property state"""
        raise NotImplementedError


class SynchronousProduct(LanguageSemantics):
    """
    Synchronous product: Program × Property

    States: ProductConfiguration(program_state, property_state)
    Actions: filtered by property operator
    Accepts: property.accepts(property_state)
    """

    def __init__(self, program: LanguageSemantics, prop: PropertySemantics):
        self._program = program
        self._property = prop

    def initials(self) -> Set[ProductConfiguration]:
        """Cartesian product of initial states"""
        prop_init = self._property.initial()
        return {
            ProductConfiguration(prog_state, prop_init)
            for prog_state in self._program.initials()
        }

    def actions(self, config: ProductConfiguration) -> Set[Any]:
        """Actions enabled by both program and property"""
        program_actions = self._program.actions(config.program)
        property_filter = self._property.actions(config.property, config.program)

        # If property returns None, don't filter (allow all program actions)
        if property_filter is None:
            return program_actions

        # If property returns empty set, block all actions
        if len(property_filter) == 0:
            return set()

        # Otherwise intersect
        return program_actions & property_filter

    def execute(
        self, config: ProductConfiguration, action: Any
    ) -> Set[ProductConfiguration]:
        """Execute action in both program and property"""
        results = set()

        # Execute in program - get ALL possible next states
        next_program_states = self._program.execute(config.program, action)

        for next_prog in next_program_states:
            # Execute in property using the RESULTING state
            next_prop = self._property.execute(config.property, action, next_prog)
            results.add(ProductConfiguration(next_prog, next_prop))

        return results

    def accepts(self, config: ProductConfiguration) -> bool:
        """Accepting if property accepts"""
        return self._property.accepts(config.property)


class AcceptingStateChecker:
    """BFS callback to find accepting states"""

    def __init__(self, product: SynchronousProduct):
        self.product = product
        self.accepting_states = []
        self.all_states = []

    def __call__(self, state: ProductConfiguration, opaque):
        self.all_states.append(state)
        if self.product.accepts(state):
            self.accepting_states.append(state)
        return False, opaque  # Don't terminate, continue search


# Example: Safety property "Never both in CS"
class MutualExclusionProperty(PropertySemantics):
    """Property: ¬(alice_CS ∧ bob_CS)"""

    def initial(self):
        return "OK"

    def accepts(self, state):
        return state == "OK"

    def actions(self, property_state, program_state):
        # Don't filter actions - we want to explore all states
        # including the ERROR states
        return None  # None means "no filtering"

    def execute(self, property_state, action, next_program_state):
        # Once in ERROR, stay in ERROR
        if property_state == "ERROR":
            return "ERROR"

        # Check if next state violates mutual exclusion
        # The check happens AFTER the transition
        if hasattr(next_program_state, "alice") and hasattr(next_program_state, "bob"):
            alice_in_cs = next_program_state.alice == "CS"
            bob_in_cs = next_program_state.bob == "CS"

            if alice_in_cs and bob_in_cs:
                return "ERROR"

        return "OK"


# Example: Liveness property "Eventually CS"
class EventuallyCSProperty(PropertySemantics):
    """Property: ◇(alice_CS ∨ bob_CS)"""

    def initial(self):
        return "WAITING"

    def accepts(self, state):
        return state == "SATISFIED"

    def actions(self, property_state, program_state):
        return None  # No action filtering

    def execute(self, property_state, action, next_program_state):
        if property_state == "SATISFIED":
            return "SATISFIED"

        # Check if someone reached CS
        if hasattr(next_program_state, "alice") and hasattr(next_program_state, "bob"):
            alice_in_cs = next_program_state.alice == "CS"
            bob_in_cs = next_program_state.bob == "CS"

            if alice_in_cs or bob_in_cs:
                return "SATISFIED"

        return "WAITING"


# Example: Deadlock detection property
class NoDeadlockProperty(PropertySemantics):
    """Property: Always has enabled actions (no deadlock)"""

    def initial(self):
        return "OK"

    def accepts(self, state):
        return state == "OK"

    def actions(self, property_state, program_state):
        return None  # No filtering

    def execute(self, property_state, action, next_program_state):
        # This property needs access to available actions
        # We'll mark it in the test instead
        return property_state


# Adapter for existing PropertySemantics
def adapt_property_actions(property_obj, property_state, program_state):
    """Helper to adapt actions() return value"""
    result = property_obj.actions(property_state, program_state)
    if result is None:
        # No filtering - allow all program actions
        from languagesemantics import LanguageSemantics

        return set()  # Will be intersected with program actions
    return result


# Wrapper to fix AB1 action ambiguity
class AB1WithUniqueActions(LanguageSemantics):
    """Wrapper around AB1Semantics that adds unique action prefixes"""

    def __init__(self):
        from alicebob_semantics import AB1Semantics

        self._ab1 = AB1Semantics()

    def initials(self):
        return self._ab1.initials()

    def actions(self, state):
        """Prefix actions with 'alice_' or 'bob_' based on who can execute them"""
        actions = []

        # Alice's actions
        if state.alice == "W":
            actions.append("alice_a1")
        elif state.alice == "a1":
            actions.append("alice_a2")
        elif state.alice == "a2":
            actions.append("alice_CS")
        elif state.alice == "CS":
            actions.append("alice_W")

        # Bob's actions
        if state.bob == "W":
            actions.append("bob_b1")
        elif state.bob == "b1":
            actions.append("bob_b2")
        elif state.bob == "b2":
            actions.append("bob_CS")
        elif state.bob == "CS":
            actions.append("bob_W")

        return actions

    def execute(self, state, action):
        from alicebob_semantics import AliceBobState

        new_state = AliceBobState(
            state.alice, state.bob, state.flag_alice, state.flag_bob
        )

        # Alice actions
        if action == "alice_a1":
            new_state.alice = "a1"
        elif action == "alice_a2":
            new_state.alice = "a2"
        elif action == "alice_CS":
            new_state.alice = "CS"
        elif action == "alice_W":
            new_state.alice = "W"

        # Bob actions
        elif action == "bob_b1":
            new_state.bob = "b1"
        elif action == "bob_b2":
            new_state.bob = "b2"
        elif action == "bob_CS":
            new_state.bob = "CS"
        elif action == "bob_W":
            new_state.bob = "W"

        return [new_state]
