from BFS_definition import BFS
from aliceEtBobImplementation import AB2


def has_deadlock_candidates(marked, graph):
    """Cherche états accessibles sans successeurs (potentiels deadlocks)."""
    deadlocks = []
    for state in marked:
        successors = graph.neighbors(state)
        if len(successors) == 0:
            deadlocks.append(state)
    return deadlocks


if __name__ == "__main__":
    print("====== AB2 (complet) =======")
    ab2 = AB2()
    marked, _ = BFS(ab2, lambda v, o: (False, o), None)
    print(f"États atteints : {marked}")

    # Test exclusion mutuelle
    exclusion_violation = any("CS" in state[0] and "CS" in state[1] for state in marked)
    try:
        assert ("CS", "CS") not in marked
    except AssertionError:
        print("❌ Violation d'exclusion mutuelle dans AB2")
    else:
        print("✅ Exclusion mutuelle AB2 OK")

    print(f"Nombre d'états : {len(marked)}")

    # Test deadlock
    deadlock_states = has_deadlock_candidates(marked, ab2)
    if deadlock_states:
        print("⚠️ États potentiels deadlock AB2 :", deadlock_states)
    else:
        print("✅ Pas de deadlock détecté dans AB2")
