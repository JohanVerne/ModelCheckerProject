from BFS_definition import BFS
from aliceEtBobImplementation import AB1


def has_deadlock_candidates(marked, graph):
    """Cherche états accessibles sans successeurs (potentiels deadlocks)."""
    deadlocks = []
    for state in marked:
        successors = graph.neighbors(state)
        if len(successors) == 0:
            deadlocks.append(state)
    return deadlocks


if __name__ == "__main__":

    print("====== AB1 =======")
    # Exécution BFS complète

    ab1 = AB1()
    marked, _ = BFS(ab1, lambda v, o: (False, o), None)
    print(marked)

    # Test exclusion mutuelle : impossible d'atteindre ('CS', 'CS')
    marked, _ = BFS(ab1, lambda v, o: (False, o), None)
    try:
        assert ("CS", "CS") not in marked
    except AssertionError:
        print("❌ Violation d'exclusion mutuelle dans AB1")
    else:
        print("✅ Exclusion mutuelle AB1 OK")

    # Test deadlock
    deadlock_states = has_deadlock_candidates(marked, ab1)
    if deadlock_states:
        print("⚠️ États potentiels deadlock AB1 :", deadlock_states)
    else:
        print("✅ Pas de deadlock détecté dans AB1")
