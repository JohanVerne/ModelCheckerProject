from BFS_definition import BFS
from aliceEtBobImplementation import AB3

def has_deadlock_candidates(marked, graph):
    """Cherche états accessibles sans successeurs (potentiels deadlocks)."""
    deadlocks = []
    for state in marked:
        successors = graph.neighbors(state)
        if len(successors) == 0:
            deadlocks.append(state)
    return deadlocks

if __name__ == "__main__":
    print("====== AB3 =======")
    ab3 = AB3()
    marked, _ = BFS(ab3, lambda v, o: (False, o), None)
    print(marked)
    
    # Test exclusion mutuelle
    exclusion_violation = any("CS" in state[0] and "CS" in state[1] 
                              for state in marked)
    assert not exclusion_violation
    print("✅ Exclusion mutuelle AB3 OK")
    
    print(f"Nombre d'états : {len(marked)}")

    # Test deadlock
    deadlock_states = has_deadlock_candidates(marked, ab3)
    if deadlock_states:
        print("⚠️ États potentiels deadlock AB3 :", deadlock_states)
    else:
        print("✅ Pas de deadlock détecté dans AB3")
