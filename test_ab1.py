from BFS_definition import BFS
from aliceEtBobImplementation import AB1

if __name__ == "__main__":

    print("====== AB1 =======")
    # Exécution BFS complète

    ab1 = AB1()
    marked, _ = BFS(ab1, lambda v, o: (False, o), None)
    print(marked)

    # Test exclusion mutuelle : impossible d'atteindre ('CS', 'CS')
    marked, _ = BFS(ab1, lambda v, o: (False, o), None)
    assert ('CS', 'CS') not in marked
    print("✅ Exclusion mutuelle AB1 OK")

    # Test deadlock : ici, on regarde s'il existe au moins un état
    # où personne n'est en CS mais où au moins un mouvement reste possible.
    # Pour un vrai deadlock, on voudrait un état sans successeur ET où quelqu'un veut la CS.
    has_deadlock_candidate = False
    for state in marked:
        neighbors = ab1.neighbors(state)
        if len(neighbors) == 0:
            has_deadlock_candidate = True
            print("⚠️ État candidat deadlock AB1:", state)

    if not has_deadlock_candidate:
        print("✅ Aucun deadlock évident dans AB1")

