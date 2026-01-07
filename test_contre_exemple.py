"""
Test de contre-exemples sur AB1/AB2/AB3.
Utilise on_entry_create_parents pour trouver un chemin vers un "mauvais" état.
"""

from BFS_definition import BFS, on_entry_create_parents
from aliceEtBobImplementation import AB1, AB2, AB3


def test_contre_exemple(graph, nom, mauvais_etat, description):
    """Teste si un mauvais état est atteignable."""
    print(f"\n=== Contre-exemple {nom} : {description} ===")

    result = BFS(graph, on_entry_create_parents, opaque=(None, graph, mauvais_etat))

    if isinstance(result[1], list):
        print("❌ CONTRE-EXEMPLE TROUVÉ !")
        print("Chemin complet :", " → ".join(map(str, result[1])))
    else:
        print("✅ Impossible d'atteindre l'état", mauvais_etat)
        print("États explorés :", len(result[0]))

    return isinstance(result[1], list)


if __name__ == "__main__":
    print("TESTS CONTRE-EXEMPLES ALICE & BOB\n")

    # AB1
    ab1 = AB1()
    test_contre_exemple(ab1, "AB1", ("CS", "CS"), "section critique simultanée")

    # AB2
    ab2 = AB2()
    test_contre_exemple(
        ab2, "AB2", ("W", "W", "UP", "UP"), "les deux drapeaux levés simultanément"
    )

    # AB3
    ab3 = AB3()

    test_contre_exemple(
        ab3, "AB3", ("CS", "CS", "UP", "UP"), "CS simultanée malgré résolution conflit"
    )

    print("\n🎯 Point 5 du PDF terminé !")
