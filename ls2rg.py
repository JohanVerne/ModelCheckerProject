from BFS_definition import RootedGraph

# Classe qui transforme une sémantique de langage en graphe enraciné
# Language Semantics to Rooted Graph (LS2RG)
class LS2RG(RootedGraph):

    # Constructeur : prend une sémantique de langage
    # ls : instance d'une classe qui implémente LanguageSemantics
    def __init__(self, ls):
        self._ls = ls  # Stocke la sémantique de langage

    # Retourne les états initiaux (racines du graphe)
    def roots(self):
        return self._ls.initials()

    # Retourne les voisins d'un état (sommet)
    # Pour un état donné, calcule tous les états successeurs possibles
    # en exécutant toutes les actions possibles depuis cet état
    def neighbors(self, state):
        successors = []
        # Récupère toutes les actions possibles depuis cet état
        actions = self._ls.actions(state)
        # Pour chaque action, exécute-la et ajoute les états résultants
        for action in actions:
            # execute retourne un ensemble d'états successeurs
            next_states = self._ls.execute(state, action)
            successors.extend(next_states)
        return successors      