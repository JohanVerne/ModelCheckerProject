from BFS_definition import Rootedgraph


class AB1(Rootedgraph):
    """Implementation of the basic Alice and Bob problem graph

    Root : Alice and Bob are inside their houses

    Automate AB1 pour Alice & Bob.
    Un état est un tuple (etat_alice, etat_bob).
    """

    def __init__(self):
        # À adapter selon votre figure, ceci est un exemple raisonnable :
        # W = waiting (maison), CS = section critique
        self.initial_state = ("W", "W")

        # transitions[state] = [liste des états successeurs]
        self.transitions = {
            ("W", "W"): [("CS", "W"), ("W", "CS")],
            ("CS", "W"): [("W", "W")],
            ("W", "CS"): [("W", "W")],
            # ajouter d'autres états si votre AB1 en prévoit plus
        }

    def roots(self):
        return [self.initial_state]

    def neighbors(self, vertex):
        return self.transitions.get(vertex, [])


class AB2(Rootedgraph):
    """
    Automate AB2 avec drapeaux pour Alice & Bob.
    Un état est (etat_alice, etat_bob, flagAlice, flagBob).
    """

    def __init__(self):
        # Etats possibles (à adapter à votre figure) :
        # W = waiting, CS = section critique, A1/A2/A3 pour les étapes d'Alice,
        # B1/B2/B3/... pour celles de Bob.
        self.initial_state = ("W", "W", "DOWN", "DOWN")

        # Transitions à compléter selon la figure d'AB2.
        # Exemple de base illustratif :
        self.transitions = {
            # Alice lève son drapeau
            ("W", "W", "DOWN", "DOWN"): [
                ("A1", "W", "UP", "DOWN"),   # {a1}/flagAlice = UP
                ("W", "B1", "DOWN", "UP"),  # {b1}/flagBob = UP
            ],

            # Ici il faudra continuer en suivant vos figures :
            # - passage par A2/A3 avec mise à jour de flagAlice
            # - passage par B2/B3 avec conditions sur flagAlice/flagBob
            # - états CS pour chacun
        }

    def roots(self):
        return [self.initial_state]

    def neighbors(self, vertex):
        return self.transitions.get(vertex, [])
