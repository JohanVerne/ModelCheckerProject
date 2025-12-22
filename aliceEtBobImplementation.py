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
