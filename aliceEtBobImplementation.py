from BFS_definition import Rootedgraph


class AB1(Rootedgraph):
    """Implementation of the basic Alice and Bob problem graph

    Root : Alice and Bob are inside their houses

    Automate AB1 pour Alice & Bob.
    Un état est un tuple (etat_alice, etat_bob).
    """

    def __init__(self):
        # I = Initial, CS = section critique
        self.initial_state = ("I", "I")

        # transitions[state] = [liste des états successeurs]
        self.transitions = {
            ("I", "I"): [("CS", "I"), ("I", "CS")],
            ("CS", "I"): [("I", "I"), ("CS", "CS")],
            ("I", "CS"): [("I", "I"), ("CS", "CS")],
            ("CS", "CS"): [("I", "CS"), ("CS", "I")],
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
        # I: Initial, W = waiting, CS = section critique,
        # flagAlice/flagBob = UP/DOWN
        self.initial_state = ("I", "I", "DOWN", "DOWN")

        self.transitions = {
            # Alice lève son drapeau ou Bob lève son drapeau et avance en W
            ("I", "I", "DOWN", "DOWN"): [
                ("I", "W", "DOWN", "UP"),
                ("W", "I", "UP", "DOWN"),
            ],
            # Bob avance en W quand Alice est en W
            ("W", "I", "UP", "DOWN"): [
                ("W", "W", "UP", "UP"),
                # Alice avance en CS si drapeau Bob down
                ("CS", "I", "UP", "DOWN"),
            ],
            # Alcie avance en W quand Bob est en W
            ("I", "W", "DOWN", "UP"): [
                ("W", "W", "UP", "UP"),
                # Bob avance en CS si drapeau Alice down
                ("I", "CS", "DOWN", "UP"),
            ],
            # SORTIE CS → drapeau down
            ("CS", "I", "UP", "DOWN"): [
                ("I", "I", "DOWN", "DOWN"),
            ],
            ("I", "CS", "DOWN", "UP"): [
                ("I", "I", "DOWN", "DOWN"),
            ],
            ("CS", "W", "UP", "UP"): [
                ("I", "W", "DOWN", "UP"),
            ],
            ("W", "CS", "UP", "UP"): [
                ("W", "I", "UP", "DOWN"),
            ],
        }

    def roots(self):
        return [self.initial_state]

    def neighbors(self, vertex):
        return self.transitions.get(vertex, [])


class AB3(Rootedgraph):
    """
    Automate AB3 : stratégie avec résolution de conflit via drapeaux.
    """

    def __init__(self):
        self.initial_state = ("I", "I", "DOWN", "DOWN")

        self.transitions = self.transitions = {
            # Alice lève son drapeau ou Bob lève son drapeau et avance en W
            ("I", "I", "DOWN", "DOWN"): [
                ("I", "W", "DOWN", "UP"),
                ("W", "I", "UP", "DOWN"),
            ],
            # Bob avance en W quand Alice est en W
            ("W", "I", "UP", "DOWN"): [
                ("W", "W", "UP", "UP"),
                # Alice avance en CS si drapeau Bob down
                ("CS", "I", "UP", "DOWN"),
            ],
            # Alcie avance en W quand Bob est en W
            ("I", "W", "DOWN", "UP"): [
                ("W", "W", "UP", "UP"),
                # Bob avance en CS si drapeau Alice down
                ("I", "CS", "DOWN", "UP"),
            ],
            # SORTIE CS → drapeau down
            ("CS", "I", "UP", "DOWN"): [
                ("I", "I", "DOWN", "DOWN"),
            ],
            ("I", "CS", "DOWN", "UP"): [
                ("I", "I", "DOWN", "DOWN"),
            ],
            ("CS", "W", "UP", "UP"): [
                ("I", "W", "DOWN", "UP"),
            ],
            ("W", "CS", "UP", "UP"): [
                ("W", "I", "UP", "DOWN"),
            ],
            # On ajoute la transition de sortie du deadlock
            ("W", "W", "UP", "UP"): [
                ("I", "W", "DOWN", "UP"),
            ],
        }

    def roots(self):
        return [self.initial_state]

    def neighbors(self, vertex):
        return self.transitions.get(vertex, [])
