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
        
            # ALICE monte drapeau → vérifie Bob → CS
            ("A1", "W", "UP", "DOWN"): [
                ("CS", "W", "UP", "DOWN"),    # {a2}[flagBob == DOWN] → CS
            ],
            
            # BOB monte drapeau → vérifie Alice → CS  
            ("W", "B1", "DOWN", "UP"): [
                ("W", "CS", "DOWN", "UP"),    # {b2}[flagAlice == DOWN] → CS
            ],
            
            # SORTIE CS → drapeau down
            ("CS", "W", "UP", "DOWN"): [
                ("W", "W", "DOWN", "DOWN"),   # {a3}/flagAlice = DOWN
            ],
            ("W", "CS", "DOWN", "UP"): [
                ("W", "W", "DOWN", "DOWN"),   # {b3}/flagBob = DOWN
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
        self.initial_state = ("W", "W", "DOWN", "DOWN")
        
        self.transitions = {
            # Identique à AB2 au début
            ("W", "W", "DOWN", "DOWN"): [
                ("A1", "W", "UP", "DOWN"),    # {a1}/flagAlice = UP
                ("W", "B1", "DOWN", "UP"),    # {b1}/flagBob = UP
            ],
            ("A1", "W", "UP", "DOWN"): [
                ("CS", "W", "UP", "DOWN"),    # {a2}[flagBob == DOWN] → CS
            ],
            ("W", "B1", "DOWN", "UP"): [
                ("W", "CS", "DOWN", "UP"),    # {b2}[flagAlice == DOWN] → CS
                
                # NOUVEAU pour AB3 : Bob voit flagAlice==UP → baisse son drapeau
                ("W", "W", "UP", "DOWN"),     # {b4}[flagAlice == UP]/flagBob = DOWN
            ],
            
            # Sorties CS
            ("CS", "W", "UP", "DOWN"): [
                ("W", "W", "DOWN", "DOWN"),   # {a3}/flagAlice = DOWN
            ],
            ("W", "CS", "DOWN", "UP"): [
                ("W", "W", "DOWN", "DOWN"),   # {b3}/flagBob = DOWN
            ],
        }

    def roots(self):
        return [self.initial_state]

    def neighbors(self, vertex):
        return self.transitions.get(vertex, [])
    
    
