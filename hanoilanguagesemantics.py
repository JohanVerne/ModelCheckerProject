from languagesemantics import LanguageSemantics
from copy import deepcopy

# Classe State pour représenter un état de Hanoi
# Utilise des listes mutables pour meilleures performances
class HanoiState:
    def __init__(self, pegs):
        # pegs est une liste de listes: [[3,2,1], [], []]
        self.pegs = pegs
    
    # Méthode pour comparer deux états (nécessaire pour BFS)
    def __eq__(self, other):
        if not isinstance(other, HanoiState):
            return False
        return self.pegs == other.pegs
    
    # Méthode pour rendre l'état hashable (utilisable dans des sets/dicts)
    def __hash__(self):
        # Convertir en tuple pour le rendre hashable
        return hash(tuple(tuple(peg) for peg in self.pegs))
    
    # Représentation lisible de l'état
    def __repr__(self):
        return f"HanoiState({self.pegs})"


# Implémentation de la sémantique du langage pour les tours de Hanoi
# Un état est un HanoiState avec 3 listes (une pour chaque tige)
# Une action est un tuple (tige_source, tige_destination)
class HanoiLanguageSemantics(LanguageSemantics):

    # Constructeur : initialise le problème avec n disques
    def __init__(self, n_disks):
        self.n = n_disks

    # Retourne les états initiaux : tous les disques sur la première tige
    def initials(self):
        initial_state = HanoiState([
            list(range(self.n, 0, -1)),  # tous les disques sur la tige 0
            [],  # tige 1 vide
            []   # tige 2 vide
        ])
        return [initial_state]

    # Retourne toutes les actions possibles depuis un état donné
    # Une action est un tuple (tige_source, tige_destination)
    def actions(self, state):
        possible_actions = []
        pegs = state.pegs

        # Pour chaque tige source (i)
        for i in range(3):
            # Si la tige est vide, on ne peut rien déplacer
            if not pegs[i]:
                continue

            # Récupère le disque du haut
            disk = pegs[i][-1]

            # Pour chaque tige destination (j)
            for j in range(3):
                # Ne peut pas déplacer sur la même tige
                if i == j:
                    continue

                # Mouvement valide si destination vide ou disque plus grand
                if not pegs[j] or pegs[j][-1] > disk:
                    # Ajoute l'action (source, destination)
                    possible_actions.append((i, j))

        return possible_actions

    # Exécute une action sur un état et retourne l'ensemble des états résultants
    def execute(self, state, action):
        source, dest = action
        
        # Deepcopy de l'état complet en une seule opération
        new_pegs = deepcopy(state.pegs)
        
        # Déplace le disque
        disk = new_pegs[source].pop()
        new_pegs[dest].append(disk)
        
        # Retourne un nouvel état
        new_state = HanoiState(new_pegs)
        return [new_state]
    
    # Méthode utilitaire pour vérifier si un état est une solution
    def is_solution(self, state):
        return (state.pegs == [[], [], list(range(self.n, 0, -1))] or
                state.pegs == [[], list(range(self.n, 0, -1)), []])
