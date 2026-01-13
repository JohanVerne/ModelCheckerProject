# Validation utilisant l'approche Language Semantics
# Ce fichier démontre comment utiliser LS2RG pour transformer une sémantique
# de langage en graphe enraciné, puis appliquer BFS dessus

from hanoilanguagesemantics import HanoiLanguageSemantics
from ls2rg import LS2RG
from BFS_definition import BFS

# Fonction pour résoudre Hanoi avec l'approche Language Semantics
def hanoi_solver_ls(n):
    # Étape 1: Créer la sémantique du langage pour Hanoi
    ls = HanoiLanguageSemantics(n)
    
    # Étape 2: Transformer la sémantique en graphe enraciné
    rg = LS2RG(ls)
    
    # Étape 3: Définir le callback pour trouver la solution
    def on_entry(state, opaque):
        # Si on trouve une solution, on l'ajoute à opaque
        if ls.is_solution(state):
            opaque.append(state)
        # Arrête le parcours dès qu'une solution est trouvée
        return (ls.is_solution(state), opaque)
    
    # Étape 4: Lancer BFS sur le graphe enraciné
    marked, final_opaque = BFS(rg, on_entry, [])
    return final_opaque, marked  # Retourne (opaque, visited)


# Test avec 3 disques
print("=== Résolution de Hanoi avec Language Semantics ===")
opaque, visited_states = hanoi_solver_ls(3)

print(f"Nombre d'états explorés : {len(visited_states)}")
print(f"Solution trouvée : {opaque}")

# Démonstration des concepts de Language Semantics
print("\n=== Démonstration des méthodes Language Semantics ===")
ls = HanoiLanguageSemantics(3)

# Afficher l'état initial
initial_state = ls.initials()[0]
print(f"État initial : {initial_state}")

# Afficher les actions possibles
actions = ls.actions(initial_state)
print(f"\nActions possibles depuis l'état initial : {actions}")

# Exécuter la première action
if actions:
    first_action = actions[0]
    print(f"\nExécution de l'action {first_action} (déplacer de tige {first_action[0]} vers tige {first_action[1]})")
    next_states = ls.execute(initial_state, first_action)
    print(f"État résultant : {next_states[0]}")
    
    # Afficher les actions possibles depuis ce nouvel état
    next_actions = ls.actions(next_states[0])
    print(f"Actions possibles depuis le nouvel état : {next_actions}")
