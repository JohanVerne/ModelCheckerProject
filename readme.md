# Model Checker Project

Implémentation progressive d'un vérificateur de modèles basé sur BFS générique pour graphes orientés avec racines.

## Fonctionnalités implémentées

### 1. Abstraction `Rootedgraph`
- Interface abstraite avec `roots()` et `neighbors(vertex)`.
- Compatible avec tous les automates Alice & Bob.

### 2. Implémentations de graphes
- **`DictionaryGraph`** : graphe générique à partir de dictionnaire d'adjacence.
- **`HanoiGraph`** : tours de Hanoï avec \(n\) disques, états comme tuples de 3 tours.

### 3. Algorithme BFS générique
- `BFS(graph, on_entry_callback, opaque)` explore tous les états accessibles.
- **Callbacks flexibles** :
  - `on_entry_check4vertex` : arrête après 4 sommets visités.
  - `on_entry_create_parents` : reconstruit le chemin racine→objectif.
- Tests sur graphes simples + Hanoï (jusqu'à 6 disques).

### 4. Automates Alice & Bob (problème d'exclusion mutuelle)
| Automate | États modélisés | Tests | Exclusion mutuelle |
|----------|-----------------|-------|--------------------|
| **AB1**  | `(alice_state, bob_state)`<br>`W`, `CS` | BFS complet<br>Exclusion `("CS", "CS")` | ✅ Vérifiée |
| **AB2**  | `(alice_state, bob_state, flagAlice, flagBob)`<br>`W`, `A1`, `B1`, `UP/DOWN` | BFS partiel (3 états) | ⏳ En cours |

## Structure des fichiers
```
├── BFS_definition.py        # BFS + abstractions + Hanoi
├── aliceEtBobImplementation.py # AB1, AB2 (classes Rootedgraph)
├── test_ab1.py             # Tests complets AB1
├── test_ab2.py             # Tests BFS AB2
└── README.md              # Ce fichier
```

## Tests fonctionnels
```bash
python test_ab1.py    # AB1: exclusion mutuelle OK
python test_ab2.py    # AB2: 3 états atteignables
```

## Progression à venir
1. Compléter transitions AB2 + tests exclusion/deadlock.
2. Implémenter AB3 (stratégie améliorée avec drapeaux).
3. Vérification deadlock pour AB2/AB3.
4. **Trace de contre-exemple** sans modifier BFS (point 5 du TP).

## Ressources
Professor's link to his course "From Zero to Model-Checking" : [https://teodorov.github.io/z2mc/](https://teodorov.github.io/z2mc/)

***