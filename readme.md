# 1ère partie - Model Checker Project

Implémentation complète d'un vérificateur de modèles basé sur BFS générique pour graphes orientés avec racines.

## Fonctionnalités implémentées

### 1. Abstraction Rootedgraph

Interface abstraite définissant `roots()` et `neighbors(vertex)`. Compatible avec tous les automates Alice & Bob.

### 2. Implémentations de graphes

- `DictionaryGraph` : graphe générique à partir de dictionnaire d'adjacence
- `HanoiGraph` : tours de Hanoï avec n disques, états représentés comme tuples de 3 tours
- `AB1`, `AB2`, `AB3` : automates Alice & Bob (détails ci-dessous)

### 3. Algorithme BFS générique

`BFS(graph, on_entry_callback, opaque)` explore tous les états accessibles depuis les racines.

Callbacks implémentés :

- `on_entry_check4vertex` : arrête après 4 sommets visités
- `on_entry_create_parents` : reconstruit le chemin racine→objectif via dictionnaire parents

### 4. Automates Alice & Bob - Exclusion mutuelle

| Automate | Représentation d'état        | États principaux                       | Exclusion mutuelle | Deadlock    |
| -------- | ---------------------------- | -------------------------------------- | ------------------ | ----------- |
| AB1      | `(alice, bob)`               | W, CS                                  | **Présente**       | Aucun       |
| AB2      | `(alice, bob, flagA, flagB)` | I, W, CS, UP/DOWN                      | Vérifiée           | **Présent** |
| AB3      | `(alice, bob, flagA, flagB)` | I, W, CS, UP/DOWN + résolution conflit | Vérifiée           | Aucun       |

### 5. Tests de vérification

Fonction `has_deadlock_candidates(marked, graph)` présente dans chaque fichier test :

- Recherche états accessibles sans successeurs (deadlocks potentiels)
- Exclusion mutuelle vérifiée par assertion sur états CS simultanés

### 6. Traces de contre-exemple

`test_contre_exemple.py` utilise `on_entry_create_parents` pour chercher des chemins vers :

- États CS simultanés `("CS", "CS", ...)`
- États drapeaux incohérents `("W", "W", "UP", "UP")`

## Structure des fichiers

```
├── BFS_definition.py              # BFS générique + HanoiGraph + abstractions
├── aliceEtBobImplementation.py    # AB1, AB2, AB3 (classes Rootedgraph)
├── test_ab1.py                    # Tests AB1 : exclusion + deadlock
├── test_ab2.py                    # Tests AB2 : exclusion + deadlock
├── test_ab3.py                    # Tests AB3 : exclusion + deadlock
├── test_contre_exemple.py         # Traces vers états "mauvais"
└── README.md                      # Documentation
```

## Commandes de test

```bash
python test_ab1.py          # AB1 : exclusion OK, pas de deadlock
python test_ab2.py          # AB2 : exclusion OK, pas de deadlock
python test_ab3.py          # AB3 : exclusion OK, deadlock suspect détecté
python test_contre_exemple.py  # Point 5 : traces impossibles vers mauvais états
```

## Couverture du sujet

1. ✅ BFS fonctionnel testé
2. ✅ Hanoi fonctionnel testé
3. ✅ AB1, AB2, AB3 encodés avec RootedGraph
4. ✅ Exclusion mutuelle vérifiée pour chaque AB
5. ✅ Deadlock vérifié pour chaque AB
6. ✅ Traces de contre-exemple sans modifier BFS

## Progression Git

Commits réguliers sur 3 semaines avec :

- Fonctionnalités testables à chaque étape
- Tests unitaires dédiés par automate
- Documentation évolutive

# 2ème partie - Language Semantics to Rooted Graph

```
LanguageSemantics          LS2RG          RootedGraph
├── initials()    ─────→   roots() ─────→   BFS
├── actions(state) ─────→  neighbors()    └── on_entry()
└── execute(state,a) ─────┘                └── visited_states
```

**Objectif** : Transformer une sémantique de langage en graphe enraciné pour BFS.

## Implémentation

| Au tableau          | Fichier                     | Implémentation              |
| ------------------- | --------------------------- | --------------------------- |
| `LanguageSemantics` | `languagesemantics.py`      | Interface abstraite         |
| `initials()`        | `hanoilanguagesemantics.py` | État initial Hanoï          |
| `actions(state)`    | `hanoilanguagesemantics.py` | Déplacements valides        |
| `execute(state,a)`  | `hanoilanguagesemantics.py` | Nouvel état après action    |
| `LS2RG`             | `ls2rg.py`                  | Adaptateur vers RootedGraph |
| `roots()`           | `ls2rg.py`                  | `self._ls.initials()`       |
| `neighbors()`       | `ls2rg.py`                  | `actions()` + `execute()`   |
| `BFS(on_entry)`     | `hanoi_solver_ls.py`        | `breadth_first_search()`    |

## Flux complet

```
HanoiLanguageSemantics → LS2RG → RootedGraph → BFS → Solution
```

# 3ème partie - Soup Semantics

**Généralisation LS2RG** pour systèmes **multi-composants couplés**.

```
SoupSemantics([Sem1,Sem2,...]) → Soup2RG → RootedGraph → BFS
```

## Implémentation

```
soupsemantics.py
├── SoupConfiguration : tuple d'états (state1, state2, ...)
├── SoupSemantics(LanguageSemantics)
│   ├── initials() : produit cartésien des initial_states
│   ├── actions() : union des actions possibles
│   └── execute() : produit cartésien des transitions
└── Soup2RG(LS2RG) : adaptateur
```

## Tests

```
soup_validation.py
├── 2×Hanoï(2) : 7 états → SoupConfig([[], [2,1]], [[], [2,1]])
└── 3×Hanoï(2) : 27 états → tous les 3 systèmes résolus
```

## Complexité

| Systèmes       | États max | Explorés BFS |
| -------------- | --------- | ------------ |
| 1 Hanoï(2)     | 3         | 3            |
| **2 Hanoï(2)** | **9**     | **7**        |
| **3 Hanoï(2)** | **27**    | **27**       |

# 4ème partie - Vérification Alice & Bob

**Model Checking** des protocoles AB1/AB2/AB3 avec `LanguageSemantics → LS2RG → BFS`.

## Fichiers

```
alicebob/
├── alicebob_semantics.py  # AliceBobState + AB1/AB2/AB3 implémentations
├── test_ab1_semantics.py
├── test_ab2_semantics.py
└── test_ab3_semantics.py
```

## Résultats

| Protocole | Exclusion | Deadlock | États |
| --------- | --------- | -------- | ----- |
| **AB1**   | 0 ✅      | **1 ❌** | 12    |
| **AB2**   | 0 ✅      | **1 ❌** | 8     |
| **AB3**   | 0 ✅      | **0 ✅** | 10    |

## Analyse

- **AB1** : Deadlock `(a2,b2)` classique
- **AB2** : Deadlock flags `UP/UP`
- **AB3** : **Correct** grâce à `b4` (Bob cède)

**AB3 assure l'exclusion mutuelle SANS deadlock** ! 🎯

# 5ème partie - Produit synchrone & Property Verification

**Produit synchrone** : Composition d'un programme avec un automate de propriété pour vérification.

```
SoupProgram × SoupSemantics → SynchronousProduct → LS2RG → BFS → Accepting States
```

## Architecture

```
synchronous_product.py
├── ProductConfiguration((program_state, property_state))
├── PropertySemantics (interface)
│   ├── initial() : état initial de la propriété
│   ├── accepts(state) : états acceptants
│   ├── actions(prop_state, prog_state) : filtrage d'actions
│   └── execute(prop_state, action, next_prog_state) : transition
├── SynchronousProduct(LanguageSemantics)
│   ├── initials() : produit cartésien des états initiaux
│   ├── actions() : intersection des actions (filtrées par propriété)
│   └── execute() : exécution parallèle programme + propriété
└── AcceptingStateChecker : callback BFS pour états acceptants
```

## Pipeline complet

```
AB1Semantics ──→ AB1WithUniqueActions ──┐
                                        ├──→ SynchronousProduct ──→ LS2RG ──→ BFS ──→ Accepting/Error states
MutualExclusionProperty ────────────────┘
```

## Propriétés implémentées

### Safety Properties (Büchi automata)

| Propriété              | Classe                    | États     | Acceptants | Description                    |
| ---------------------- | ------------------------- | --------- | ---------- | ------------------------------ |
| **Exclusion mutuelle** | `MutualExclusionProperty` | OK, ERROR | OK         | `¬(alice_CS ∧ bob_CS)`         |
| **Absence deadlock**   | `NoDeadlockProperty`      | OK, ERROR | OK         | Toujours une action disponible |

### Liveness Properties (Büchi automata)

| Propriété   | Classe                 | États              | Acceptants | Description            |
| ----------- | ---------------------- | ------------------ | ---------- | ---------------------- |
| **Progrès** | `EventuallyCSProperty` | WAITING, SATISFIED | SATISFIED  | `◇(alice_CS ∨ bob_CS)` |

## Fichiers

```
synchronous_product.py
├── ProductConfiguration           # États du produit
├── PropertySemantics              # Interface propriété
├── SynchronousProduct             # Composition programme × propriété
├── MutualExclusionProperty        # Propriété sûreté
├── EventuallyCSProperty           # Propriété vivacité
├── NoDeadlockProperty             # Détection deadlock
└── AB1WithUniqueActions           # Wrapper pour AB1

full_pipeline.py
├── test_product_structure()       # Test configuration produit
├── test_ab1_with_mutex_property() # AB1 viole exclusion mutuelle ❌
├── test_ab2_with_mutex_property() # AB2 respecte exclusion ✅
└── test_ab3_with_mutex_property() # AB3 respecte exclusion ✅
```

## Résultats de vérification

| Protocole | Wrapper              | Propriété          | États explorés | Violations | Résultat     |
| --------- | -------------------- | ------------------ | -------------- | ---------- | ------------ |
| **AB1**   | AB1WithUniqueActions | Exclusion mutuelle | 16+            | **> 0**    | ❌ VIOLATION |
| **AB2**   | AB2Semantics         | Exclusion mutuelle | 8              | **0**      | ✅ CORRECT   |
| **AB3**   | AB3Semantics         | Exclusion mutuelle | 10             | **0**      | ✅ CORRECT   |

## Commandes de test

```bash
python full_pipeline.py
```

## Extension vers Büchi automata complets

Les propriétés implémentées sont des automates de Büchi simplifiés. Pour un model checker complet, il faudrait :

- Support des cycles acceptants (liveness complète)
- Algorithmes de recherche de cycles (Nested DFS, Tarjan)
- Opérateurs temporels LTL (Always, Until, Release)
- Négation de propriétés (complémentation Büchi)

_Reste à faire : Tester tous les Alice et Bob avec différents patrons / toutes les propriétés ..._

# Bilan

✅ **Produit synchrone opérationnel**  
✅ **Propriétés safety vérifiables**  
✅ **Détection violations AB1**  
✅ **Validation correctness AB2/AB3**  
✅ **Pipeline complet fonctionnel**

## Ressources

Professor's link to his course "From Zero to Model-Checking" : [https://teodorov.github.io/z2mc/](https://teodorov.github.io/z2mc/)
