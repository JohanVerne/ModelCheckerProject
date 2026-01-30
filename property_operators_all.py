"""
5ème partie - Property Operator Interface - COMPLET

Implémente toutes les propriétés Büchi du PDF Alice-Bob-Soup:
- P1: Exclusion mutuelle (never A.CS & B.CS)
- P2: Absence de deadlock (never deadlock)
- P3: Au moins un progresse (eventually A.CS or B.CS)
- P4: Demande garantit entrée (flagUp → eventually CS)
- P5: Progrès non contesté (uncontested progress)
"""

from abc import ABC, abstractmethod
from typing import Callable, Any, List, Tuple, Set
from languagesemantics import LanguageSemantics
from alicebob_semantics import AliceBobState, AB1Semantics, AB2Semantics, AB3Semantics
from ls2rg import LS2RG
from BFS_definition import BFS


# ============================================================================
# INTERFACE Opérateur de Propriété
# ============================================================================

class PropertyOperator(LanguageSemantics):
    """Interface pour opérateur de propriété × système"""

    def __init__(self, system_sem: LanguageSemantics):
        self.system = system_sem

    @abstractmethod
    def check_violation(self, sys_state: Any) -> bool:
        """Retourne True si violation détectée"""
        pass


# ============================================================================
# P1: EXCLUSION MUTUELLE (Safety - Büchi)
# ============================================================================

class MutualExclusionOperator(PropertyOperator):
    """
    P1: never (A.CS & B.CS)

    Automate Büchi:
    - État 1 : normal (boucle sur !A.CS | !B.CS)
    - État 0 : violation (acceptant, piège sur A.CS & B.CS)
    """

    def __init__(self, system_sem: LanguageSemantics):
        super().__init__(system_sem)

    def initials(self) -> List[Tuple[str, Any]]:
        sys_initials = self.system.initials()
        return [("NORMAL", s) for s in sys_initials]

    def actions(self, composed_state: Tuple[str, Any]) -> List[Any]:
        prop_state, sys_state = composed_state
        if prop_state == "VIOLATION":
            return []  # Piège
        return self.system.actions(sys_state)

    def execute(self, composed_state: Tuple[str, Any], action: Any) -> List[Tuple[str, Any]]:
        prop_state, sys_state = composed_state
        next_sys_states = self.system.execute(sys_state, action)

        result = []
        for next_sys in next_sys_states:
            # Vérifier exclusion mutuelle
            if self.check_violation(next_sys):
                result.append(("VIOLATION", next_sys))
            else:
                result.append(("NORMAL", next_sys))
        return result

    def check_violation(self, sys_state: AliceBobState) -> bool:
        """A.CS & B.CS"""
        return sys_state.alice == 'CS' and sys_state.bob == 'CS'


# ============================================================================
# P2: ABSENCE DE DEADLOCK (Safety - Büchi)
# ============================================================================

class NoDeadlockOperator(PropertyOperator):
    """
    P2: never deadlock

    Deadlock = état sans successeur (hors états finaux)
    """

    def initials(self) -> List[Tuple[str, Any]]:
        sys_initials = self.system.initials()
        return [("CHECKING", s) for s in sys_initials]

    def actions(self, composed_state: Tuple[str, Any]) -> List[Any]:
        prop_state, sys_state = composed_state
        if prop_state == "DEADLOCK":
            return []
        return self.system.actions(sys_state)

    def execute(self, composed_state: Tuple[str, Any], action: Any) -> List[Tuple[str, Any]]:
        prop_state, sys_state = composed_state
        next_sys_states = self.system.execute(sys_state, action)

        result = []
        for next_sys in next_sys_states:
            if self.check_violation(next_sys):
                result.append(("DEADLOCK", next_sys))
            else:
                result.append(("CHECKING", next_sys))
        return result

    def check_violation(self, sys_state: AliceBobState) -> bool:
        """Deadlock = 0 actions possibles"""
        return len(self.system.actions(sys_state)) == 0


# ============================================================================
# P3: AU MOINS UN PROGRESSE (Liveness - Büchi)
# ============================================================================

class AtLeastOneInCSOperator(PropertyOperator):
    """
    P3: always eventually (A.CS or B.CS)

    Automate Büchi avec 2 états:
    - x : normal (peut boucler sur q ou !q)
    - y : acceptant (cycle sur !q = violation)

    Violation = cycle infini sans jamais passer par CS
    """

    def initials(self) -> List[Tuple[str, Any]]:
        sys_initials = self.system.initials()
        return [("NORMAL", s) for s in sys_initials]

    def actions(self, composed_state: Tuple[str, Any]) -> List[Any]:
        prop_state, sys_state = composed_state
        return self.system.actions(sys_state)

    def execute(self, composed_state: Tuple[str, Any], action: Any) -> List[Tuple[str, Any]]:
        prop_state, sys_state = composed_state
        next_sys_states = self.system.execute(sys_state, action)

        result = []
        for next_sys in next_sys_states:
            in_cs = next_sys.alice == 'CS' or next_sys.bob == 'CS'

            if in_cs:
                # Retour à normal
                result.append(("NORMAL", next_sys))
            else:
                # Pas en CS : passe à acceptant (potentiel cycle)
                result.append(("ACCEPTING_CYCLE", next_sys))
        return result

    def check_violation(self, sys_state: AliceBobState) -> bool:
        """Violation si jamais en CS"""
        return sys_state.alice != 'CS' and sys_state.bob != 'CS'


# ============================================================================
# P4: DEMANDE GARANTIT ENTRÉE (Liveness - Büchi)
# ============================================================================

class EventualEntryOperator(PropertyOperator):
    """
    P4: (flagAlice=UP → eventually A.CS) & (flagBob=UP → eventually B.CS)

    Automate Büchi avec 3 états:
    - 0 : normal
    - 1 : Alice attend (acceptant si boucle infinie)
    - 2 : Bob attend (acceptant si boucle infinie)
    """

    def initials(self) -> List[Tuple[str, Any]]:
        sys_initials = self.system.initials()
        return [("NORMAL", s) for s in sys_initials]

    def actions(self, composed_state: Tuple[str, Any]) -> List[Any]:
        prop_state, sys_state = composed_state
        return self.system.actions(sys_state)

    def execute(self, composed_state: Tuple[str, Any], action: Any) -> List[Tuple[str, Any]]:
        prop_state, sys_state = composed_state
        next_sys_states = self.system.execute(sys_state, action)

        result = []
        for next_sys in next_sys_states:
            # p0 = flagAlice UP, q0 = Alice@CS
            alice_wants = next_sys.flag_alice == 'UP'
            alice_in_cs = next_sys.alice == 'CS'

            # p1 = flagBob UP, q1 = Bob@CS
            bob_wants = next_sys.flag_bob == 'UP'
            bob_in_cs = next_sys.bob == 'CS'

            # Transitions selon automate P4
            if prop_state == "NORMAL":
                if alice_wants and not alice_in_cs:
                    result.append(("ALICE_WAITING", next_sys))
                elif bob_wants and not bob_in_cs:
                    result.append(("BOB_WAITING", next_sys))
                else:
                    result.append(("NORMAL", next_sys))

            elif prop_state == "ALICE_WAITING":
                if alice_in_cs:
                    # Alice entre : retour normal
                    result.append(("NORMAL", next_sys))
                else:
                    # Reste en attente (cycle = violation)
                    result.append(("ALICE_WAITING", next_sys))

            elif prop_state == "BOB_WAITING":
                if bob_in_cs:
                    result.append(("NORMAL", next_sys))
                else:
                    result.append(("BOB_WAITING", next_sys))

        return result

    def check_violation(self, sys_state: AliceBobState) -> bool:
        """Starvation détectée"""
        alice_starving = sys_state.flag_alice == 'UP' and sys_state.alice != 'CS'
        bob_starving = sys_state.flag_bob == 'UP' and sys_state.bob != 'CS'
        return alice_starving or bob_starving


# ============================================================================
# P5: PROGRÈS NON CONTESTÉ (Fairness - Büchi)
# ============================================================================

class UncontestedProgressOperator(PropertyOperator):
    """
    P5: uncontested progress

    Si Alice attend et Bob n'est pas intéressé → Alice doit progresser
    Si Bob attend et Alice n'est pas intéressée → Bob doit progresser

    Automate Büchi:
    - 0 : normal
    - 1 : Alice attend sans contestation (acceptant si boucle)
    - 2 : Bob attend sans contestation (acceptant si boucle)
    """

    def initials(self) -> List[Tuple[str, Any]]:
        sys_initials = self.system.initials()
        return [("NORMAL", s) for s in sys_initials]

    def actions(self, composed_state: Tuple[str, Any]) -> List[Any]:
        prop_state, sys_state = composed_state
        return self.system.actions(sys_state)

    def execute(self, composed_state: Tuple[str, Any], action: Any) -> List[Tuple[str, Any]]:
        prop_state, sys_state = composed_state
        next_sys_states = self.system.execute(sys_state, action)

        result = []
        for next_sys in next_sys_states:
            # Prédicats
            a_in_cs = next_sys.alice == 'CS'
            a_waiting = next_sys.alice in ['W', 'a1'] and not a_in_cs
            b_not_interested = next_sys.bob == 'W' and next_sys.flag_bob == 'DOWN'

            b_in_cs = next_sys.bob == 'CS'
            b_waiting = next_sys.bob in ['W', 'b1'] and not b_in_cs
            a_not_interested = next_sys.alice == 'W' and next_sys.flag_alice == 'DOWN'

            # Transitions
            if prop_state == "NORMAL":
                if not a_in_cs and a_waiting and b_not_interested:
                    result.append(("ALICE_UNCONTESTED", next_sys))
                elif a_not_interested and not b_in_cs and b_waiting:
                    result.append(("BOB_UNCONTESTED", next_sys))
                else:
                    result.append(("NORMAL", next_sys))

            elif prop_state == "ALICE_UNCONTESTED":
                if a_in_cs:
                    result.append(("NORMAL", next_sys))
                else:
                    result.append(("ALICE_UNCONTESTED", next_sys))

            elif prop_state == "BOB_UNCONTESTED":
                if b_in_cs:
                    result.append(("NORMAL", next_sys))
                else:
                    result.append(("BOB_UNCONTESTED", next_sys))

        return result

    def check_violation(self, sys_state: AliceBobState) -> bool:
        """Unfairness détectée"""
        a_in_cs = sys_state.alice == 'CS'
        a_waiting = sys_state.alice in ['W', 'a1'] and not a_in_cs
        b_not_interested = sys_state.bob == 'W' and sys_state.flag_bob == 'DOWN'

        b_in_cs = sys_state.bob == 'CS'
        b_waiting = sys_state.bob in ['W', 'b1'] and not b_in_cs
        a_not_interested = sys_state.alice == 'W' and sys_state.flag_alice == 'DOWN'

        return (not a_in_cs and a_waiting and b_not_interested) or                (a_not_interested and not b_in_cs and b_waiting)


# ============================================================================
# VÉRIFICATION COMPLÈTE
# ============================================================================

def verify_property(protocol_name: str, property_name: str, operator_class):
    """Vérifie une propriété sur un protocole"""

    # Sélection du protocole
    protocols = {
        'AB1': AB1Semantics(),
        'AB2': AB2Semantics(),
        'AB3': AB3Semantics()
    }

    if protocol_name not in protocols:
        raise ValueError(f"Protocole {protocol_name} inconnu")

    system = protocols[protocol_name]
    operator = operator_class(system)

    # Callback BFS pour détecter violations
    def on_entry(composed_state, opaque):
        prop_state, sys_state = composed_state

        # Détecter états d'acceptation (violations Büchi)
        is_accepting = prop_state in [
            "VIOLATION", "DEADLOCK", 
            "ACCEPTING_CYCLE", "ALICE_WAITING", "BOB_WAITING",
            "ALICE_UNCONTESTED", "BOB_UNCONTESTED"
        ]

        if is_accepting:
            opaque['violations'].append((prop_state, sys_state))

        opaque['visited_count'] += 1
        return False, opaque  # Continue exploration

    # Exécution BFS
    rg = LS2RG(operator)
    initial_opaque = {'violations': [], 'visited_count': 0}
    visited, results = BFS(rg, on_entry, initial_opaque)

    # Résultats
    satisfied = len(results['violations']) == 0

    print(f"\n{'='*70}")
    print(f"{protocol_name} × {property_name}")
    print(f"{'='*70}")
    print(f"États explorés : {results['visited_count']}")
    print(f"Violations : {len(results['violations'])}")
    print(f"Propriété : {'✅ SATISFAITE' if satisfied else '❌ INSATISFAITE'}")

    if not satisfied:
        print(f"\nContre-exemples (premiers 3):")
        for i, (prop_state, sys_state) in enumerate(results['violations'][:3]):
            print(f"  {i+1}. {prop_state} → {sys_state}")

    return satisfied, results


def verify_all_properties():
    """Vérifie toutes les propriétés sur AB1, AB2, AB3"""

    properties = [
        ('P1: Exclusion', MutualExclusionOperator),
        ('P2: No Deadlock', NoDeadlockOperator),
        ('P3: At Least One In CS', AtLeastOneInCSOperator),
        ('P4: Eventual Entry', EventualEntryOperator),
        ('P5: Uncontested Progress', UncontestedProgressOperator)
    ]

    protocols = ['AB1', 'AB2', 'AB3']

    # Tableau récapitulatif
    results_table = {}

    for protocol in protocols:
        results_table[protocol] = {}
        for prop_name, prop_class in properties:
            satisfied, _ = verify_property(protocol, prop_name, prop_class)
            results_table[protocol][prop_name] = satisfied

    # Affichage tableau
    print(f"\n\n{'='*70}")
    print("TABLEAU RÉCAPITULATIF - PROPRIÉTÉS BÜCHI")
    print(f"{'='*70}\n")

    print(f"| Protocole | P1 | P2 | P3 | P4 | P5 |")
    print(f"|-----------|----|----|----|----|-----|")

    for protocol in protocols:
        row = [protocol]
        for prop_name, _ in properties:
            result = results_table[protocol][prop_name]
            row.append("✅" if result else "❌")
        print(f"| {' | '.join(row)} |")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║   VÉRIFICATION COMPLÈTE ALICE & BOB - PROPRIÉTÉS BÜCHI (P1-P5)   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)

    verify_all_properties()

    print("\n✅ Vérification terminée!")
    print("📝 Voir readme_updated.md pour la documentation complète")
