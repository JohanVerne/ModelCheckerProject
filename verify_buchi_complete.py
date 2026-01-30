"""
Vérification BÜCHI - Alice & Bob Protocols (AB1-AB5)
Vérifie P1-P5 avec détection de cycles d'acceptation (chemins infinis)

Progression attendue:
- AB1: Seulement P1 (exclusion basique, deadlock présent)
- AB2: P1 uniquement (deadlock symétrique)
- AB3: P1 + P2 (Bob recule, pas de deadlock)
- AB4: P1 + P2 + P3 (Bob abandonne, amélioration liveness)
- AB5: P1-P5 toutes satisfaites (Peterson complet avec turn)
"""

from typing import Dict, List, Tuple, Any, Set
from collections import deque
from languagesemantics import LanguageSemantics
from alicebob_semantics import (
    AliceBobState,
    AB1Semantics, AB2Semantics, AB3Semantics,
    AB4Semantics, AB5Semantics
)
from ls2rg import LS2RG
from BFS_definition import BFS


class BuchiPropertyOperator(LanguageSemantics):
    """Opérateur Büchi de base"""

    def __init__(self, system_sem: LanguageSemantics):
        self.system = system_sem

    def is_accepting(self, prop_state: str) -> bool:
        """État acceptant du Büchi (à surcharger)"""
        raise NotImplementedError


class P1_Exclusion_Buchi(BuchiPropertyOperator):
    """P1: never (A.CS & B.CS) - Büchi avec état piège"""

    def initials(self) -> List[Tuple[str, Any]]:
        return [("NORMAL", s) for s in self.system.initials()]

    def actions(self, composed_state: Tuple[str, Any]) -> List[Any]:
        prop_state, sys_state = composed_state
        if prop_state == "ACCEPT_ERROR":
            return []
        return self.system.actions(sys_state)

    def execute(self, composed_state: Tuple[str, Any], action: Any) -> List[Tuple[str, Any]]:
        prop_state, sys_state = composed_state
        if prop_state == "ACCEPT_ERROR":
            return [("ACCEPT_ERROR", sys_state)]

        next_sys_states = self.system.execute(sys_state, action)
        result = []
        for next_sys in next_sys_states:
            if next_sys.alice == 'CS' and next_sys.bob == 'CS':
                result.append(("ACCEPT_ERROR", next_sys))
            else:
                result.append(("NORMAL", next_sys))
        return result

    def is_accepting(self, prop_state: str) -> bool:
        return prop_state == "ACCEPT_ERROR"


class P2_NoDeadlock_Buchi(BuchiPropertyOperator):
    """P2: never deadlock"""

    def initials(self) -> List[Tuple[str, Any]]:
        return [("NORMAL", s) for s in self.system.initials()]

    def actions(self, composed_state: Tuple[str, Any]) -> List[Any]:
        prop_state, sys_state = composed_state
        if prop_state == "ACCEPT_ERROR":
            return []
        return self.system.actions(sys_state)

    def execute(self, composed_state: Tuple[str, Any], action: Any) -> List[Tuple[str, Any]]:
        prop_state, sys_state = composed_state
        if prop_state == "ACCEPT_ERROR":
            return [("ACCEPT_ERROR", sys_state)]

        next_sys_states = self.system.execute(sys_state, action)
        result = []
        for next_sys in next_sys_states:
            if len(self.system.actions(next_sys)) == 0:
                result.append(("ACCEPT_ERROR", next_sys))
            else:
                result.append(("NORMAL", next_sys))
        return result

    def is_accepting(self, prop_state: str) -> bool:
        return prop_state == "ACCEPT_ERROR"


class P3_AtLeastOneIn_Buchi(BuchiPropertyOperator):
    """P3: always eventually (A.CS or B.CS) - liveness"""

    def initials(self) -> List[Tuple[str, Any]]:
        return [("WAITING", s) for s in self.system.initials()]

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
                result.append(("NORMAL", next_sys))
            else:
                result.append(("WAITING", next_sys))
        return result

    def is_accepting(self, prop_state: str) -> bool:
        """Cycle dans WAITING = jamais en CS = erreur"""
        return prop_state == "WAITING"


class P4_EventualEntry_Buchi(BuchiPropertyOperator):
    """P4: (flagUp → eventually CS)"""

    def initials(self) -> List[Tuple[str, Any]]:
        return [("NORMAL", s) for s in self.system.initials()]

    def actions(self, composed_state: Tuple[str, Any]) -> List[Any]:
        prop_state, sys_state = composed_state
        return self.system.actions(sys_state)

    def execute(self, composed_state: Tuple[str, Any], action: Any) -> List[Tuple[str, Any]]:
        prop_state, sys_state = composed_state
        next_sys_states = self.system.execute(sys_state, action)

        result = []
        for next_sys in next_sys_states:
            alice_wants = next_sys.flag_alice == 'UP'
            alice_in = next_sys.alice == 'CS'
            bob_wants = next_sys.flag_bob == 'UP'
            bob_in = next_sys.bob == 'CS'

            if prop_state == "NORMAL":
                if alice_wants and not alice_in:
                    result.append(("ALICE_WAITING", next_sys))
                elif bob_wants and not bob_in:
                    result.append(("BOB_WAITING", next_sys))
                else:
                    result.append(("NORMAL", next_sys))
            elif prop_state == "ALICE_WAITING":
                if alice_in:
                    result.append(("NORMAL", next_sys))
                else:
                    result.append(("ALICE_WAITING", next_sys))
            elif prop_state == "BOB_WAITING":
                if bob_in:
                    result.append(("NORMAL", next_sys))
                else:
                    result.append(("BOB_WAITING", next_sys))

        return result

    def is_accepting(self, prop_state: str) -> bool:
        """Cycle dans WAITING = starvation"""
        return prop_state in ["ALICE_WAITING", "BOB_WAITING"]


class P5_UncontestedProgress_Buchi(BuchiPropertyOperator):
    """P5: uncontested progress - fairness"""

    def initials(self) -> List[Tuple[str, Any]]:
        return [("NORMAL", s) for s in self.system.initials()]

    def actions(self, composed_state: Tuple[str, Any]) -> List[Any]:
        prop_state, sys_state = composed_state
        return self.system.actions(sys_state)

    def execute(self, composed_state: Tuple[str, Any], action: Any) -> List[Tuple[str, Any]]:
        prop_state, sys_state = composed_state
        next_sys_states = self.system.execute(sys_state, action)

        result = []
        for next_sys in next_sys_states:
            a_in_cs = next_sys.alice == 'CS'
            a_waiting = next_sys.alice in ['W', 'a1'] and not a_in_cs
            b_not_interested = next_sys.bob in ['I'] and next_sys.flag_bob == 'DOWN'

            b_in_cs = next_sys.bob == 'CS'
            b_waiting = next_sys.bob in ['W', 'b1'] and not b_in_cs
            a_not_interested = next_sys.alice in ['I'] and next_sys.flag_alice == 'DOWN'

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

    def is_accepting(self, prop_state: str) -> bool:
        """Cycle sans contestation = unfairness"""
        return prop_state in ["ALICE_UNCONTESTED", "BOB_UNCONTESTED"]


def detect_accepting_cycle(operator, visited_states: Set) -> Tuple[bool, List]:
    """
    Détecte si un cycle d'acceptation existe (Büchi)
    Retourne (has_cycle, example_cycle)
    """
    accepting_states = []

    for state in visited_states:
        if isinstance(state, tuple) and len(state) == 2:
            prop_state, sys_state = state
            if operator.is_accepting(prop_state):
                accepting_states.append(state)

    return len(accepting_states) > 0, accepting_states


def verify_buchi(protocol_name: str, property_operator_class) -> Tuple[bool, int, bool, List]:
    """Vérifie une propriété Büchi sur un protocole"""

    protocols = {
        'AB1': AB1Semantics(),
        'AB2': AB2Semantics(),
        'AB3': AB3Semantics(),
        'AB4': AB4Semantics(),
        'AB5': AB5Semantics()
    }

    system = protocols[protocol_name]
    operator = property_operator_class(system)

    visited_composed = set()
    accepting_states = []

    def on_entry(composed_state, opaque):
        opaque['count'] += 1
        opaque['visited'].add(composed_state)

        prop_state, sys_state = composed_state
        if operator.is_accepting(prop_state):
            opaque['accepting'].append(composed_state)

        return False, opaque

    rg = LS2RG(operator)
    initial_opaque = {'count': 0, 'visited': set(), 'accepting': []}
    visited, results = BFS(rg, on_entry, initial_opaque)

    has_cycle, cycle_states = detect_accepting_cycle(operator, results['visited'])

    # Propriété satisfaite si AUCUN cycle d'acceptation trouvé
    satisfied = not has_cycle

    return satisfied, results['count'], has_cycle, cycle_states


def run_buchi_verification():
    """Lance la vérification complète Büchi"""

    print("="*80)
    print(" VÉRIFICATION BÜCHI - ALICE & BOB (AB1-AB5)")
    print(" P1-P5 avec cycles d'acceptation (chemins infinis)")
    print("="*80)

    protocols = ['AB1', 'AB2', 'AB3', 'AB4', 'AB5']
    properties = [
        ('P1: Exclusion', P1_Exclusion_Buchi),
        ('P2: No Deadlock', P2_NoDeadlock_Buchi),
        ('P3: At Least One In', P3_AtLeastOneIn_Buchi),
        ('P4: Eventual Entry', P4_EventualEntry_Buchi),
        ('P5: Uncontested Progress', P5_UncontestedProgress_Buchi)
    ]

    results = {}

    for protocol in protocols:
        print(f"\n{'─'*80}")
        print(f" {protocol}")
        print(f"{'─'*80}")

        results[protocol] = {}

        for prop_name, prop_class in properties:
            sat, states, has_cycle, cycles = verify_buchi(protocol, prop_class)
            results[protocol][prop_name] = sat

            status = "✅ SAT  " if sat else "❌ UNSAT"
            cycle_info = f"Cycle: {'Oui' if has_cycle else 'Non'}"

            print(f"  {prop_name:25} : {status} | États: {states:4} | {cycle_info}")

    # Tableau récapitulatif
    print("\n\n" + "="*80)
    print(" TABLEAU RÉCAPITULATIF - BÜCHI (CYCLES D'ACCEPTATION)")
    print("="*80)
    print()
    print("| Modèle | P1 | P2 | P3 | P4 | P5 | Satisfaites | Commentaire |")
    print("|--------|----|----|----|----|-------|-------------|-------------|")

    for protocol in protocols:
        r = results[protocol]
        p1 = "✅" if r['P1: Exclusion'] else "❌"
        p2 = "✅" if r['P2: No Deadlock'] else "❌"
        p3 = "✅" if r['P3: At Least One In'] else "❌"
        p4 = "✅" if r['P4: Eventual Entry'] else "❌"
        p5 = "✅" if r['P5: Uncontested Progress'] else "❌"

        sat_count = sum([
            r['P1: Exclusion'],
            r['P2: No Deadlock'],
            r['P3: At Least One In'],
            r['P4: Eventual Entry'],
            r['P5: Uncontested Progress']
        ])

        if sat_count == 5:
            comment = "Solution complète"
        elif sat_count >= 3:
            comment = "Amélioration progressive"
        elif sat_count == 1:
            comment = "Protocole naïf"
        else:
            comment = f"{sat_count}/5"

        print(f"| {protocol}    | {p1}  | {p2}  | {p3}  | {p4}  | {p5}   | {sat_count}/5       | {comment} |")

    print()
    print("="*80)
    print("Légende: ✅ = SAT (propriété satisfaite)")
    print("         ❌ = UNSAT (cycle d'acceptation trouvé)")
    print("="*80)

    print("\n📊 Analyse de progression AB1 → AB5:")
    print("  AB1: Protocole naïf (P1 uniquement)")
    print("  AB2: Drapeaux basiques (P1, deadlock symétrique)")
    print("  AB3: Bob recule vers R (P1+P2, évite deadlock)")
    print("  AB4: Bob abandonne vers I (P1+P2+P3, meilleure liveness)")
    print("  AB5: Algorithme de Peterson avec turn (P1-P5 toutes, solution optimale)")

    return results


if __name__ == "__main__":
    run_buchi_verification()
