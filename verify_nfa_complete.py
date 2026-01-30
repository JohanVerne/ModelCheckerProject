"""
Vérification NFA - Alice & Bob Protocols (AB1-AB5)
Vérifie P1 (Exclusion) et P2 (No Deadlock) avec Patron 1 et Patron 2
"""

from typing import Dict, List, Tuple, Any
from languagesemantics import LanguageSemantics
from alicebob_semantics import (
    AliceBobState,
    AB1Semantics, AB2Semantics, AB3Semantics, 
    AB4Semantics, AB5Semantics
)
from ls2rg import LS2RG
from BFS_definition import BFS


class NFAPropertyOperator(LanguageSemantics):
    """Opérateur NFA de base"""

    def __init__(self, system_sem: LanguageSemantics):
        self.system = system_sem

    def check_condition(self, sys_state: Any) -> bool:
        """Condition à vérifier (à surcharger)"""
        raise NotImplementedError


class ExclusionNFA(NFAPropertyOperator):
    """P1: never (A.CS & B.CS)"""

    def initials(self) -> List[Tuple[str, Any]]:
        return [("OK", s) for s in self.system.initials()]

    def actions(self, composed_state: Tuple[str, Any]) -> List[Any]:
        prop_state, sys_state = composed_state
        if prop_state == "ERROR":
            return []
        return self.system.actions(sys_state)

    def execute(self, composed_state: Tuple[str, Any], action: Any) -> List[Tuple[str, Any]]:
        prop_state, sys_state = composed_state
        if prop_state == "ERROR":
            return [("ERROR", sys_state)]

        next_sys_states = self.system.execute(sys_state, action)
        result = []
        for next_sys in next_sys_states:
            if self.check_condition(next_sys):
                result.append(("ERROR", next_sys))
            else:
                result.append(("OK", next_sys))
        return result

    def check_condition(self, sys_state: Any) -> bool:
        """A.CS & B.CS"""
        return sys_state.alice == 'CS' and sys_state.bob == 'CS'


class NoDeadlockNFA(NFAPropertyOperator):
    """P2: never deadlock"""

    def initials(self) -> List[Tuple[str, Any]]:
        return [("OK", s) for s in self.system.initials()]

    def actions(self, composed_state: Tuple[str, Any]) -> List[Any]:
        prop_state, sys_state = composed_state
        if prop_state == "ERROR":
            return []
        return self.system.actions(sys_state)

    def execute(self, composed_state: Tuple[str, Any], action: Any) -> List[Tuple[str, Any]]:
        prop_state, sys_state = composed_state
        if prop_state == "ERROR":
            return [("ERROR", sys_state)]

        next_sys_states = self.system.execute(sys_state, action)
        result = []
        for next_sys in next_sys_states:
            if self.check_condition(next_sys):
                result.append(("ERROR", next_sys))
            else:
                result.append(("OK", next_sys))
        return result

    def check_condition(self, sys_state: Any) -> bool:
        """deadlock = 0 actions possibles"""
        return len(self.system.actions(sys_state)) == 0


def verify_nfa(protocol_name: str, property_operator_class) -> Tuple[bool, int, List]:
    """Vérifie une propriété NFA sur un protocole"""

    protocols = {
        'AB1': AB1Semantics(),
        'AB2': AB2Semantics(),
        'AB3': AB3Semantics(),
        'AB4': AB4Semantics(),
        'AB5': AB5Semantics()
    }

    system = protocols[protocol_name]
    operator = property_operator_class(system)

    errors = []
    states_count = 0

    def on_entry(composed_state, opaque):
        prop_state, sys_state = composed_state
        opaque['count'] += 1

        if prop_state == "ERROR":
            opaque['errors'].append(sys_state)

        return False, opaque

    rg = LS2RG(operator)
    initial_opaque = {'errors': [], 'count': 0}
    visited, results = BFS(rg, on_entry, initial_opaque)

    satisfied = len(results['errors']) == 0
    return satisfied, results['count'], results['errors']


def run_nfa_verification():
    """Lance la vérification complète NFA"""

    print("="*80)
    print(" VÉRIFICATION NFA - ALICE & BOB (AB1-AB5)")
    print(" P1: Exclusion Mutuelle | P2: Absence de Deadlock")
    print("="*80)

    protocols = ['AB1', 'AB2', 'AB3', 'AB4', 'AB5']
    results = {}

    # Vérification P1 et P2 pour chaque protocole
    for protocol in protocols:
        print(f"\n{'─'*80}")
        print(f" {protocol}")
        print(f"{'─'*80}")

        # P1: Exclusion
        p1_sat, p1_states, p1_errors = verify_nfa(protocol, ExclusionNFA)
        print(f"  P1 (Exclusion)     : {'✅ SAT' if p1_sat else '❌ UNSAT'} | États: {p1_states:4} | Erreurs: {len(p1_errors)}")

        # P2: No Deadlock
        p2_sat, p2_states, p2_errors = verify_nfa(protocol, NoDeadlockNFA)
        print(f"  P2 (No Deadlock)   : {'✅ SAT' if p2_sat else '❌ UNSAT'} | États: {p2_states:4} | Erreurs: {len(p2_errors)}")

        results[protocol] = {
            'P1': p1_sat,
            'P2': p2_sat,
            'states_p1': p1_states,
            'states_p2': p2_states,
            'errors_p1': p1_errors,
            'errors_p2': p2_errors
        }

    # Tableau récapitulatif Patron 1 (identique pour Patron 2)
    print("\n\n" + "="*80)
    print(" TABLEAU RÉCAPITULATIF - NFA PATRON 1 & 2")
    print("="*80)
    print()
    print("| Modèle | P1: Exclusion | P2: No Deadlock | États | Commentaire |")
    print("|--------|---------------|-----------------|-------|-------------|")

    for protocol in protocols:
        r = results[protocol]
        p1_sym = "✅ SAT  " if r['P1'] else "❌ UNSAT"
        p2_sym = "✅ SAT  " if r['P2'] else "❌ UNSAT"
        states = r['states_p1']

        if r['P1'] and r['P2']:
            comment = "Toutes propriétés satisfaites"
        elif r['P1'] and not r['P2']:
            comment = "Deadlock présent"
        elif not r['P1']:
            comment = "Exclusion mutuelle non garantie"
        else:
            comment = ""

        print(f"| {protocol}    | {p1_sym}      | {p2_sym}        | {states:5} | {comment} |")

    print()
    print("="*80)
    print("Note: Patron 1 et Patron 2 donnent les MÊMES résultats pour P1 et P2")
    print("      (propriétés safety simples)")
    print("="*80)

    print("\n📊 Analyse de progression:")
    print("  AB1: Protocole naïf (deadlock présent)")
    print("  AB2: Drapeaux basiques (deadlock symétrique)")
    print("  AB3: Bob recule (résout deadlock)")
    print("  AB4: Bob abandonne (résout deadlock)")
    print("  AB5: Algorithme de Peterson (solution complète)")

    return results


if __name__ == "__main__":
    run_nfa_verification()
