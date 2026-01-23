"""
5ème partie - Property Operator Interface

**Langage Propriété contrôle LanguageSemantics** via 3 méthodes :
- initial() : état initial opérateur
- actions(λ,état) : filtre actions selon prédicat  
- execute(pas,cfg_suivante) : met à jour état opérateur
"""

from abc import ABC, abstractmethod
from typing import Callable, Any, List, Tuple
from languagesemantics import LanguageSemantics
from hanoilanguagesemantics import HanoiLanguageSemantics, HanoiState
from ls2rg import LS2RG
from BFS_definition import BFS

# 1. INTERFACE Opérateur (hérite LanguageSemantics)
class PropertyOperator(LanguageSemantics):
    """Langage Propriété × Système"""
    pass

# 2. LANGAGE DE PROPRIÉTÉ : "Always Progress"
class AlwaysProgressOperator(PropertyOperator):
    """Propriété : jamais bloqué (toujours progression possible)"""
    
    def __init__(self, system_sem: LanguageSemantics):
        self.system = system_sem
        self.prop_state = "CHECKING"  # CHECKING | FOUND_BLOCK | SATISFIED
    
    def initial(self) -> Tuple[str, Any]:
        """État composé initial"""
        return ("CHECKING", self.system.initials()[0])
    
    def initials(self) -> List[Tuple[str, Any]]:
        return [self.initial()]
    
    # INTERFACE LanguageSemantics
    def actions(self, composed_state: Tuple[str, Any]) -> List[Any]:
        """Toutes actions système (contrôle via on_entry)"""
        prop_state, sys_state = composed_state
        if prop_state == "FOUND_BLOCK":
            return []  # Arrêt sur violation
        return self.system.actions(sys_state)
    
    def execute(self, composed_state: Tuple[str, Any], 
                action: Any) -> List[Tuple[str, Any]]:
        """Transition état composé"""
        prop_state, sys_state = composed_state
        next_sys_states = self.system.execute(sys_state, action)
        
        result = []
        for next_sys in next_sys_states:
            # VIOLATION si système bloqué (0 actions ET pas solution)
            sys_actions = len(self.system.actions(next_sys))
            is_sol = self.system.is_solution(next_sys)
            
            if sys_actions == 0 and not is_sol:
                result.append(("FOUND_BLOCK", next_sys))
            else:
                result.append(("CHECKING", next_sys))
        return result

# 3. PRÉDICAT du langage de propriétés
def progress_predicate(sys_state: HanoiState, action: Any) -> bool:
    """λ : action mène à progression (pas deadlock immédiat)"""
    hanoi = HanoiLanguageSemantics(3)
    next_states = hanoi.execute(sys_state, action)
    return any(len(hanoi.actions(ns)) > 0 for ns in next_states)

# 4. VÉRIFICATION avec contrôle opérateur
def verify_always_progress():
    """Langage Propriété contrôle le système"""
    hanoi = HanoiLanguageSemantics(3)
    operator = AlwaysProgressOperator(hanoi)
    
    def on_entry(composed_state, opaque):
        prop_state, sys_state = composed_state
        
        # 1. Filtre actions selon prédicat λ
        available_actions = operator.actions(composed_state)
        safe_actions = [a for a in available_actions 
                       if progress_predicate(sys_state, a)]
        
        # 2. Stats
        opaque['safe_ratio'].append(len(safe_actions) / len(available_actions))
        
        # 3. Violation propriété
        if prop_state == "FOUND_BLOCK":
            opaque['violations'].append(sys_state)
            return True, opaque  # TERMINATE
        
        return False, opaque
    
    # 5. BFS sur espace contrôlé
    rg = LS2RG(operator)
    initial_opaque = {'violations': [], 'safe_ratio': []}
    visited, results = BFS(rg, on_entry, initial_opaque)
    
    print("=== Always Progress Verification ===")
    print(f"Violations (blocage) : {len(results['violations'])}")
    print(f"Ratio actions safe : {sum(results['safe_ratio'])/len(results['safe_ratio']):.2f}")
    print(f"États composés explorés : {len(visited)}")
    
    return len(results['violations']) == 0

if __name__ == "__main__":
    is_safe = verify_always_progress()
    print(f"Hanoï(3) satisfait Always Progress : {is_safe}")
