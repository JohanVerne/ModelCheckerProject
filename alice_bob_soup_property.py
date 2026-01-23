"""
Alice-Bob Soup + Property Operator 
✅ EXÉCUTABLE machine prof
"""

from languagesemantics import LanguageSemantics
from soupsemantics import SoupSemantics, SoupConfiguration
from ls2rg import LS2RG
from BFS_definition import BFS
from alicebob_semantics import AliceBobState, AB3Semantics

class FairnessOperator(LanguageSemantics):
    """Fairness sur Soup Alice-Bob"""
    
    def __init__(self, soup_sem: SoupSemantics):
        self.soup = soup_sem
        self.alice_turn = True
    
    def initials(self):
        return [("FAIR", self.soup.initials()[0])]
    
    def actions(self, composed_state):
        prop_state, soup_config = composed_state
        if prop_state == "UNFAIR":
            return []
        return self.soup.actions(soup_config)
    
    def execute(self, composed_state, action):
        prop_state, soup_config = composed_state
        next_configs = self.soup.execute(soup_config, action)
        
        result = []
        for next_config in next_configs:
            # Accès CORRECT aux états de la soup
            ab1_state = next_config.states[0]  # 1er AB3
            ab2_state = next_config.states[1]  # 2e AB3
            
            # Starvation si AB1 fait 3 CS d'affilée
            alice_cs = ab1_state.alice == 'CS'
            unfair = alice_cs and not self.alice_turn
            
            new_turn = not self.alice_turn
            new_prop = "UNFAIR" if unfair else "FAIR"
            
            result.append((new_prop, next_config))
        return result

def verify_ab_soup_fairness():
    # Soup : 2 AB3 indépendants
    ab1 = AB3Semantics()
    ab2 = AB3Semantics()
    soup = SoupSemantics([ab1, ab2])
    
    fairness_op = FairnessOperator(soup)
    
    def on_entry(composed, opaque):
        prop_state, soup_config = composed
        
        if prop_state == "UNFAIR":
            opaque['starvation'].append(soup_config)
            return True, opaque
        
        # Stats exclusion mutuelle
        ab1_state = soup_config.states[0]
        ab2_state = soup_config.states[1]
        em_violation = (ab1_state.alice == 'CS' and ab2_state.bob == 'CS')
        opaque['em_violations'] += em_violation
        
        return False, opaque
    
    rg = LS2RG(fairness_op)
    initial_opaque = {'starvation': [], 'em_violations': 0}
    visited, results = BFS(rg, on_entry, initial_opaque)
    
    print("=== Alice-Bob Soup(AB3×AB3) + Fairness ===")
    print(f"Starvation : {len(results['starvation'])}")
    print(f"Exclusion mutuelle violations : {results['em_violations']}")
    print(f"États soup composés : {len(visited)}")

if __name__ == "__main__":
    verify_ab_soup_fairness()
