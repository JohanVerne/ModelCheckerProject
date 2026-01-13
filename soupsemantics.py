from languagesemantics import LanguageSemantics
from ls2rg import LS2RG
from itertools import product
from copy import deepcopy

class SoupConfiguration:
    """Représente une configuration d'une soupe : un tuple d'états (un par système)"""
    
    def __init__(self, states):
        self.states = tuple(states)  # Immutable tuple of states
    
    def __eq__(self, other):
        if not isinstance(other, SoupConfiguration):
            return False
        return self.states == other.states
    
    def __hash__(self):
        return hash(self.states)
    
    def __repr__(self):
        return f"SoupConfig({list(self.states)})"

class SoupSemantics(LanguageSemantics):
    """Combine plusieurs LanguageSemantics en un système couplé"""
    
    def __init__(self, semantics_list):
        self._semantics = semantics_list  # Liste de LanguageSemantics
    
    def initials(self):
        # Produit cartésien des états initiaux de chaque système
        initial_states = [sem.initials() for sem in self._semantics]
        initial_configs = []
        for state_tuple in product(*initial_states):
            initial_configs.append(SoupConfiguration(state_tuple))
        return initial_configs
    
    def actions(self, config):
        """Actions possibles : union des actions de TOUS les systèmes"""
        all_actions = set()
        for i, state in enumerate(config.states):
            sem = self._semantics[i]
            all_actions.update(sem.actions(state))
        return list(all_actions)
    
    def execute(self, config, action):
        """Applique l'action à TOUS les systèmes simultanément"""
        next_configs = []
        
        # Pour chaque système, applique l'action et récupère les successeurs
        next_states_lists = []
        for i, state in enumerate(config.states):
            sem = self._semantics[i]
            next_states = sem.execute(state, action)
            next_states_lists.append(next_states)
        
        # Produit cartésien des successeurs
        for next_tuple in product(*next_states_lists):
            next_configs.append(SoupConfiguration(next_tuple))
        
        return next_configs
    
    def is_solution(self, config):
        """Solution si TOUS les systèmes sont résolus"""
        return all(
            sem.is_solution(state) 
            for sem, state in zip(self._semantics, config.states)
        )

# Adaptateur Soup → RootedGraph (identique à LS2RG)
class Soup2RG(LS2RG):
    """Spécialisation de LS2RG pour les soupes"""
    def __init__(self, soup):
        super().__init__(soup)
