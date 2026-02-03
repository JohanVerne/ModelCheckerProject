from languagesemantics import LanguageSemantics
from soupsemantics import SoupSemantics, Soup2RG
from ls2rg import LS2RG
from BFS_definition import BFS

class AliceBobState:
    """État : (alice_pos, bob_pos, flags)"""
    def __init__(self, alice, bob, flag_alice='DOWN', flag_bob='DOWN'):
        self.alice = alice  # 'W', 'CS', 'a1', 'a2', 'a3'
        self.bob = bob      # 'W', 'CS', 'b1', 'b2', 'b3', 'b4'
        self.flag_alice = flag_alice
        self.flag_bob = flag_bob
    
    def __eq__(self, other):
        return (self.alice, self.bob, self.flag_alice, self.flag_bob) == \
               (other.alice, other.bob, other.flag_alice, other.flag_bob)
    
    def __hash__(self):
        return hash((self.alice, self.bob, self.flag_alice, self.flag_bob))
    
    def __repr__(self):
        return f"AB({self.alice},{self.bob},{self.flag_alice},{self.flag_bob})"

from languagesemantics import LanguageSemantics

class AB1Semantics(LanguageSemantics):
    """Protocole AB1 : pas de drapeaux, juste la progression vers CS"""

    def initials(self):
        return [AliceBobState('W', 'W')]  # pas de flags utilisés ici

    def actions(self, state):
        actions = []

        # Alice
        if state.alice == 'W':
            actions.append('a1')
        elif state.alice == 'a1':
            actions.append('a2')
        elif state.alice == 'a2':
            actions.append('CS')
        elif state.alice == 'CS':
            actions.append('W')

        # Bob
        if state.bob == 'W':
            actions.append('b1')
        elif state.bob == 'b1':
            actions.append('b2')
        elif state.bob == 'b2':
            actions.append('CS')
        elif state.bob == 'CS':
            actions.append('W')

        return actions

    def execute(self, state, action):
        new_state = AliceBobState(
            state.alice, state.bob, state.flag_alice, state.flag_bob
        )

        if action.startswith('a'):
            if action == 'a1':
                new_state.alice = 'a1'
            elif action == 'a2':
                new_state.alice = 'a2'
            elif action == 'CS':
                new_state.alice = 'CS'
            elif action == 'W':
                new_state.alice = 'W'
        else:
            if action == 'b1':
                new_state.bob = 'b1'
            elif action == 'b2':
                new_state.bob = 'b2'
            elif action == 'CS':
                new_state.bob = 'CS'
            elif action == 'W':
                new_state.bob = 'W'

        return [new_state]

    def is_exclusion_violation(self, state):
        return state.alice == 'CS' and state.bob == 'CS'

    def is_deadlock(self, state):
        # deadlock classique a2/b2
        return state.alice == 'a2' and state.bob == 'b2'


class AB2Semantics(LanguageSemantics):
    """Protocole AB2 : stratégie simple à base de drapeaux"""

    def initials(self):
        return [AliceBobState('W', 'W', 'DOWN', 'DOWN')]

    def actions(self, state):
        actions = []

        # Alice (hisse son drapeau, entre en CS si Bob est DOWN, puis rebaisse)
        if state.alice == 'W' and state.flag_alice == 'DOWN':
            actions.append('a1')  # flagAlice UP
        elif state.alice == 'a1' and state.flag_bob == 'DOWN':
            actions.append('a2')  # CS
        elif state.alice == 'a2':
            actions.append('a3')  # retourne en W et flagAlice DOWN

        # Bob (symétrique)
        if state.bob == 'W' and state.flag_bob == 'DOWN':
            actions.append('b1')  # flagBob UP
        elif state.bob == 'b1' and state.flag_alice == 'DOWN':
            actions.append('b2')  # CS
        elif state.bob == 'b2':
            actions.append('b3')  # retourne en W et flagBob DOWN

        return actions

    def execute(self, state, action):
        new_state = AliceBobState(
            state.alice, state.bob, state.flag_alice, state.flag_bob
        )

        # Alice
        if action == 'a1':
            new_state.alice = 'a1'
            new_state.flag_alice = 'UP'
        elif action == 'a2':
            new_state.alice = 'CS'
        elif action == 'a3':
            new_state.alice = 'W'
            new_state.flag_alice = 'DOWN'

        # Bob
        elif action == 'b1':
            new_state.bob = 'b1'
            new_state.flag_bob = 'UP'
        elif action == 'b2':
            new_state.bob = 'CS'
        elif action == 'b3':
            new_state.bob = 'W'
            new_state.flag_bob = 'DOWN'

        return [new_state]

    def is_exclusion_violation(self, state):
        return state.alice == 'CS' and state.bob == 'CS'

    def is_deadlock(self, state):
        # exemple de situation de blocage : les deux en attente avec leurs drapeaux UP
        return (
            state.alice == 'a1'
            and state.bob == 'b1'
            and state.flag_alice == 'UP'
            and state.flag_bob == 'UP'
        )


class AB3Semantics(LanguageSemantics):
    """Protocole AB3 : Bob baisse son drapeau s'il voit celui d'Alice hissé"""

    def initials(self):
        return [AliceBobState('W', 'W', 'DOWN', 'DOWN')]

    def actions(self, state):
        actions = []

        # Alice : même logique que AB2
        if state.alice == 'W' and state.flag_alice == 'DOWN':
            actions.append('a1')  # flagAlice UP
        elif state.alice == 'a1' and state.flag_bob == 'DOWN':
            actions.append('a2')  # CS
        elif state.alice == 'a2':
            actions.append('a3')  # retourne en W, flagAlice DOWN

        # Bob AB3 :
        # il hisse son drapeau, mais si celui d'Alice est UP, il choisit de le baisser (b4)
        if state.bob == 'W' and state.flag_bob == 'DOWN':
            actions.append('b1')  # flagBob UP
        elif state.bob == 'b1':
            if state.flag_alice == 'UP':
                actions.append('b4')  # baisse son drapeau s'il voit Alice UP
            else:
                actions.append('b2')  # sinon, il peut aller en CS
        elif state.bob == 'b2':
            actions.append('b3')  # sortie de CS, retour en W + flag DOWN
        elif state.bob == 'b4':
            actions.append('b3')  # passe par b3 pour retourner à W

        return actions

    def execute(self, state, action):
        new_state = AliceBobState(
            state.alice, state.bob, state.flag_alice, state.flag_bob
        )

        # Alice (comme AB2)
        if action == 'a1':
            new_state.alice = 'a1'
            new_state.flag_alice = 'UP'
        elif action == 'a2':
            new_state.alice = 'CS'
        elif action == 'a3':
            new_state.alice = 'W'
            new_state.flag_alice = 'DOWN'

        # Bob AB3
        elif action == 'b1':
            new_state.bob = 'b1'
            new_state.flag_bob = 'UP'
        elif action == 'b2':
            new_state.bob = 'CS'
        elif action == 'b3':
            new_state.bob = 'W'
            new_state.flag_bob = 'DOWN'
        elif action == 'b4':
            # Bob baisse son drapeau mais reste dans un état intermédiaire
            new_state.bob = 'b4'
            new_state.flag_bob = 'DOWN'

        return [new_state]

    def is_exclusion_violation(self, state):
        return state.alice == 'CS' and state.bob == 'CS'

    def is_deadlock(self, state):
        # idéalement AB3 doit éviter tout deadlock ; on peut garder un test défensif
        return False
    

class AB4Semantics(LanguageSemantics):
    """Bob abandonne complètement si Alice a son drapeau UP"""

    def initials(self):
        return [AliceBobState('W', 'W', 'DOWN', 'DOWN')]

    def actions(self, state):
        actions = []

        # ALICE (identique à AB2/AB3)
        if state.alice == 'W' and state.flag_alice == 'DOWN':
            actions.append('a1')
        elif state.alice == 'a1' and state.flag_bob == 'DOWN':
            actions.append('a2')
        elif state.alice == 'a2':
            actions.append('a3')

        # BOB avec ABANDON
        if state.bob == 'W' and state.flag_bob == 'DOWN':
            actions.append('b1')
        elif state.bob == 'b1':
            if state.flag_alice == 'UP':
                actions.append('b4')  # ABANDON
            else:
                actions.append('b2')  # Entre en CS
        elif state.bob == 'b2':
            actions.append('b3')

        return actions

    def execute(self, state, action):
        new_state = AliceBobState(
            state.alice, state.bob, state.flag_alice, state.flag_bob
        )

        # ALICE
        if action == 'a1':
            new_state.alice = 'a1'
            new_state.flag_alice = 'UP'
        elif action == 'a2':
            new_state.alice = 'CS'
        elif action == 'a3':
            new_state.alice = 'W'
            new_state.flag_alice = 'DOWN'

        # BOB
        elif action == 'b1':
            new_state.bob = 'b1'
            new_state.flag_bob = 'UP'
        elif action == 'b2':
            new_state.bob = 'CS'
        elif action == 'b3':
            new_state.bob = 'W'
            new_state.flag_bob = 'DOWN'
        elif action == 'b4':
            # ABANDON : retour direct à W avec flag DOWN
            new_state.bob = 'W'
            new_state.flag_bob = 'DOWN'

        return [new_state]

class AB5Semantics(LanguageSemantics):
    """Les DEUX processus peuvent reculer temporairement"""

    def initials(self):
        return [AliceBobState('W', 'W', 'DOWN', 'DOWN')]

    def actions(self, state):
        actions = []

        # ALICE avec RECUL
        if state.alice == 'W' and state.flag_alice == 'DOWN':
            actions.append('a1')
        elif state.alice == 'a1':
            if state.flag_bob == 'DOWN':
                actions.append('a2')  # Entre en CS
            elif state.flag_bob == 'UP':
                actions.append('a4')  # RECUL
        elif state.alice == 'a4':
            if state.flag_bob == 'DOWN' or state.bob == 'CS':
                actions.append('a1')  # Retente
        elif state.alice == 'a2':
            actions.append('a3')
        elif state.alice == 'CS':
            actions.append('a3')

        # BOB avec RECUL (symétrique)
        if state.bob == 'W' and state.flag_bob == 'DOWN':
            actions.append('b1')
        elif state.bob == 'b1':
            if state.flag_alice == 'DOWN':
                actions.append('b2')  # Entre en CS
            elif state.flag_alice == 'UP':
                actions.append('b4')  # RECUL
        elif state.bob == 'b4':
            if state.flag_alice == 'DOWN' or state.alice == 'CS':
                actions.append('b1')  # Retente
        elif state.bob == 'b2':
            actions.append('b3')
        elif state.bob == 'CS':
            actions.append('b3')

        return actions

    def execute(self, state, action):
        new_state = AliceBobState(
            state.alice, state.bob, state.flag_alice, state.flag_bob
        )

        # ALICE
        if action == 'a1':
            new_state.alice = 'a1'
            new_state.flag_alice = 'UP'
        elif action == 'a2':
            new_state.alice = 'CS'
        elif action == 'a3':
            new_state.alice = 'W'
            new_state.flag_alice = 'DOWN'
        elif action == 'a4':
            # RECUL : garde flag UP
            new_state.alice = 'a4'
            new_state.flag_alice = 'UP'  # ← Flag reste UP

        # BOB
        elif action == 'b1':
            new_state.bob = 'b1'
            new_state.flag_bob = 'UP'
        elif action == 'b2':
            new_state.bob = 'CS'
        elif action == 'b3':
            new_state.bob = 'W'
            new_state.flag_bob = 'DOWN'
        elif action == 'b4':
            # RECUL : garde flag UP
            new_state.bob = 'b4'
            new_state.flag_bob = 'UP'  # ← Flag reste UP

        return [new_state]



def ab_validation(protocol='AB1'):
    sem = AB1Semantics()  # ou AB2Semantics(), AB3Semantics()
    rg = LS2RG(sem)
    
    def on_entry(state, opaque):
        if sem.is_exclusion_violation(state):
            opaque['violation'].append(state)
        if sem.is_deadlock(state):
            opaque['deadlock'].append(state)
        return (False, opaque)  # Explore tout
    
    visited, results = BFS(rg, on_entry, {'violation':[], 'deadlock':[]})
    return results, visited
