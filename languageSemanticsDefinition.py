from BFS_definition import RootedGraph


class LanguageSemantics:

    def __init__(self, states):
        self.initStates = states

    def initials(self) -> set:
        return self.initStates

    def actions(self, state: set) -> set:
        """Returns a set of possible actions in a given state."""
        pass

    def execute(self, action: set, state: set) -> set:
        """Returns the resulting state after executing an action in a given state."""
        pass


class LS2RG(RootedGraph):
    """Converts a Language Semantics to a Rooted Graph"""

    def __init__(self, langSem: LanguageSemantics):
        self.langSem = langSem

    def roots(self):
        return list(self.langSem.initials())

    def neighbors(self, vertex):
        actions = self.langSem.actions(vertex) if self.langSem.actions(vertex) else []
        neighbors = []
        for action in actions:
            new_state = (
                self.langSem.execute(action, vertex)
                if self.langSem.execute(action, vertex)
                else None
            )
            if new_state is not None:
                neighbors.append(new_state)
        return neighbors


if __name__ == "__main__":
    # Example usage
    langSem = LanguageSemantics(states={("state1"), ("state2")})
    rg = LS2RG(langSem)
    print("Roots:", rg.roots())
    print("Neighbors of state1:", rg.neighbors("state1"))
