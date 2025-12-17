from abc import ABC, abstractmethod
from collections import deque


class Rootedgraph(ABC):
    @abstractmethod
    def roots(self):  # return all graph roots
        pass

    @abstractmethod
    def neighbors(self, vertex):  # return all neighbors if vertex
        pass


class DictionaryGraph(Rootedgraph):
    def __init__(self, graph=None, roots=None):
        self.graph = graph if graph is not None else {}
        self._roots = roots if roots is not None else []

    def roots(self):
        return self._roots

    def neighbors(self, vertex):
        return self.graph.get(vertex, [])


class HanoiGraph(Rootedgraph):
    def __init__(self, n_disks: int):
        self.n_disks = n_disks
        self._roots = [(tuple(range(n_disks, 0, -1)), (), ())]  # All disks on tower A

    def roots(self):
        if self.n_disks <= 0:
            return [((), (), ())]  # No disks, all towers empty
        return self._roots

    def neighbors(self, vertex):
        # vertex : state of the towers as a tuple of 3 tuples
        neighbors = []
        for current_tower in range(3):
            if vertex[current_tower]:  # If the tower is not empty
                disk = vertex[current_tower][-1]  # Get the top disk
                for target_tower in range(3):
                    if target_tower != current_tower and (
                        not vertex[target_tower] or disk < vertex[target_tower][-1]
                    ):  # Check if move is valid

                        # Create new state by moving the disk
                        new_towers = list(
                            map(list, vertex)
                        )  # Convert tuples to lists for mutability + deep copy
                        # Convert state from tuples to lists for mutability
                        new_towers[current_tower].pop()
                        # Remove disk from current tower
                        new_towers[target_tower].append(disk)
                        # Add disk to target tower
                        neighbors.append(
                            (tuple(map(tuple, new_towers)))
                        )  # Convert back to tuple of tuples and add to new states (neighbors)
        return neighbors


def on_entry_check4vertex(vertex, opaque: int):
    """Appelée sur chaque nouveau sommet visité
    Ici, on arrête la recherche après avoir visité 4 sommets

    Args:
        vertex : Le sommet qui vient d'être découvert
        opaque : Données passées / accumulées (peut être n'importe quoi)

    Returns:
        (terminate, new_opaque) :
            terminate : booléen - True pour arrêter BFS, False pour continuer
            new_opaque : Valeur mise à jour de opaque
    """
    if opaque == 3:
        return (True, opaque)
    return (False, opaque + 1)


def path_to_objective(parents: dict, startNode):
    path = []
    current = startNode
    while current is not None:
        path.append(current)
        current = parents[current]
    path.reverse()
    return path


def on_entry_create_parents(vertex, opaque: tuple[dict, Rootedgraph, callable]):
    """On entry function that builds parent dictionary

    Args:
        vertex : The vertex that has just been discovered
        opaque (dict, Rootedgraph, callable): The parent dictionnary being built (key : child, value : parent), the graph to determine parents of children and the objective function
    Returns:
        (terminate, new_opaque) :
            terminate : bool - True to stop BFS, False to continue
            new_opaque : (dict, Rootedgraph, callable) - Updated opaque value (new parent dictionary, same graph, and same objective function)
    """
    # Initialize opaque if not provided
    if opaque is None or opaque[0] is None:
        parentsDict = {}
        graph = opaque[1] if opaque and len(opaque) > 1 else None
        if graph is None:  # If no graph provided, we can't check neighbors
            raise ValueError("Graph must be provided in opaque value")
        objective = opaque[2]
        opaque = (parentsDict, graph, objective)

    parentsDict = opaque[0]
    graph = opaque[1]
    objective = opaque[2]

    # Add the vertex to the parent dictionary if not already present
    if vertex not in parentsDict:
        parentsDict[vertex] = None  # Root has no parent
    for neighbor in graph.neighbors(vertex):
        if neighbor not in parentsDict:
            parentsDict[neighbor] = vertex
            if (
                neighbor == objective
            ):  # If we reached the objective, stop BFS and return the path from the root to the objective
                return (
                    True,
                    path_to_objective(parentsDict, neighbor),
                )
    return (False, (parentsDict, graph, objective))


def BFS(graph: Rootedgraph, on_entry: callable, opaque=None):
    marked = set()
    queue = deque()  # double ended queue : can pop from both ends
    for root in graph.roots():
        queue.append(root)

    while queue:
        v = queue.popleft()
        if v not in marked:
            marked.add(v)
            terminate, opaque = on_entry(v, opaque)
            if terminate:
                return marked, opaque
            for neighbors in graph.neighbors(v):
                if neighbors not in marked:  # Prevent re-adding already marked nodes
                    queue.append(neighbors)
    return marked, opaque


if __name__ == "__main__":
    """
    graph = {0: [1, 2], 1: [0, 2], 2: [0, 3, 4, 5], 3: [2], 4: [2, 5], 5: [2]}
    graph2 = {0: [1, 2], 1: [0, 2], 2: [1, 2, 3], 3: [2, 4], 4: [3], 6: [7], 7: [6]}
    print("====== BFS =======")
    print(BFS(DictionaryGraph(graph, [0]), on_entry_check4vertex, 0))
    print(BFS(DictionaryGraph(graph, [0, 5]), on_entry_check4vertex, 0))
    # print(BFS(DictionaryGraph(graph, [0, 5])))
    # print(BFS(DictionaryGraph(graph2, [0, 7])))
    # print(BFS(DictionaryGraph(graph2, [6])))
    print("============")
    print(BFS(DictionaryGraph(graph2, [0]), lambda v, o: (v == 3, o), 0))
    print(BFS(DictionaryGraph(graph2, [0, 6]), lambda v, o: (v == 3, o), 0))

    print("====== HANOI =======")
    hanoi_graph = HanoiGraph(3)
    print(BFS(hanoi_graph, lambda v, o: (v == ((), (), (3, 2, 1)), o))[0])
    print(len(BFS(hanoi_graph, lambda v, o: (v == ((), (), (3, 2, 1)), o))[0]))

    hanoi_graph = HanoiGraph(6)
    print(
        BFS(
            hanoi_graph,
            lambda v, o: (v == ((), (), (6, 5, 4, 3, 2, 1)), o),
        )[0]
    )
    print(
        len(
            BFS(
                hanoi_graph,
                lambda v, o: (v == ((), (), (6, 5, 4, 3, 2, 1)), o),
            )[0]
        )
    )
    """
    hanoi_graph = HanoiGraph(3)
    result = BFS(
        hanoi_graph,
        on_entry_create_parents,
        opaque=(None, hanoi_graph, ((), (), (3, 2, 1))),
    )
    print(result[1])
