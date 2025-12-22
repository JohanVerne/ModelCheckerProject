from BFS_definition import BFS
from aliceEtBobImplementation import AB1

if __name__ == "__main__":
    # ... vos autres tests ...

    print("====== AB1 =======")
    ab1 = AB1()
    marked, _ = BFS(ab1, lambda v, o: (False, o), None)
    print(marked)
