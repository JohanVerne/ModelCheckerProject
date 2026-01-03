from aliceEtBobImplementation import AB2
from BFS_definition import BFS

if __name__ == "__main__":

    print("====== AB2 =======")
    ab2 = AB2()
    marked, _ = BFS(ab2, lambda v, o: (False, o), None)
    print(marked)

    # Test exclusion mutuelle (à ajuster quand tu auras les états CS définis)
    assert ("CS", "CS", "UP", "UP") not in marked
    print("✅ Exclusion mutuelle AB2 OK")   