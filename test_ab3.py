from BFS_definition import BFS
from aliceEtBobImplementation import AB3

if __name__ == "__main__":
    print("====== AB3 =======")
    ab3 = AB3()
    marked, _ = BFS(ab3, lambda v, o: (False, o), None)
    print(marked)
    
    # Test exclusion mutuelle
    exclusion_violation = any("CS" in state[0] and "CS" in state[1] 
                              for state in marked)
    assert not exclusion_violation
    print("✅ Exclusion mutuelle AB3 OK")
    
    print(f"Nombre d'états : {len(marked)}")
