from languagesemantics import LanguageSemantics
from ls2rg import LS2RG
from BFS_definition import BFS, RootedGraph
from alicebob_semantics import AB3Semantics

if __name__ == "__main__":
    sem = AB3Semantics()
    rg = LS2RG(sem)

    def on_entry(state, opaque):
        if sem.is_exclusion_violation(state):
            opaque['violation'].append(state)
        if sem.is_deadlock(state):
            opaque['deadlock'].append(state)
        return (False, opaque)
    
    visited, results = BFS(rg, on_entry, {'violation':[], 'deadlock':[]})
    print("=== AB3 ===")
    print(f"Exclusion violation : {len(results['violation'])}")
    print(f"Deadlock : {len(results['deadlock'])}")
    print(f"États explorés : {len(visited)}")
