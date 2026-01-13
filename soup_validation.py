from hanoilanguagesemantics import HanoiLanguageSemantics
from soupsemantics import SoupSemantics, Soup2RG
from BFS_definition import BFS

def soup_hanoi_solver(n1, n2):
    # 2 tours de Hanoï couplées
    hanoi1 = HanoiLanguageSemantics(n1)
    hanoi2 = HanoiLanguageSemantics(n2)
    
    soup = SoupSemantics([hanoi1, hanoi2])
    rg = Soup2RG(soup)
    
    def on_entry(config, opaque):
        if soup.is_solution(config):
            opaque.append(config)
        return (soup.is_solution(config), opaque)
    
    marked, solutions = BFS(rg, on_entry, [])
    return solutions, marked

# Test : 2 tours de Hanoï à 2 disques
print("=== Soup Semantics : 2 Tours de Hanoï(2) ===")
solutions, visited = soup_hanoi_solver(2, 2)
print(f"États explorés : {len(visited)}")
print(f"Solution : {solutions}")


# Test : 3 tours de Hanoï à 2 disques
print("\n=== Soup Semantics : 3 Tours de Hanoï(2) ===")
hanoi1 = HanoiLanguageSemantics(2)
hanoi2 = HanoiLanguageSemantics(2) 
hanoi3 = HanoiLanguageSemantics(2)

soup3 = SoupSemantics([hanoi1, hanoi2, hanoi3])
rg3 = Soup2RG(soup3)

def on_entry3(config, opaque):
    if soup3.is_solution(config):
        opaque.append(config)
    return (soup3.is_solution(config), opaque)

marked3, solutions3 = BFS(rg3, on_entry3, [])
print(f"États explorés : {len(marked3)}")
print(f"Solution : {solutions3}")

