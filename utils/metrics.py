from typing import List, Tuple, Dict
import numpy as np
from scipy.spatial.distance import cdist

Point = Tuple[float, float]

def get_pareto_front(points: List[Point]) -> List[Point]:
    """
    Computes the Pareto front for a set of 2D points.
    Assumes that lower values for both dimensions are better.

    Args:
        points: A list of (ECE, SH) tuples.

    Returns:
        A list of points on the Pareto front, sorted by the first dimension (ECE).
    """
    sorted_points = sorted(points)

    pareto_front = []
    last_sh = float('inf')

    for point in sorted_points:
        ece, sh = point
        if sh < last_sh:
            pareto_front.append(point)
            last_sh = sh
            
    return pareto_front

def compute_hv(
    method_points: Dict[str, List[Point]]
) -> Dict[str, float]:
    """
    Calculates the hypervolume ratio for each method.

    The ratio is the hypervolume of a method's individual Pareto front divided
    by the hypervolume of the global Pareto front (from all points combined).
    A score closer to 1.0 is better.

    Args:
        method_points: A dictionary where keys are method names and values are
                       lists of (ECE, SH) points for that method.

    Returns:
        A dictionary with method names as keys and their hypervolume ratio as values.
    """
    all_points = [p for points_list in method_points.values() for p in points_list]
    global_front = get_pareto_front(all_points)

    # --- Determine a reasonable reference point ---
    # It should be strictly worse (larger) than any point on the global front.
    # We add a 10% buffer based on the span of the data for robustness.
    max_ece = max(p[0] for p in all_points)
    max_sh = max(p[1] for p in all_points)
    min_ece = min(p[0] for p in all_points)
    min_sh = min(p[1] for p in all_points)
    
    ece_buffer = (max_ece - min_ece) * 0.1 if (max_ece - min_ece) > 0 else 1.0
    sh_buffer = (max_sh - min_sh) * 0.1 if (max_sh - min_sh) > 0 else 1.0

    ref_point = (max_ece + ece_buffer, max_sh + sh_buffer)

    def _calculate_hypervolume(front: List[Point], ref: Point) -> float:
            if not front:
                return 0.0
            
            # Front is assumed to be sorted by ECE (first dimension)
            total_area = 0.0
            ref_ece, ref_sh = ref
            last_sh = ref_sh

            for ece, sh in front:
                width = ref_ece - ece
                height = last_sh - sh
                total_area += width * height
                last_sh = sh
                
            return total_area

    global_hv = _calculate_hypervolume(global_front, ref_point)

    results = {}
    for name, points in method_points.items():
        individual_front = get_pareto_front(points)
        individual_hv = _calculate_hypervolume(individual_front, ref_point)
        results[name] = individual_hv / global_hv

    return results

def compute_igd_discrete(
    method_points: Dict[str, List[Point]]
) -> Dict[str, float]:
    """
    Calculates the Inverted Generational Distance (IGD) for each method.

    IGD measures the average distance from the points on the global front to
    the nearest point in a given method's solution set. A lower score is better.

    Args:
        method_points: A dictionary where keys are method names and values are
                       lists of (ECE, SH) points for that method.

    Returns:
        A dictionary with method names as keys and their IGD score as values.
    """
    all_points = [p for points_list in method_points.values() for p in points_list]

    global_front_Z = get_pareto_front(all_points)

    results = {}
    Z_np = np.array(global_front_Z)
    for name, points_A in method_points.items():
        A_np = np.array(points_A)
        distances_matrix = cdist(Z_np, A_np)
        min_distances = np.min(distances_matrix, axis=1)

        # IGD = [ (1/|Z|) * sum(d_i^p) ] ^ (1/p)
        igd_score = np.mean(min_distances ** 2) ** (1/2)
        results[name] = igd_score
        
    return results



if __name__ == '__main__':
    # Method A: Points are close to or on the "true" Pareto front.
    # Method B: Good, but consistently dominated by Method A.
    # Method C: Points are clearly worse and scattered.

    mock_data = {
        'Method A': [
            (0.1, 0.9), (0.2, 0.6), (0.4, 0.3), (0.7, 0.2)
        ],
        'Method B': [
            (0.15, 0.95), (0.25, 0.65), (0.45, 0.35), (0.75, 0.25)
        ],
        'Method C': [
            (0.5, 0.8), (0.6, 0.5), (0.8, 0.6), (0.9, 0.4)
        ]
    }

    print("--- Hypervolume Ratio Analysis (Higher is Better) ---")
    hv_ratios = compute_hv(mock_data)
    for method, ratio in sorted(hv_ratios.items(), key=lambda item: item[1], reverse=True):
        print(f"{method}: {ratio:.4f}")

    print("\n" + "="*50 + "\n")

    print("--- Inverted Generational Distance (IGD) Analysis (Lower is Better) ---")
    igd_scores = compute_igd_discrete(mock_data)
    for method, score in sorted(igd_scores.items(), key=lambda item: item[1]):
        print(f"{method}: {score:.4f}")
        
    # --- Visualization Points (for understanding) ---
    all_points = [p for points_list in mock_data.values() for p in points_list]
    global_pf = get_pareto_front(all_points)
    print("\nGlobal Pareto Front is composed of points from Method A:")
    print(global_pf)