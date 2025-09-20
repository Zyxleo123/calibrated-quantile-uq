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

def _create_staircase_front(
    points: List[Point], 
    step_size: float
) -> List[Point]:
    """
    Creates a denser set of points representing a staircase front using a
    fixed interval length between samples on the ECE axis.
    
    Args:
        points: The original list of (ECE, SH) points.
        step_size: The desired distance between sampled points on the ECE axis.
        
    Returns:
        A new, denser list of points representing the staircase.
    """
    sorted_points = sorted(points, key=lambda p: p[0])
    interpolated_points = []
    for i in range(len(sorted_points) - 1):
        p1_ece, p1_sh = sorted_points[i]
        p2_ece, _ = sorted_points[i+1]
        
        interpolated_points.append((p1_ece, p1_sh))
        
        if p2_ece <= p1_ece:
            continue
        current_x = p1_ece + step_size
        while current_x < p2_ece:
            interpolated_points.append((current_x, p1_sh))
            current_x += step_size

    interpolated_points.append(sorted_points[-1])
    return interpolated_points


def compute_igd_staircase(
    method_points: Dict[str, List[Point]],
    ece_step_size: float = None,
) -> Dict[str, float]:
    """
    Calculates the Inverted Generational Distance (IGD) for each method
    using a staircase interpolation with a fixed step size.

    Args:
        method_points: A dictionary where keys are method names and values are
                       lists of (ECE, SH) points for that method.
        ece_step_size: The fixed interval length between interpolated points
                       on the ECE (horizontal) axis.

    Returns:
        A dictionary with method names as keys and their IGD score as values.
    """
    all_points = [p for points_list in method_points.values() for p in points_list]
    global_front_Z = get_pareto_front(all_points)
    if ece_step_size is not None:
        global_front_Z = _create_staircase_front(global_front_Z, ece_step_size)
        
    results = {}
    Z_np = np.array(global_front_Z)
    for name, points_A in method_points.items():
        A_np = np.array(points_A)
        distances_matrix = cdist(Z_np, A_np)
        min_distances = np.min(distances_matrix, axis=1)

        # IGD = [ (1/|Z|) * sum(d_i^p) ] ^ (1/p)
        igd_score = np.mean(min_distances ** 2) ** 0.5
        results[name] = igd_score
        
    return results

def compute_gd_staircase(
    method_points: Dict[str, List[Point]],
    ece_step_size: float=None,
) -> Dict[str, float]:
    """
    Calculates the Generational Distance (GD) for each method
    using a staircase interpolation for the Pareto front.

    Args:
        method_points: A dictionary where keys are method names and values are
                       lists of (ECE, SH) points for that method.
        ece_step_size: The fixed interval length between interpolated points
                       on the ECE (horizontal) axis for the reference front.

    Returns:
        A dictionary with method names as keys and their GD score as values.
    """
    all_points = [p for points_list in method_points.values() for p in points_list]
    global_front_Z = get_pareto_front(all_points)
    if ece_step_size is not None:
        global_front_Z = _create_staircase_front(global_front_Z, ece_step_size)
        
    results = {}
    Z_np = np.array(global_front_Z)
    for name, points_A in method_points.items():
        A_np = np.array(points_A)
        distances_matrix = cdist(A_np, Z_np)
        min_distances = np.min(distances_matrix, axis=1)

        # GD = [ (1/|A|) * sum(d_i^p) ] ^ (1/p) with p=2
        gd_score = (np.mean(min_distances ** 2)) ** 0.5
        results[name] = gd_score
        
    return results

if __name__ == '__main__':
    # Method A: Points are exactly on the "true" Pareto front.
    # Method B: Good, but consistently dominated by Method A.
    # Method C: Points are clearly worse and scattered, far from the front.

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

    print("--- Inverted Generational Distance (Discrete) (Lower is Better) ---")
    igd_scores_discrete = compute_igd_staircase(mock_data)
    for method, score in sorted(igd_scores_discrete.items(), key=lambda item: item[1]):
        print(f"{method}: {score:.4f}")
    
    print("\n" + "="*50 + "\n")

    print("--- Inverted Generational Distance (Staircase) (Lower is Better) ---")
    igd_scores_staircase = compute_igd_staircase(mock_data, ece_step_size=0.01)
    for method, score in sorted(igd_scores_staircase.items(), key=lambda item: item[1]):
        print(f"{method}: {score:.4f}")

    print("\n" + "="*50 + "\n")

    print("--- Generational Distance (Discrete) (Lower is Better) ---")
    gd_scores_discrete = compute_gd_staircase(mock_data)
    for method, score in sorted(gd_scores_discrete.items(), key=lambda item: item[1]):
        print(f"{method}: {score:.4f}")
    
    print("\n" + "="*50 + "\n")

    print("--- Generational Distance (Staircase) (Lower is Better) ---")
    gd_scores_staircase = compute_gd_staircase(mock_data, ece_step_size=0.01)
    for method, score in sorted(gd_scores_staircase.items(), key=lambda item: item[1]):
        print(f"{method}: {score:.4f}")
        
    # --- Visualization Points (for understanding) ---
    all_points = [p for points_list in mock_data.values() for p in points_list]
    global_pf = get_pareto_front(all_points)
    print("\nGlobal Pareto Front is composed of points from Method A:")
    print(global_pf)