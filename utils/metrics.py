from typing import List, Tuple, Dict
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

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


def compute_igd(
    method_points: Dict[str, List[Point]],
    ece_step_size: float = 0.0,
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
    if ece_step_size:
        global_front_Z = _create_staircase_front(global_front_Z, ece_step_size)
        
    results = {}
    Z_np = np.array(global_front_Z)
    for name, points_A in method_points.items():
        A_np = np.array(points_A)
        distances_matrix = cdist(Z_np, A_np)
        min_distances = np.min(distances_matrix, axis=1)

        # IGD = (1/|Z|) [ sum(d_i^p) ] ^ (1/p)
        igd_score = np.sum(min_distances ** 2) ** 0.5 / len(Z_np)
        results[name] = igd_score
        
    return results

def compute_gd(
    method_points: Dict[str, List[Point]],
    ece_step_size: float=0.0,
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
    if ece_step_size:
        global_front_Z = _create_staircase_front(global_front_Z, ece_step_size)
        
    results = {}
    Z_np = np.array(global_front_Z)
    for name, points_A in method_points.items():
        A_np = np.array(points_A)
        distances_matrix = cdist(A_np, Z_np)
        min_distances = np.min(distances_matrix, axis=1)

        # GD = (1/|A|) [ sum(d_i^p) ] ^ (1/p) with p=2
        gd_score = np.sum(min_distances ** 2) ** 0.5 / len(A_np)
        results[name] = gd_score
        
    return results

def compute_igd_plus(
    method_points: Dict[str, List[Point]],
    ece_step_size: float = 0.0,
    p: int = 1,
) -> Dict[str, float]:
    """
    Calculates the Inverted Generational Distance Plus (IGD+) for each method.

    This metric measures how far the points in the true Pareto front are from the
    obtained set of points. It only considers distances where the obtained point
    does not dominate the true front point. A lower value is better.

    The formula is: IGD+(A, Z) = (1/|Z|) * Σ_{z∈Z} min_{a∈A} ||(a - z)+||_p
    where (v)+ = max(0, v) applied element-wise.

    Args:
        method_points: A dictionary where keys are method names and values are
                       lists of (ECE, SH) points (the obtained set A) for that method.
        ece_step_size: The fixed interval length for staircase interpolation
                       of the reference Pareto front (the true front Z).
        p: The order of the norm used for distance calculation (e.g., 1 for
           Manhattan, 2 for Euclidean).

    Returns:
        A dictionary with method names as keys and their IGD+ score as values.
    """
    all_points = [p for points_list in method_points.values() for p in points_list]
    global_front_Z = get_pareto_front(all_points)
    if ece_step_size:
        global_front_Z = _create_staircase_front(global_front_Z, ece_step_size)

    results = {}
    Z_np = np.array(global_front_Z)
    k, m = Z_np.shape

    for name, points_A in method_points.items():
        A_np = np.array(points_A)
        n, _ = A_np.shape
        diff = A_np.reshape(1, n, m) - Z_np.reshape(k, 1, m)
        plus_diff = np.maximum(0, diff)

        # distance_matrix[i, j] = ||(A_np[j] - Z_np[i])+||_p
        distances_matrix = np.linalg.norm(plus_diff, ord=p, axis=2)
        min_distances = np.min(distances_matrix, axis=1)
        igd_plus_score = np.mean(min_distances)
        results[name] = igd_plus_score
        
    return results

def compute_gd_plus(
    method_points: Dict[str, List[Point]],
    ece_step_size: float = 0.0,
    p: int = 1,
) -> Dict[str, float]:
    """
    Calculates the Generational Distance Plus (GD+) for each method.

    This metric measures how far the obtained points are from the true Pareto
    front. It only considers distances where the obtained point does not dominate
    a corresponding true front point. A lower value is better.

    The formula is: GD+(A, Z) = (1/|A|) * Σ_{a∈A} min_{z∈Z} ||(a - z)+||_p
    where (v)+ = max(0, v) applied element-wise.

    Args:
        method_points: A dictionary where keys are method names and values are
                       lists of (ECE, SH) points (the obtained set A) for that method.
        ece_step_size: The fixed interval length for staircase interpolation
                       of the reference Pareto front (the true front Z).
        p: The order of the norm used for distance calculation (e.g., 1 for
           Manhattan, 2 for Euclidean).

    Returns:
        A dictionary with method names as keys and their GD+ score as values.
    """
    # Z is the reference Pareto front (approximated true front)
    all_points = [p for points_list in method_points.values() for p in points_list]
    global_front_Z = get_pareto_front(all_points)
    if ece_step_size:
        global_front_Z = _create_staircase_front(global_front_Z, ece_step_size)
        
    results = {}
    Z_np = np.array(global_front_Z)

    k, m = Z_np.shape
    for name, points_A in method_points.items():
        # A is the obtained front from a method
        A_np = np.array(points_A)
        n, _ = A_np.shape

        diff = A_np.reshape(n, 1, m) - Z_np.reshape(1, k, m)
        plus_diff = np.maximum(0, diff)
        # distances_matrix[i, j] = ||(A_np[i] - Z_np[j])+||_p
        distances_matrix = np.linalg.norm(plus_diff, ord=p, axis=2)
        min_distances = np.min(distances_matrix, axis=1)
        gd_plus_score = np.mean(min_distances)
        results[name] = gd_plus_score
        
    return results

if __name__ == '__main__':
    from glob import glob
    import sys
    sys.path.append('/zfsauton2/home/yixiz/calibrated-quantile-uq')
    from plots.plot_utils import load_pickle
    from plots.plot_seeds import BASELINE_NAMES
    from collections import defaultdict
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', '-d', type=str, required=True,
                        help='Directory containing pickle files with ECE and SH data.')

    args = parser.parse_args()
    pkl_files = glob(os.path.join(args.data_dir, '*.pkl'))
    method_points = defaultdict(list)  # method_name -> List of (ECE, SH)
    for pkl_path in pkl_files:
        method_name = [method for method in BASELINE_NAMES if method in os.path.basename(pkl_path)][0]
        data = load_pickle(pkl_path)
        method_points[method_name].extend(
            list(zip(data['te_ece_controlled'], data['te_sharp_score_controlled']))
        )
    hv_results = compute_hv(method_points)
    igd_results = compute_igd(method_points)
    igdp_results = compute_igd_plus(method_points)
    gd_results = compute_gd(method_points, ece_step_size=0.001)
    gdp_results = compute_gd_plus(method_points, ece_step_size=0.001)
    print("Hypervolume Ratios:")
    for method, score in hv_results.items():
        print(f"  {method}: {score:.4f}")
    print("Inverted Generational Distances (IGD):")
    for method, score in igd_results.items():
        print(f"  {method}: {score:.4f}")
    print("Inverted Generational Distances Plus (IGD+):")
    for method, score in igdp_results.items():
        print(f"  {method}: {score:.4f}")
    print("Generational Distances (GD):")
    for method, score in gd_results.items():
        print(f"  {method}: {score:.4f}")
    print("Generational Distances Plus (GD+):")
    for method, score in gdp_results.items():
        print(f"  {method}: {score:.4f}")
