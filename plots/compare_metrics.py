import json
import os
import numpy as np
from typing import Dict, Any, List, Tuple
from plot_utils import load_pickle, safe_get

# makeshift fix
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def gather_all_metrics(dataset_name: str, our_results_path: str, reported_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Gathers metrics from our model's pickle file and the reported data JSON."""
    all_points = []
    dataset_key = dataset_name.title()
    reported_metrics = safe_get(reported_data, dataset_key, {})
    
    for method, metrics in reported_metrics.items():
        all_points.append({"name": method, "ece": metrics[0], "sharpness": metrics[1], "source": "Reported"})

    data = load_pickle(our_results_path)

    ece_list = safe_get(data, 'te_ece_list_best', [])
    sharpness_list = safe_get(data, 'te_sharp_score_list_best', [])
    recal_ece_list = safe_get(data, 'recal_te_ece_list_best', [])
    recal_sharpness_list = safe_get(data, 'recal_te_sharp_score_list_best', [])
    thresholds = safe_get(data, 'thresholds', [])

    for ece, sharpness, thr in zip(ece_list, sharpness_list, thresholds):
        if ece is None or sharpness is None:
            continue
        all_points.append({"name": f"Our Method (thr={thr:.2f})", "ece": ece, "sharpness": sharpness, "source": "Ours"})

    for ece, sharpness, thr in zip(recal_ece_list, recal_sharpness_list, thresholds):
        if ece is None or sharpness is None:
            continue
        all_points.append({"name": f"Our Method (recal, thr={thr:.2f})", "ece": ece, "sharpness": sharpness, "source": "Ours"})

    return all_points


def find_pareto_frontier(points: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Identifies the Pareto frontier from a list of points (lower is better for both metrics)."""
    frontier, dominated = [], []
    for p1 in points:
        is_dominated = False
        for p2 in points:
            if p1 is p2: continue
            if (p2['ece'] <= p1['ece'] and p2['sharpness'] <= p1['sharpness']) and \
               (p2['ece'] < p1['ece'] or p2['sharpness'] < p1['sharpness']):
                is_dominated = True
                break
        if is_dominated: dominated.append(p1)
        else: frontier.append(p1)
    return frontier, dominated


def rank_points(points: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Ranks points by a weighted score and returns the ranked list and the weights used."""
    weights = {'w_ece': 0, 'w_sharp': 0, 'var_ece': 0, 'var_sharp': 0}
    if len(points) < 2:
        for p in points: p['score'] = 0
        return points, weights

    all_ece = [p['ece'] for p in points]
    all_sharpness = [p['sharpness'] for p in points]
    
    var_ece = np.var(all_ece)
    var_sharpness = np.var(all_sharpness)
    epsilon = 1e-9
    w_ece = 1 / (var_ece + epsilon)
    w_sharp = 1 / (var_sharpness + epsilon)
    
    weights.update({'w_ece': w_ece, 'w_sharp': w_sharp, 'var_ece': var_ece, 'var_sharp': var_sharpness})

    for p in points:
        p['score'] = w_ece * p['ece'] + w_sharp * p['sharpness']
        
    return sorted(points, key=lambda x: x['score']), weights


def format_report_string(dataset_name: str, frontier_points: List, dominated_points: List, ranked_points: List, weights: Dict) -> str:
    """Formats the complete analysis into a single string for printing or file saving."""
    report_lines = []
    
    report_lines.append("=" * 70)
    report_lines.append(f"  Analysis Report for Dataset: {dataset_name.title()}")
    report_lines.append("=" * 70)

    report_lines.append("\nCalculated Weights for Ranking:")
    report_lines.append(f"  - Var(ECE): {weights['var_ece']:.4f} -> Weight(ECE): {weights['w_ece']:.4f}")
    report_lines.append(f"  - Var(Sharpness): {weights['var_sharp']:.4f} -> Weight(Sharpness): {weights['w_sharp']:.4f}")

    report_lines.append("\n--- Pareto Frontier Analysis ---")
    report_lines.append("Models that are not dominated by any other model:")
    frontier_points_sorted = sorted(frontier_points, key=lambda x: x['ece'])
    for p in frontier_points_sorted:
        report_lines.append(f"  - {p['name']:<25} | ECE: {p['ece']:>5.2f}, Sharpness: {p['sharpness']:>5.2f}")

    report_lines.append("\n--- Dominated Models ---")
    if dominated_points:
        dominated_points_sorted = sorted(dominated_points, key=lambda x: x['name'])
        for p in dominated_points_sorted:
            report_lines.append(f"  - {p['name']:<25} | ECE: {p['ece']:>5.2f}, Sharpness: {p['sharpness']:>5.2f}")
    else:
        report_lines.append("  None.")

    report_lines.append("\n--- Overall Ranking (based on weighted score) ---")
    header = f"{'Rank':<5} {'Method Name':<25} {'ECE':<8} {'Sharpness':<12} {'Score':<10}"
    report_lines.append(header)
    report_lines.append("-" * (len(header) + 2))
    for i, p in enumerate(ranked_points):
        is_frontier = any(p is pf for pf in frontier_points)
        marker = "*" if is_frontier else " "
        report_lines.append(f"{i+1:<5} {p['name']:<25} {p['ece']:<8.2f} {p['sharpness']:<12.2f} {p['score']:<10.2f} {marker}")
    report_lines.append("-" * (len(header) + 2))
    report_lines.append("'*' indicates model is on the Pareto frontier.")
    
    return "\n".join(report_lines)


# --- Main Orchestration Function ---

def compare_and_rank_methods(
    dataset_name: str, 
    results_path: str,
    output_path: str,
    reported_data_path: str = "data/beyond_pinball_report.json"
):
    """
    Main function to perform comparison, ranking, and save the report to a file.

    Args:
        dataset_name (str): The name of the dataset to analyze (e.g., 'boston').
        results_dir (str): Directory containing our .pkl result files.
        reported_data_path (str): Path to the JSON file with reported metrics.
        output_path (str, optional): If provided, the report is saved to this file path. 
                                     Otherwise, it's printed to the console.
    """
    print(f"\n>>> Starting analysis for dataset: {dataset_name.title()}")
    # 1. Load reported data
    try:
        with open(reported_data_path, 'r') as f:
            reported_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Reported data file not found at '{reported_data_path}'. Aborting.")
        return

    # 2. Perform analyses
    all_points = gather_all_metrics(dataset_name, results_path, reported_data)
    if not all_points:
        print("No metrics could be gathered for comparison. Aborting.")
        return

    frontier_points, dominated_points = find_pareto_frontier(all_points)
    ranked_points, weights = rank_points(all_points)

    # 3. Generate the report string
    report_content = format_report_string(dataset_name, frontier_points, dominated_points, ranked_points, weights)
    
    # 4. Output the report to file
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
            
    with open(output_path, 'w') as f:
        f.write(report_content)
    print(f"Analysis complete. Report for '{dataset_name}' saved to: {output_path}")


if __name__ == '__main__':
    from glob import glob

    results_dir = "results"
    results_files = glob(os.path.join(results_dir, "*.pkl"))
    for results_file in results_files:
        dataset_name = os.path.basename(results_file).split("_")[0]
        compare_and_rank_methods(
            dataset_name=dataset_name,
            results_path=results_file,
            output_path=f"reports/{dataset_name}_report.txt"
        )