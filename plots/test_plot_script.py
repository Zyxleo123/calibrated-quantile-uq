import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plots.plot_metrics import plot_training_stats, compare_ece_sharpness, compare_scoring_rules, calibration_plot
from plot_utils import load_pickle

def main():
    pkl = "boston_lossscaled_batch_cal_ens1_bootFalse_seed0_thres0.0-0.2.pkl"
    if not os.path.exists(pkl):
        print(f"Test pickle not found: {pkl}. Please place it in the current directory to run the demo.")
        return
    data = load_pickle(pkl)
    outdir = "./results"
    os.makedirs(outdir, exist_ok=True)

    plot_training_stats(data, outpath=os.path.join(outdir, "training_stats.png"))
    compare_ece_sharpness(data, outpath=os.path.join(outdir, "ece_sharpness.png"))
    compare_scoring_rules(data, outpath=os.path.join(outdir, "scoring_rules.png"))
    calibration_plot(data, outpath=os.path.join(outdir, "calibration_plot.png"))
    print("Plots saved to", outdir)

if __name__ == "__main__":
    main()
