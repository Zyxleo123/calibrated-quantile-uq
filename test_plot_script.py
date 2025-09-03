import os
import sys
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plots.plot_metrics import plot_training_stats, compare_ece_sharpness, compare_scoring_rules, calibration_plot, plot_ece_sharpness
from plots.plot_utils import load_pickle

def main():
    pkls = glob.glob("results/*.pkl")
    for pkl in pkls:
        data = load_pickle(pkl)
        # outdir = "/home/scratch/yixiz/results"
        outdir = "results"
        if os.path.isdir(os.path.join(outdir, pkl)):
            print("Directory already exists:", os.path.join(outdir, pkl))
            continue
        pkl, ext = os.path.splitext(os.path.basename(pkl))
        plot_training_stats(data, outpath=os.path.join(outdir, pkl + "/training_stats.png"))
        compare_ece_sharpness(data, outpath=os.path.join(outdir, pkl + "/ece_sharpness.png"))
        compare_scoring_rules(data, outpath=os.path.join(outdir, pkl + "/scoring_rules.png"))
        calibration_plot(data, outpath=os.path.join(outdir, pkl + "/calibration_plot.png"))
        plot_ece_sharpness(data, outpath=os.path.join(outdir, pkl + "/ece_sharpness.png"))
        print(f"Plot of {pkl} saved to {outdir}")

if __name__ == "__main__":
    main()
