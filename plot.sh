echo "Clearing metric_log/ before running experiments..."
rm metric_log/*

echo "Computing metrics and plotting..."
python plots/plot_seeds.py -n lump
python plots/plot_seeds.py -n lump -r

echo "Plotting Pareto fronts..."
python plots/plot_pf.py -n lump

echo "Reporting mean and std error..."
python plots/plot_mean.py -n lump