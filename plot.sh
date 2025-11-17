EXP_NAME=$1

echo "Clearing metric_log/ before running experiments..."
rm -r metric_log/$EXP_NAME

echo "Computing metrics and plotting..."
python -u plots/plot_seeds.py -n $EXP_NAME
python -u plots/plot_seeds.py -n $EXP_NAME -r

echo "Plotting Pareto fronts..."
python -u plots/plot_pf.py -n $EXP_NAME

echo "Reporting mean and std error..."
python -u plots/plot_mean.py -n $EXP_NAME
python -u plots/plot_mean.py -n $EXP_NAME -r