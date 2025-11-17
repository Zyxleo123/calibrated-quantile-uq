gpus=(7 6 0 1 2 3 4 5)
datasets=("protein" "concrete" "elevator" "energy" "facebook" "kin8nm" "fusion" "naval" "diamonds" "power" "wine" "yacht" "boston")
loss=("calipso" "maqr" "batch_qr" "batch_int" "batch_cal" "mpaic" "batch_QRT")

gpu_index=0
for dataset in "${datasets[@]}"; do
    for l in "${loss[@]}"; do
        gpu=${gpus[$gpu_index]}
        echo "Running inference for dataset: $dataset, loss: $l on GPU: $gpu"
        python -u main_inference.py --data "$dataset" --loss "$l" --num_ep 1000 --nl 8 --hs 256 --seed 1 --gpu "$gpu" &
        
        gpu_index=$(( (gpu_index + 1) % ${#gpus[@]} ))
        # To avoid overloading a single GPU, wait for all background processes to finish every time we loop through all GPUs
        if [ $gpu_index -eq 0 ]; then
            wait
        fi
    done
done