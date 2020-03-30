#!/bin/bash

min_alpha=0.1
max_alpha=5
min_hidden_size=5
max_hidden_size=45
# lambdas=(0.0001 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 50 100 500 1000)
# alphas=(0.0001 0.001 0.005 0.01 0.05)

id=0
for size in $(seq $min_hidden_size 5 $max_hidden_size); do
   echo Starting size=$size
   python neural_networks.py --alpha 0.005 --hidden_layer_size $size --file_id $id --Lambda 0 --num_iterations 1000
   id=$((id + 1))
done
# for alpha in ${alphas[@]}; do
#    echo Starting alpha=$alpha
#    python neural_networks.py --alpha $alpha --file_id $id --Lambda 0 --num_iterations 1000
#    id=$((id + 1))
# done
# 
# for alpha in $(seq $min_alpha 0.1 $max_alpha); do
#    echo Starting alpha=$alpha
#    python neural_networks.py --alpha ${alpha/,/.} --file_id $id --Lambda 0 --num_iterations 1000
#    id=$((id + 1))
# done

# for lambda in ${lambdas[@]}; do
#    echo Starting lambda=$lambda
#    python neural_networks.py --Lambda $lambda --theta_file_id $id
#    id=$((id + 1))
# done
