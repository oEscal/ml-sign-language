#!/bin/bash

min_alpha=0.000003
max_alpha=0.000009
step_alpha=0.000001
min_hidden_size=75
max_hidden_size=100
step_hidden_size=5
min_num_iterations=100
step_num_iterations=100
max_num_iterations=1000

# lambdas=(0.0001 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 50 100 500 1000)
alphas=(0.0001 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 50 100 500 1000) # 0.0001 0.001 0.005 0.01 0.05)

id=0
# for num_iterations in $(seq $min_num_iterations $step_num_iterations $max_num_iterations); do
#    echo Starting number iterations=$num_iterations
#    python neural_networks.py --alpha 0.00005 --file_id $id --Lambda 0 --num_iterations $num_iterations
#    id=$((id + 1))
# done
# for hidden_size in $(seq $min_hidden_size $step_hidden_size $max_hidden_size); do
#    echo Starting hidden size=$hidden_size
#    python neural_networks.py --alpha 0.00005 --file_id $id --Lambda 0 --num_iterations 1000 --hidden_layer_size $hidden_size
#    id=$((id + 1))
# done
# for alpha in $(seq $min_alpha $step_alpha $max_alpha); do
#    echo Starting alpha=$alpha
#    python neural_networks.py --alpha ${alpha/,/.} --file_id $id --Lambda 0 --num_iterations 1000
#    id=$((id + 1))
# done
for alpha in ${alphas[@]}; do
   echo Starting alpha=$alpha
   python neural_networks.py --alpha $alpha --file_id $id --Lambda 0 --num_iterations 1000
   id=$((id + 1))
done

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
