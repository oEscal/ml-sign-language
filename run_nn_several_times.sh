#!/bin/bash

min_alpha=0.001
max_alpha=0.01
step_alpha=0.001
min_hidden_size=5
max_hidden_size=70
step_hidden_size=5
min_num_iterations=100
step_num_iterations=100
max_num_iterations=1000
min_batch=16
max_batch=256
step_mult_batch=2

alphas=(0.0000001 0.000001 0.00001 0.0001 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 50 100 500 1000)
default_alpha=1.4e-05

id=0


if [[ $1 == big_steps ]]; then
   for alpha in $(seq $min_alpha $step_alpha $max_alpha); do
      echo Starting alpha=$alpha
      python neural_networks.py --alpha ${alpha/,/.} --file_id $id --Lambda 0 --num_iterations 1000
      id=$((id + 1))
   done
elif [[ $1 == study_alpha ]]; then
   for alpha in ${alphas[@]}; do
      echo Starting alpha=$alpha
      python neural_networks.py --alpha $alpha --file_id $id --Lambda 0 --num_iterations 1000
      id=$((id + 1))
   done
elif [[ $1 == study_iterations ]]; then
   for num_iterations in $(seq $min_num_iterations $step_num_iterations $max_num_iterations); do
      echo Starting number iterations=$num_iterations
      python neural_networks.py --alpha $default_alpha --file_id $id --Lambda 0 --num_iterations $num_iterations
      id=$((id + 1))
   done
elif [[ $1 == study_hidden ]]; then
   for hidden_size in $(seq $min_hidden_size $step_hidden_size $max_hidden_size); do
      echo Starting hidden size=$hidden_size
      python neural_networks.py --alpha $default_alpha --file_id $id --Lambda 0 --num_iterations 1000 --hidden_layer_size $hidden_size
      id=$((id + 1))
   done
elif [[ $1 == study_batch ]]; then
   batch=$min_batch
   while [ $batch -le $max_batch ]; do
      echo Starting batch size=$((batch + 2))
      python neural_networks.py --alpha 1.4e-05 --file_id $id --Lambda 0 --num_iterations 1000 --batch_size $batch
      id=$((id + 1))
      batch=$((batch * step_mult_batch))
   done
fi
