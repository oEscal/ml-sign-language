#!/bin/bash

alphas=(0.0001 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 50 100 500 1000)
lambdas=(0.0001 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 50 100 500 1000)

id=0
for alpha in ${alphas[@]}; do
   echo Starting alpha=$alpha
   python neural_networks.py --alpha $alpha --theta_file_id $id
   id=$((id + 1))
done

for lambda in ${lambdas[@]}; do
   echo Starting lambda=$lambda
   python neural_networks.py --Lambda $lambda --theta_file_id $id
   id=$((id + 1))
done
