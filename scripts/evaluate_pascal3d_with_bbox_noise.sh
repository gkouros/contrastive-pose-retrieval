#!/bin/bash
mkdir -p "/users/visics/gkouros/projects/models/pascal3d/$1/evaluation"
touch "/users/visics/gkouros/projects/models/pascal3d/$1/evaluation/main.log"
cd "/users/visics/gkouros/projects/thesis/"

NOISE_VALUES=( 0.0 0.25 0.5 0.75 1.0 )
for NOISE in "${NOISE_VALUES[@]}"
do
    echo Running experiment with bbox noise=$NOISE
    python3 scripts/evaluate_pascal3d.py \
        --root_dir="/esat/topaz/gkouros/datasets/pascal3d" \
        --runs_path="/users/visics/gkouros/projects/models/pascal3d/" \
        --experiment="$1" \
        --from_scratch \
        --noevaluate_inference_time \
        --occlusion_level=0 \
        --notrain_plus_db \
        --nopositive_from_db \
        --bbox_noise=$NOISE \
        --use_hdf5 \
        --nobgr \
        2>&1 | tee -a "/users/visics/gkouros/projects/models/pascal3d/$1/evaluation/main.log"
done
