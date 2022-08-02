#!/bin/bash
mkdir -p "/users/visics/gkouros/projects/models/pascal3d/$1/evaluation"
touch "/users/visics/gkouros/projects/models/pascal3d/$1/evaluation/main.log"
cd "/users/visics/gkouros/projects/thesis/"

for LEVEL in {0..3}
do
    python3 scripts/evaluate_pascal3d.py \
        --root_dir="/esat/topaz/gkouros/datasets/pascal3d" \
        --runs_path="/users/visics/gkouros/projects/models/pascal3d/" \
        --experiment="$1" \
        --nofrom_scratch \
        --noevaluate_inference_time \
        --occlusion_level=$LEVEL \
        --notrain_plus_db \
        --nopositive_from_db \
        --bbox_noise=0 \
        --use_hdf5 \
        --nobgr \
        2>&1 | tee -a "/users/visics/gkouros/projects/models/pascal3d/$1/evaluation/main.log"
done