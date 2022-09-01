#!/bin/bash

WEIGHTS_PATH= "/esat/topaz/gkouros/models/pascal3d/trained_models"

mkdir -p "$WEIGHTS_PATH/evaluation"
touch "$WEIGHTS_PATH/pascal3d/evaluation/main.log"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

for LEVEL in {0..3}
do
    python3 scripts/evaluate_pascal3d.py \
        --dataset_path="/esat/topaz/gkouros/datasets/pascal3d" \
        --weights_path="/esat/topaz/gkouros/models/pascal3d/trained_models" \
        --object_category="car" \
        --nofrom_scratch \
        --noevaluate_inference_time \
        --occlusion_level=$LEVEL \
        --notrain_plus_db \
        --nopositive_from_db \
        --bbox_noise=0 \
        --nouse_hdf5 \
        --nobgr \
        2>&1 | tee -a "/esat/topaz/gkouros/models/evaluation/main.log"
done