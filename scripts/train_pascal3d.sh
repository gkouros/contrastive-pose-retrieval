#!/bin/bash

EXPERIMENT=$1
LOSS="weighted_contrastive"
CHECKPOINT=""
POSITIVE_TYPE="normals"
DATASET="pascal3d"
LABELLING="quat"

# add executable
python3 train_pascal3d.py \
    --logs \
    --logdir="/users/visics/gkouros/projects/models/pascal3d/" \
    --dataset_path="/esat/topaz/gkouros/datasets/" \
    --checkpoint="$CHECKPOINT" \
    --object_category="car" \
    --positive_type="$POSITIVE_TYPE" \
    --nopositive_from_db \
    --horizontal_flip \
    --subset_ratio=1 \
    --use_miner \
    --labelling_method="$LABELLING" \
    --num_workers=4 \
    --num_epochs=1000 \
    --num_iters_per_epoch=0 \
    --batch_size=32 \
    --embedding_size=512 \
    --trunk_lr=0.0001 \
    --trunk_wd=0.0005 \
    --embedder_lr=0.001 \
    --embedder_wd=0.0005 \
    --margin=1.0 \
    --loss=$LOSS \
    --patience=1000 \
    --lr_scheduler="exp" \
    --lr_decay_patience=1 \
    --lr_decay=1.0 \
    --k=0 \
    --backbone="resnet50" \
    --type_of_triplets="all" \
    --multimodal \
    --experiment=$EXPERIMENT-$LOSS-$POSITIVE_TYPE-$LABELLING \
    --primary_metric='accuracy_at_10' \
    --pose_positive_threshold=5 \
    --nouse_hdf5 \
    --nouse_fixed_cad_model \
    --occlusion_scale=0.5 \
    --weight_mode="cad" \
    --bbox_noise=0.1
