#!/bin/bash

for CATEGORY in aeroplane bicycle boat bottle bus car chair diningtable motorbike sofa train tvmonitor
do
    echo "Encoding category $CATEGORY"
    python3 scripts/encode_reference_set.py \
        --category=$CATEGORY \
        --weights_path=$(pwd)/trained_models/
done