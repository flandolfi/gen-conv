#!/bin/bash

OPTS="--model CustomDGCNN --dataset ModelNet40 --batch_size 8 --num_workers 16"

for SEED in {0..9}; do
    python -m benchmark train $OPTS --valid_size 0 --max_epochs 60 \
        --checkpoint_name final_$SEED --seed $SEED
done

python -m benchmark test $OPTS --checkpoint_name "final_*" --devices 1
