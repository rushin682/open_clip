python -m training.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to "wandb" \
    --dataset-type "h5" \
    --train-data "HTAN-WUSTL_train.csv" \
    --val-data "HTAN-WUSTL_val.csv" \
    --lock-image \
    --lock-image-unlocked-groups -1 \
    --warmup 500 \
    --batch-size 128 \
    --lr 1e-3 \
    --wd 0.1 \
    --epochs 30 \
    --workers 0 \
    --wandb-project-name "ctranspath" \
    --model "spatial_clip_snn-ctranspath"

: ' model options
    - spatial_clip_snn-ctranspath
    - spatial_clip_scgpt-ctranspath
    - spatial_clip_scVI-ctranspath

    Need to preprocess the gexpressions before giving input to the gexp encoder.
'
