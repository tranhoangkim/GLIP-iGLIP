export CUDA_VISIBLE_DEVICES=4
nohup python demo/finetune_iglip_global_threshold.py > iglip.log &

# export CUDA_VISIBLE_DEVICES=3
# nohup python demo/finetune_glip_global_threshold.py > glip.log &

# SAVE_DIR='./results/GMOT_40_finetune'
# rm -r $SAVE_DIR 