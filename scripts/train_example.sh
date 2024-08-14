export NCCL_DEBUG=INFO
export GPUS_PER_NODE=1
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=24212
CONFIG="config/spt_base_cfg.json"

torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    scripts/train_example.py \
    --config ${CONFIG} 