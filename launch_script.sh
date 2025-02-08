node_list=$SLURM_STEP_NODELIST
master_addr=$(scontrol show hostname ${node_list} | head -n1)
export MASTER_ADDR=$master_addr
apptainer exec --bind /mnt:/mnt --nv  /mnt/petrelfs/share_data/tongronglei/maniskill2.sif /mnt/petrelfs/zhangtianyi1/anaconda3/envs/RTtest/bin/python rt1_pytorch/train_sim_more_action.py