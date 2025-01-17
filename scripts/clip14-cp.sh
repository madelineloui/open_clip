#!/bin/bash
##Slurm sbatch options
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:volta:2
#SBATCH --exclusive -O

# Loading the required modules
source /etc/profile
module load anaconda/2023a

# Activate conda env
source activate open-clip

# enter the src folder of the open_clip repository
cd src

# set the training args

torchrun --nproc_per_node 2 -m open_clip_train.main \
    --batch-size 90 \
    --precision amp \
    --workers 1 \
    --report-to tensorboard \
    --save-frequency 1 \
    --logs /home/gridsan/manderson/open_clip/logs \
    --dataset-type csv \
    --csv-separator ',' \
    --train-data /home/gridsan/manderson/vlm4rs/fmow/fmow_mm_data.csv \
    --csv-img-key filepath \
    --csv-caption-key caption \
    --warmup 1000 \
    --lr=5e-6 \
    --wd=0.1 \
    --epochs=32 \
    --model ViT-L-14 \
    --pretrained /home/gridsan/manderson/open_clip/vit_large_patch14_clip_224.openai/open_clip_pytorch_model.bin