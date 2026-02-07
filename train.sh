#!/bin/bash

# Tiktok Style
python train.py --config=configs/CIFAR10_VQ_ae.yaml \
    wandb_name='VQ_Stage_1_LQ'  \
    vq_type=VQ \
    use_tail_dropout=False \
    tied_timestep=False \
    l2_v_target=False \
    Decoder.query_type=learnable \
    use_noise_query=False \
    diffusion_decoder=False \
    tail_dropout_p=0.0 \

# One-D-Piece/ImageFolder Style
python train.py --config=configs/CIFAR10_VQ_ae.yaml \
    wandb_name='VQ_Stage_1_LQ_TD_P0.5'  \
    vq_type=VQ \
    use_tail_dropout=True \
    tied_timestep=False \
    l2_v_target=False \
    Decoder.query_type=learnable \
    use_noise_query=False \
    diffusion_decoder=False \
    tail_dropout_p=0.5 \

# flowmo style
python train.py --config=configs/CIFAR10_VQ_ae.yaml \
    wandb_name='VQ_Stage_1_NQ'  \
    vq_type=VQ \
    use_tail_dropout=False \
    tied_timestep=False \
    l2_v_target=True \
    Decoder.query_type=noise \
    use_noise_query=True \
    diffusion_decoder=True \
    tail_dropout_p=0.0 \

# flextok style
python train.py --config=configs/CIFAR10_VQ_ae.yaml \
    wandb_name='VQ_Stage_1_NQ_TD'  \
    vq_type=VQ \
    use_tail_dropout=True \
    tied_timestep=False \
    l2_v_target=True \
    Decoder.query_type=noise \
    use_noise_query=True \
    diffusion_decoder=True \
    tail_dropout_p=0.5 \


# selftok style
python train.py --config=configs/CIFAR10_VQ_ae.yaml \
    wandb_name='VQ_Stage_1_NQ_TD_Tied'  \
    vq_type=VQ \
    use_tail_dropout=True \
    tied_timestep=True \
    l2_v_target=True \
    Decoder.query_type=noise \
    use_noise_query=True \
    diffusion_decoder=True \
    tail_dropout_p=1.0 \
