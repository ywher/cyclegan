# maps
# python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --use_wandb

# cityscapes
# CUDA_VISIBLE_DEVICES=4 python train.py --dataroot ./datasets/cityscapes_seg_2_img --name cityscapes_cyclegan --model cycle_gan 
# --use_wandb

#########################
### cityscapes 2 acdc ###
#########################
# load_size=1024
# crop_size=512
# city 2 acdc dark
# CUDA_VISIBLE_DEVICES=0 python train.py \
# --dataroot ./datasets/city2dark \
# --name city2dark_cyclegan \
# --model cycle_gan \
# --preprocess scale_width_and_crop \
# --load_size $load_size \
# --crop_size $crop_size

# city 2 acdc fog
# CUDA_VISIBLE_DEVICES=5 python train.py \
# --dataroot ./datasets/city2fog \
# --name city2fog_cyclegan \
# --model cycle_gan \
# --preprocess scale_width_and_crop \
# --load_size $load_size \
# --crop_size $crop_size

# city 2 acdc rain
# CUDA_VISIBLE_DEVICES=4 python train.py \
# --dataroot ./datasets/city2rain \
# --name city2rain_cyclegan \
# --model cycle_gan \
# --preprocess scale_width_and_crop \
# --load_size $load_size \
# --crop_size $crop_size

# city 2 acdc snow
# CUDA_VISIBLE_DEVICES=0 python train.py \
# --dataroot ./datasets/city2snow \
# --name city2snow_cyclegan \
# --model cycle_gan \
# --preprocess scale_width_and_crop \
# --load_size $load_size \
# --crop_size $crop_size

###################
### acdc 2 acdc ###
###################
load_size=910
crop_size=512
# acdc 2 acdc dark
CUDA_VISIBLE_DEVICES=0 python train.py \
--dataroot ./datasets/acdc2dark \
--name acdc2dark_cyclegan \
--model cycle_gan \
--preprocess scale_width_and_crop \
--load_size $load_size \
--crop_size $crop_size \
--serial_batches

# acdc 2 acdc fog
# CUDA_VISIBLE_DEVICES=5 python train.py \
# --dataroot ./datasets/acdc2fog \
# --name acdc2fog_cyclegan \
# --model cycle_gan \
# --preprocess scale_width_and_crop \
# --load_size $load_size \
# --crop_size $crop_size \
# --serial_batches

# acdc 2 acdc rain
# CUDA_VISIBLE_DEVICES=4 python train.py \
# --dataroot ./datasets/acdc2rain \
# --name acdc2rain_cyclegan \
# --model cycle_gan \
# --preprocess scale_width_and_crop \
# --load_size $load_size \
# --crop_size $crop_size \
# --serial_batches

# acdc 2 acdc snow
# CUDA_VISIBLE_DEVICES=0 python train.py \
# --dataroot ./datasets/acdc2snow \
# --name acdc2snow_cyclegan \
# --model cycle_gan \
# --preprocess scale_width_and_crop \
# --load_size $load_size \
# --crop_size $crop_size \
# --serial_batches
