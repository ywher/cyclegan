# maps
# python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --use_wandb

# cityscapes
# CUDA_VISIBLE_DEVICES=4 python train.py --dataroot ./datasets/cityscapes_seg_2_img --name cityscapes_cyclegan --model cycle_gan 
# --use_wandb

# city 2 acdc dark
# CUDA_VISIBLE_DEVICES=0 python train.py \
# --dataroot ./datasets/city2dark \
# --name city2dark_cyclegan \
# --model cycle_gan \
# --preprocess scale_width_and_crop \
# --load_size 1024 \
# --crop_size 512

# city 2 acdc fog
# CUDA_VISIBLE_DEVICES=5 python train.py \
# --dataroot ./datasets/city2fog \
# --name city2fog_cyclegan \
# --model cycle_gan \
# --preprocess scale_width_and_crop \
# --load_size 1024 \
# --crop_size 512

# city 2 acdc rain
# CUDA_VISIBLE_DEVICES=4 python train.py \
# --dataroot ./datasets/city2rain \
# --name city2rain_cyclegan \
# --model cycle_gan \
# --preprocess scale_width_and_crop \
# --load_size 1024 \
# --crop_size 512

# city 2 acdc snow
CUDA_VISIBLE_DEVICES=0 python train.py \
--dataroot ./datasets/city2snow \
--name city2snow_cyclegan \
--model cycle_gan \
--preprocess scale_width_and_crop \
--load_size 1024 \
--crop_size 512
