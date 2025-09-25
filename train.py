"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp


def format_time(seconds):
    """格式化时间显示（秒转换为时分秒格式）"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


if __name__ == "__main__":
    opt = TrainOptions().parse()  # get training options
    opt.device = init_ddp()
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print(f"The number of training images = {dataset_size}")

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations
    
    # 添加训练时间统计变量
    training_start_time = time.time()  # 训练开始时间
    total_epochs = opt.n_epochs + opt.n_epochs_decay  # 总训练轮数
    completed_epochs = 0  # 已完成的轮数
    
    print(f"开始训练，总共需要训练 {total_epochs} 轮")
    
    # 将训练开始信息写入日志
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(training_start_time))
    with open(visualizer.log_name, "a") as log_file:
        log_file.write(f"[训练开始] 开始时间: {start_time_str}, 总轮数: {total_epochs}\n")
    
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()
        # Set epoch for DistributedSampler
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(total_iters, losses)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        model.update_learning_rate()  # update learning rates at the end of every epoch

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
            model.save_networks("latest")
            model.save_networks(epoch)

        print(f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time:.0f} sec")
        
        # 计算并显示预计剩余训练时间
        completed_epochs += 1
        current_time = time.time()
        elapsed_time = current_time - training_start_time
        
        if completed_epochs > 0:
            avg_time_per_epoch = elapsed_time / completed_epochs
            remaining_epochs = total_epochs - completed_epochs
            estimated_remaining_time = avg_time_per_epoch * remaining_epochs
            
            # 计算预计结束时间
            estimated_end_time = current_time + estimated_remaining_time
            end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(estimated_end_time))
            
            elapsed_str = format_time(elapsed_time)
            remaining_str = format_time(estimated_remaining_time)
            
            progress_message = (f"训练进度: {completed_epochs}/{total_epochs} ({100*completed_epochs/total_epochs:.1f}%) | "
                             f"已用时间: {elapsed_str} | 预计剩余时间: {remaining_str} | "
                             f"预计结束时间: {end_time_str}")
            
            print(progress_message)
            print("-" * 80)
            
            # 将训练进度信息写入日志文件
            with open(visualizer.log_name, "a") as log_file:
                log_file.write(f"[训练进度统计] {progress_message}\n")

    # 训练完成后的总结
    total_training_time = time.time() - training_start_time
    total_time_str = format_time(total_training_time)
    completion_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    completion_message = f"训练完成！总用时: {total_time_str}, 完成时间: {completion_time_str}"
    print("=" * 80)
    print(completion_message)
    print("=" * 80)
    
    # 将训练完成信息写入日志
    with open(visualizer.log_name, "a") as log_file:
        log_file.write(f"[训练完成] {completion_message}\n")
        log_file.write("=" * 80 + "\n")
    
    cleanup_ddp()
