# -*- coding: utf-8 -*-
"""
Cityscapes数据集预处理脚本

功能：
    将Cityscapes数据集转换为适用于pix2pix和CycleGAN训练的格式
    
输入：
    - gtFine目录：包含语义分割标签图像的目录
    - leftImg8bit目录：包含街景照片的目录
    
输出：
    - pix2pix格式：将语义分割图和真实照片水平拼接成512x256的图像
    - CycleGAN格式：将语义分割图和真实照片分别保存在A、B两个目录中
"""

import os
import glob
from PIL import Image

help_msg = """
数据集可以从 https://cityscapes-dataset.com 下载。
请下载数据集 [gtFine_trainvaltest.zip] 和 [leftImg8bit_trainvaltest.zip] 并解压它们。
gtFine 包含语义分割标签。使用 --gtFine_dir 指定解压后的 gtFine_trainvaltest 目录路径。
leftImg8bit 包含行车记录仪拍摄的照片。使用 --leftImg8bit_dir 指定解压后的 leftImg8bit_trainvaltest 目录路径。
处理后的图像将保存到 --output_dir 目录中。

使用示例：

python prepare_cityscapes_dataset.py --gtFine_dir ./gtFine/ --leftImg8bit_dir ./leftImg8bit --output_dir ./datasets/cityscapes/
"""

def load_resized_img(path):
    """
    加载并调整图像大小
    
    输入：
        path (str): 图像文件路径
        
    输出：
        PIL.Image: 调整为256x256大小的RGB图像
    """
    return Image.open(path).convert('RGB').resize((256, 256))

def check_matching_pair(segmap_path, photo_path):
    """
    检查语义分割图和照片是否匹配
    
    输入：
        segmap_path (str): 语义分割图路径
        photo_path (str): 照片路径
        
    功能：
        验证两个文件名是否对应同一张图像，确保数据配对正确
    """
    # 从文件名中提取标识符，去除特定后缀
    segmap_identifier = os.path.basename(segmap_path).replace('_gtFine_color', '')
    photo_identifier = os.path.basename(photo_path).replace('_leftImg8bit', '')
        
    # 断言两个标识符必须相同，否则抛出错误
    assert segmap_identifier == photo_identifier, \
        "[%s] 和 [%s] 似乎不匹配。终止处理。" % (segmap_path, photo_path)
    

def process_cityscapes(gtFine_dir, leftImg8bit_dir, output_dir, phase):
    """
    处理Cityscapes数据集
    
    输入：
        gtFine_dir (str): 语义分割标签目录路径
        leftImg8bit_dir (str): 街景照片目录路径  
        output_dir (str): 输出目录路径
        phase (str): 数据集阶段，'train' 或 'val'
        
    输出：
        在output_dir中创建处理后的数据集：
        - train/test目录：包含pix2pix格式的拼接图像
        - train/testA目录：包含CycleGAN的A域图像（真实照片）
        - train/testB目录：包含CycleGAN的B域图像（语义分割图）
    """
    # 将验证集映射为测试集，训练集保持不变
    save_phase = 'test' if phase == 'val' else 'train'
    savedir = os.path.join(output_dir, save_phase)
    
    # 创建必要的目录结构
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(savedir + 'A', exist_ok=True)  # CycleGAN的A域（真实照片）
    os.makedirs(savedir + 'B', exist_ok=True)  # CycleGAN的B域（语义分割图）
    print("目录结构已在 %s 准备完成" % output_dir)
    
    # 构建语义分割图文件路径模式并获取所有匹配的文件
    segmap_expr = os.path.join(gtFine_dir, phase) + "/*/*_color.png"
    segmap_paths = glob.glob(segmap_expr)
    segmap_paths = sorted(segmap_paths)

    # 构建街景照片文件路径模式并获取所有匹配的文件
    photo_expr = os.path.join(leftImg8bit_dir, phase) + "/*/*_leftImg8bit.png"
    photo_paths = glob.glob(photo_expr)
    photo_paths = sorted(photo_paths)

    # 确保语义分割图和照片数量一致
    assert len(segmap_paths) == len(photo_paths), \
        "找到 %d 张匹配 [%s] 的图像，和 %d 张匹配 [%s] 的图像。数量不一致，终止处理。" % (len(segmap_paths), segmap_expr, len(photo_paths), photo_expr)

    # 遍历所有图像对进行处理
    for i, (segmap_path, photo_path) in enumerate(zip(segmap_paths, photo_paths)):
        # 检查当前图像对是否匹配
        check_matching_pair(segmap_path, photo_path)
        
        # 加载并调整图像大小到256x256
        segmap = load_resized_img(segmap_path)  # 语义分割图
        photo = load_resized_img(photo_path)    # 真实照片

        # 为pix2pix创建数据：将两张图像水平拼接成一张512x256的图像
        sidebyside = Image.new('RGB', (512, 256))
        sidebyside.paste(segmap, (256, 0))  # 语义分割图放在右侧
        sidebyside.paste(photo, (0, 0))     # 真实照片放在左侧
        savepath = os.path.join(savedir, "%d.jpg" % i)
        sidebyside.save(savepath, format='JPEG', subsampling=0, quality=100)

        # 为CycleGAN创建数据：将两张图像分别保存在不同目录中
        # A域：真实照片
        savepath = os.path.join(savedir + 'A', "%d_A.jpg" % i)
        photo.save(savepath, format='JPEG', subsampling=0, quality=100)
        
        # B域：语义分割图
        savepath = os.path.join(savedir + 'B', "%d_B.jpg" % i)
        segmap.save(savepath, format='JPEG', subsampling=0, quality=100)
        
        # 每处理10%的图像就打印一次进度
        if i % (len(segmap_paths) // 10) == 0:
            print("%d / %d: 最后保存的图像路径 %s" % (i, len(segmap_paths), savepath))



if __name__ == '__main__':
    """
    主函数：处理命令行参数并执行数据集预处理
    
    命令行参数：
        --gtFine_dir: Cityscapes gtFine目录的路径（必需）
        --leftImg8bit_dir: Cityscapes leftImg8bit目录的路径（必需）  
        --output_dir: 输出图像保存目录的路径（必需）
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gtFine_dir', type=str, required=True,
                        help='Cityscapes gtFine目录的路径。')
    parser.add_argument('--leftImg8bit_dir', type=str, required=True,
                        help='Cityscapes leftImg8bit_trainvaltest目录的路径。')
    parser.add_argument('--output_dir', type=str, required=True,
                        default='./datasets/cityscapes',
                        help='输出图像的保存目录。')
    opt = parser.parse_args()

    print(help_msg)
    
    print('为验证集准备Cityscapes数据集')
    process_cityscapes(opt.gtFine_dir, opt.leftImg8bit_dir, opt.output_dir, "val")
    print('为训练集准备Cityscapes数据集')
    process_cityscapes(opt.gtFine_dir, opt.leftImg8bit_dir, opt.output_dir, "train")

    print('完成')

    

