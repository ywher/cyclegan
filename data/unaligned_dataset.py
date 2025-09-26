import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class UnalignedDataset(BaseDataset):
    """
    非对齐/未配对数据集类
    
    用途：
        这个数据集类可以加载非对齐/未配对的数据集，主要用于CycleGAN等无监督图像翻译任务。
    
    数据结构要求：
        需要两个目录来存放来自不同域的训练图像：
        - 域A：'/path/to/data/trainA' 
        - 域B：'/path/to/data/trainB'
        
    训练命令：
        可以使用数据集标志 '--dataroot /path/to/data' 来训练模型。
        
    测试时类似地需要准备：
        - '/path/to/data/testA' 
        - '/path/to/data/testB' 目录
    
    关键特性：
        - 支持不同大小的A域和B域数据集
        - 通过serial_batches参数控制数据配对方式
        - 适用于风格迁移、域适应等任务
    """

    def __init__(self, opt):
        """初始化数据集类。

        参数：
            opt (Option类) -- 存储所有实验标志；需要是BaseOptions的子类
        """
        BaseDataset.__init__(self, opt)
        # 创建A域和B域的数据路径
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")  # 创建路径 '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")  # 创建路径 '/path/to/data/trainB'

        # 加载图像路径列表
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # 从 '/path/to/data/trainA' 加载图像
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # 从 '/path/to/data/trainB' 加载图像
        self.A_size = len(self.A_paths)  # 获取数据集A的大小
        self.B_size = len(self.B_paths)  # 获取数据集B的大小
        
        # 根据转换方向确定输入输出通道数
        btoA = self.opt.direction == "BtoA"  # 是否为B到A的转换
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc   # 输入图像的通道数
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # 输出图像的通道数
        
        # 创建图像变换（预处理）函数
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """返回一个数据点及其元数据信息。

        参数：
            index (int)      -- 用于数据索引的随机整数

        返回一个包含A、B、A_paths和B_paths的字典：
            A (tensor)       -- 输入域中的图像
            B (tensor)       -- 目标域中对应的图像
            A_paths (str)    -- A域图像路径
            B_paths (str)    -- B域图像路径
        """
        A_path = self.A_paths[index % self.A_size]  # 确保索引在范围内
        
        # serial_batches参数的关键作用：
        # - 如果为True：按顺序选择B域图像，保持A、B配对的一致性
        # - 如果为False：随机选择B域图像，避免固定的A-B配对关系
        if self.opt.serial_batches:  # 顺序批次模式：用于测试或需要可重现结果的情况
            index_B = index % self.B_size
        else:  # 随机批次模式：用于训练，通过随机配对增加数据多样性
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        
        # 加载图像并转换为RGB格式
        A_img = Image.open(A_path).convert("RGB")
        B_img = Image.open(B_path).convert("RGB")
        
        # 应用图像变换（调整大小、裁剪、数据增强等）
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        """返回数据集中图像的总数。

        由于我们有两个可能包含不同数量图像的数据集，
        我们取两者的最大值作为数据集长度。
        这确保了所有图像都能在训练过程中被访问到。
        """
        return max(self.A_size, self.B_size)
