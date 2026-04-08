import torch
import time
import sys
import os

# ==========================================
# 1. 环境准备：将项目根目录加入 Python 路径
# ==========================================
# 这一步很重要，否则找不到 projects.mmdet3d_plugin
sys.path.append('/home/UniAD')

try:
    from projects.mmdet3d_plugin import UniAD
    from mmcv import Config
    print("✅ 成功导入 UniAD 库")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("💡 提示: 请确保你已经执行了 pip install -v -e . (即编译安装过 UniAD)")
    sys.exit(1)

# ==========================================
# 2. 模拟 UniAD 的复杂输入数据
# ==========================================
# UniAD 通常接受一个字典列表，或者直接是一个大字典
# 这里我们模拟最核心的输入：图像 (img_inputs)

# 参数设定 (UniAD 默认通常是 6 个相机, 3 个时刻, 3x360x640 或 3x900x1600)
# 为了测试速度，我们使用较小的尺寸 3x360x640
B = 2  # Batch Size
N = 6  # 相机数量
C = 3  # 通道数
H = 360 # 高度
W = 640 # 宽度

print(f"正在生成模拟数据: Batch={B}, Cameras={N}, Size={H}x{W}...")

# 模拟图像输入: (B, N, 3, H, W)
# 注意：UniAD 的输入通常是一个 list 或 dict，取决于具体实现版本
# 这里我们构造一个最通用的 Tensor 形式
fake_img = torch.randn(B, N, C, H, W).cuda()

# 构造输入字典 (模拟 mmdet 的 data dict 结构)
# 注意：UniAD 的 forward_train 需要特定的 key
data = {
    'img': [fake_img], # 很多版本包装在 list 里
    'img_metas': [
        {'filename': ['cam0.jpg']*N, 'box_type_3d': 'LiDAR'}, 
        {'filename': ['cam0.jpg']*N, 'box_type_3d': 'LiDAR'}
    ] # 这里必须有两个元素对应 B=2，否则会报错
}

print("✅ 数据生成完毕")

# ==========================================
# 3. 初始化模型
# ==========================================
print("正在加载模型配置...")
# 这里我们直接用代码构建一个最简配置，避免读取外部 config 文件的麻烦
# 如果你想用真实的配置文件，可以替换这部分为 Config.fromfile(...)

cfg_dict = dict(
    use_grid_mask=True,
    video_test_mode=False,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        with_cp=True),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=5),
    pts_bbox_head=dict(
        type='PETRHead', # 简化版 Head
        num_classes=10,
        in_channels=256,
        num_query=300,
        num_points=4,
        transformer=dict(
            type='PETRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=256,
                            num_levels=1),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        max_num=50)))

from mmcv import Config
cfg = Config(cfg_dict)

print("正在实例化 UniAD 模型...")
try:
    model = UniAD(**cfg_dict).cuda()
    print("✅ 模型实例化成功")
except Exception as e:
    print(f"❌ 模型实例化失败: {e}")
    print("💡 提示: UniAD 的结构非常复杂，如果这里报错，可能是配置参数缺失。")
    sys.exit(1)

# ==========================================
# 4. 运行 Batch 测试 (训练模式)
# ==========================================
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("🚀 开始跑 Batch 测试 (5个 Iteration)...")
start_time = time.time()

try:
    for i in range(5):
        optimizer.zero_grad()
        
        # 关键点：调用 forward_train 而不是 forward
        # UniAD 继承自 mmdet，训练时通常用 forward_train
        # 输入 data 字典，通常需要解包
        
        # 注意：UniAD 的输入处理非常繁琐，这里做一个通用的尝试
        # 如果报错 "missing keys"，说明 data 字典缺字段
        
        losses = model(return_loss=True, **data)
        
        # 计算总 Loss (UniAD 返回的是一个字典)
        loss = sum(_loss for _loss in losses.values())
        
        loss.backward()
        optimizer.step()
        
        print(f"Iter [{i+1}/5] Loss: {loss.item():.4f}")

    print("-" * 30)
    print("✅ 恭喜！Batch 测试通过！")
    print(f"⏱️ 耗时: {time.time() - start_time:.2f} 秒")
    print("💡 下一步：可以开始跑全量数据的 Epoch 测试了。")

except Exception as e:
    print(f"❌ 测试失败: {e}")
    print("💡 提示: 检查 data 字典的 key 是否与模型 forward_train 需要的匹配。")
