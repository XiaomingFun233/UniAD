按 README 的官方路径，UniAD 运行流程是这 7 步：

1. 安装环境  
见 [README.md](/home/UniAD/README.md) 和 [INSTALL.md](/home/UniAD/docs/INSTALL.md)  
核心版本是：`torch 2.0.1`、`mmcv-full 1.6.1`、`mmdet 2.26.0`、`mmdet3d 1.0.0rc6`。

2. 下载仓库并安装依赖  
```bash
git clone https://github.com/OpenDriveLab/UniAD.git
cd UniAD
pip install -r requirements.txt
```

3. 准备 checkpoint  
放到 `UniAD/ckpts/`（`uniad_base_track_map.pth`、`uniad_base_e2e.pth` 等），见 [INSTALL.md](/home/UniAD/docs/INSTALL.md)。

4. 准备 nuScenes 数据与 info 文件  
见 [DATA_PREP.md](/home/UniAD/docs/DATA_PREP.md)，最终目录要像文档里的结构：`data/nuscenes`、`data/infos`、`data/others`。

5. 先跑官方评测样例（验证环境是否通）  
见 [TRAIN_EVAL.md](/home/UniAD/docs/TRAIN_EVAL.md)  
```bash
cd /home/UniAD
./tools/uniad_dist_eval.sh ./projects/configs/stage1_track_map/base_track_map.py ./ckpts/uniad_base_track_map.pth 8
```

6. 训练  
- Stage1（track+map）  
```bash
./tools/uniad_dist_train.sh ./projects/configs/stage1_track_map/base_track_map.py N_GPUS
```
- Stage2（端到端）通常以上一阶段权重初始化（在 config 里设 `load_from`），再跑同样训练脚本。  
入口在 [README.md](/home/UniAD/README.md) 与 [TRAIN_EVAL.md](/home/UniAD/docs/TRAIN_EVAL.md)。

7. 评测与可视化  
- 评测：`./tools/uniad_dist_eval.sh CONFIG CKPT N_GPUS`  
- 可视化：见 [TRAIN_EVAL.md](/home/UniAD/docs/TRAIN_EVAL.md) 的 `visualization` 小节。

一句话总结：`安装环境 -> 准备数据/权重 -> 先跑 stage1 评测样例 -> stage1 训练 -> stage2 训练 -> 评测/可视化`。