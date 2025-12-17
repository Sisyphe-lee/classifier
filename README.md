## 项目简介
简单 CNN 岩石分类器，数据集目录 `homework/{train,val,test}` 下每个类别一个子文件夹（包含 garbage 类）。

## 环境准备
- Python ≥ 3.8
- PyTorch + torchvision（支持 GPU 最佳）
- 依赖安装示例：`pip install torch torchvision pillow`

## 训练
默认从 `homework/train` 和 `homework/val` 读取数据，指标实时写入 `training_log.csv`，每轮在 `homework/ckpt` 保存一次。
- 单机（单/多卡 DataParallel）：  
  `python train.py --data-root homework --epochs 15 --batch-size 32 --val-interval 100`
- 分布式多卡（DDP + torchrun）：  
  `torchrun --nproc_per_node=4 train.py --data-root homework --distributed --val-interval 100`
- 常用可调参数：`--image-size`（默认 224），`--max-val-batches`（限制验证批次数以加速），`--ckpt-dir`（检查点目录），`--log-csv`（日志路径）。

## 推理
使用训练产生的检查点预测单张图片类别：
`python inference.py /path/to/image.jpg --checkpoint /path/to/epoch_002.pth --device cuda:0`

输出示例：`Predicted: sandstone with confidence 0.87 (from epoch 5)`
