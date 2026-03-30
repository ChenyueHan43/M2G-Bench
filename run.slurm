#!/bin/bash
#SBATCH --job-name=GS_Baseline      # 给你的任务起个帅气的名字
#SBATCH --nodes=1                   # 申请 1 个节点
#SBATCH --cpus-per-task=8           # 申请 8 个 CPU 核心
#SBATCH --mem=64GB                  # 申请 64GB 内存
#SBATCH --gres=gpu:1                # 申请 1 张显卡 (H200)
#SBATCH --time=04:00:00             # 申请 4 个小时的时长
#SBATCH --output=result_%j.log      # 把运行过程的输出保存到这个日志文件里

echo " 任务开始排队并成功获取节点，开始工作！"

# 1. 激活你的 Conda 环境 (这行是为了确保后台能识别 conda)
source ~/.bashrc
conda activate graphsearch

# 2. 进入你的代码目录
cd /scratch/ch5085/graphsearch

# 3. 安装极速下载神器 (如果你之前没安过的话)
pip install -U huggingface_hub hf_transfer

# 4. 开启满血加速，自动把 65GB 的 Qwen 模型下载到你的 scratch 根目录
echo " 开始极速下载 Qwen2.5-32B 模型..."
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir /scratch/ch5085/Qwen2.5-32B-Instruct --local-dir-use-symlinks False

# 5. 运行你的 Python 代码！
echo " 模型下载完毕，开始跑实验！"
#  注意：把下面这行的 main.py 换成你实际用来跑实验的文件名（比如 agent.py 或 run.py）
python run_experiment.py

echo " 全部任务结束！"
