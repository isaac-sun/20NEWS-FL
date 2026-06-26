帮我把这个压缩包里的项目改成在我的电脑上能跑的版本：

设备：带有 NVIDIA 显卡的游戏本，CUDA 可用

1. 解压，安装依赖：torch, scikit-learn, numpy, pandas, matplotlib, seaborn, openpyxl, tqdm, transformers
2. 把 main.py 里 Config 中的 device 改成 "cuda"，batch_size 改成 32，num_rounds 改成 30，num_mc_samples 改成 30
3. 运行 python main.py，跑完把 results 文件夹压缩发给我
