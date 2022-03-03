save_dir=../../results/udr
CUDA_VISIBLE_DEVICES="" python train.py --save-dir $save_dir udr --config-file ../../config/train/udr_7_dims_0826/udr_large.json

export PYTHONPATH="/datamirror/yindaz/RL-CC/src"
nohup python train.py --save-dir ../../results/udr udr --config-file ../../config/train/udr_7_dims_0826/udr_large.json &