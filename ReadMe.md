python -m workspace.src.train.chord_train

CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --use_env workspace.src.train.chord_train.py

/workspace 폴더 위에서