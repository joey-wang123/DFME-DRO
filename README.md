# Defending against Data-Free Model Extraction by  Distributionally Robust Defensive Training (NeurIPS 2023)




## Package Requirements
- Pytorch 1.12.1


## Model Extraction With Proposed Defense
## Step1: DRO Defensive Training

`cd teacher-train`

### CIFAR10

`python teacher-train_DRO.py --dataset cifar10 --model resnet34_8x  --method SVGD`


### CIFAR100

`python teacher-train_DRO.py --dataset cifar100 --model resnet34_8x  --method SVGD`

## Step2: DFME Defense Experiment

`cd DFME`

### CIFAR10

`python3 train_cifar10.py --model resnet34_8x --dataset cifar10 --scale 0.27 --batch_size 64 --ckpt 'path/to/your/DROmodel'    --device 0 --grad_m 1 --query_budget 20 --log_dir 'path/to/your/log/dir'  --lr_G 1e-4 --student_model resnet18_8x --loss l1;`

### CIFAR100

`python3 train_cifar100.py --model resnet34_8x --dataset cifar100 --scale 0.2 --batch_size 64 --ckpt  'path/to/your/DROmodel'  --device 0 --grad_m 1 --query_budget 200 --log_dir 'path/to/your/log/dir' --lr_G 1e-4 --student_model  resnet18_8x --loss l1;`

## Citation

If you find our paper or this resource helpful, please consider cite:

```
@inproceedings{wang2023defending,
  title={Defending against Data-Free Model Extraction by Distributionally Robust Defensive Training},
  author={Wang, Zhenyi and Shen, Li and Liu, Tongliang and Duan, Tiehang and Zhu, Yanjun and Zhan, Donglin and Doermann, David and Gao, Mingchen},
  booktitle={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023}
}
```