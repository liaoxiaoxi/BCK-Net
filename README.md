# BCK-Net
BCK-Net: Background-Consistent Koopman Dynamics for Infrared Video Small Target Detection
By Xiaoxi Liao, Kai Wang*, Wei Jiang, Hongke Zhang

<img width="1414" height="523" alt="image" src="https://github.com/user-attachments/assets/53a8fe0a-3143-476d-940d-3409c824a3da" />

<img width="1403" height="530" alt="image" src="https://github.com/user-attachments/assets/d2e69a77-b106-4b17-86d9-67c044812c83" />

## Requirements

* Python 3.8
* torch 2.1.2+cu118
* torchvision 0.16.2+cu118
* opencv-python 4.13.0.92
* numpy 1.26.4
* Pillow 10.3.0
* tqdm 4.64.1
* matplotlib 3.8.2
* imageio 2.34.0

You can install the main dependencies with:

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python==4.13.0.92 numpy==1.26.4 pillow==10.3.0 tqdm==4.64.1 matplotlib==3.8.2 imageio==2.34.0
```

## To test

```bash
python test.py \
  --dataset_name daub \
  --root DAUB \
  --val_meta DAUB/test.txt \
  --T_clip 40 \
  --checkpoint ./checkpoints_bck_ktvm/best_f1.pth \
  --save_dir ./test_results_daub \
  --save_vis
```

For other datasets, change `--dataset_name`, `--root`, and the corresponding meta files. For example:

```bash
# IRDST
python test.py \
  --dataset_name irdst \
  --root IRDST \
  --val_meta IRDST/test.txt \
  --T_clip 40 \
  --checkpoint ./checkpoints_bck_ktvm/best_f1.pth \
  --save_dir ./test_results_irdst \
  --save_vis

# IRSTD-UAV
python test.py \
  --dataset_name irstd_uav \
  --root IRSTD-UAV \
  --val_meta test \
  --T_clip 40 \
  --checkpoint ./checkpoints_bck_ktvm/best_f1.pth \
  --save_dir ./test_results_irstd_uav \
  --save_vis

# NUDT-MIRSDT
python test.py \
  --dataset_name nudt_mirsdt \
  --root NUDT-MIRSDT \
  --val_meta test \
  --T_clip 40 \
  --checkpoint ./checkpoints_bck_ktvm/best_f1.pth \
  --save_dir ./test_results_nudt_mirsdt \
  --save_vis
```

## To train

```bash
python train.py \
  --dataset_name daub \
  --root DAUB \
  --train_meta DAUB/train.txt \
  --val_meta DAUB/test.txt \
  --T_clip 40 \
  --batch_size 4 \
  --epochs 6000 \
  --lr 5e-5 \
  --save_dir ./checkpoints_bck_ktvm
```

For other datasets, use:

```bash
# IRDST
python train.py \
  --dataset_name irdst \
  --root IRDST \
  --train_meta IRDST/train.txt \
  --val_meta IRDST/test.txt \
  --T_clip 40 \
  --batch_size 4 \
  --epochs 6000 \
  --lr 5e-5 \
  --save_dir ./checkpoints_bck_irdst

# IRSTD-UAV
python train.py \
  --dataset_name irstd_uav \
  --root IRSTD-UAV \
  --train_meta train \
  --val_meta test \
  --T_clip 40 \
  --batch_size 4 \
  --epochs 6000 \
  --lr 5e-5 \
  --save_dir ./checkpoints_bck_irstd_uav

# NUDT-MIRSDT
python train.py \
  --dataset_name nudt_mirsdt \
  --root NUDT-MIRSDT \
  --train_meta train \
  --val_meta test \
  --T_clip 40 \
  --batch_size 4 \
  --epochs 6000 \
  --lr 5e-5 \
  --save_dir ./checkpoints_bck_nudt_mirsdt
```

### Dataset Download

The four moving infrared small target detection datasets used in this work are publicly available. Their download links can be found below.

* **DAUB**
  DAUB is an infrared dim-small aircraft target detection and tracking dataset under ground/air backgrounds.

  * [Science Data Bank](https://www.scidb.cn/en/detail?dataSetId=720626420933459968)

* **IRDST**
  IRDST is a large-scale infrared dim small target detection dataset used in RDIAN.

  * [RDIAN GitHub](https://github.com/sun11999/RDIAN)
  * [Dataset Page](https://xzbai.buaa.edu.cn/datasets.html)

* **IRSTD-UAV**
  IRSTD-UAV is a UAV-based moving infrared small target detection dataset containing real-world infrared video sequences with complex backgrounds.

  * [TDCNet GitHub](https://github.com/IVPLabX/TDCNet)

* **NUDT-MIRSDT**
  NUDT-MIRSDT is a multi-frame infrared small target detection dataset with mask and point-level annotations. It is used to evaluate moving infrared small target detection under normal and low-SNR conditions.

  * [DTUM GitHub](https://github.com/TinaLRJ/Multi-frame-infrared-small-target-detection-DTUM)


Please make sure the paths in `train.txt` and `test.txt` are consistent with your local dataset structure.

