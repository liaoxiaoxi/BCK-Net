# koopman_dataset_factory.py

from typing import Tuple

# 你的三个 dataset 文件名按你现在的命名来
from daub_dataset_dg_final import DAUBDataset
# from irdst_dataset1 import IRDSTDataset
# from irstd_uav_dataset import IRSTDUAVDataset


def build_dataset(
    dataset_name: str,
    root: str,
    train_meta: str,
    val_meta: str,
    T_clip: int = 5,
):
    """
    Returns:
        train_set, val_set

    dataset_name:
        'daub'
        'irdst'
        'irstd_uav'
    """

    dataset_name = dataset_name.lower()

    if dataset_name == "daub":
        train_set = DAUBDataset(root=root, list_file=train_meta, T_clip=T_clip)
        val_set = DAUBDataset(root=root, list_file=val_meta, T_clip=T_clip)
        return train_set, val_set

    elif dataset_name == "irdst":
        train_set = IRDSTDataset(root=root, list_file=train_meta, T_clip=T_clip, target_size=(512, 512))
        val_set = IRDSTDataset(root=root, list_file=val_meta, T_clip=T_clip, target_size=(512, 512))
        return train_set, val_set

    elif dataset_name == "irstd_uav":
        # IRSTDUAVDataset 的接口是 split，不是 list_file
        train_set = IRSTDUAVDataset(root=root, split=train_meta, T_clip=T_clip)
        val_set = IRSTDUAVDataset(root=root, split=val_meta, T_clip=T_clip)
        return train_set, val_set

    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")