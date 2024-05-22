from mpi4py import MPI
from torch.utils.data import DataLoader

from datasets.zurich_raw2rgb_dataset import ZurichRAW2RGBMultiGPU
from datasets.synthetic_burst_train_set import SyntheticBurst
from datasets.burstsr_dataset import BurstSRDataset

def load_data_bsr(
    *,
    data_dir,
    batch_size,
    burst_size=8,
    deterministic=False,
):
    if not data_dir:
        raise ValueError("unspecified data directory")

    train_zurich_raw2rgb = ZurichRAW2RGBMultiGPU(root=data_dir,  split='train', shard=MPI.COMM_WORLD.Get_rank(), num_shards=MPI.COMM_WORLD.Get_size(),)
    train_dataset = SyntheticBurst(train_zurich_raw2rgb, burst_size=burst_size, crop_sz=256)

    if deterministic:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1, pin_memory=True
            )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True
            )
    while True:
        yield from train_loader



def load_data_bsr_real(
    *,
    data_dir,
    batch_size,
    burst_size=8,
    deterministic=False,
):
    if not data_dir:
        raise ValueError("unspecified data directory")

    train_dataset = BurstSRDataset(data_dir, burst_size=burst_size, crop_sz=32, split='train')


    if deterministic:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1, pin_memory=True
            )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True
            )
    while True:
        yield from train_loader


##########################################################################################
############## shadow removal ############################################################
from datasets.deshadow_dataset import create_dataset

def load_data_deshadow(
    *,
    dataroot_LQ,
    dataroot_GT,
    phase,
    name,
    batch_size,
    dataroot_mask=None,
    crop_size=None,
    use_flip=False,
    use_rot=False,
    use_swap=False,
    color=None,
    deterministic=False,
):
    if not dataroot_LQ:
        raise ValueError("unspecified data directory")

    train_dataset = create_dataset(
        dataroot_LQ,
        dataroot_GT,
        phase,
        name,
        dataroot_mask=dataroot_mask,
        crop_size=crop_size,
        use_flip=use_flip,
        use_rot=use_rot,
        use_swap=use_swap,
        color=color,)

    if deterministic:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1, pin_memory=True
            )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True
            )
    while True:
        yield from train_loader

