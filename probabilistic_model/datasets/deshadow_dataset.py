"""create dataset and dataloader"""
import logging

import torch
import torch.utils.data

import cv2
import random
import numpy as np

import datasets.deshadow_utils as util


def create_dataset(
        dataroot_LQ,
        dataroot_GT,
        phase,
        name,
        dataroot_mask=None,
        crop_size=None,
        use_flip=False,
        use_rot=False,
        use_swap=False,
        color=None,
    ):
    if phase == "train":
        dataset = LQGTDataset(
                    dataroot_LQ,
                    dataroot_GT,
                    phase,
                    dataroot_mask=dataroot_mask,
                    LR_size=crop_size,
                    GT_size=crop_size,
                    use_flip=use_flip,
                    use_rot=use_rot,
                    use_swap=use_swap,
                    color=color,)
    elif phase == "valid":
        dataset = LQDataset(
                    dataroot_LQ,
                    phase,
                    dataroot_mask=dataroot_mask,
                    LR_size=crop_size,
                    use_flip=use_flip,
                    use_rot=use_rot,
                    use_swap=use_swap,
                    color=color,)
    elif phase == "train_val":
        dataset = LQGTDataset_(
                    dataroot_LQ,
                    dataroot_GT,
                    phase,
                    dataroot_mask=dataroot_mask,
                    LR_size=crop_size,
                    GT_size=crop_size,
                    use_flip=use_flip,
                    use_rot=use_rot,
                    use_swap=use_swap,
                    color=color,)

    logger = logging.getLogger("base")
    logger.info(
        "Dataset [{:s} - {:s}] is created.".format(
            dataset.__class__.__name__, name
        )
    )
    return dataset

class LQGTDataset(torch.utils.data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(
        self, 
        dataroot_LQ,
        dataroot_GT,
        phase,
        dataroot_mask=None,
        LR_size=None,
        GT_size=None,
        use_flip=False,
        use_rot=False,
        use_swap=False,
        color=None,
    ):
        super().__init__()
        self.phase = phase
        self.dataroot_mask = dataroot_mask
        self.use_flip = use_flip
        self.use_rot = use_rot
        self.use_swap = use_swap
        self.color = color,
        self.LR_size, self.GT_size = LR_size, GT_size
        self.LR_env, self.GT_env, self.mask_env = None, None, None  # environment for lmdb

        # read image list from image files
        self.LR_paths = util.get_image_paths(
            "img", dataroot_LQ
        )  # LR list
        self.GT_paths = util.get_image_paths(
            "img", dataroot_GT
        )  # GT list
        if dataroot_mask != None:
            self.mask_paths = util.get_image_paths(
                "img", self.dataroot_mask
            )  # mask list   
        if self.LR_paths and self.GT_paths:
            assert len(self.LR_paths) == len(
                self.GT_paths
            ), "GT and LR datasets have different number of images - {}, {}.".format(
                len(self.LR_paths), len(self.GT_paths)
            )
        self.random_scale_list = [1]

    def __getitem__(self, index):

        GT_path, LR_path = None, None
        scale = 1

        # get GT image
        GT_path = self.GT_paths[index]
        resolution = None
        img_GT = util.read_img(
            self.GT_env, GT_path, resolution
        )  # return: Numpy float32, HWC, BGR, [0,1]

        # modcrop in the validation / test phase
        if self.phase != 'train':
            img_GT = util.modcrop(img_GT, scale)

        # get LR image
        LR_path = self.LR_paths[index]
        img_LR = util.read_img(self.LR_env, LR_path, resolution)

        if self.dataroot_mask != None:
            #get mask image
            mask_path = self.mask_paths[index]
            img_mask = util.read_img(self.mask_env, mask_path, resolution)

        if self.phase == 'train':
            H, W, C = img_LR.shape
            assert self.LR_size == self.GT_size // scale, "GT size does not match LR size"

            # randomly crop
            rnd_h = random.randint(0, max(0, H - self.LR_size))
            rnd_w = random.randint(0, max(0, W - self.LR_size))
            img_LR = img_LR[rnd_h : rnd_h + self.LR_size, rnd_w : rnd_w + self.LR_size, :]
            if self.dataroot_mask != None:
                img_mask = img_mask[rnd_h : rnd_h + self.LR_size, rnd_w : rnd_w + self.LR_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[
                rnd_h_GT : rnd_h_GT + self.GT_size, rnd_w_GT : rnd_w_GT + self.GT_size, :
            ]

            # augmentation - flip, rotate
            if self.dataroot_mask != None:
                img_LR, img_GT, img_mask = util.augment(
                    [img_LR, img_GT, img_mask],
                    self.use_flip,
                    self.use_rot,
                    "LQGT",
                    self.use_swap,
                )
            else :
                img_LR, img_GT = util.augment(
                    [img_LR, img_GT],
                    self.use_flip,
                    self.use_rot,
                    "LQGT",
                    self.use_swap,
                )

        elif self.LR_size is not None:
            H, W, C = img_LR.shape
            assert self.LR_size == self.GT_size // scale, "GT size does not match LR size"

            if self.LR_size < H and self.LR_size < W:
                # center crop
                rnd_h = H // 2 - self.LR_size//2
                rnd_w = W // 2 - self.LR_size//2
                img_LR = img_LR[rnd_h : rnd_h + self.LR_size, rnd_w : rnd_w + self.LR_size, :]
                if self.dataroot_mask != None:
                    img_mask = img_mask[rnd_h : rnd_h + self.LR_size, rnd_w : rnd_w + self.LR_size, :]
                rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[
                    rnd_h_GT : rnd_h_GT + self.GT_size, rnd_w_GT : rnd_w_GT + self.GT_size, :
                ]

        # change color space if necessary
        if self.color:
            H, W, C = img_LR.shape
            img_LR = util.channel_convert(C, self.color, [img_LR])[
                0
            ]  # TODO during val no definition
            img_GT = util.channel_convert(img_GT.shape[2], self.color, [img_GT])[
                0
            ]
            if self.dataroot_mask != None:
                img_mask = util.channel_convert(C, self.color, [img_mask])[
                    0
                ]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
        ).float()
        img_LR = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))
        ).float()
        if self.dataroot_mask != None:
            img_mask = torch.from_numpy(
                np.ascontiguousarray(np.transpose(img_mask, (2, 0, 1)))
            ).float()

        if LR_path is None:
            LR_path = GT_path

        img_LR = img_LR*2-1
        img_GT = img_GT*2-1

        if self.dataroot_mask != None:
            img_LR = torch.cat([img_LR, img_mask], dim=0)
        return {"LQ": img_LR, "GT": img_GT, "LQ_path": LR_path, "GT_path": GT_path}

    def __len__(self):
        return len(self.GT_paths)



class LQDataset(torch.utils.data.Dataset):
    """
    Read LR (Low Quality, here is LR) and LR image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(
        self, 
        dataroot_LQ,
        phase,
        dataroot_mask=None,
        LR_size=None,
        use_flip=False,
        use_rot=False,
        use_swap=False,
        color=None,
    ):
        super().__init__()
        self.phase = phase
        self.dataroot_mask = dataroot_mask
        self.use_flip = use_flip
        self.use_rot = use_rot
        self.use_swap = use_swap
        self.color = color,
        self.LR_size = LR_size
        self.LR_paths = None
        self.LR_env, self.mask_env = None, None  # environment for lmdb

        # read image list from image files
        self.LR_paths = util.get_image_paths(
            "img", dataroot_LQ
        )  # LR list
        if dataroot_mask != None:
            self.mask_paths = util.get_image_paths(
                "img", self.dataroot_mask
            )  # mask list   
        self.random_scale_list = [1]

    def __getitem__(self, index):

        LR_path = None
        scale = 1

        # get LR image
        LR_path = self.LR_paths[index]
        resolution = None
        img_LR = util.read_img(
            self.LR_env, LR_path, resolution
        )  # return: Numpy float32, HWC, BGR, [0,1]

        if self.dataroot_mask != None:
            #get mask image
            mask_path = self.mask_paths[index]
            img_mask = util.read_img(self.mask_env, mask_path, resolution)

        # modcrop in the validation / test phase
        if self.phase != "train":
            img_LR = util.modcrop(img_LR, scale)

        if self.phase == "train":
            H, W, C = img_LR.shape

            rnd_h = random.randint(0, max(0, H - self.LR_size))
            rnd_w = random.randint(0, max(0, W - self.LR_size))
            img_LR = img_LR[rnd_h : rnd_h + self.LR_size, rnd_w : rnd_w + self.LR_size, :]
            if self.dataroot_mask != None:
                img_mask = img_mask[rnd_h : rnd_h + self.LR_size, rnd_w : rnd_w + self.LR_size, :]

            # augmentation - flip, rotate
            if self.dataroot_mask != None:
                img_LR, img_mask = util.augment(
                    [img_LR, img_mask],
                    self.use_flip,
                    self.use_rot,
                    "LQGT",
                    self.use_swap,
                )
            else :
                img_LR = util.augment(
                    img_LR,
                    self.use_flip,
                    self.use_rot,
                    "LQGT",
                    self.use_swap,
                )

        # change color space if necessary
        if self.color:
            H, W, C = img_LR.shape
            img_LR = util.channel_convert(C, self.color, [img_LR])[
                0
            ]  # TODO during val no definition
            if self.dataroot_mask != None:
                img_mask = util.channel_convert(C, self.color, [img_mask])[
                    0
                ]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_LR.shape[2] == 3:
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_LR = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))
        ).float()
        if self.dataroot_mask != None:
            img_mask = torch.from_numpy(
                np.ascontiguousarray(np.transpose(img_mask, (2, 0, 1)))
            ).float()

        img_LR = img_LR*2-1

        if self.dataroot_mask != None:
            img_LR = torch.cat([img_LR, img_mask], dim=0)

        return {"LQ": img_LR, "LQ_path": LR_path}

    def __len__(self):
        return len(self.LR_paths)

class LQGTDataset_(torch.utils.data.Dataset):

    def __init__(
        self, 
        dataroot_LQ,
        dataroot_GT,
        phase,
        dataroot_mask=None,
        LR_size=None,
        GT_size=None,
        use_flip=False,
        use_rot=False,
        use_swap=False,
        color=None,
    ):
        super().__init__()
        self.phase = phase,
        self.dataroot_mask = dataroot_mask
        self.use_flip = use_flip,
        self.use_rot = use_rot,
        self.use_swap = use_swap,
        self.color = color,
        self.LR_size, self.GT_size = LR_size, GT_size
        self.LR_env, self.GT_env, self.mask_env = None, None, None  # environment for lmdb

        # read image list from image files
        self.LR_paths = util.get_image_paths(
            "img", dataroot_LQ
        )[:100]  # LR list
        self.GT_paths = util.get_image_paths(
            "img", dataroot_GT
        )[:100]  # GT list
        if dataroot_mask != None:
            self.mask_paths = util.get_image_paths(
                "img", self.dataroot_mask
            )[:100]  # GT list   
        if self.LR_paths and self.GT_paths:
            assert len(self.LR_paths) == len(
                self.GT_paths
            ), "GT and LR datasets have different number of images - {}, {}.".format(
                len(self.LR_paths), len(self.GT_paths)
            )
        self.random_scale_list = [1]


    def __getitem__(self, index):

        GT_path, LR_path = None, None
        scale = 1

        # get GT image
        GT_path = self.GT_paths[index]
        resolution = None
        img_GT = util.read_img(
            self.GT_env, GT_path, resolution
        )  # return: Numpy float32, HWC, BGR, [0,1]

        # modcrop in the validation / test phase
        if self.phase != 'train':
            img_GT = util.modcrop(img_GT, scale)

        # get LR image
        LR_path = self.LR_paths[index]
        img_LR = util.read_img(self.LR_env, LR_path, resolution)

        if self.dataroot_mask != None:
            #get mask image
            mask_path = self.mask_paths[index]
            img_mask = util.read_img(self.mask_env, mask_path, resolution)

        if self.phase == 'train':
            H, W, C = img_LR.shape
            assert self.LR_size == self.GT_size // scale, "GT size does not match LR size"

            # randomly crop
            rnd_h = random.randint(0, max(0, H - self.LR_size))
            rnd_w = random.randint(0, max(0, W - self.LR_size))
            img_LR = img_LR[rnd_h : rnd_h + self.LR_size, rnd_w : rnd_w + self.LR_size, :]
            if self.dataroot_mask != None:
                img_mask = img_mask[rnd_h : rnd_h + self.LR_size, rnd_w : rnd_w + self.LR_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[
                rnd_h_GT : rnd_h_GT + self.GT_size, rnd_w_GT : rnd_w_GT + self.GT_size, :
            ]

            # augmentation - flip, rotate
            if self.dataroot_mask != None:
                img_LR, img_GT, img_mask = util.augment(
                    [img_LR, img_GT, img_mask],
                    self.use_flip,
                    self.use_rot,
                    "LQGT",
                    self.use_swap,
                )
            else :
                img_LR, img_GT = util.augment(
                    [img_LR, img_GT],
                    self.use_flip,
                    self.use_rot,
                    "LQGT",
                    self.use_swap,
                )

        elif self.LR_size is not None:
            H, W, C = img_LR.shape
            assert self.LR_size == self.GT_size // scale, "GT size does not match LR size"

            if self.LR_size < H and self.LR_size < W:
                # center crop
                rnd_h = H // 2 - self.LR_size//2
                rnd_w = W // 2 - self.LR_size//2
                img_LR = img_LR[rnd_h : rnd_h + self.LR_size, rnd_w : rnd_w + self.LR_size, :]
                if self.dataroot_mask != None:
                    img_mask = img_mask[rnd_h : rnd_h + self.LR_size, rnd_w : rnd_w + self.LR_size, :]
                rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[
                    rnd_h_GT : rnd_h_GT + self.GT_size, rnd_w_GT : rnd_w_GT + self.GT_size, :
                ]

        # change color space if necessary
        if self.color:
            H, W, C = img_LR.shape
            img_LR = util.channel_convert(C, self.color, [img_LR])[
                0
            ]  # TODO during val no definition
            img_GT = util.channel_convert(img_GT.shape[2], self.color, [img_GT])[
                0
            ]
            if self.dataroot_mask != None:
                img_mask = util.channel_convert(C, self.color, [img_mask])[
                    0
                ]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
        ).float()
        img_LR = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))
        ).float()
        if self.dataroot_mask != None:
            img_mask = torch.from_numpy(
                np.ascontiguousarray(np.transpose(img_mask, (2, 0, 1)))
            ).float()

        if LR_path is None:
            LR_path = GT_path

        img_LR = img_LR*2-1
        img_GT = img_GT*2-1

        if self.dataroot_mask != None:
            img_LR = torch.cat([img_LR, img_mask], dim=0)
        return {"LQ": img_LR, "GT": img_GT, "LQ_path": LR_path, "GT_path": GT_path}

    def __len__(self):
        return len(self.GT_paths)