import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist
import cv2
import csv

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults_deshadow_mask,
    create_model_and_diffusion_nafnet_prepred,
    add_dict_to_argparser,
    args_to_dict,
)

from pytorch_lightning import seed_everything
seed_everything(13)

######################################## Model and Dataset ########################################################
from datasets.deshadow_dataset import create_dataset
from torch.utils.data.dataloader import DataLoader

import datasets.deshadow_utils as util

####################################### calculate score ###########################################################
from utils.metrics import PSNR, SSIM, LPIPS

def save_npz(all_images, name, pre_model):
    arr = np.array(all_images)
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_{name}_{pre_model}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)


def write_scores_to_csv(out_path, start_step, pre_model, scores):
    header = ['pre-model', 'start step'] + list(scores.keys())
    mode = 'w' if not os.path.isfile(out_path) else 'a'
    with open(out_path, mode, newline='') as f:
        writer = csv.writer(f)
        if mode == 'w':
            writer.writerow(header)
        writer.writerow([pre_model] + [start_step] + [i[0] for i in scores.values()])


def compute_score(all_images_gt_raw, all_images_pred_raw, start_step, pre_model):
    metrics = ('psnr', 'ssim', 'lpips')
    device = 'cpu'
    boundary_ignore = 40
    metrics_all = {}
    scores = {}
    for m in metrics:
        if m == 'psnr':
            loss_fn = PSNR(boundary_ignore=boundary_ignore)
        elif m == 'ssim':
            loss_fn = SSIM(boundary_ignore=boundary_ignore, use_for_loss=False)
        elif m == 'lpips':
            loss_fn = LPIPS(boundary_ignore=boundary_ignore)
            loss_fn.to(device)
        else:
            raise ValueError(f"Unknown metric: {m}")
        metrics_all[m] = loss_fn
        scores[m] = []

    scores = {k: [] for k, v in scores.items()}
    all_images_gt_raw = th.cat(all_images_gt_raw)
    all_images_pred_raw = th.cat(all_images_pred_raw)

    for m, m_fn in metrics_all.items():
        metric_value = m_fn(all_images_pred_raw, all_images_gt_raw).cpu().item()
        scores[m].append(metric_value)
        logger.log(f"{m} is {metric_value}")

    out_path = os.path.join(logger.get_dir(), f"score.csv")
    write_scores_to_csv(out_path, start_step, pre_model, scores)

##################################################################################################################
        

def main():
    args = create_argparser().parse_args()
    start_step_list = [int(i) for i in args.start_step_list.split(",")]
    dist_util.setup_dist(int(args.gpu_id))
    logger.configure(args.output_path)

    seed = 42
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.use_deterministic_algorithms = True

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_nafnet_prepred(
        **args_to_dict(args, model_and_diffusion_defaults_deshadow_mask().keys())
    )
    logger.log("load model:"+args.model_path)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    dataset = create_dataset(
        dataroot_LQ=args.dataroot_LQ,
        dataroot_GT=args.dataroot_GT,
        phase="valid",
        name="Test_Dataset",
        dataroot_mask=args.dataroot_mask,
        use_flip=False,
        use_rot=False,
        use_swap=False,
        color=None,)
    data_loader = DataLoader(dataset, batch_size=args.batch_size)

    pre_pred_paths = util.get_image_paths("img", args.dataroot_pre_pred)


    logger.log("sampling...")
    for start_step in start_step_list:
        logger.log("#"*10,"start_step=",start_step,"#"*10)
        all_images_gt = []
        all_images_pred = []
        all_images_gt_raw = []
        all_images_pred_raw = []
        for index, d in enumerate(data_loader):
            print(index)
            lq = d["LQ"]

            lq = lq.to("cuda")

            ############### pre pred ##################
            pre_pred_path = pre_pred_paths[index]
            pre_pred_ = util.read_img(None, pre_pred_path)
            pre_pred_ = th.unsqueeze(th.from_numpy(np.ascontiguousarray(np.transpose(pre_pred_, (2, 0, 1)))).float(), dim=0)
            pre_pred_ = pre_pred_[:, [2, 1, 0]]

            ############ convert to sample for diffusion ###################
            pre_pred_ = pre_pred_*2-1
            pre_pred = (pre_pred_).clone().detach()

            ############## Diffusion ################
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            if start_step != 0:
                sample = sample_fn(
                    model,
                    (args.batch_size, 3, pre_pred.shape[2], pre_pred.shape[3]),
                    pre_pred,
                    start_step,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=lq,
                )
            else :
                sample = pre_pred

            sample = ((sample + 1) / 2).clamp(0.0, 1.0)

            all_images_pred_raw.append(sample.to("cpu"))

            # Save predictions as png
            # pred
            sample = sample[:, [2, 1, 0]]
            sample = sample[0].to("cpu").permute(1, 2, 0).numpy()* 255
            all_images_pred.append(sample)

            if args.save_png:
                cv2.imwrite('{}/{}.png'.format(logger.get_dir(), "0"*(4-len(str(index)))+str(index)), sample)

        dist.barrier()


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=1,
        use_ddim=False,
        model_path="",
        dataroot_LQ="",
        dataroot_GT="",
        dataroot_mask="",
        dataroot_pre_pred="",
        output_path="",
        gpu_id=0,
        crop_size=None,
        save_png=True,
        start_step_list="1,2,3,4,5,6,8,10,15,20,30,40,50,60,70,80",
    )
    defaults.update(model_and_diffusion_defaults_deshadow_mask())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()

