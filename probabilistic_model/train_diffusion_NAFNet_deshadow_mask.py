import sys
sys.path.append(".")
import argparse
import wandb
import torch as th

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data_deshadow
from guided_diffusion.resample import (
    create_named_schedule_sampler,
    create_named_schedule_sampler_startstep
)
from guided_diffusion.script_util import (
    model_and_diffusion_defaults_deshadow_mask,
    create_model_and_diffusion_nafnet,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoopDeshadow


def main():
    args = create_argparser().parse_args()

    ## wandb
    if args.debug:
        mode = "online"
    else:
        mode = "disabled"

    dist_util.setup_dist_wandb(args,"learning_shadowremoval",mode)
    logger.configure(dir=args.log_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_nafnet(
        **args_to_dict(args, model_and_diffusion_defaults_deshadow_mask().keys())
    )

    model.to(dist_util.dev())
    if args.start_step == -1:
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    else:
        schedule_sampler = create_named_schedule_sampler_startstep(args.schedule_sampler, diffusion, args.start_step)

    logger.log("creating data loader...")
    data = load_data_deshadow(
        dataroot_LQ=args.dataroot_LQ,
        dataroot_GT=args.dataroot_GT,
        phase="train",
        name="Train_Dataset",
        batch_size=args.batch_size,
        dataroot_mask=args.dataroot_mask,
        crop_size=int(args.crop_size),
        use_flip=True,
        use_rot=True,
        use_swap=False,
        color="RGB",
        deterministic=False,
    )

    logger.log("training...")
    TrainLoopDeshadow(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        wandb=wandb,
    ).run_loop()



def create_argparser():
    defaults = dict(
        dataroot_LQ="/home-local/tokoro/shadow_removal/ntire24_shrem_train_inp",
        dataroot_mask="/home-local/kawai/NTIRE2024/shadow_removal/official/train/mask_SASMA",
        dataroot_GT="/home-local/tokoro/shadow_removal/ntire24_shrem_train_gt",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        log_dir="./logs",
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_id=0,
        start_step=-1,
        crop_size=None,
        debug=False,
    )
    defaults.update(model_and_diffusion_defaults_deshadow_mask())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
