from copy import deepcopy
from datetime import timedelta
from pprint import pprint
import numpy as np
import os

import torch
import torch.distributed as dist
import wandb
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed
from tqdm import tqdm

from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import (
    get_data_parallel_group,
    set_data_parallel_group,
    set_sequence_parallel_group,
)
from opensora.acceleration.plugin import ZeroSeqParallelPlugin
from opensora.datasets import prepare_dataloader, prepare_variable_dataloader, save_sample
from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from opensora.utils.ckpt_utils import create_logger, load, model_sharding, record_model_param_shape, save
from opensora.utils.config_utils import (
    create_experiment_workspace,
    create_tensorboard_writer,
    parse_configs,
    save_training_config,
)
from opensora.utils.misc import all_reduce_mean, format_numel_str, get_model_numel, requires_grad, to_torch_dtype
from opensora.utils.train_utils import MaskGenerator, update_ema


def calculate_weight_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        param_norm = param.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def calculate_gradient_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def ensure_parent_directory_exists(file_path):
    directory_path = os.path.dirname(file_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")


def log_sample(model, text_encoder, vae, scheduler, coordinator, cfg, epoch, exp_dir, global_step, dtype, device):
    model = model.eval()
    text_encoder.y_embedder = (
        model.module.y_embedder
    )  # hack for classifier-free guidance
    save_dir = os.path.join(
        exp_dir, f"epoch{epoch}-global_step{global_step + 1}"
    )

    prompts = [
        "Two pirate ships battling each other with canons as they sail inside a cup filled with coffee. The two pirate ships are fully in view, and the camera view is an aeriel view looking down at a 45 degree angle at the ships.",
        "Stunning aerial view of a road winding through a dark green forest. Mountains appear in the distance.",
        "Flying through a modern city.",
        "A savanna full of animals.",
        "An archeological site in the mayan djungle. A mayan pyramid rises over the green landscape.",
        "a close-up view of a vibrant scene from nature. The foreground is dominated by a cluster of small, white flowers with green leaves, their delicate petals and stems creating a sense of softness and freshness. These flowers are the main focus of the scene, their bright white color contrasting with the surrounding greenery.  In the background, there's a blurred landscape that appears to be a field or meadow, filled with tall, dry grasses that have turned a warm, golden color, suggesting either the onset of autumn or the end of a dry season. The grasses are scattered with small white flowers that echo the ones in the foreground, creating a sense of continuity and harmony in the scene.  The style of the scene is realistic, with a shallow depth of field that blurs the background, drawing the viewer's attention to the flowers in the foreground. The lighting in the scene is soft and diffused, with no harsh shadows, which enhances the natural and serene atmosphere of the scene. The colors are rich and saturated, particularly the whites of the flowers and the golden hues of the grasses, which contribute to the overall vibrancy of the scene",
        "a serene scene of a small town nestled at the foot of a mountain range. The town, with its cluster of houses and buildings, is enveloped by a lush expanse of trees and shrubbery. The sky overhead is clear, casting a soft, diffused morning light over the landscape. In the distance, the mountains rise majestically, their peaks obscured by thin layers of snow. The colors in the scene are predominantly green and gray, reflecting the natural hues of the landscape. The green of the vegetation contrasts with the gray of the mountains and the overcast sky, creating a harmonious balance of colors.  Despite the absence of any discernible action or movement, the scene conveys a sense of tranquility and peace, as if time has slowed down in this secluded corner of the world. There are no texts or countable objects in the scene, and the relative positions of the objects suggest a typical layout for a small town, with houses and buildings scattered around a central area. Overall, the scene presents a picturesque snapshot of a quiet moment in a peaceful mountain town."
    ]

    with torch.no_grad():
        rng = np.random.default_rng(seed=0)
        image_size = (256, 256)
        num_frames = 16
        fps = 8
        input_size = (num_frames, *image_size)
        latent_size = vae.get_latent_size(input_size)
        z = rng.normal(size=(len(prompts), vae.out_channels, *latent_size))
        z = torch.tensor(z, device=device, dtype=float).to(dtype=dtype)
        samples = scheduler.sample(
            model,
            text_encoder,
            z=z,
            prompts=prompts,
            device=device,
            additional_args=dict(
                height=torch.tensor(
                    [image_size[0]], device=device, dtype=dtype
                ).repeat(len(prompts)),
                width=torch.tensor(
                    [image_size[1]], device=device, dtype=dtype
                ).repeat(len(prompts)),
                num_frames=torch.tensor(
                    [num_frames], device=device, dtype=dtype
                ).repeat(len(prompts)),
                ar=torch.tensor(
                    [image_size[0] / image_size[1]],
                    device=device,
                    dtype=dtype,
                ).repeat(len(prompts)),
                fps=torch.tensor(
                    [fps], device=device, dtype=dtype
                ).repeat(len(prompts)),
            ),
        )
        samples = vae.decode(samples.to(dtype))

        # 4.4. save samples
        if coordinator.is_master():
            for sample_idx, sample in enumerate(samples):
                save_path = os.path.join(
                    save_dir, f"sample_{sample_idx}"
                )
                ensure_parent_directory_exists(save_dir)
                save_sample(
                    sample,
                    fps=fps,
                    save_path=save_path,
                )
                if cfg.wandb:
                    wandb.log(
                        {
                            f"eval/prompt_{sample_idx}": wandb.Video(
                                os.path.abspath(save_path + ".mp4"),
                                caption=prompts[sample_idx],
                                format="mp4",
                                fps=fps,
                            )
                        },
                        step=global_step,
                    )

    model = model.train()
    text_encoder.y_embedder = None




def main():
    # ======================================================
    # 1. args & cfg
    # ======================================================
    cfg = parse_configs(training=True)
    exp_name, exp_dir = create_experiment_workspace(cfg)
    save_training_config(cfg._cfg_dict, exp_dir)

    # ======================================================
    # 2. runtime variables & colossalai launch
    # ======================================================
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert cfg.dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg.dtype}"

    # 2.1. colossalai init distributed training
    # we set a very large timeout to avoid some processes exit early
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(1024)
    coordinator = DistCoordinator()
    device = get_current_device()
    dtype = to_torch_dtype(cfg.dtype)

    # 2.2. init logger, tensorboard & wandb
    if not coordinator.is_master():
        logger = create_logger(None)
    else:
        print("Training configuration:")
        pprint(cfg._cfg_dict)
        logger = create_logger(exp_dir)
        logger.info(f"Experiment directory created at {exp_dir}")

        writer = create_tensorboard_writer(exp_dir)
        if cfg.wandb:
            PROJECT="que_sora_sora"
            wandb.init(project=PROJECT, name=exp_name, config=cfg._cfg_dict)

    # 2.3. initialize ColossalAI booster
    if cfg.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=cfg.dtype,
            initial_scale=2**16,
            max_norm=cfg.grad_clip,
        )
        set_data_parallel_group(dist.group.WORLD)
    elif cfg.plugin == "zero2-seq":
        plugin = ZeroSeqParallelPlugin(
            sp_size=cfg.sp_size,
            stage=2,
            precision=cfg.dtype,
            initial_scale=2**16,
            max_norm=cfg.grad_clip,
        )
        set_sequence_parallel_group(plugin.sp_group)
        set_data_parallel_group(plugin.dp_group)
    else:
        raise ValueError(f"Unknown plugin {cfg.plugin}")
    booster = Booster(plugin=plugin)

    # ======================================================
    # 3. build dataset and dataloader
    # ======================================================
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info(f"Dataset contains {len(dataset)} samples.")
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
    )
    # TODO: use plugin's prepare dataloader
    if cfg.bucket_config is None:
        dataloader = prepare_dataloader(**dataloader_args)
    else:
        dataloader = prepare_variable_dataloader(
            bucket_config=cfg.bucket_config,
            num_bucket_build_workers=cfg.num_bucket_build_workers,
            **dataloader_args,
        )
    if cfg.dataset.type == "VideoTextDataset":
        total_batch_size = cfg.batch_size * dist.get_world_size() // cfg.sp_size
        logger.info(f"Total batch size: {total_batch_size}")

    # ======================================================
    # 4. build model
    # ======================================================
    # 4.1. build model
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS)
    input_size = (dataset.num_frames, *dataset.image_size)
    latent_size = vae.get_latent_size(input_size)
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length
    )
    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        f"Trainable model params: {format_numel_str(model_numel_trainable)}, Total model params: {format_numel_str(model_numel)}"
    )

    # 4.2. create ema
    ema = deepcopy(model).to(torch.float32).to(device)
    requires_grad(ema, False)
    ema_shape_dict = record_model_param_shape(ema)

    # 4.3. move to device
    vae = vae.to(device, dtype)
    model = model.to(device, dtype)

    # 4.4. build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # 4.5. setup optimizer
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=0,
        adamw_mode=True,
    )
    lr_scheduler = None

    # 4.6. prepare for training
    if cfg.grad_checkpoint:
        set_grad_checkpoint(model)
    model.train()
    update_ema(ema, model, decay=0, sharded=False)
    ema.eval()
    if cfg.mask_ratios is not None:
        mask_generator = MaskGenerator(cfg.mask_ratios)

    # =======================================================
    # 5. boost model for distributed training with colossalai
    # =======================================================
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )
    torch.set_default_dtype(torch.float)
    logger.info("Boost model for distributed training")
    if cfg.dataset.type == "VariableVideoTextDataset":
        num_steps_per_epoch = dataloader.batch_sampler.get_num_batch() // dist.get_world_size()
    else:
        num_steps_per_epoch = len(dataloader)

    # =======================================================
    # 6. training loop
    # =======================================================
    start_epoch = start_step = log_step = sampler_start_idx = acc_step = 0
    running_loss = 0.0
    sampler_to_io = dataloader.batch_sampler if cfg.dataset.type == "VariableVideoTextDataset" else None
    # 6.1. resume training
    if cfg.load is not None:
        logger.info("Loading checkpoint")
        ret = load(
            booster,
            model,
            ema,
            optimizer,
            lr_scheduler,
            cfg.load,
            sampler=sampler_to_io if not cfg.start_from_scratch else None,
        )
        if not cfg.start_from_scratch:
            start_epoch, start_step, sampler_start_idx = ret
        logger.info(f"Loaded checkpoint {cfg.load} at epoch {start_epoch} step {start_step}")

        
        optim_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Overwriting loaded learning rate from {optim_lr} to config lr={cfg.lr}")
        for g in optimizer.param_groups:
            g["lr"] = cfg.lr
    logger.info(f"Training for {cfg.epochs} epochs with {num_steps_per_epoch} steps per epoch")

    if cfg.dataset.type == "VideoTextDataset":
        dataloader.sampler.set_start_index(sampler_start_idx)
    model_sharding(ema)

    # log prompts for pre-training ckpt
    log_sample(model, text_encoder, vae, scheduler, coordinator, cfg, start_epoch, exp_dir, start_step, dtype, device)

    # 6.2. training loop
    for epoch in range(start_epoch, cfg.epochs):
        if cfg.dataset.type == "VideoTextDataset":
            dataloader.sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info(f"Beginning epoch {epoch}...")

        with tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:
            for step, batch in pbar:
                x = batch.pop("video").to(device, dtype)  # [B, C, T, H, W]
                y = batch.pop("text")
                # Visual and text encoding
                with torch.no_grad():
                    # Prepare visual inputs
                    x = vae.encode(x)  # [B, C, T, H/P, W/P]
                    # Prepare text inputs
                    model_args = text_encoder.encode(y)

                # Mask
                if cfg.mask_ratios is not None:
                    mask = mask_generator.get_masks(x)
                    model_args["x_mask"] = mask
                else:
                    mask = None

                # Video info
                for k, v in batch.items():
                    model_args[k] = v.to(device, dtype)

                # Diffusion
                t = torch.randint(0, scheduler.num_timesteps, (x.shape[0],), device=device)
                loss_dict = scheduler.training_losses(model, x, t, model_args, mask=mask)

                # Backward & update
                loss = loss_dict["loss"].mean()
                booster.backward(loss=loss, optimizer=optimizer)
                optimizer.step()
                optimizer.zero_grad()

                # Update EMA
                update_ema(ema, model.module, optimizer=optimizer)

                # Log loss values:
                all_reduce_mean(loss)
                running_loss += loss.item()
                global_step = epoch * num_steps_per_epoch + step
                log_step += 1
                acc_step += 1

                # Log to tensorboard
                if coordinator.is_master() and (global_step + 1) % cfg.log_every == 0:
                    avg_loss = running_loss / log_step
                    pbar.set_postfix({"loss": avg_loss, "step": step, "global_step": global_step})
                    running_loss = 0
                    log_step = 0
                    writer.add_scalar("loss", loss.item(), global_step)

                    weight_norm = calculate_weight_norm(model)
                    gradient_norm = calculate_gradient_norm(model)

                    if cfg.wandb:
                        wandb.log(
                            {
                                "iter": global_step,
                                "epoch": epoch,
                                "loss": loss.item(),
                                "avg_loss": avg_loss,
                                "acc_step": acc_step,
                                "lr": optimizer.param_groups[0]["lr"],
                                "weight_norm": weight_norm,
                                "gradient_norm": gradient_norm,
                            },
                            step=global_step,
                        )

                # Save checkpoint
                if cfg.ckpt_every > 0 and (global_step + 1) % cfg.ckpt_every == 0:
                    save(
                        booster,
                        model,
                        ema,
                        optimizer,
                        lr_scheduler,
                        epoch,
                        step + 1,
                        global_step + 1,
                        cfg.batch_size,
                        coordinator,
                        exp_dir,
                        ema_shape_dict,
                        sampler=sampler_to_io,
                    )
                    logger.info(
                        f"Saved checkpoint at epoch {epoch} step {step + 1} global_step {global_step + 1} to {exp_dir}"
                    )

                    # log prompts for each checkpoints
                    log_sample(model, text_encoder, vae, scheduler, coordinator, cfg, epoch, exp_dir, global_step, dtype, device)


        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        if cfg.dataset.type == "VideoTextDataset":
            dataloader.sampler.set_start_index(0)
        if cfg.dataset.type == "VariableVideoTextDataset":
            dataloader.batch_sampler.set_epoch(epoch + 1)
            print("Epoch done, recomputing batch sampler")
        start_step = 0


if __name__ == "__main__":
    main()
