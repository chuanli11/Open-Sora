from copy import deepcopy
from datetime import timedelta
from pprint import pprint
import numpy as np
import os
import random
import time

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
from opensora.registry import DATASETS, MODELS, SCHEDULERS, SCHEDULERS_INFERENCE, build_module
from opensora.utils.ckpt_utils import create_logger, load, model_sharding, record_model_param_shape, save
from opensora.utils.config_utils import (
    create_experiment_workspace,
    create_tensorboard_writer,
    parse_configs,
    save_training_config,
)
from opensora.utils.misc import all_reduce_mean, format_numel_str, get_model_numel, requires_grad, to_torch_dtype
from opensora.utils.train_utils import MaskGenerator, update_ema


from torch.optim.lr_scheduler import LRScheduler as _LRScheduler
from typing import List
class WarmupScheduler(_LRScheduler):
    """Starts with a log space warmup lr schedule until it reaches N epochs then applies
    the specific scheduler (For example: ReduceLROnPlateau).

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        warmup_epochs (int): Number of epochs to warmup lr in log space until starting applying the scheduler.
        after_scheduler (:class:`torch.optim.lr_scheduler`): After warmup_epochs, use this scheduler.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, warmup_epochs: int, after_scheduler: _LRScheduler, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        self.min_lr  = 1e-7
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
                self.finished = True
            return self.after_scheduler.get_lr()

        return [self.min_lr * ((lr / self.min_lr) ** ((self.last_epoch + 1) / self.warmup_epochs)) for lr in self.base_lrs]

    def step(self, epoch: int = None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
        else:
            return super().step(epoch)

class ConstantWarmupLR(WarmupScheduler):
    """Multistep learning rate scheduler with warmup.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        warmup_steps (int, optional): Number of warmup steps, defaults to 0.
        gamma (float, optional): Multiplicative factor of learning rate decay, defaults to 0.1.
        num_steps_per_epoch (int, optional): Number of steps per epoch, defaults to -1.
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(
        self,
        optimizer,
        factor: float,
        warmup_steps: int = 0,
        last_epoch: int = -1,
        **kwargs,
    ):
        base_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=-1)
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)


def save_rng_state():
    rng_state = {
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'numpy': np.random.get_state(),
        'random': random.getstate()
    }
    return rng_state


def load_rng_state(rng_state):
    torch.set_rng_state(rng_state['torch'])
    if rng_state['torch_cuda'] is not None:
        torch.cuda.set_rng_state_all(rng_state['torch_cuda'])
    np.random.set_state(rng_state['numpy'])
    random.setstate(rng_state['random'])


#from mmengine.runner import set_random_seed
def set_seed_custom(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #set_random_seed(seed=seed)


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


z_log = None
def log_sample(model, text_encoder, vae, scheduler, coordinator, cfg, epoch, exp_dir, global_step, dtype, device):
    global z_log
    rng_state = save_rng_state()
    back_to_train_model = model.training
    back_to_train_vae = vae.training
    vae = vae.eval()
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
        image_size = (256, 256)
        num_frames = 16
        fps = 8
        input_size = (num_frames, *image_size)
        latent_size = vae.get_latent_size(input_size)
        if z_log is None:
            rng = np.random.default_rng(seed=42)
            z_log = rng.normal(size=(len(prompts), vae.out_channels, *latent_size))
        z = torch.tensor(z_log, device=device, dtype=float).to(dtype=dtype)
        set_seed_custom(42)
        samples = scheduler.sample(
            model,
            text_encoder,
            z=z,
            prompts=prompts,
            device=device,
            additional_args=dict(
                height=torch.tensor([image_size[0]], device=device, dtype=dtype).repeat(len(prompts)),
                width=torch.tensor([image_size[1]], device=device, dtype=dtype).repeat(len(prompts)),
                num_frames=torch.tensor([num_frames], device=device, dtype=dtype).repeat(len(prompts)),
                ar=torch.tensor([image_size[0] / image_size[1]], device=device, dtype=dtype).repeat(len(prompts)),
                fps=torch.tensor([fps], device=device, dtype=dtype).repeat(len(prompts)),
            ),
        )
        samples = vae.decode(samples.to(dtype))

        # 4.4. save samples
        if coordinator.is_master():
            for sample_idx, sample in enumerate(samples):
                save_path = os.path.join(
                    save_dir, f"sample_{sample_idx}"
                )
                ensure_parent_directory_exists(save_path)
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

    if back_to_train_model:
        model = model.train()
    if back_to_train_vae:
        vae = vae.train()
    text_encoder.y_embedder = None

    load_rng_state(rng_state)


measure_apt = False
def hook_fn(module, input, output):
    global measure_apt
    with torch.no_grad():
        if measure_apt:
            act_sign = torch.nn.functional.relu(output.detach().sign()).type(torch.bool)
            module.hashes = hash(act_sign)
        else:
            module.hashes = None

# bin -> numbers
def bin2numbers(input):
    numbers = []
    for i in range((input.size(1)-1)//64 + 1):
        block = input[:,i*64:(i+1)*64]
        base = 2**torch.arange(block.size(1), device=input.device)

        # ensure that base is at correct dimension
        base = base.unsqueeze(0)
        for i in range(len(input.shape)-2):
            base = base.unsqueeze(i+2)

        # pytorch does not support uint64_t, but int64_t
        # thus, we interpret last bit as sign
        if block.size(1) == 64:
            base[:,-1] = -1

        num = (block*base).sum(1)
        numbers.append(num)
    return torch.stack(numbers, dim=1)


def hash(act, labels=None):
    FIRSTH = 37 # prime
    A = 54059 # a prime
    B = 76963 # another prime
    # C = 86969 # yet another prime

    device = act.device

    numbers = bin2numbers(act)
    if labels is not None:
        numbers = torch.cat([numbers,labels],1)
    numbers = numbers * B
    result = FIRSTH;
    d = numbers.shape[1];
    for i in range(d):
        number = numbers[:,i]
        result = number ^ (result * A);
    return result;


def compute_hashes(model):
    all_hashes = []
    for module in model.modules():
        if isinstance(module, torch.nn.SiLU):
            all_hashes.append(module.hashes)
    return all_hashes


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
    scheduler_inference = build_module(cfg.scheduler_inference, SCHEDULERS)

    # 4.5. setup optimizer
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=0,
        adamw_mode=True,
    )
    lr_scheduler = ConstantWarmupLR(optimizer, factor=1, warmup_steps=200, last_epoch=-1)

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


    # Register hooks to capture activations
    for module in model.modules():
        if isinstance(module, torch.nn.SiLU):
            module.register_forward_hook(hook_fn)


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
            None,# lr_scheduler,
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
    first_global_step = start_epoch * num_steps_per_epoch + start_step
    if coordinator.is_master():
        log_sample(model, text_encoder, vae, scheduler_inference, coordinator, cfg, start_epoch, exp_dir, first_global_step, dtype, device)

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
            iteration_times = []
            for step, batch in pbar:
                start_time = time.time()
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

                global measure_apt
                next_global_step = epoch * num_steps_per_epoch + step
                apt_every = 25
                if coordinator.is_master() and next_global_step % (cfg.log_every * apt_every) == 0:
                    measure_apt = True
                else:
                    measure_apt = False

                # Diffusion
                t = torch.randint(0, scheduler.num_timesteps, (x.shape[0],), device=device)
                loss_dict = scheduler.training_losses(model, x, t, model_args, mask=mask)

                # Backward & update
                loss = loss_dict["loss"].mean()
                if coordinator.is_master() and measure_apt:
                    metrics_before = compute_hashes(model)
                booster.backward(loss=loss, optimizer=optimizer)
                optimizer.step()
                optimizer.zero_grad()
                if lr_scheduler is not None:
                    lr_scheduler.step()

                loss_dict = scheduler.training_losses(model, x, t, model_args, mask=mask)
                loss = loss_dict["loss"].mean()
                if coordinator.is_master() and measure_apt:
                    metrics_after = compute_hashes(model)
                    APTs = torch.tensor([(b == a).sum()/(a.shape[0] * np.prod(a.shape[2:])) for (b,a) in zip(metrics_before, metrics_after)])
                else:
                    APTs = None

                # Update EMA
                update_ema(ema, model.module, optimizer=optimizer)

                # Log loss values:
                all_reduce_mean(loss)
                running_loss += loss.item()
                global_step = epoch * num_steps_per_epoch + step
                log_step += 1
                acc_step += 1
                iteration_times.append(time.time() - start_time)


                # Log to tensorboard
                if coordinator.is_master() and global_step % cfg.log_every == 0:
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
                                "avg_iteration_time": sum(iteration_times) / len(iteration_times),
                                "iter": global_step,
                                "epoch": epoch,
                                "loss": loss.item(),
                                "avg_loss": avg_loss,
                                "acc_step": acc_step,
                                "lr": optimizer.param_groups[0]["lr"],
                                "weight_norm": weight_norm,
                                "gradient_norm": gradient_norm,
                                **({"APT": APTs.mean().item()} if APTs is not None else {})
                            },
                            step=global_step,
                        )
                        iteration_times = []

                # Save checkpoint
                if cfg.ckpt_every > 0 and global_step % cfg.ckpt_every == 0 and global_step != 0:
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
                if coordinator.is_master() and global_step % 250 == 0:
                    log_sample(model, text_encoder, vae, scheduler_inference, coordinator, cfg, epoch, exp_dir, global_step, dtype, device)


        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        if cfg.dataset.type == "VideoTextDataset":
            dataloader.sampler.set_start_index(0)
        if cfg.dataset.type == "VariableVideoTextDataset":
            dataloader.batch_sampler.set_epoch(epoch + 1)
            print("Epoch done, recomputing batch sampler")
        start_step = 0


if __name__ == "__main__":
    main()
