import os
import gc
import math
import time
import psutil
import warnings

import numpy as np
import torch
import csv
from torch import distributed as dist

from factory.data_loader_factory import DataLoaderFactory
from callbacks.neptune_logger import CustomNeptuneLogger
from dataset.custom_dataset import CustomYOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.checks import check_version
from ultralytics.utils.torch_utils import de_parallel

from ultralytics.utils import (
    LOGGER,
    RANK,
    TQDM,
    __version__,
    colorstr
)

TORCH_1_13 = check_version(torch.__version__, "1.13.0")


class CustomTrainer(DetectionTrainer):
    def __init__(self, config, overrides, config_path):
        super().__init__(overrides=overrides)
        self.config = config
        self.overrides = overrides
        self.config_path = config_path


    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."

        dataset = self.build_dataset(dataset_path, mode, batch_size)

        # Use your custom DataLoader factory
        if mode == 'train':
            train_loader, _, my_sampler = DataLoaderFactory.create(config=self.config, train_dataset=dataset,
                                                                   val_dataset=dataset, args=self.args)
            return train_loader

        elif mode == 'val':
            _, val_loader, _ = DataLoaderFactory.create(config=self.config, train_dataset=dataset,
                                                        val_dataset=dataset,args=self.args)
            return val_loader


    def _do_train(self, world_size=1):
            """Train completed, evaluate and plot if specified by arguments."""
            if world_size > 1:
                self._setup_ddp(world_size)
            self._setup_train(world_size)

            args_path = os.path.join(self.save_dir, "args.yaml")

            # Create CustomNeptune logger
            if self.config.callbacks.neptune_logger_callback:
                neptune_logger = CustomNeptuneLogger(token=self.config.callbacks.neptune_token,
                                                            project=self.config.callbacks.neptune_project,
                                                            config=self.config,
                                                            data_config_path=self.overrides['data'],
                                                            yolo_args=args_path,
                                                            config_path=self.config_path,
                                                            device=self.overrides['device'])
                neptune_logger.start_logging(self.model)
  
            # Init model criterion
            if getattr(self.model, "criterion", None) is None:
                self.model.criterion = self.model.init_criterion()

            nb = len(self.train_loader)  # number of batches
            nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
            last_opt_step = -1
            self.epoch_time = None
            self.epoch_time_start = time.time()
            self.train_time_start = time.time()
            self.run_callbacks("on_train_start")
            LOGGER.info(
                f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                f"Logging results to {colorstr('bold', self.save_dir)}\n"
                f'Starting training for ' + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
            )
            epoch = self.start_epoch
            self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
            while True:
                self.epoch = epoch
                self.run_callbacks("on_train_epoch_start")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                    self.scheduler.step()

                self.model.train()
                if RANK != -1:
                    self.train_loader.sampler.set_epoch(epoch)
                pbar = enumerate(self.train_loader)

                if RANK in {-1, 0}:
                    LOGGER.info(self.progress_string())
                    pbar = TQDM(enumerate(self.train_loader), total=nb)
                self.tloss = None
                for i, batch in pbar:
                    inputs, labels, indices = batch['img'], batch['cls'], batch['indices']
                    self.run_callbacks("on_train_batch_start")
                    # Warmup
                    ni = i + nb * epoch
                    if ni <= nw:
                        xi = [0, nw]  # x interp
                        self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                        for j, x in enumerate(self.optimizer.param_groups):
                            # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                            x["lr"] = np.interp(
                                ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                            )
                            if "momentum" in x:
                                x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                    # Forward
                    with autocast(self.amp):
                        batch = self.preprocess_batch(batch)
                        #self.loss, self.loss_items = self.model(batch)
                        preds = self.model.forward(batch["img"])
                        self.loss, self.loss_items = self.model.criterion(preds, batch)
                        if RANK != -1:
                            self.loss *= world_size
                        self.tloss = (
                            (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                        )

                    # Backward
                    self.scaler.scale(self.loss).backward()

                    # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                    if ni - last_opt_step >= self.accumulate:
                        self.optimizer_step()
                        last_opt_step = ni

                        # Timed stopping
                        if self.args.time:
                            self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                            if RANK != -1:  # if DDP training
                                broadcast_list = [self.stop if RANK == 0 else None]
                                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                                self.stop = broadcast_list[0]
                            if self.stop:  # training time exceeded
                                break

                    # Update CustomSampler weights
                    if self.config.data_loader.prioritized_sampler:
                        self.model.eval()
                        with torch.no_grad():
                            eval_preds = self.model(batch["img"])
                            
                        self.train_loader.sampler.update_weights(eval_preds, batch)
                        del eval_preds
                        gc.collect()
                        torch.cuda.empty_cache()
                        self.model.train()
                    
                    # Log memory after each batch
                    if self.config.callbacks.neptune_logger_callback:
                        memory = psutil.virtual_memory()
                        neptune_logger.log_metric(f"memory/total_memory", memory.total)
                        neptune_logger.log_metric(f"memory/available_memory", memory.available)
                        neptune_logger.log_metric(f"memory/used_memory", memory.used)
                        neptune_logger.log_metric(f"memory/free_memory", memory.free)
                        neptune_logger.log_metric(f"memory/percent_memory", memory.percent)
                    
                    # Log
                    mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                    loss_len = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                    if RANK in {-1, 0}:
                        pbar.set_description(
                            ("%11s" * 2 + "%11.4g" * (2 + loss_len))
                            % (f"{epoch + 1}/{self.epochs}", mem, *losses, batch["cls"].shape[0], batch["img"].shape[-1])
                        )
                        self.run_callbacks("on_batch_end")
                        if self.args.plots and ni in self.plot_idx:
                            self.plot_training_samples(batch, ni)

                    self.run_callbacks("on_train_batch_end")

                self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
                self.run_callbacks("on_train_epoch_end")
                if RANK in {-1, 0}:
                    final_epoch = epoch + 1 >= self.epochs
                    self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                    # Validation
                    if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                        self.metrics, self.fitness = self.validate()
                    self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})

                    # Update metrics to Neptune
                    if self.config.callbacks.neptune_logger_callback:
                        neptune_logger.log_metric(f"loss/train/box_loss", self.tloss[0])
                        neptune_logger.log_metric(f"loss/train/cls_loss", self.tloss[1])
                        neptune_logger.log_metric(f"loss/train/dfl_loss", self.tloss[2])

                        neptune_logger.log_metric(f"loss/val/box_loss", self.metrics["val/box_loss"])
                        neptune_logger.log_metric(f"loss/val/cls_loss", self.metrics["val/cls_loss"])
                        neptune_logger.log_metric(f"loss/val/dfl_loss", self.metrics["val/dfl_loss"])

                        neptune_logger.log_metric(f"metrics/precision", self.metrics["metrics/precision(B)"])
                        neptune_logger.log_metric(f"metrics/recall", self.metrics["metrics/recall(B)"])   
                        if self.metrics["metrics/precision(B)"] + self.metrics["metrics/recall(B)"] != 0:
                            f1 = (2 * (self.metrics["metrics/precision(B)"] * self.metrics["metrics/recall(B)"])
                                / (self.metrics["metrics/precision(B)"] + self.metrics["metrics/recall(B)"])) 
                        else:
                            f1 = 0    
                        neptune_logger.log_metric(f"metrics/F1", f1)     
                        neptune_logger.log_metric(f"metrics/mAP50", self.metrics["metrics/mAP50(B)"])
                        neptune_logger.log_metric(f"metrics/mAP50-95", self.metrics["metrics/mAP50-95(B)"])

                        neptune_logger.log_metric(f"learning_rate/pg0", self.lr["lr/pg0"])
                        neptune_logger.log_metric(f"learning_rate/pg1", self.lr["lr/pg1"])
                        neptune_logger.log_metric(f"learning_rate/pg2", self.lr["lr/pg2"])

                        neptune_logger.log_metric(f"epoch", self.epoch + 1)

                    self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                    if self.args.time:
                        self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                    # Save model
                    if self.args.save or final_epoch:
                        self.save_model()
                        self.run_callbacks("on_model_save")

                # Scheduler
                t = time.time()
                self.epoch_time = t - self.epoch_time_start
                self.epoch_time_start = t
                if self.args.time:
                    mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                    self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                    self._setup_scheduler()
                    self.scheduler.last_epoch = self.epoch  # do not move
                    self.stop |= epoch >= self.epochs  # stop if exceeded epochs
                self.run_callbacks("on_fit_epoch_end")
                gc.collect()
                torch.cuda.empty_cache()  # clear GPU memory at end of epoch, may help reduce CUDA out of memory errors

                # Early Stopping
                if RANK != -1:  # if DDP training
                    broadcast_list = [self.stop if RANK == 0 else None]
                    dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                    self.stop = broadcast_list[0]
                if self.stop:
                    break  # must break all DDP ranks
                epoch += 1

            if RANK in {-1, 0}:
                # Do final val with best.pt
                LOGGER.info(
                    f"\n{epoch - self.start_epoch + 1} epochs completed in "
                    f"{(time.time() - self.train_time_start) / 3600:.3f} hours."
                )
                self.final_eval()
                if self.args.plots:
                    self.plot_metrics()
                self.run_callbacks("on_train_end")

            # Save sampler infos
            if self.config.data_loader.prioritized_sampler:
                label_change_count = self.train_loader.sampler.get_label_change_count()
                fit_count = self.train_loader.sampler.get_fit_count()
                self.save_custom_sampler_stats(label_change_count, "label_change_count")
                self.save_custom_sampler_stats(fit_count, "fit_count")


            gc.collect()
            torch.cuda.empty_cache()
            self.run_callbacks("teardown")


    def save_custom_sampler_stats(self, images_dict, output_csv_name):
        sorted_images_dict = dict(sorted(images_dict.items(), key=lambda item: item[1], reverse=True))

        output_csv = f"{self.save_dir}/{output_csv_name}.csv"

        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            writer.writerow(["Index", "Count", "Image"])
            
            for index, data in sorted_images_dict.items():
                img = self.train_loader.dataset.get_image_and_label(index)['im_file']
                writer.writerow([index, data, img])


def build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32, multi_modal=False):
    """Build YOLO Dataset."""
    return CustomYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def autocast(enabled: bool, device: str = "cuda"):
    """
    Get the appropriate autocast context manager based on PyTorch version and AMP setting.

    This function returns a context manager for automatic mixed precision (AMP) training that is compatible with both
    older and newer versions of PyTorch. It handles the differences in the autocast API between PyTorch versions.

    Args:
        enabled (bool): Whether to enable automatic mixed precision.
        device (str, optional): The device to use for autocast. Defaults to 'cuda'.

    Returns:
        (torch.amp.autocast): The appropriate autocast context manager.

    Note:
        - For PyTorch versions 1.13 and newer, it uses `torch.amp.autocast`.
        - For older versions, it uses `torch.cuda.autocast`.

    Example:
        ```python
        with autocast(amp=True):
            # Your mixed precision operations here
            pass
        ```
    """
    if TORCH_1_13:
        return torch.amp.autocast(device, enabled=enabled)
    else:
        return torch.cuda.amp.autocast(enabled)