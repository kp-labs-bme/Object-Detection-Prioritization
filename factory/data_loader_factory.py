import torch
from typing import Tuple, Union
from torch.utils.data import DataLoader

from factory.base_factory import BaseFactory
from dataset.custom_detection_sampler import CustomWeightedRandomDetectionSampler


class DataLoaderFactory(BaseFactory):
    @classmethod
    def create(cls, **kwargs) -> Tuple[DataLoader, DataLoader, Union[None, CustomWeightedRandomDetectionSampler]]:
        train_dataset, val_dataset = kwargs["train_dataset"], kwargs["val_dataset"]
        config = kwargs["config"]
        args = kwargs["args"]

        if bool(config.data_loader.prioritized_sampler):
            c_const = config.data_loader.c_constant
            explore_type = config.data_loader.explore_type
            exploit_type = config.data_loader.exploit_type
            sampler = CustomWeightedRandomDetectionSampler(args,
                                                  c_const=c_const,
                                                  explore_type=explore_type,
                                                  exploit_type=exploit_type,
                                                  num_samples=train_dataset.ni)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        cuda_available = torch.cuda.is_available()
        device = 'cuda' if cuda_available else ''

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=config.data_loader.batch_size_train,
                                  num_workers=config.data_loader.num_workers,
                                  sampler=sampler,
                                  shuffle=shuffle,
                                  pin_memory=cuda_available,
                                  pin_memory_device=device,
                                  collate_fn=getattr(train_dataset, "collate_fn", None),
)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=config.data_loader.batch_size_train,
                                num_workers=config.data_loader.num_workers,
                                pin_memory=cuda_available,
                                pin_memory_device=device,
                                collate_fn=getattr(val_dataset, "collate_fn", None))

        return train_loader, val_loader, sampler
