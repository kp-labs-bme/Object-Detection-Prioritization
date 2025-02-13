import torch
from ultralytics.data.dataset import YOLODataset


class CustomYOLODataset(YOLODataset):
    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        data = self.transforms(self.get_image_and_label(index))
        data["index"] = index  # Add the index to the data
        return data


    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))

        indices = [b['index'] for b in batch]  # extract indices

        for i, k in enumerate(keys):
            if k == "index":
                continue  # skip the index key as we handle it separately
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            elif k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value

        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)

        new_batch["indices"] = torch.tensor(indices)  # add indices to the batch

        return new_batch
