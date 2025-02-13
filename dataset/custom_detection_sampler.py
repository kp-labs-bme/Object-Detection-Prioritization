import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from typing import Iterator

from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou


class CustomWeightedRandomDetectionSampler(WeightedRandomSampler):
    def __init__(self, args, c_const, explore_type, exploit_type, weights=None, num_samples=None, replacement=True):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if weights is None:
            weights = torch.ones(num_samples, dtype=torch.double, device=device)
        super(CustomWeightedRandomDetectionSampler, self).__init__(
            weights,        # type: torch.tensor
            num_samples,    # type: int
            replacement     # type: bool
        )

        self.args = args
        self.device = device
        self.weights = weights
        self.fit_counts = {}
        self.label_change_count = {}
        self.last_preds = {}
        self.init_pb_error = 0.001
        self.cp = c_const
        self.explore_type = explore_type
        self.exploit_type = exploit_type


    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        yield from iter(rand_tensor.tolist())

    def get_fit_count(self) -> torch.tensor:
        return self.fit_counts
    
    def get_label_change_count(self) -> torch.tensor:
        return self.label_change_count
    
    def get_weights(self) -> torch.tensor:
        return self.weights
    
    def update_weights(self, preds, batch):
        """Update the sampler's weights"""
        preds = self.postprocess(preds)
        for si, pred in enumerate(preds):
            npr = len(pred)
            pbatch = self._prepare_batch(si, batch)
            index, cls, bbox = int(pbatch.pop("index")), pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            if npr == 0:
                continue

            # Predictions
            predn = self._prepare_pred(pred, pbatch)

            if self.explore_type == "fit_count":
                if index not in self.fit_counts:
                    self.fit_counts[index] = 1
                    self.last_preds[index] = predn
                    continue

                self.fit_counts[index] += 1

                max_fit_count = max(self.fit_counts.values())
                log_term = torch.log(torch.tensor(max_fit_count) + self.init_pb_error)
                explore = self.cp * torch.sqrt(2 * log_term / (self.fit_counts[index] + self.init_pb_error))

            if index not in self.last_preds:
                self.last_preds[index] = predn
                continue

            # Evaluate
            #if nl:
            last_bboxes = self.last_preds[index][:, :4]
            last_cls = self.last_preds[index][:, 5]

            # Count bounding boxes label changes
            label_change_count = self._process_batch(predn, last_bboxes, last_cls)
            if index not in self.label_change_count:
                self.label_change_count[index] = label_change_count
            else:
                self.label_change_count[index] = self.label_change_count[index] + label_change_count

            self.last_preds[index] = predn

            max_label_change = max(self.label_change_count.values())
            exploit = self.label_change_count[index] / max_label_change if max_label_change != 0 else 0

            # Update weights
            self.weights[index] = (exploit + explore).detach()


            # Clear memory after processing batch
            del pbatch, predn, cls, bbox, index
            torch.cuda.empty_cache()

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
         # Initialize a boolean array to keep track of correct matches
        correct = torch.zeros((pred_classes.shape[0], 1), dtype=torch.bool)

        # Calculate the correct class matrix
        correct_class = true_classes[:, None] == pred_classes

        iou = iou * correct_class  # zero out the wrong classes

        # Convert IoU to numpy for easier manipulation
        iou_np = iou.detach().cpu().numpy()

        # Get matches where IoU is above the threshold
        matches = np.array(np.nonzero(iou_np >= 0.5)).T

        # Check if there are any matches
        if matches.shape[0] > 0:
            if matches.shape[0] > 1:
                # Sort matches by IoU in descending order and remove duplicates
                matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

            # Convert matches back to a torch tensor
            matches = torch.tensor(matches)

            # Update the correct array based on matches
            correct[matches[:, 1].long()] = True

        # Calculate the number of false values in the correct tensor
        num_false = correct.numel() - torch.sum(correct).item()

        # Clear unused tensors to free memory 
        del correct, correct_class, iou, iou_np, matches

        return num_false
        
    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)


    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        
        # Indexing the batch
        index = batch["indices"][si]
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1).to(self.device)  # Move to the correct device
        bbox = batch["bboxes"][idx].to(self.device)  # Move to the correct device
        ori_shape = batch["ori_shape"][si]
        imgsz = torch.tensor(batch["img"].shape[2:], device=self.device)  # Convert to tensor on the correct device
        
        # Convert imgsz to Python int values
        imgsz_int = imgsz.cpu().numpy().tolist()  # Convert tensor to a list of integers
        
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * imgsz[[1, 0, 1, 0]]  # target boxes
            # Use the converted Python int values for scale_boxes
            ops.scale_boxes(imgsz_int, bbox, ori_shape)  # native-space labels
        
        return {"index": index, "cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz_int}
    

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"])#, ratio_pad=pbatch["ratio_pad"]
          # native-space pred
        return predn
    

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=None,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
        )
