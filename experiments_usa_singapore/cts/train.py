import imp
import importlib
import inspect
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from kornia.geometry.depth import depth_to_3d
from kornia.losses.depth_smooth import inverse_depth_smoothness_loss
from sklearn import ensemble
from torch import nn
from torchmetrics import JaccardIndex

import wandb
from lib.losses import Loss
from lib.optimizers import Optimizer
from lib.utils.visualize import draw_points_image_labels


class FilterDepth(nn.Module):
    def __init__(self, kernel: int = 9, threshold: float = 0.1):
        super().__init__()
        self.threshold = threshold
        self.max_pool = nn.MaxPool2d(kernel, 1, kernel // 2)

    def forward(self, x, train=True):
        with torch.no_grad():

            orig_depth = x
            max_depth = x.max()

            # compute min pooling
            x = self.max_pool(
                torch.where(
                    x > 0,
                    max_depth - x,
                    torch.tensor(0.0, dtype=x.dtype, device=x.device),
                )
            )
            pooled_depth = torch.where(
                x > 0, max_depth - x, torch.tensor(0.0, dtype=x.dtype, device=x.device)
            )

            # apply threshold
            mask = orig_depth > 0
            diff = torch.abs(pooled_depth - orig_depth)[mask] / pooled_depth[mask]
            filtered_depth = torch.zeros_like(orig_depth)
            filtered_depth[mask] = torch.where(
                diff < self.threshold,
                orig_depth[mask],
                torch.tensor(0.0, dtype=orig_depth.dtype, device=orig_depth.device),
            )

            return filtered_depth


class TrainModel(pl.LightningModule):
    def __init__(
        self,
        # models
        # a single module name or multiple names
        # (a single loss and optimizer for a single model or a dict name, value
        # for multiple models)
        model_modules: str | list[str] = None,
        optimizer: Optimizer | dict[str, Optimizer] | None = None,
        loss: Loss | None = None,
        # training params
        train_kwargs={},
        # model params
        model_kwargs={},
    ):
        super().__init__()
        self.automatic_optimization = False
        self.resume = True
        # params
        self.train_log_step = train_kwargs.get("train_log_images_step", 2000)
        self.val_log_step = train_kwargs.get("val_log_images_step", 500)
        self.test_log_step = train_kwargs.get("test_log_images_step", 500)
        self.lambda_xm_src = train_kwargs.get("lambda_xm_src", 0.8)
        self.lambda_xm_trg = train_kwargs.get("lambda_xm_trg", 0.1)
        self.depth_filter = FilterDepth()
        self.start_densification = train_kwargs.get("start_densification", 40000)
        # model info
        self.num_classes = (
            model_kwargs["num_classes"]
            if model_modules is isinstance(model_modules, str)
            else model_kwargs[model_modules[0]]["num_classes"]
        )
        self.class_names = train_kwargs["class_names"]
        self.rgb_palette = np.array(train_kwargs["class_palette"] + [[255, 255, 255]])
        self.rgb_palette = self.rgb_palette.astype(int)
        # assert len(self.class_names) == self.num_classes

        # load models
        self.loss = loss
        self.modules_name = model_modules
        self.model = _load_models(model_modules, optimizer, loss, **model_kwargs)
        model_modules[0]

        self.best_source_iou = 0
        self.best_target_iou = 0
        self.best_source_iou_3d = 0
        self.best_target_iou_3d = 0
        self.best_source_iou_avg = 0
        self.best_target_iou_avg = 0

        # metrics
        self.segm_iou_train = JaccardIndex(
            num_classes=self.num_classes, average="none", task="multiclass"
        ).to(self.device)
        self.segm_iou_train.reset()
        self.segm_iou_val_source = JaccardIndex(
            num_classes=self.num_classes, average="none", task="multiclass"
        ).to(self.device)
        self.segm_iou_val_source.reset()
        self.segm_iou_val_target = JaccardIndex(
            num_classes=self.num_classes, average="none", task="multiclass"
        ).to(self.device)
        self.segm_iou_val_target.reset()
        self.segm_iou_test_target = JaccardIndex(
            num_classes=self.num_classes, average="none", task="multiclass"
        ).to(self.device)
        self.segm_iou_test_target.reset()

        self.segm_ious_splits_2d = {
            "train": self.segm_iou_train,
            "val/source": self.segm_iou_val_source,
            "val/target": self.segm_iou_val_target,
            "test/target": self.segm_iou_test_target,
        }

        self.segm_iou_train_3d = JaccardIndex(
            num_classes=self.num_classes, average="none", task="multiclass"
        ).to(self.device)
        self.segm_iou_train_3d.reset()
        self.segm_iou_val_source_3d = JaccardIndex(
            num_classes=self.num_classes, average="none", task="multiclass"
        ).to(self.device)
        self.segm_iou_val_source_3d.reset()
        self.segm_iou_val_target_3d = JaccardIndex(
            num_classes=self.num_classes, average="none", task="multiclass"
        ).to(self.device)
        self.segm_iou_val_target_3d.reset()
        self.segm_iou_test_target_3d = JaccardIndex(
            num_classes=self.num_classes, average="none", task="multiclass"
        ).to(self.device)
        self.segm_iou_test_target_3d.reset()

        self.segm_ious_splits_3d = {
            "train": self.segm_iou_train_3d,
            "val/source": self.segm_iou_val_source_3d,
            "val/target": self.segm_iou_val_target_3d,
            "test/target": self.segm_iou_test_target_3d,
        }

        self.segm_iou_train_avg = JaccardIndex(
            num_classes=self.num_classes, average="none", task="multiclass"
        ).to(self.device)
        self.segm_iou_train_avg.reset()
        self.segm_iou_val_source_avg = JaccardIndex(
            num_classes=self.num_classes, average="none", task="multiclass"
        ).to(self.device)
        self.segm_iou_val_source_avg.reset()
        self.segm_iou_val_target_avg = JaccardIndex(
            num_classes=self.num_classes, average="none", task="multiclass"
        ).to(self.device)
        self.segm_iou_val_target_avg.reset()
        self.segm_iou_test_target_avg = JaccardIndex(
            num_classes=self.num_classes, average="none", task="multiclass"
        ).to(self.device)
        self.segm_iou_test_target_avg.reset()

        self.segm_ious_splits_avg = {
            "train": self.segm_iou_train_avg,
            "val/source": self.segm_iou_val_source_avg,
            "val/target": self.segm_iou_val_target_avg,
            "test/target": self.segm_iou_test_target_avg,
        }
        self.l1_loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)

    def configure_optimizers(self):
        if isinstance(self.model, nn.ModuleDict):
            optimizers, schedulers = zip(
                *[m.build_optimizer() for m in self.model.values()]
            )
            return list(optimizers), [sc for sc in schedulers if sc is not None]
        else:
            optimizer, scheduler = self.model.build_optimizer()
            return [optimizer], [scheduler] if scheduler is not None else []

    def forward(self, *args, model_name: str | None = None, **kwargs):
        return _dispatch(model_name, self.model)(*args, **kwargs)

    def densify_point_cloud_eval(
        self,
        preds_2d_fe,
        dense_depth,
        batch,
        threshold=0.9,
        discard_percentage=0.2,
    ):
        # mask background
        mask_background = torch.zeros_like(preds_2d_fe["seg_logit_2d"].argmax(1)).bool()
        probs_2d, predicted_class_2d = F.softmax(
            preds_2d_fe["seg_logit_2d"], dim=1
        ).max(1)

        # mask also non confident predictions
        for c in range(self.num_classes):
            elem_class_c_2d = predicted_class_2d == c

            if elem_class_c_2d.sum() > 0:
                treshold_class_c = torch.quantile(
                    probs_2d[elem_class_c_2d], threshold, interpolation="nearest"
                ).detach()
                non_confident_class_c = probs_2d[elem_class_c_2d] < treshold_class_c

                above_thresh_indeces_class_c = torch.where(
                    non_confident_class_c == False
                )[0]
                discard_class_c = above_thresh_indeces_class_c[
                    torch.randperm(len(above_thresh_indeces_class_c))
                ]
                discard_class_c = discard_class_c[
                    : int(len(above_thresh_indeces_class_c) * discard_percentage)
                ]

                non_confident_class_c[discard_class_c] = True
                mask_background[elem_class_c_2d] = non_confident_class_c

        # mask pixels outiside a certian range of depth because depth network at early stage is a mess
        dense_depth[mask_background.unsqueeze(1)] = 0
        dense_depth[dense_depth < 1] = 0
        dense_depth[dense_depth > 80] = 0

        dense_depth[batch["depth"] > 0] = 0
        keep_mask_2d = (dense_depth > 0).squeeze(1)
        intrinsics = batch["intrinsics"]

        # prepare input for 3D network (feature vectors of 1s and batch index for each point)
        dense_lidar = depth_to_3d(dense_depth, intrinsics)

        # original_coords = batch["pts_cam_coord"] * 20
        # min_value_original_points = original_coords.min(0)[0]

        dense_lidar = dense_lidar.permute((0, 2, 3, 1)).reshape(
            dense_lidar.shape[0], -1, 3
        )
        coords_lidar = dense_lidar.clone()

        dense_lidar = dense_lidar * 20  # 20 cm voxels
        dense_lidar = dense_lidar - batch["min_values"].unsqueeze(1)
        dense_lidar += batch["offsets"].unsqueeze(1)
        dense_lidar = dense_lidar.long()

        # only use voxels inside receptive field
        points_inside = (dense_lidar.min(2)[0] >= 0) * (dense_lidar.max(2)[0] < 4096)

        keep_mask_2d = torch.logical_and(
            keep_mask_2d, points_inside.reshape(keep_mask_2d.shape)
        )
        keep_mask = keep_mask_2d.flatten()
        batch_idxs = (
            torch.arange(0, dense_depth.shape[0], device=dense_depth.device)[
                :, None, None
            ]
            .repeat(1, dense_depth.shape[-2], dense_depth.shape[-1])
            .flatten()
            .reshape(dense_depth.shape[0], -1, 1)
        )
        dense_lidar = torch.cat([dense_lidar, batch_idxs], 2)
        dense_lidar = dense_lidar.reshape(-1, 4)[keep_mask]
        coords_lidar = coords_lidar.reshape(-1, 3)[keep_mask]
        feature_vector_dense = torch.ones(
            (dense_lidar.shape[0], 1),
            device=batch["x"][1].device,
            dtype=batch["x"][1].dtype,
        )
        batch["x_dense"] = (
            torch.cat([batch["x"][0], dense_lidar], dim=0),
            torch.cat([batch["x"][1], feature_vector_dense], dim=0),
        )
        eval_points = torch.zeros((len(batch["x_dense"][0])))
        eval_points[: len(batch["x"][0])] = 1

        pseudo_dense_gt = batch["seg_labels_2d"][keep_mask_2d].long()
        pseudo_dense_gt = torch.cat([batch["seg_label"], pseudo_dense_gt], dim=0)

        return eval_points.bool(), pseudo_dense_gt

    def _log_images(self, img, pred_depth, gt_depth, stage, step):
        _, _, h, w = img.shape
        px = 1 / plt.rcParams["figure.dpi"]
        figure = plt.figure(tight_layout=True, figsize=(2 * w * px, 1 * h * px))

        # img
        ax = figure.add_subplot(1, 3, 1)
        ax.set_axis_off()
        img = img[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        ax.imshow(img)

        ax = figure.add_subplot(1, 3, 2)
        ax.set_axis_off()
        pred_depth = pred_depth[0, 0].detach().cpu().numpy()
        ax.imshow(pred_depth, cmap="magma_r")

        ax = figure.add_subplot(1, 3, 3)
        ax.set_axis_off()
        pred_depth = gt_depth[0, 0].detach().cpu().numpy()
        ax.imshow(pred_depth, cmap="magma_r")

        log_folder = f"imgs-{stage}"

        self.loggers[0].experiment.log(
            {f"{log_folder}": wandb.Image(figure)}
        )            
        plt.close(figure)

    def cc(self, a, b):
        a = a.view(-1)
        b = b.view(-1)
        g_s_m = pd.Series(a.cpu().detach().numpy())
        g_a_d = pd.Series(b.cpu().detach().numpy())
        corr_gust = round(g_s_m.corr(g_a_d), 4)
        #     lst.append(corr_gust)
        return torch.tensor(0.1 * (1 - corr_gust) / 2).cuda()

    def cross_modal_loss(self, input, target):
        return F.kl_div(
            F.log_softmax(input, dim=1),
            F.softmax(target.detach(), dim=1),
            reduction="none",
        ).sum(1).mean() + self.cc(
            F.log_softmax(input, dim=1),
            F.softmax(target.detach(), dim=1),
        )

    def mask_sky(self, depth):
        mask_sky = depth > 0
        mask_sky = F.max_pool2d(
            mask_sky.float(), kernel_size=17, stride=1, padding=17 // 2
        )
        return ~mask_sky.bool()

    def densify_point_cloud(
        self,
        preds_2d_fe,
        dense_depth,
        batch,
        threshold=0.9,
        use_gt=False,
        discard_percentage=0.2,
    ):
        # mask background
        mask_background = torch.zeros_like(preds_2d_fe["seg_logit_2d"].argmax(1)).bool()
        probs_2d, predicted_class_2d = F.softmax(
            preds_2d_fe["seg_logit_2d"], dim=1
        ).max(1)

        # mask also non confident predictions
        for c in range(self.num_classes):
            elem_class_c_2d = predicted_class_2d == c

            if elem_class_c_2d.sum() > 0:
                treshold_class_c = torch.quantile(
                    probs_2d[elem_class_c_2d], threshold, interpolation="nearest"
                ).detach()
                non_confident_class_c = probs_2d[elem_class_c_2d] < treshold_class_c

                above_thresh_indeces_class_c = torch.where(
                    non_confident_class_c == False
                )[0]
                discard_class_c = above_thresh_indeces_class_c[
                    torch.randperm(len(above_thresh_indeces_class_c))
                ]
                discard_class_c = discard_class_c[
                    : int(len(above_thresh_indeces_class_c) * discard_percentage)
                ]

                non_confident_class_c[discard_class_c] = True
                mask_background[elem_class_c_2d] = non_confident_class_c

        # mask pixels outiside a certian range of depth because depth network at early stage is a mess
        dense_depth[mask_background.unsqueeze(1)] = 0
        dense_depth[dense_depth < 1] = 0
        dense_depth[dense_depth > 80] = 0

        # copy back GT into the densified dense depth
        dense_depth[batch["depth"] > 0] = batch["depth"][batch["depth"] > 0]
        keep_mask_2d = (dense_depth > 0).squeeze(1)
        intrinsics = batch["intrinsics"]

        # prepare input for 3D network (feature vectors of 1s and batch index for each point)
        dense_lidar = depth_to_3d(dense_depth, intrinsics).contiguous()
        dense_lidar = dense_lidar.permute((0, 2, 3, 1)).reshape(-1, 3)
        coords_lidar = dense_lidar.clone()

        dense_lidar = dense_lidar * 20  # 20 cm voxels
        dense_lidar = dense_lidar - dense_lidar.min(0)[0]
        dense_lidar = dense_lidar.long()
        # only use voxels inside receptive field
        points_inside = (dense_lidar.min(1)[0] >= 0) * (dense_lidar.max(1)[0] < 4096)

        keep_mask_2d = torch.logical_and(
            keep_mask_2d, points_inside.reshape(keep_mask_2d.shape)
        )
        keep_mask = keep_mask_2d.flatten()
        batch_idxs = (
            torch.arange(0, dense_depth.shape[0], device=dense_depth.device)[
                :, None, None
            ]
            .repeat(1, dense_depth.shape[-2], dense_depth.shape[-1])
            .flatten()
            .reshape(-1, 1)
        )
        dense_lidar = torch.cat([dense_lidar, batch_idxs], 1)
        dense_lidar = dense_lidar[keep_mask]
        coords_lidar = coords_lidar[keep_mask]

        feature_vector_dense = torch.ones(
            (dense_lidar.shape[0], 1),
            device=batch["x"][1].device,
            dtype=batch["x"][1].dtype,
        )
        batch["x_dense"] = (dense_lidar, feature_vector_dense)

        # now prepare supervision for the newly added points (i.e. from the 2D network)
        pseudo_dense_gt = preds_2d_fe["seg_logit_2d"].argmax(1).detach()

        # use original GT when avaialable
        if use_gt:
            mask_supervised_points_2d = batch["seg_labels_2d"] != -100
            pseudo_dense_gt[mask_supervised_points_2d] = batch["seg_labels_2d"][
                mask_supervised_points_2d
            ].long()

        pseudo_dense_gt = pseudo_dense_gt[keep_mask_2d]

        return keep_mask_2d, pseudo_dense_gt, coords_lidar

    def _generic_step(self, batch, stage, step=None, log_step=1000):
        self.model.train()
        data_batch_src = batch["source"]
        data_batch_trg = batch["target"]

        optimizer_2d, optimizer_3d = self.optimizers()
        optimizer_2d.zero_grad()
        optimizer_3d.zero_grad()
        scheduler_2d, scheduler_3d = self.lr_schedulers()

        data_batch_src["filtered_depth"] = self.depth_filter(data_batch_src["depth"])
        preds_2d_fe, out_2D_feature, img_indices, preds_2d_be = self(
            data_batch_src, model_name=self.modules_name[0]
        )
        preds_3d_fe, _, preds_3d_be = self(
            data_batch_src, model_name=self.modules_name[1]
        )

        seg_loss_src_2d = self.loss(
            "segmentation",
            pred=preds_2d_fe["seg_logit"],
            gt=data_batch_src["seg_label"],
        )

        depth_loss_src = (
            inverse_depth_smoothness_loss(
                preds_2d_fe["depth"].float(), data_batch_src["img"]
            )
            * 0.01
        )
        l1_depth_loss_src = (
            self.l1_loss_fn(
                preds_2d_fe["depth"][data_batch_src["filtered_depth"] > 0],
                data_batch_src["filtered_depth"][data_batch_src["filtered_depth"] > 0],
            )
            * 0.1
        )

        dense_depth = preds_2d_fe["depth"].float().detach().clone()
        # mask sky
        mask_sky = self.mask_sky(data_batch_src["depth"])
        dense_depth[mask_sky] = 0

        dense_depth_plot = dense_depth.clone()

        loss_2d = []
        loss_3d = []
        if self.global_step <= self.start_densification:
            seg_loss_src_3d = self.loss(
                "segmentation",
                pred=preds_3d_fe["seg_logit"],
                gt=data_batch_src["seg_label"],
            )
            loss_3d.append(seg_loss_src_3d)

        loss_2d.append(seg_loss_src_2d + l1_depth_loss_src + depth_loss_src)

        seg_logit_2d_avg = preds_2d_be["seg_logit_avg"]
        seg_logit_2d_max = preds_2d_be["seg_logit_max"]
        seg_logit_2d_min = preds_2d_be["seg_logit_min"]
        seg_logit_3d_point = preds_3d_be["seg_logit_point"]

        xm_loss_src_2d_avg = self.cross_modal_loss(
            seg_logit_2d_avg, preds_3d_fe["seg_logit"]
        )
        xm_loss_src_2d_max = self.cross_modal_loss(
            seg_logit_2d_max, preds_3d_fe["seg_logit"]
        )
        xm_loss_src_2d_min = self.cross_modal_loss(
            seg_logit_2d_min, preds_3d_fe["seg_logit"]
        )

        xm_loss_src_2d = (
            xm_loss_src_2d_avg + xm_loss_src_2d_max + xm_loss_src_2d_min
        ) / 3

        xm_loss_src_3d = self.cross_modal_loss(
            seg_logit_3d_point, preds_2d_fe["seg_logit"]
        )

        loss_2d.append(self.lambda_xm_src * xm_loss_src_2d)
        loss_3d.append(self.lambda_xm_src * xm_loss_src_3d)

        self.manual_backward(sum(loss_2d))
        self.manual_backward(sum(loss_3d))

        if self.global_step > self.start_densification:
            labels_first_pc_original = data_batch_src["seg_label"][
                data_batch_src["x"][0][:, 3] == 0
            ]
            coords_lidar_original = data_batch_src["pts_cam_coord"][
                data_batch_src["x"][0][:, 3] == 0
            ]

            keep_mask_2d, pseudo_dense_gt, coords_lidar = self.densify_point_cloud(
                preds_2d_fe,
                dense_depth.clone(),
                data_batch_src,
                threshold=0.9,
                use_gt=True,
                discard_percentage=0.9,
            )
            preds_3d_fe_dense, _, _ = self(
                data_batch_src, use_dense=True, model_name=self.modules_name[1]
            )

            seg_loss_src_3d_dense = self.loss(
                "segmentation",
                pred=preds_3d_fe_dense["seg_logit"],
                gt=pseudo_dense_gt,
            )

            if step % 1000 == 0:
                dense_indeces = torch.nonzero(keep_mask_2d[0])
                labels_first_pc = pseudo_dense_gt[
                    data_batch_src["x_dense"][0][:, 3] == 0
                ]
                draw_points_image_labels(
                    data_batch_src["img"][0].cpu(),
                    preds_2d_fe["seg_logit_2d"].argmax(1).detach().cpu(),
                    coords_lidar[: labels_first_pc.shape[0]],
                    coords_lidar_original,
                    dense_indeces.cpu(),
                    labels_first_pc.cpu(),
                    data_batch_src["img_indices"][0],
                    labels_first_pc_original.cpu(),
                    dense_depth_plot,
                    data_batch_src["depth"],
                    stage=stage,
                    current_epoch=self.current_epoch,
                    logger=self.loggers[0],
                    step=step,
                    color_palette_type="NuScenes",
                )
                
            self.manual_backward(seg_loss_src_3d_dense)
            seg_loss_src_3d = seg_loss_src_3d_dense

        if step % 2000 == 0:
            self._log_images(
                data_batch_src["img"],
                dense_depth,
                data_batch_src["depth"],
                stage,
                step,
            )

        ######### target domain optimization #########
        data_batch_trg["filtered_depth"] = self.depth_filter(data_batch_trg["depth"])
        preds_2d_fe, out_2D_feature, img_indices, preds_2d_be = self(
            data_batch_trg, model_name=self.modules_name[0]
        )
        preds_3d_fe, out_3D_feature, preds_3d_be = self(
            data_batch_trg, model_name=self.modules_name[1]
        )
        dense_depth = preds_2d_fe["depth"].float().detach().clone()

        # mask sky
        mask_sky = self.mask_sky(data_batch_trg["depth"])
        dense_depth[mask_sky] = 0

        dense_depth_plot = dense_depth.clone()

        loss_2d = []
        loss_3d = []
        seg_logit_2d_avg = preds_2d_be["seg_logit_avg"]
        seg_logit_2d_max = preds_2d_be["seg_logit_max"]
        seg_logit_2d_min = preds_2d_be["seg_logit_min"]
        seg_logit_3d_point = preds_3d_be["seg_logit_point"]

        xm_loss_trg_2d_avg = self.cross_modal_loss(
            seg_logit_2d_avg, preds_3d_fe["seg_logit"]
        )
        xm_loss_trg_2d_max = self.cross_modal_loss(
            seg_logit_2d_max, preds_3d_fe["seg_logit"]
        )
        xm_loss_trg_2d_min = self.cross_modal_loss(
            seg_logit_2d_min, preds_3d_fe["seg_logit"]
        )

        xm_loss_trg_2d = (
            xm_loss_trg_2d_avg + xm_loss_trg_2d_max + xm_loss_trg_2d_min
        ) / 3

        xm_loss_trg_3d = self.cross_modal_loss(
            seg_logit_3d_point, preds_2d_fe["seg_logit"]
        )

        loss_2d.append(self.lambda_xm_trg * xm_loss_trg_2d)
        loss_3d.append(self.lambda_xm_trg * xm_loss_trg_3d)
        depth_loss_trg = (
            inverse_depth_smoothness_loss(
                preds_2d_fe["depth"].float(), data_batch_trg["img"]
            )
            * 0.01
        )
        l1_depth_loss_trg = (
            self.l1_loss_fn(
                preds_2d_fe["depth"][data_batch_trg["filtered_depth"] > 0],
                data_batch_trg["filtered_depth"][data_batch_trg["filtered_depth"] > 0],
            )
            * 0.1
        )

        loss_2d.append(depth_loss_trg)
        loss_2d.append(l1_depth_loss_trg)

        self.manual_backward(sum(loss_2d))
        self.manual_backward(sum(loss_3d))

        self.log_dict(
            {
                f"{stage}/loss_segmentation": seg_loss_src_2d,
                f"{stage}/loss_segmentation_3d": seg_loss_src_3d,
                f"{stage}/xm_loss_src_2d": xm_loss_src_2d,
                f"{stage}/xm_loss_tgt_2d": xm_loss_trg_2d,
                f"{stage}/xm_loss_src_3d": xm_loss_src_3d,
                f"{stage}/xm_loss_tgt_3d": xm_loss_trg_3d,
            },
            prog_bar=True,
            add_dataloader_idx=False,
        )

        optimizer_2d.step()
        optimizer_3d.step()
        scheduler_2d.step()
        scheduler_3d.step()

        if step % 2000 == 0:
            self._log_images(
                data_batch_trg["img"],
                dense_depth,
                data_batch_trg["depth"],
                stage,
                step + 1,
            )

    def training_step(self, batch, batch_idx):
        return self._generic_step(batch, "train", step=self.global_step)

    def _generic_step_val(self, batch, stage, step=None, log_step=1000):
        self.model.eval()
        batch["filtered_depth"] = self.depth_filter(batch["depth"])
        preds_seg_2d, _, _, _ = self(batch, model_name=self.modules_name[0])
        dense_depth = preds_seg_2d["depth"].float().detach().clone()
        mask_sky = self.mask_sky(batch["depth"])
        dense_depth[mask_sky] = 0
        preds_seg_2d = preds_seg_2d["seg_logit"]

        preds_seg_3d, _, _ = self(batch, model_name=self.modules_name[1])
        preds_seg_3d = preds_seg_3d["seg_logit"]

        loss_2d = self.loss(
            "segmentation",
            pred=preds_seg_2d,
            gt=batch["seg_label"],
        )

        loss_3d = self.loss(
            "segmentation",
            pred=preds_seg_3d,
            gt=batch["seg_label"],
        )

        ensembl_pred = (preds_seg_2d + preds_seg_3d) / 2
        self.segm_ious_splits_2d[stage](
            preds_seg_2d.argmax(1)[batch["seg_label"] != -100],
            batch["seg_label"][batch["seg_label"] != -100],
        )
        self.segm_ious_splits_3d[stage](
            preds_seg_3d.argmax(1)[batch["seg_label"] != -100],
            batch["seg_label"][batch["seg_label"] != -100],
        )
        self.segm_ious_splits_avg[stage](
            ensembl_pred.argmax(1)[batch["seg_label"] != -100],
            batch["seg_label"][batch["seg_label"] != -100],
        )

        self.log_dict(
            {
                f"{stage}/loss_segmentation": loss_2d,
                f"{stage}/loss_segmentation_3d": loss_3d,
            },
            prog_bar=True,
            add_dataloader_idx=False,
        )

        self._log_images(
            batch["img"],
            dense_depth,
            batch["depth"],
            stage,
            log_step,
        )

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            label = "val/target"
        else:
            label = "test/target"

        out = self._generic_step_val(
            batch,
            label,
            self._val_step,
            log_step=self.val_log_step,
        )
        self._val_step += 1
        return {"loss": out, "domain": label}

    def test_step(self, batch, _):
        label = "test/target"
        out = self._generic_step_val(
            batch,
            label,
            self._test_step,
            log_step=self.test_log_step,
        )
        self._test_step += 1
        return {"loss": out, "domain": label}

    # def on_train_epoch_start(self) -> None:
    # self._reset_metrics()

    def on_validation_start(self):
        # self._reset_metrics()
        self._val_step = 0

    def on_test_start(self) -> None:
        # self._reset_metrics()
        self._test_step = 0

    # def _reset_metrics(self):
    #     for _, m in self.segm_ious_splits_2d.items():
    #         m.reset()
    #     for _, m in self.segm_ious_splits_3d.items():
    #         m.reset()

    def _evaluation_end(self, stage):

        # 2D
        segm_iou = self.segm_ious_splits_2d[stage]
        # print(segm_iou.confmat)
        iou_2d = segm_iou.compute().mean()

        if "val" in stage or "test" in stage:
            print("2d")
            print(
                [
                    (class_name, iou.item())
                    for class_name, iou in zip(self.class_names, segm_iou.compute())
                ]
            )

        if stage == "val/source":
            if iou_2d > self.best_source_iou:
                self.log("best_source_iou", iou_2d)
                self.best_source_iou = iou_2d
        elif stage == "val/target":
            if iou_2d > self.best_target_iou:
                self.best_target_iou = iou_2d
                self.log("best_target_iou", iou_2d)

        # 3D
        segm_iou = self.segm_ious_splits_3d[stage]
        # print(segm_iou.confmat)
        iou_3d = segm_iou.compute().mean()
        if "val" in stage or "test" in stage:
            print("3d")
            print(
                [
                    (class_name, iou.item())
                    for class_name, iou in zip(self.class_names, segm_iou.compute())
                ]
            )

        if stage == "val/source":
            if iou_3d > self.best_source_iou_3d:
                self.log("best_source_iou_3d", iou_3d)
                self.best_source_iou_3d = iou_3d
        elif stage == "val/target":
            if iou_3d > self.best_target_iou_3d:
                self.best_target_iou_3d = iou_3d
                self.log("best_target_iou_3d", iou_3d)

        # AVG
        segm_iou = self.segm_ious_splits_avg[stage]
        iou_avg = segm_iou.compute().mean()

        if stage == "val/source":
            if iou_avg > self.best_source_iou_avg:
                self.log("best_source_iou_avg", iou_avg)
                self.best_source_iou_avg = iou_avg
        elif stage == "val/target":
            if iou_avg > self.best_target_iou_avg:
                self.best_target_iou_avg = iou_avg
                self.log("best_target_iou_avg", iou_avg)

        if "val" in stage or "test" in stage:
            print("avg")
            print(
                [
                    (class_name, iou.item())
                    for class_name, iou in zip(self.class_names, segm_iou.compute())
                ]
            )

        self.log_dict(
            {
                f"{stage}/iou": iou_2d,
                f"{stage}/iou_3d": iou_3d,
                f"{stage}/iou_avg": iou_avg,
                # **{
                #     f"{stage}/iou-{cl}": segm_iou[idx]
                #     for idx, cl in enumerate(self.class_names)
                # },
            },
            add_dataloader_idx=False,
        )

        self.segm_ious_splits_2d[stage].reset()
        self.segm_ious_splits_3d[stage].reset()
        self.segm_ious_splits_avg[stage].reset()

    def train_epoch_end(self) -> None:
        self._evaluation_end("train")

    def validation_epoch_end(self, out):
        if len(out) > 0:
            stage = out[0][0]["domain"]
            self._evaluation_end(stage)
            stage = out[1][0]["domain"]
            self._evaluation_end(stage)

    def test_epoch_end(self, out) -> None:
        # stage = out[0]["domain"]
        # print(stage)
        self._evaluation_end("test/target")

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint["best_source_iou"] = self.best_source_iou
        checkpoint["best_target_iou"] = self.best_target_iou
        checkpoint["best_source_iou_3d"] = self.best_source_iou_3d
        checkpoint["best_target_iou_3d"] = self.best_target_iou_3d
        checkpoint["best_source_iou_avg"] = self.best_source_iou_avg
        checkpoint["best_target_iou_avg"] = self.best_target_iou_avg

    def on_load_checkpoint(self, checkpoint) -> None:
        self.best_source_iou = checkpoint["best_source_iou"]
        self.best_target_iou = checkpoint["best_target_iou"]
        self.best_source_iou_3d = checkpoint["best_source_iou_3d"]
        self.best_target_iou_3d = checkpoint["best_target_iou_3d"]
        self.best_source_iou_avg = checkpoint["best_source_iou_avg"]
        self.best_target_iou_avg = checkpoint["best_target_iou_avg"]


# Utilities


def _dispatch(key: str | None, items: dict | Any | None):
    if items is None:
        raise ValueError("any model registered for training")
    elif not isinstance(items, (nn.ModuleDict, dict)):
        return items
    elif key is not None:
        return items[key]
    raise ValueError("Multiple models found, choose one with model_name")


class ModelWrapper(nn.Module):
    def __init__(
        self,
        module_name: str,
        optimizer: Optimizer | None = None,
        **args,
    ):
        super().__init__()

        # loss and optimizer
        self.optimizer = optimizer
        self.name = module_name

        # load specific model
        model_mod = importlib.import_module(module_name)
        self.signature = model_mod.signature
        self.dependencies = model_mod.dependencies
        model_params = {
            n: v.default
            for n, v in inspect.signature(model_mod.Model).parameters.items()
        }
        model_params.update({k: v for k, v in args.items() if k in model_params})
        self.model_parameters = model_params
        self.model = model_mod.Model(**model_params)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def build_optimizer(self):
        opt, scheduler = self.optimizer.build(self.parameters())
        return [opt, scheduler]

    def get_model(self, script: bool = False):
        model = self.model
        if script:
            model = torch.jit.script(model)
        return model


def _load_models(model_modules, optimizer, loss, **kwargs):
    # prepare models
    out_model = None
    if isinstance(model_modules, (list, tuple, set)):
        if len(model_modules) == 0:
            raise ValueError("invalid empty model_modules list")
        out_model = nn.ModuleDict()
        for name in model_modules:
            args = kwargs[name] if name in kwargs else {}
            out_model[name] = ModelWrapper(
                name,
                optimizer[name] if optimizer is not None else None,
                **args,
            )
    elif isinstance(model_modules, str):
        out_model = ModelWrapper(model_modules, optimizer, **kwargs)
    elif model_modules is None:
        out_model = None
    else:
        raise ValueError(f"invalid model_modules type {type(model_modules)}")

    return out_model
