"""
Data Augmenting (from albumentations)
"""

import albumentations as A
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from ..utils import segmentation

# to compose transformations

__all__ = ["Augmenting", "Lambda", "PointCloudScaleTranslate"]


class Augmenting:
    def __init__(
        self,
        transforms: list[str],
        img_targets: dict[str, str],
        struct_targets: dict[str, str],
    ):

        # define transforms
        img_augs = {
            "brightness": A.RandomBrightnessContrast,
            "colorjitter": lambda: A.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4
            ),
            "blur": A.GaussianBlur,
            "noise": A.GaussNoise,
            "normalize": A.Normalize,
            "emboss": A.Emboss,
            "invert": A.InvertImg,
            "motionblur": A.MotionBlur,
            "chanshuffle": A.ChannelShuffle,
            "chandrop": A.ChannelDropout,
            "pcd_random_rotate_flip": PointCloudRandomRotateFlip,
        }

        struct_augs = {
            "hflip": lambda: HorizontalFlip(p=0.5),
            "rotate": lambda: Rotate(5),
        }

        all_aug_names = set(img_augs.keys()).union(set(struct_augs.keys()))
        for tr in transforms:
            if tr not in all_aug_names:
                raise ValueError(f"{tr} not available")

        # compose transforms
        img_transforms = _AlbumAdapter(
            *[Tr() for name, Tr in img_augs.items() if name in transforms],
            targets=img_targets,
        )

        struct_transforms = _AlbumAdapter(
            *[Tr() for name, Tr in struct_augs.items() if name in transforms],
            targets=struct_targets,
        )

        self.transforms = [img_transforms, struct_transforms]
        self._transform_names = transforms

    def add_custom_transform(self, *transforms, targets, name=None):
        self.transforms.append(_AlbumAdapter(*transforms, targets=targets))
        if name:
            self._transform_names.append(name)

    def __repr__(self):
        names = sorted(self._transform_names)
        if names:
            return ",".join(names)
        else:
            return "no_augmenting"

    def __call__(self, batch):
        out = batch
        for tr in self.transforms:
            out = tr(out)
        return out


class _AlbumAdapter:
    def __init__(self, *transforms, targets: dict[str, str]):

        self.targets = targets.copy()
        self._to_substitute = None
        if "image" not in self.targets:
            self._to_substitute = next(iter(self.targets.keys()))
        self.transform = A.Compose(
            list(transforms),
            additional_targets={
                n if n != self._to_substitute else "image": v
                for n, v in self.targets.items()
            },
        )

    def __call__(self, batch: dict):
        out = batch.copy()

        # apply transform
        transformed = self.transform(
            **{
                n if n != self._to_substitute else "image": batch[n]
                for n in self.targets.keys()
            }
        )
        if self._to_substitute:
            transformed[self._to_substitute] = transformed["image"]
            del transformed["image"]

        out.update(transformed)
        return out


# to natively handle depth and augmenting in albumentations

__all__ = [
    "HorizontalFlip",
    "Rotate",
    "Lambda",
]


class DualTransform(A.DualTransform):
    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
            "depth": self.apply_to_depth_,
            "segmentation": self.apply_to_segmentation_,
            "intrinsics": self.apply_to_intrinsics,
            "point_cloud": self.apply_to_point_cloud,
        }

    def apply_to_depth_(self, img, **params):
        return self.apply_to_depth(
            img,
            **{
                k: cv2.INTER_NEAREST if k == "interpolation" else v
                for k, v in params.items()
            },
        )

    def apply_to_segmentation_(self, img, **params):
        return self.apply_to_segmentation(
            img,
            **{
                k: cv2.INTER_NEAREST if k == "interpolation" else v
                for k, v in params.items()
            },
        )


class HorizontalFlip(DualTransform, A.HorizontalFlip):
    def apply_to_depth(self, *args, **kwargs):
        return super().apply(*args, **kwargs)

    def apply_to_segmentation(self, *args, **kwargs):
        return super().apply(*args, **kwargs)

    def apply_to_intrinsics(self, K, *args, **kwargs):
        return K

    def apply_to_point_cloud(self, pcd, *args, **kwargs):
        return pcd


class Rotate(DualTransform, A.HorizontalFlip):
    def apply_to_depth(self, *args, **kwargs):
        return super().apply(*args, **kwargs)

    def apply_to_segmentation(self, *args, **kwargs):
        return super().apply(*args, **kwargs)

    def apply_to_intrinsics(self, K, *args, **kwargs):
        return K

    def apply_to_point_cloud(self, pcd, *args, **kwargs):
        return pcd


class MapLabels(DualTransform):
    def __init__(
        self,
        all_labels: np.ndarray | list[str],
        mapping: dict[str, tuple[set[str], int]],
    ):
        super().__init__(always_apply=True, p=1.0)
        self.map_labels = segmentation.MapLabels(all_labels, mapping)

    def apply_to_segmentation(self, img, **params):
        return self.map_labels(img)

    def apply_to_intrinsics(self, K, *args, **kwargs):
        return K

    def apply_to_point_cloud(self, pcd, *args, **kwargs):
        return pcd


class Lambda(DualTransform, A.Lambda):
    def apply_to_depth(self, *args, **kwargs):
        return super().apply(*args, **kwargs)

    def apply_to_segmentation(self, *args, **kwargs):
        return super().apply(*args, **kwargs)

    def apply_to_intrinsics(self, *args, **kwargs):
        return super().apply(*args, **kwargs)

    def apply_to_point_cloud(self, *args, **kwargs):
        return super().apply(*args, **kwargs)


# Point cloud independent transformations


class PointCloudScaleTranslate(DualTransform):
    def __init__(
        self, scale: int = 20, full_scale: int = 4096, translate: bool = False
    ):
        super().__init__(always_apply=True, p=1.0)
        self.scale = scale
        self.full_scale = full_scale
        self.translate = translate

    def get_params(self) -> dict:
        return {"translate": np.random.rand(3)}

    def apply(self, elem, *args, **kwargs):
        return elem

    def apply_to_depth(self, *args, **kwargs):
        return super().apply(*args, **kwargs)

    def apply_to_segmentation(self, *args, **kwargs):
        return super().apply(*args, **kwargs)

    def apply_to_intrinsics(self, *args, **kwargs):
        return super().apply(*args, **kwargs)

    def apply_to_point_cloud(self, pcd, *args, **kwargs):

        # scale
        mask = ~np.isnan(pcd[..., 0])
        pcd_masked = pcd[mask] * self.scale
        pcd_masked = pcd_masked - pcd_masked.min()

        # translate
        if self.translate:
            pcd_masked += (
                np.clip(self.full_scale - pcd_masked.max() - 0.001, 0, None)
                * kwargs["translate"]
            )

        pcd[mask] = pcd_masked
        return pcd


class PointCloudRandomRotateFlip(DualTransform):
    def __init__(
        self,
        rot: float = 0.1,
        random_flip_x: bool = True,
        random_flip_y: bool = False,
        rot_y: float = 6.2831,
    ):
        super().__init__(always_apply=True, p=1.0)
        self.rot = rot
        self.random_flip_x = random_flip_x
        self.random_flip_y = random_flip_y
        self.rot_y = rot_y

    def get_params(self) -> dict:
        return {
            "rotvec": np.random.randn(3) * self.rot,
            "flip_x": np.random.randint(0, 2) if self.random_flip_x else 1,
            "flip_y": np.random.randint(0, 2) if self.random_flip_y else 1,
            "theta": np.random.rand() * self.rot_y,
        }

    def apply(self, elem, *args, **kwargs):
        return elem

    def apply_to_depth(self, *args, **kwargs):
        return super().apply(*args, **kwargs)

    def apply_to_segmentation(self, *args, **kwargs):
        return super().apply(*args, **kwargs)

    def apply_to_intrinsics(self, *args, **kwargs):
        return super().apply(*args, **kwargs)

    def apply_to_point_cloud(self, pcd, *args, **kwargs):

        # random rotate, flip
        rot_matrix = R.from_rotvec(kwargs["rotvec"]).as_matrix().astype(np.float32)
        rot_matrix[0, 0] *= kwargs["flip_x"]
        rot_matrix[1, 1] *= kwargs["flip_y"]
        if self.rot_y > 0:
            y_rot_matrix = np.array(
                [
                    [np.cos(theta := kwargs["theta"]), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)],
                ],
            )
            rot_matrix = np.matmul(rot_matrix, y_rot_matrix)

        mask = ~np.isnan(pcd[..., 0])
        pcd[mask] = np.matmul(pcd[mask], rot_matrix)
        return pcd
