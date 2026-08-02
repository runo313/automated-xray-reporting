#!/usr/bin/env python3
"""
RAD-DINO encoder: a DINOv2 ViT-B/14 self-supervised on 882k chest radiographs.

Drop-in alternative to PretrainedCNNEncoder. Exposes the same interface:

    forward(images) -> (pooled, spatial)
        pooled:  (batch, 768)
        spatial: (batch, num_regions, 768)

so train_classifier.py, train_decoder.py, the ablation arms, eval_decoder.py
and predict.py all work by swapping the encoder construction line.

Why this encoder is worth comparing against the DenseNet:

  DenseNet-chex features were fine-tuned to predict 14 binary labels, so the
  only information the training signal preserved is what helps those 14 yes/no
  decisions. Laterality, device tip position and severity gradation got no
  gradient. RAD-DINO never saw labels at all; a self-supervised objective
  preserves what is visually salient instead. If that matters for report
  generation, the 'image'-only ablation arm (currently 0.271 macro clinical F1
  on DenseNet features) should improve the most.

  It also covers the self-supervised-pretraining scope item without spending a
  day losing to Microsoft's 64-GPU DINOv2 run on a single L4.

CAVEAT to state in any writeup: RAD-DINO's training set includes CheXpert
(223,648 images), which is essentially this project's entire dataset. The
pretraining was self-supervised, so no labels or reports leaked, but the
encoder did learn representations from images that appear in this test split.
The DenseNet baseline has the same problem in a stronger form, having been
trained supervised on CheXpert labels. Neither encoder is clean; check
overlap against training_images.csv from the model card and say so plainly.

Two preprocessing notes:

  1. RAD-DINO expects ImageNet-normalized 3-channel input, not the
     [-1024, 1024] single-channel range the rest of this pipeline uses. Rather
     than adding a second transform path through dataloader.py and risking the
     normalization mismatch that cost this project a day already, the encoder
     converts internally and defaults to assuming xrv-range input. Pass
     input_range='imagenet' if you ever feed it already-correct tensors.

  2. Patch grid depends on input size: 224px gives 16x16 = 256 patches, 518px
     (the pretraining resolution) gives 37x37 = 1369. Both are far more than
     the DenseNet's 49 regions. Default is to pool down to 7x7 so cross-
     attention cost and scale stay comparable to the existing decoder runs;
     set pool_to=None to keep the full grid.

Note the feature dimension is 768, not the DenseNet's 1024. Classifier and
decoder both take feature_dim as a constructor argument, so nothing breaks,
but checkpoints are NOT interchangeable between encoders. Swapping to this
encoder means retraining the classifier first, then the decoder on top of it.

Place at: src/xray_report/models/encoders/vit_pretrained.py
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.xray_report.models.encoders.base import BaseEncoder

# ImageNet statistics, what DINOv2-derived models expect.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

XRV_SCALE = 1024.0


class RadDinoEncoder(BaseEncoder, nn.Module):
    """
    Args:
        model_name: HuggingFace id. 'microsoft/rad-dino' is the CXR one;
            'facebook/dinov2-base' is the natural-image control if you want
            to separate "self-supervised" from "self-supervised on CXRs".
        pool_to: side length of the pooled patch grid, e.g. 7 gives 49
            regions to match the DenseNet. None keeps the native grid.
        freeze_backbone: no gradients through the ViT.
        input_range: 'xrv' if images arrive in [-1024, 1024] from the existing
            dataloader (the default), 'imagenet' if they are already
            ImageNet-normalized 3-channel tensors, 'unit' for [0, 1].
        check_input_range: warn once if the input does not look like the
            declared range. Cheap insurance against the exact class of bug
            that produced chance-level AUC earlier in this project.
    """

    def __init__(self, model_name='microsoft/rad-dino', pool_to=7,
                 freeze_backbone=False, input_range='xrv',
                 check_input_range=True):
        super().__init__()

        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "RadDinoEncoder needs the transformers library:\n"
                "    pip install transformers"
            ) from e

        if input_range not in ('xrv', 'imagenet', 'unit'):
            raise ValueError(f"unknown input_range: {input_range}")

        self.model_name = model_name
        self.pool_to = pool_to
        self.input_range = input_range
        self.check_input_range = check_input_range
        self.freeze_backbone = freeze_backbone
        self._range_checked = False

        self.backbone = AutoModel.from_pretrained(model_name)
        self.feature_dim = self.backbone.config.hidden_size          # 768
        self.patch_size = self.backbone.config.patch_size            # 14

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Registered as buffers so .to(device) moves them with the module.
        self.register_buffer(
            '_mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer(
            '_std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    # ------------------------------------------------------------- preproc

    def _verify_input(self, images):
        if not self.check_input_range or self._range_checked:
            return
        self._range_checked = True

        abs_max = images.abs().max().item()
        if self.input_range == 'xrv' and abs_max < 100:
            warnings.warn(
                f"input_range='xrv' but the input's max magnitude is "
                f"{abs_max:.2f}. These tensors do not look like the "
                "[-1024, 1024] range. Passing the wrong range here silently "
                "destroys the pretrained features.",
                RuntimeWarning,
            )
        elif self.input_range in ('imagenet', 'unit') and abs_max > 100:
            warnings.warn(
                f"input_range='{self.input_range}' but the input's max "
                f"magnitude is {abs_max:.2f}, which looks like xrv-range "
                "data. Pass input_range='xrv' instead.",
                RuntimeWarning,
            )

    def _to_imagenet(self, images):
        """Convert whatever came in into ImageNet-normalized 3-channel."""
        if self.input_range == 'imagenet':
            x = images
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            return x

        if self.input_range == 'xrv':
            # (2 * p - 1) * 1024  ->  p in [0, 1]
            x = (images / XRV_SCALE + 1.0) / 2.0
        else:                                  # 'unit'
            x = images

        x = x.clamp(0.0, 1.0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(f"expected 1 or 3 channels, got {x.shape[1]}")

        return (x - self._mean) / self._std

    # ------------------------------------------------------------- forward

    def forward(self, images):
        """
        Args:
            images: (batch, 1 or 3, H, W). H and W should be multiples of the
                patch size (14); 224 gives a 16x16 grid.

        Returns:
            pooled:  (batch, feature_dim)
            spatial: (batch, num_regions, feature_dim)
        """
        self._verify_input(images)
        x = self._to_imagenet(images)

        h, w = x.shape[-2:]
        if h % self.patch_size or w % self.patch_size:
            raise ValueError(
                f"input {h}x{w} is not divisible by patch size "
                f"{self.patch_size}; use 224 (16x16 grid) or 518 (37x37)"
            )

        out = self.backbone(pixel_values=x)
        tokens = out.last_hidden_state              # (B, 1 + N, D), CLS first
        patches = tokens[:, 1:, :]                  # drop CLS

        gh, gw = h // self.patch_size, w // self.patch_size

        if self.pool_to is not None and self.pool_to < gh:
            # (B, N, D) -> (B, D, gh, gw) -> pool -> (B, R, D)
            grid = patches.transpose(1, 2).reshape(
                patches.size(0), self.feature_dim, gh, gw)
            grid = F.adaptive_avg_pool2d(grid, (self.pool_to, self.pool_to))
            spatial = grid.flatten(2).transpose(1, 2)
        else:
            spatial = patches

        # Mean over patches, matching how PretrainedCNNEncoder pools, so the
        # classifier head sees the same kind of vector. The CLS token is the
        # other reasonable choice; mean-pooling keeps the two encoders
        # comparable, which matters more here than squeezing out performance.
        pooled = spatial.mean(dim=1)

        return pooled, spatial

    @property
    def num_regions(self):
        if self.pool_to is not None:
            return self.pool_to * self.pool_to
        raise ValueError(
            "num_regions is only fixed when pool_to is set; with the native "
            "grid it depends on input resolution"
        )


# ------------------------------------------------------------------ smoke

if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description="Smoke-test the RAD-DINO encoder.")
    ap.add_argument('--model-name', default='microsoft/rad-dino')
    ap.add_argument('--image-root', default='data/images')
    ap.add_argument('--parquet', default='data/chexpert_plus_fixed.parquet')
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--pool-to', type=int, default=7)
    ap.add_argument('--image-size', type=int, default=224)
    args = ap.parse_args()

    import os

    import pandas as pd
    from PIL import Image
    from torchvision import transforms

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    encoder = RadDinoEncoder(model_name=args.model_name,
                             pool_to=args.pool_to).to(device).eval()
    print(f"loaded {args.model_name}")
    print(f"  hidden size {encoder.feature_dim}, patch {encoder.patch_size}")
    print(f"  params {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M")

    # Reuse the project's existing transform, so this tests the real path.
    class XRVNormalize:
        def __call__(self, x):
            if x.shape[0] > 1:
                x = x.mean(dim=0, keepdim=True)
            return (2.0 * x - 1.0) * 1024.0

    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        XRVNormalize(),
    ])

    df = pd.read_parquet(args.parquet)
    df = df[df['split'] == 'val'].head(args.n)

    imgs = []
    for rel in df['path_to_image']:
        path = os.path.join(args.image_root, rel.split('/', 1)[1])
        if os.path.exists(path):
            with Image.open(path) as im:
                imgs.append(tf(im))
    batch = torch.stack(imgs).to(device)
    print(f"\ninput {tuple(batch.shape)}  "
          f"range [{batch.min():.0f}, {batch.max():.0f}]")

    with torch.no_grad():
        pooled, spatial = encoder(batch)

    print(f"pooled  {tuple(pooled.shape)}   (want (N, {encoder.feature_dim}))")
    print(f"spatial {tuple(spatial.shape)}  "
          f"(want (N, {args.pool_to**2}, {encoder.feature_dim}))")

    across = pooled.std(dim=0).mean().item()
    within = pooled.std(dim=1).mean().item()
    print(f"\nfeature std ACROSS images: {across:.4f}")
    print(f"feature std WITHIN an image: {within:.4f}")

    if across < 1e-3:
        print("\n  FAIL: features are constant across images. Check the")
        print("  input range conversion before training anything.")
    else:
        print("\n  PASS: features vary per image.")

    print("""
Next: this encoder outputs 768-dim features, not 1024, so the existing
classifier and decoder checkpoints will NOT load on top of it. Retrain in
order — classifier first, then decoder:

  python3 -m src.xray_report.training.train_classifier \\
      --parquet-path data/chexpert_plus_fixed.parquet \\
      --vocab-path data/vocab.pkl --image-root data/images \\
      --checkpoint-dir checkpoints/cls_raddino --encoder raddino \\
      --num-epochs 5 --amp

Compare its macro AUC against the DenseNet's 0.790 before spending time on
the decoder. If the classifier is clearly worse, the encoder swap is not
going to help the generation arm either.
""")
