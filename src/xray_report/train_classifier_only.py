#!/usr/bin/env python3
"""Classifier-only training: encoder + classifier head, no decoder, for
producing a properly-trained encoder before joint training."""

import argparse
import os
import time
import torch
import torch.optim as optim
import pandas as pd

from src.xray_report.config import LABEL_COLS, redirect_output
from src.xray_report.dataloader import build_dataloaders
from src.xray_report.utils.vocabulary import load_vocab
from src.xray_report.models.encoders.cnn_pretrained import PretrainedCNNEncoder
from src.xray_report.models.classifier_head import ClassifierHead
from src.xray_report.models.losses import MaskedBCELoss, compute_pos_weight


class ClassifierFitModel:
    def __init__(self, n_iter, encoder, classifier, train_loader, val_loader,
                 optimizer, bce_loss, random_state=42, checkpoint_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.n_iter = n_iter
        torch.manual_seed(random_state)

        self.encoder = encoder.to(self.device)
        self.classifier = classifier.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.bce_loss = bce_loss.to(self.device)

        self.checkpoint_dir = checkpoint_dir
        self.best_val_loss = float('inf')
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        self.train_losses = []
        self.val_losses = []

    def fit(self):
        start_time = time.time()

        for epoch in range(self.n_iter):
            epoch_start = time.time()
            self.encoder.train()
            self.classifier.train()
            running_loss = 0.0

            for images, labels, mask, text_ids in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)

                self.optimizer.zero_grad()
                pooled, _ = self.encoder(images)
                logits = self.classifier(pooled)
                loss = self.bce_loss(logits, labels, mask)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            self.train_losses.append(avg_loss)

            avg_val_loss = self.validate()
            self.val_losses.append(avg_val_loss)

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{self.n_iter} — cls_loss: {avg_loss:.4f} — val_loss: {avg_val_loss:.4f} — time: {epoch_time:.1f}s")

            if self.checkpoint_dir:
                self.save_checkpoint(epoch, avg_val_loss, "last.pt")
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.save_checkpoint(epoch, avg_val_loss, "best.pt")

        print(f"Training complete in {time.time() - start_time:.1f}s")

    def validate(self):
        self.encoder.eval()
        self.classifier.eval()
        running_loss = 0.0

        with torch.no_grad():
            for images, labels, mask, text_ids in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)

                pooled, _ = self.encoder(images)
                logits = self.classifier(pooled)
                loss = self.bce_loss(logits, labels, mask)
                running_loss += loss.item()

        return running_loss / len(self.val_loader)

    def save_checkpoint(self, epoch, val_loss, filename):
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_loss': val_loss,
        }
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"saved checkpoint: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifier-only pretraining (encoder + classifier head).")
    parser.add_argument('--parquet-path', required=True)
    parser.add_argument('--vocab-path', required=True)
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--log-dir', default='logs')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=7)
    parser.add_argument('--encoder-lr', type=float, default=1e-5)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(args.log_dir, f"classifier_only_{timestamp}.log")
    redirect_output(log_path)
    print(f"logging to {log_path}")
    print(f"args: {vars(args)}")

    df = pd.read_parquet(args.parquet_path)
    vocab = load_vocab(args.vocab_path)

    loaders = build_dataloaders(df, args.image_root, vocab['token_to_idx'],
                                 batch_size=args.batch_size, num_workers=args.num_workers)

    encoder = PretrainedCNNEncoder()
    classifier = ClassifierHead(feature_dim=encoder.feature_dim, num_labels=len(LABEL_COLS))

    train_df = df[df['split'] == 'train']
    #pos_weight = compute_pos_weight(train_df, LABEL_COLS)
    bce_loss = MaskedBCELoss(pos_weight=None)

    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': args.encoder_lr},
        {'params': classifier.parameters(), 'lr': args.head_lr},
    ])

    trainer = ClassifierFitModel(
        n_iter=args.num_epochs,
        encoder=encoder,
        classifier=classifier,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        optimizer=optimizer,
        bce_loss=bce_loss,
        random_state=args.random_state,
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer.fit()