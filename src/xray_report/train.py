#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
import pandas as pd
import time
import torch.optim as optim
import os


from src.xray_report.config import LABEL_COLS, DEFAULT_MAX_LEN, redirect_output
from src.xray_report.dataloader import build_dataloaders
from src.xray_report.utils.vocabulary import load_vocab

from src.xray_report.models.encoders.cnn_pretrained import PretrainedCNNEncoder
from src.xray_report.models.classifier_head import ClassifierHead
from src.xray_report.models.decoders.rnn_attention import AttentionDecoder
from src.xray_report.models.losses import MaskedBCELoss, MaskedCrossEntropyLoss, compute_pos_weight
from src.xray_report.models.model import XRayReportModel

class FitModel():
    def __init__(self,n_iter, random_state, model,train_loader, val_loader,
                  optimizer, bce_loss, ce_loss, lambda_weight, checkpoint_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {self.device}')

        self.n_iter = n_iter
        self.random_state = random_state
        torch.manual_seed(self.random_state)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.bce_loss = bce_loss.to(self.device)
        self.ce_loss = ce_loss.to(self.device)
        self.lambda_weight= lambda_weight

        self.checkpoint_dir = checkpoint_dir
        self.best_val_loss = float('inf')
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        self.train_losses=[]
        self.val_losses_=[]

    def fit(self):
        start_time = time.time()

        for epoch in range (self.n_iter):
            epoch_start = time.time()
            self.model.train()
            running_loss= 0.0
            for images, labels,mask,text_ids in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)
                text_ids = text_ids.to(self.device)

                self.optimizer.zero_grad()
                
                cls_logits, dec_logits, _ = self.model(images,text_ids,labels)
                cls_loss = self.bce_loss(cls_logits, labels, mask)
                gen_loss = self.ce_loss(dec_logits, text_ids[:, 1:])
                loss = cls_loss + self.lambda_weight * gen_loss

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            self.train_losses.append(avg_loss)
            avg_val_loss = self.validate(self.val_loader)
            self.val_losses.append(avg_val_loss)

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{self.n_iter} — loss: {avg_loss:.4f} — time: {epoch_time:.1f}s")

            if self.checkpoint_dir:
                self.save_checkpoint(epoch, avg_val_loss, filename="last.pt")
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save_checkpoint(epoch, avg_val_loss, filename="best.pt")

        total_time = time.time() - start_time
        print(f"Training complete in {total_time:.1f}s")

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for images, labels, mask, text_ids in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)
                text_ids = text_ids.to(self.device)

                cls_logits, dec_logits, _ = self.model(images, text_ids, labels)
                cls_loss = self.bce_loss(cls_logits, labels, mask)
                gen_loss = self.ce_loss(dec_logits, text_ids[:, 1:])
                loss = cls_loss + self.lambda_weight * gen_loss

                running_loss += loss.item()

        return running_loss / len(val_loader)

    def save_checkpoint(self, epoch, val_loss, filename):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_loss': val_loss,
        }
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"saved checkpoint: {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        start_epoch = checkpoint['epoch'] + 1
        print(f"resumed from checkpoint at epoch {start_epoch}")
        return start_epoch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CheXpert automated reporting model.")
    parser.add_argument('--log-dir', default='logs', help="Directory to write training logs")
    parser.add_argument('--encoder-type', default='resnet50', help="Encoder variant, for log naming")
    parser.add_argument('--decoder-type', default='rnn', help="Decoder variant, for log naming")
    parser.add_argument('--parquet-path', required=True, help="Path to merged labels parquet file")
    parser.add_argument('--vocab-path', required=True, help="Path to saved vocabulary pickle")
    parser.add_argument('--image-root', required=True, help="Root folder containing downloaded images")
    parser.add_argument('--checkpoint-dir', required=True, help="Directory to save model checkpoints")
    parser.add_argument('--resume-from', default=None, help="Path to checkpoint to resume from")

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--encoder-lr', type=float, default=1e-5)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--lambda-weight', type=float, default=1.0)
    parser.add_argument('--max-len', type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    #log file
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_filename = f"{args.encoder_type}_{args.decoder_type}_{timestamp}.log"
    log_path = os.path.join(args.log_dir, log_filename)

    redirect_output(log_path)
    print(f"logging to {log_path}")
    print(f"args: {vars(args)}")

    # data
    df = pd.read_parquet(args.parquet_path)
    vocab = load_vocab(args.vocab_path)
    token_to_idx = vocab['token_to_idx']

    loaders = build_dataloaders(
        df, args.image_root, token_to_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_len=args.max_len,
    )

    encoder = PretrainedCNNEncoder()
    classifier = ClassifierHead(feature_dim=encoder.feature_dim, num_labels=len(LABEL_COLS))
    decoder = AttentionDecoder(
        vocab_size=len(token_to_idx),
        embed_dim=256,
        hidden_size=512,
        findings_embed_dim=64,
        num_labels=len(LABEL_COLS),
        feature_dim=encoder.feature_dim,
    )
    model = XRayReportModel(encoder, classifier, decoder)

    # losses 
    train_df = df[df['split'] == 'train']
    pos_weight = compute_pos_weight(train_df, LABEL_COLS)
    bce_loss = MaskedBCELoss(pos_weight=pos_weight)
    ce_loss = MaskedCrossEntropyLoss(pad_idx=token_to_idx['<pad>'])

    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': args.encoder_lr},
        {'params': model.classifier.parameters(), 'lr': args.head_lr},
        {'params': model.decoder.parameters(), 'lr': args.head_lr},
    ])

    trainer = FitModel(
        n_iter=args.num_epochs,
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        optimizer=optimizer,
        bce_loss=bce_loss,
        ce_loss=ce_loss,
        lambda_weight=args.lambda_weight,
        random_state=args.random_state,
        checkpoint_dir=args.checkpoint_dir
    )
    start_epoch = 0
    if args.resume_from:
        start_epoch = trainer.load_checkpoint(args.resume_from)
    trainer.fit()
