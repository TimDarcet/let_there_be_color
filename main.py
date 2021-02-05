import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model import LTBC
from data import places365DataModule


def main(args):
    # Settings
    pl.seed_everything(42)
    from_checkpoint = False
    checkpoint = 'lightning_logs/version_4/checkpoints/epoch=6-step=6390.ckpt'
    
    # Handle the data
    dm = places365DataModule(args.data_folder, batch_size=args.batch_size)

    # Define model
    model = LTBC(alpha=args.alpha, rightsize=True, classify=True)

    # Exp logger
    logger = TensorBoardLogger('logs/tensorboard_logs')

    # Define training
    trainer = pl.Trainer(gpus=1,
                         num_nodes=args.n_nodes,
                         accelerator='ddp',
                         auto_select_gpus=True,
                         max_epochs=args.epochs,
                         callbacks=[ModelCheckpoint(monitor='val_loss')],
                         logger=logger)

    # Train
    trainer.fit(model, dm)


if __name__ ==  '__main__':
    parser = ArgumentParser()
    parser.add_argument('--n_nodes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--data_folder', type=str, default="../places365_standard")
    args = parser.parse_args()
    main(args)
