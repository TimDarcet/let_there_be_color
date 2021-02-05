from model import CoolModel
from pytorch_lightning import Trainer
from test_tube import Experiment
#from data import MNISTDataModule
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
import os
from argparse import ArgumentParser

def main(args):
    model = CoolModel()
    
    # train on 80 GPUs across 10 nodes
    trainer = Trainer(max_epochs=1,
                      gpus=1,
                      num_nodes=args.n_nodes,
                      accelerator='ddp')
    trainer.fit(model, MNISTDataModule(os.getcwd()))


if __name__ ==  '__main__':
    parser = ArgumentParser()
    parser.add_argument('--n_nodes', type=int, default=2)
    args = parser.parse_args()
    main(args)
