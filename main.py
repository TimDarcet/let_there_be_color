from model import CoolModel
from pytorch_lightning import Trainer
from test_tube import Experiment
#from data import MNISTDataModule
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
import os


def main():
    model = CoolModel()
    
    # train on 80 GPUs across 10 nodes
    trainer = Trainer(max_epochs=1,
                      gpus=1,
                      num_nodes=1)
                      accelerator='ddp')
    trainer.fit(model, MNISTDataModule(os.getcwd()))


if __name__ ==  '__main__':
    main()
