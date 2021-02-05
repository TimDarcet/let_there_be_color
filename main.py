from model import CoolModel
from pytorch_lightning import Trainer
from test_tube import Experiment

def main():
    model = CoolModel()
    exp = Experiment(save_dir=os.getcwd())
    
    # train on 80 GPUs across 10 nodes
    trainer = Trainer(experiment=exp, 
                      max_nb_epochs=1, 
                      gpus=[0,], 
                      nb_gpu_nodes=10)
    trainer.fit(model)
  
if __name__ ==  '__main__':
    main()
