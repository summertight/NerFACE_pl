# Built-in
import os
import logging

# Deep-Learning Framework
import hydra
import pytorch_lightning as pl
#from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

# Modules
from runners import get_runner
from data import get_dm
from utils.callbacks import get_checkpoint_callback

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

log = logging.getLogger(__name__)
PATH = os.getcwd()


"""
python train.py --config-name=config.yaml gpu=[0] 
"""


@hydra.main(config_path="./configs/", config_name="")
def interactive_run(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, True)
    print(OmegaConf.to_yaml(cfg))

    # Get data, model, logger
    data_module = get_dm(cfg)
    data_module.setup()##TODO done
    runner = get_runner(cfg)
    # logger = TensorBoardLogger('tb_logs')

    job_id = os.getcwd().split('/')[-1][9:]
    #logger = WandbLogger(name=job_id, project='nerface_wild')
    cfg.working_dir = PATH # reset working directory to root directory of proj.
    
    # Setup GPUs
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, cfg.gpu))
    cfg.gpu = list(range(len(cfg.gpu)))

    # Compute validation interval (epoch)
    # val_interval = int(1000 // cfg.dataset['data_size'])
    # val_interval = 1 if val_interval < 1 else val_interval
    val_interval = 10

    # Set trainer
    checkpoint_callback = get_checkpoint_callback(criterion='step', save_frequency=cfg.train_params.checkpoint_freq)
    trainer = pl.Trainer(gpus=cfg.gpu, 
                        deterministic=True, 
                        # max_epochs=cfg.train_params.num_epochs, 
                        max_steps=cfg.train_params.num_iters,
                        callbacks=[checkpoint_callback],
                        check_val_every_n_epoch=val_interval,
                        num_sanity_val_steps=2,
                        accelerator="ddp",
                        precision=32)

    print(runner)
    
    # Train
    trainer.fit(runner,
                data_module.train_dataloader(),
                data_module.valid_dataloader())

if __name__ == "__main__":
    interactive_run()