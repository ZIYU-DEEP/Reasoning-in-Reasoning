from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import WandbLogger
from loguru import logger

import os
import wandb
import torch

from generator.model import RetrievalAugmentedGenerator
from generator.datamodule import GeneratorDataModule
from common import set_logger

class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        
        # ------------------------------------------------------------------
        # New arguments to load a pretrained checkpoint weights
        parser.add_argument("--init_ckpt_path", type=str, default=None,
                            help="Path to the checkpoint file to init the model.")
        parser.add_argument("--init_ckpt_filename", type=str, default=None,
                            help="Filename of the checkpoint to init the model.")
        # ------------------------------------------------------------------
         
        # ------------------------------------------------------------------
        # Linking arguments
        parser.link_arguments("model.model_name", "data.model_name")
        parser.link_arguments("data.max_inp_seq_len", "model.max_inp_seq_len")
        parser.link_arguments("data.max_oup_seq_len", "model.max_oup_seq_len")
         # ------------------------------------------------------------------

    def before_fit(self):
    
        # ------------------------------------------------------------------
        # Initializing with trained weights
        logger.info(f'Config: {self.config}')
        init_ckpt_path = self.config.fit.init_ckpt_path
        init_ckpt_filename = self.config.fit.init_ckpt_filename
        
        if init_ckpt_path:
            ckpt_file = os.path.join(init_ckpt_path, init_ckpt_filename)
            ckpt_module = torch.load(ckpt_file)['module']
            self.model.generator.load_state_dict(ckpt_module, strict=False)
            logger.info(f"Model loaded from checkpoint: {init_ckpt_path}")
        else:
            logger.info("No checkpoint provided; training from scratch.")
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Set the wandb logger
        # Check the path
        log_dir = self.config.fit.trainer.default_root_dir
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        
        # Set the key words
        train_mode = 'ckpt' if init_ckpt_path else 'scratch'
        lr = self.config.fit.model.lr
        bs = self.config.fit.data.batch_size
        ms = self.config.fit.trainer.max_steps
        gpus = self.config.fit.trainer.devices
        
        # Notice by default, it will take control of the ckpt saving process
        wandb_logger = WandbLogger(
            project=f"reprover-{self.config.fit.model.gen_type}",
            name=f"{train_mode}-lr{lr}-bs{bs}-steps{ms}-gpus{gpus}",
            save_dir=self.config.fit.trainer.default_root_dir,
            log_model=False,  # Do not upload model weights to wandb cloud
        )
        self.trainer.logger = wandb_logger
        logger.info("Wandb logger setup complete")
        # ------------------------------------------------------------------
        
def main():
    
     # ------------------------------------------------------------------
    # Set the logger
    set_logger(verbose=True)
    logger.info(f"PID: {os.getpid()}.")
    logger.info(f"Starting the training process.")
     # ------------------------------------------------------------------
    
     # ------------------------------------------------------------------
    # Set the client and train
    cli = CustomCLI(model_class=RetrievalAugmentedGenerator, 
                    datamodule_class=GeneratorDataModule,
                    save_config_callback=None,
                    run=True)
    
    logger.info(f"Configuration loaded and training is complete.")
     # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
