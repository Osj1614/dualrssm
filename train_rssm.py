from rssm.trainer import SSFactorModelTrainer
import pyrallis


cfg = pyrallis.parse(config_class=SSFactorModelTrainer.Config)
trainer = SSFactorModelTrainer(cfg)
trainer.train(cfg.n_epochs)