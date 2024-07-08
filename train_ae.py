from ae_pricing.trainer import AEPricingTrainer
import pyrallis


cfg = pyrallis.parse(config_class=AEPricingTrainer.Config)
trainer = AEPricingTrainer(cfg)
trainer.run()