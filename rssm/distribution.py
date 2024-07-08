__all__ = [ "MultiCategoricalWithLogits"]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D



class MultiCategoricalWithLogits(D.Distribution):
    def __init__(self, logits, num_classes: int = 16):
        super().__init__(validate_args=False)
        self.orig_shape = logits.shape # (..., K * num_classes)
        self.num_classes = num_classes
        if self.orig_shape[-1] % num_classes != 0:
            raise ValueError("Last dimension of logits must be divisible by num_classes.")
        self.num_categoricals = self.orig_shape[-1] // num_classes

        self.dist = D.Categorical(logits=logits.reshape(*self.orig_shape[:-1], self.num_categoricals, num_classes))
    

    def rsample(self, sample_shape=torch.Size()):
        return (
            nn.functional.one_hot(self.dist.sample(sample_shape), num_classes=self.num_classes)
            + self.dist.probs - self.dist.probs.detach()  # Straight-through
        ).reshape(*self.orig_shape)
        
    

    def sample(self, sample_shape=torch.Size()):
        return nn.functional.one_hot(self.dist.sample(sample_shape), num_classes=self.num_classes).float().reshape(self.orig_shape)


    def entropy(self):
        return self.dist.entropy()#.reshape(*self.orig_shape[:-1])
    

    def enumerate_support(self, expand=True):
        support = self.dist.enumerate_support(expand=expand)
        return support.reshape(self.orig_shape)
    

    @property
    def mode(self):
        return nn.functional.one_hot(self.dist.mode, num_classes=self.num_classes).float().reshape(*self.orig_shape)  # Add straight-through?
    
    @property
    def mean(self):  # for compatibility with continuous distributions
        return self.mode
    
    
    

@D.kl.register_kl(MultiCategoricalWithLogits, MultiCategoricalWithLogits)
def _kl_multicategorical_with_logits(p: MultiCategoricalWithLogits, q: MultiCategoricalWithLogits):
    return D.kl.kl_divergence(p.dist, q.dist).reshape(*p.orig_shape[:-1], p.num_categoricals, -1)


if __name__ == "__main__":
    B, S, C = 2, 3, 64
    logits1 = torch.empty(B, S, C).normal_()
    logits2 = torch.empty(B, S, C).normal_()

    dist1 = MultiCategoricalWithLogits(logits1)
    dist2 = MultiCategoricalWithLogits(logits2)

    print(dist1.entropy().shape)
    print(torch.distributions.kl_divergence(dist1, dist2).shape)