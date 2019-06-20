if __name__ == "__main__":
    from fastai import tabular
    from fastai.torch_core import *
    from fastai.layers import *
    from torch import functional as F
    import torch

    class JaccardLoss(nn.Module):
        def forward(self, logits: Tensor, target: Tensor):
            n_classes = logits.shape[1]
            target_1_hot = torch.eye(n_classes)[target.squeeze(1)]
            target_1_hot = target_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)

            true_1_hot = target_1_hot.type(logits.type())
            intersection = torch.sum(probas * true_1_hot, dim=[0, 2, 3])
            cardinality = torch.sum(probas + true_1_hot, dim=[0, 2, 3])
            union = cardinality - intersection
            jacc = (intersection / (union + 1e-7)).mean()
            return 1 - jacc

    def JaccardFlat(*args, axis=-1, **kwargs):
        return FlattenedLoss(JaccardLoss, *args, axis=axis, **kwargs)