from fastai.callbacks import *
from typing import *

from .pointnet import *
from .pointnet.utils.pytorch_utils import SharedMLP

__all__ = ['PCNet']


class PCNDecoder(nn.Module):
    def __init__(self, features_hook: Hook, grid_size=4, grid_scale=0.05):
        super().__init__()

        self.features_hook = features_hook

        self.grid_size = grid_size
        self.grid_scale = grid_scale
        grid = torch.meshgrid([
            torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size),
            torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size),
        ])
        self.grid = torch.stack(grid, dim=-1).reshape((-1, 2))  # type: Tensor

        self.shared_mlp = SharedMLP([133, 512, 3])

    def forward(self, xyz: Tensor):
        features = self.features_hook.stored.transpose(1, 2)

        assert len(xyz.shape) == len(features.shape) == 3
        assert xyz.shape[0] == features.shape[0]
        assert xyz.shape[1] == features.shape[1] or features.shape[1] == 1
        assert xyz.shape[2] == 3

        bs = xyz.shape[0]
        num_course = xyz.shape[1]

        # If given global features, tile to the same size local features would be
        if features.shape[1] == 1:
            features = features.expand(-1, num_course, -1)

        grid_feat = self.grid.to(xyz.device).repeat(num_course, 1).expand(bs, -1, -1)
        point_feat = xyz.repeat_interleave(self.grid_size ** 2, dim=1)
        global_or_local_feat = features.repeat_interleave(self.grid_size ** 2, dim=1)

        feat = torch.cat([grid_feat, point_feat, global_or_local_feat], dim=2)

        center = xyz.repeat_interleave(self.grid_size ** 2, dim=1)
        fine_local = self.shared_mlp(feat.transpose(1, 2).unsqueeze(dim=3)).transpose(1, 2).squeeze(dim=3)
        fine = fine_local + center

        return fine


class PCNet(nn.Module):
    def __init__(self, encoder: Union[Pointnet2SSGSeg, Pointnet2MSGSeg]):
        super().__init__()

        self.encoder = encoder
        self.decoder = PCNDecoder(features_hook=hook_output(encoder.FP_modules[0]), grid_size=4, grid_scale=0.04)

    def forward(self, pointcloud):
        self.encoder(pointcloud)
        return self.decoder(pointcloud[..., :3])
