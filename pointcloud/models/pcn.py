from fastai.torch_core import *
from fastai.layers import *

from .pointnet.utils.pytorch_utils import SharedMLP

__all__ = ['PCNet']


class PCNDecoder(nn.Module):
    def __init__(self, grid_size=4, grid_scale=0.05):
        super().__init__()

        self.grid_size = grid_size
        self.grid_scale = grid_scale
        grid = torch.meshgrid([
            torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size),
            torch.linspace(-self.grid_scale, self.grid_scale, self.grid_size),
        ])
        self.grid = torch.stack(grid, dim=-1).reshape((-1, 2))  # type: Tensor

        self.shared_mlp = SharedMLP([512, 512, 3])

    def forward(self, xyz: Tensor, features: Tensor):
        assert len(xyz.shape) == len(features.shape) == 3
        assert xyz.shape[0] == features.shape[0]
        assert xyz.shape[1] == features.shape[1] or features.shape[1] == 1
        assert xyz.shape[2] == 3

        bs = xyz.shape[0]
        num_course = xyz.shape[1]

        # If given global features, tile to the same size local features would be
        if features.shape[1] == 1:
            features = features.expand(-1, num_course, -1)

        grid_feat = self.grid.repeat(num_course, 1).expand(bs, -1, -1)
        point_feat = xyz.repeat_interleave(self.grid_size ** 2, dim=1)
        global_or_local_feat = features.repeat_interleave(self.grid_size ** 2, dim=1)

        feat = torch.cat([grid_feat, point_feat, global_or_local_feat], dim=2)

        center = xyz.repeat_interleave(self.grid_size ** 2, dim=1)
        fine = self.shared_mlp(feat) + center

        return fine


class PCNet(nn.Module):
    def __init__(self, encoder:nn.Module):
        super().__init__()

        # encoder has an output either of BxF or BxNxF. For now, assume the latter. The
        # former case I'll worry about later.

        self.encoder = encoder
        self.decoder = PCNDecoder(grid_size=4, grid_scale=0.04)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x[:, :3], x[:, 3:])
