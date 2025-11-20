from pathlib import Path

import torch
import torch.nn as nn

SRC_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):    
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.register_buffer("input_mean", torch.tensor([0.03162788227200508, 11.780111312866211]))
        self.register_buffer("input_std", torch.tensor([5.585452079772949, 5.992654800415039]))

        input_dim = n_track * 2 * 2
        output_dim = n_waypoints * 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),  # Output 3 waypoints × 2D
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        x = torch.cat([track_left, track_right], dim=1)  # (B, 2 * n_track, 2)
        x = x.view(x.size(0), -1, 2)
        x = (x- self.input_mean) / self.input_std
        x = x.view(x.size(0), -1)
        out = self.net(x)  # (B, output_dim)
        return out.view(-1, self.n_waypoints, 2)


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        dropout = 0.1,
        num_layers = 2
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.query_embed = nn.Embedding(n_waypoints, d_model)

        self.input_proj = nn.Linear(2, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        B = track_left.size(0)
        device = track_left.device

        track = torch.cat([track_left, track_right], dim=1)

        memory = self.input_proj(track)

        query = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        decoded = self.decoder(tgt=query, memory=memory)

        return self.output_proj(decoded)


class CNNPlanner(torch.nn.Module):
    class DownBlock(nn.Module):
        def __init__(self, in_channels, out_channels, num_layers=2):
            super().__init__()
            layers = []

            # First layer: downsampling
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU())

            # Remaining layers: regular convs with same output channels
            for _ in range(num_layers - 1):
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
                layers.append(nn.ReLU())

            self.block = nn.Sequential(*layers)

        def forward(self, x):
            return self.block(x)
    
    class UpBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.up(x))
    
    def __init__(
        self,
        n_waypoints: int = 3,
        in_channels: int = 3,
        num_classes: int = 3,
        channels_l0 = 64,
        n_blocks = 2
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)
        
        # ENCODER
        # Special first layer
        self.special_conv = nn.Conv2d(in_channels, channels_l0, kernel_size=11, stride=2, padding=5)
        self.relu = nn.ReLU()

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # Loop for blocks
        c1 = channels_l0
        channels = [in_channels, channels_l0]
        for _ in range(n_blocks):
            c2 = c1 * 2
            self.encoder_blocks.append(self.DownBlock(c1, c2))
            c1 = c2
            channels.append(c2)

        # Build decoder
        c2 = c1 // 2
        for i in range(n_blocks):  # +1 for symmetry with special_conv
            self.decoder_blocks.append(self.UpBlock(c1, c2))
            c1 = c2 + channels[len(channels) - 2 - i]
            c2 = c1 // 2

        # Final UBlock to recover spatial dimensions
        self.up_final = self.UpBlock(c1, c2)

        # Final Layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_layer = nn.Linear(c2, n_waypoints * 2)

        # Segmentation head
        self.seg_head = nn.Conv2d(c2, num_classes, kernel_size=3, padding=1)

    def forward(self, image: torch.Tensor, train = False, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        encoder_feats = []

        # Special first layer
        x = self.relu(self.special_conv(x))     # (B, 64, H/2, W/2)
        
        for block in self.encoder_blocks:
            encoder_feats.append(x) # Will hold all skip connection features, skips last encoder block
            x = block(x)    

        for i, block in enumerate(self.decoder_blocks):
            x = block(x)
            skip = encoder_feats[len(encoder_feats) - 1 -i]
            x = torch.cat([x, skip], dim=1) # Skip Connection

        # Final Up Block to Recover spatial dims of original images
        x = self.up_final(x)

        # Find waypoints from images
        y = self.global_avg_pool(x)
        y = y.view(x.size(0), -1)
        waypoints = self.final_layer(y).view(-1, self.n_waypoints, 2)
        
        return waypoints


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = SRC_DIR.parent / "models" / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    models_dir = SRC_DIR.parent / "models"
    models_dir.mkdir(exist_ok=True)
    output_path = models_dir / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
