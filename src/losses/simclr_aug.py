import torch
import numpy as np
from torch.nn.functional import interpolate

# ----------------------------
# 1️⃣ Cutout / Time Mask
# ----------------------------
class Cutout:
    def __init__(self, perc: float = 0.1):
        """
        Randomly zero out a contiguous portion of the time series.
        perc: fraction of sequence length to mask
        """
        assert 0.0 <= perc <= 1.0, "perc must be between 0 and 1"
        self.perc = perc

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        B, T, D = x.shape
        new_x = x.clone()
        win_len = max(1, int(self.perc * T))  # at least 1
        start = np.random.randint(0, max(1, T - win_len + 1))
        end = start + win_len
        new_x[:, start:end, :] = 0.0
        return new_x


# ----------------------------
# 2️⃣ Jitter (Add Gaussian Noise)
# ----------------------------

class Jitter:
    def __init__(self, sigma: float = 0.02):
        """
        Add small Gaussian noise to the signal.
        sigma: noise standard deviation relative to signal scale
        """
        assert sigma >= 0.0, "sigma must be non-negative"
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.sigma
        return x + noise


# ----------------------------
# 3️⃣ Scaling (Amplitude Scaling)
# ----------------------------
class Scaling:
    def __init__(self, sigma: float = 0.05):
        """
        Randomly scale each channel of each sample.
        sigma: standard deviation of scaling factor around 1.0
        """
        assert sigma >= 0.0, "sigma must be non-negative"
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        # scaling factor per sample and channel
        factor = 1.0 + torch.randn(B, D, device=x.device) * self.sigma
        factor = factor.unsqueeze(1)  # shape: (B,1,D) to broadcast over time
        return x * factor


# ----------------------------
# 4️⃣ Window Slice (Random Crop + Interpolation)
# ----------------------------
class WindowSlice:
    def __init__(self, reduce_ratio: float = 0.8, diff_len: bool = True):
        """
        Randomly crops a portion of the time series and interpolates back to original length.
        reduce_ratio: fraction of original length to crop
        diff_len: if True, each sample can have different crop
        """
        assert 0.0 < reduce_ratio <= 1.0
        self.reduce_ratio = reduce_ratio
        self.diff_len = diff_len

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        B, T, D = x.shape
        target_len = max(1, int(np.ceil(self.reduce_ratio * T)))

        # If crop length >= original, just return
        if target_len >= T:
            return x

        x_t = x.transpose(1, 2)  # shape: (B, D, T) for interpolate

        if self.diff_len:
            starts = np.random.randint(0, T - target_len + 1, size=B)
            cropped = torch.stack([x_t[i, :, starts[i]:starts[i]+target_len] for i in range(B)], dim=0)
        else:
            start = np.random.randint(0, T - target_len + 1)
            cropped = x_t[:, :, start:start+target_len]

        # interpolate back to original length
        interpolated = interpolate(cropped, size=T, mode='linear', align_corners=False)
        return interpolated.transpose(1, 2)  # back to (B, T, D)


# ----------------------------
# Example Compose
# ----------------------------
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

# Example usage:
# transform = Compose([
#     Cutout(0.1),
#     Jitter(0.02),
#     Scaling(0.05),
#     WindowSlice(0.8)
# ])
# x_aug = transform(x)

