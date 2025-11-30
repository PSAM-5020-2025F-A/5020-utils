from IPython.display import display
from PIL import Image as PImage

from torch import nn, no_grad, Tensor
from torch import cat, randn_like
from torch import log as t_log, exp as t_exp

from torch.distributions import Normal
from torch.nn import functional as F

from torchvision.utils import make_grid as tv_make_grid

def get_num_params(m):
  psum = sum(p.numel() for p in m.parameters() if p.requires_grad)
  return psum

def get_labels(model, inputs):
  batch_size = 15
  model.eval()
  labels = []
  with no_grad():
    for batch_start in list(range(0, len(inputs), batch_size)):
      y_pred = model(inputs[batch_start:batch_start + batch_size]).argmax(dim=1)
      labels += [l.item() for l in y_pred]
  return labels

def NormalizeMinMax(min=0.0, max=1.0):
  def mmn(t):
    return (t - t.min()) / (t.max() - t.min()) * (max - min) + min
  return mmn

# B x C x H x W
def batch_to_sized_grid(batch, max_dim=500):
  grid_t = (255 * tv_make_grid(batch, normalize=True, scale_each=True)).permute(1,2,0)
  gh,gw = grid_t.shape[0:2]
  nimg = PImage.new("RGB", (gw, gh))
  pxs = grid_t.int().reshape(-1, 3).tolist()
  nimg.putdata([tuple(p) for p in pxs])

  scale = max_dim / max(gw, gh)
  nh,nw = int(scale * gh), int(scale * gw)
  return nimg.resize((nw, nh))

def display_activation_grids(layer_activations, sample_idx, max_imgs=64, max_dim=720):
  for layer,actvs in layer_activations.items():
    sample_actvs = actvs[sample_idx, :max_imgs]
    batch = sample_actvs.unsqueeze(1)
    print(f"\n{layer}: {actvs.shape[-2]} x {actvs.shape[-1]}")
    display(batch_to_sized_grid(batch, max_dim))

def display_kernel_grids(layer_kernels, max_imgs=64, max_dim=256):
  for layer,kernels in layer_kernels.items():
    n_channels = 3 if kernels.shape[1] == 3 else 1
    batch = kernels[:max_imgs, :n_channels]
    print(f"\n{layer}: {kernels.shape[-2]} x {kernels.shape[-1]}")
    display(batch_to_sized_grid(batch, max_dim))


def KLDivergenceLoss(mu2=0, sig2=1):
  if mu2 == 0 and sig2 == 1:
    def comp_kl(mu1, sig1):
      return -0.5 * (1.0 + 2 * t_log(sig1) - mu1.pow(2) - sig1.pow(2)).sum()
    return comp_kl

  log_sig2 = t_log(sig2)
  two_sig2_sq = 2 * sig2.pow(2)
  def comp_kl(mu1, sig1):
    return (log_sig2 - t_log(sig1) + (sig1.pow(2) + (mu1 - mu2).pow(2)) / two_sig2_sq - 0.5).sum()
  return comp_kl


class VAE(nn.Module):
  def __init__(self, in_size, hidden_size=1024, latent_size=16, decode_activation=None):
    super().__init__()
    self.flat = nn.Flatten(start_dim=1)
    self.in2hid = nn.Linear(in_size, hidden_size)
    self.hid2mean = nn.Linear(hidden_size, latent_size)
    self.hid2std = nn.Linear(hidden_size, latent_size)
    self.z2hid = nn.Linear(latent_size, hidden_size)
    self.hid2dec = nn.Linear(hidden_size, in_size)

    self.decact = nn.Identity()
    if decode_activation == "sigmoid":
      self.decact = nn.Sigmoid()
    elif decode_activation == "tanh":
      self.decact = nn.Tanh()

  def encode(self, x):
    x = self.flat(x)
    hid = F.silu(self.in2hid(x))
    mean = self.hid2mean(hid)
    logvar = self.hid2std(hid)
    std = t_exp(0.5*logvar)
    return mean, std

  def sample(self, mu, sigma):
    z = mu + sigma * randn_like(sigma)
    return z

  def decode(self, z):
    hid = F.silu(self.z2hid(z))
    dec = self.decact(self.hid2dec(hid))
    return dec

  def forward(self, x):
    mu, sig = self.encode(x)
    z = self.sample(mu, sig)
    return self.decode(z), mu, sig


class CVAE(VAE):
  def __init__(self, in_size, hidden_size=1024, latent_size=16, cond_size=4):
    super().__init__(in_size, hidden_size, latent_size)
    self.in2hid = nn.Linear(in_size + cond_size, hidden_size)
    self.z2hid = nn.Linear(latent_size + cond_size, hidden_size)

  def encode(self, x, cond):
    x = self.flat(x)
    xc = cat((x, cond), dim=1)
    return super().encode(xc)

  def decode(self, z, cond):
    zc = cat((z, cond), dim=1)
    return super().decode(zc)

  def forward(self, x, cond):
    mu, sig = self.encode(x, cond)
    z = self.sample(mu, sig)
    return self.decode(z, cond), mu, sig
