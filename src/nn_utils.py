import torch

from IPython.display import display
from PIL import Image as PImage
from torchvision.utils import make_grid as tv_make_grid

def get_num_params(m):
  psum = sum(p.numel() for p in m.parameters() if p.requires_grad)
  return psum

def get_labels(model, inputs):
  model.eval()
  with torch.no_grad():
    y_pred = model(inputs).argmax(dim=1)
    return [l.item() for l in y_pred]

def NormalizeMinMax(min=0.0, max=1.0):
  def mmn(t):
    return (t - t.min()) / (t.max() - t.min()) * (max - min) + min
  return mmn

# B x C x H x W
def batch_to_sized_grid(batch, max_dim=500):
  grid_t = (255 * tv_make_grid(batch, normalize=True, scale_each=True)).permute(1,2,0)
  gh,gw = grid_t.shape[0:2]
  scale = max_dim / max(gw, gh)
  nh,nw = int(scale * gh), int(scale * gw)
  nimg = PImage.new("RGB", (gw, gh))

  # TODO: TEST
  nimg.putdata(grid_t[:,:,:3].int().reshape(-1, 3).tolist())
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
