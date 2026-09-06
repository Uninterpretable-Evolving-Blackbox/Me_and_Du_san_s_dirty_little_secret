import sys
import torch

src, dst = sys.argv[1], sys.argv[2]
sd = torch.load(src, map_location="cpu")
torch.save({k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}, dst)
