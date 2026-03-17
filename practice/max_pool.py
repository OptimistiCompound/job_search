import torch
import torch.nn as nn
from typing import Tuple

class MaxPool2D:
    def __init__(self):
        pass
    
    def generate_regions(self, image: torch.Tensor):
        h, w, _ = image.shape
        new_h, new_w = h // 2, w // 2
        for i in range(new_h):
            for j in range(new_w):
                img = image[2 * i : (2 * i + 2), 2 * j : (2 * j + 2)]
                yield img, i, j

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (h, w, c)
        Returns:
            output: (h // 2, w // 2, c)
        """
        h, w, num_filters = x.shape
        output = torch.zeros((h // 2, w // 2, num_filters))

        for img, i, j in self.generate_regions(x):
            output[i, j] = torch.amax(img, dim=(0, 1)) # calc max on (h, w)

        return output
    
##################################################

class VectorizeMaxPool2D:
    def __init__(self):
        pass
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (h, w, c)
        Returns:
            output: (h // 2, w // 2, c)
        """
        h, w, c = x.shape
        output = torch.zeros(h // 2, w // 2, c)
        
        # reshape: h w c -> h//2 2 w//2 2 c
        x = x.reshape(h // 2, 2, w // 2, 2, c)
        x = x.amax(dim=(1, 3)) # squeeze: h//2 w//2 c
        
        return x