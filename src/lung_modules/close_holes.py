import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class CloseHoles:
    arr_2d: np.ndarray
    def run(self, option=True):
        if option:
            self.arr_2d = 1 - self.arr_2d
        components, matrix, stats, _ = cv2.connectedComponentsWithStats(
            self.arr_2d, connectivity=8
        )
        sizes = stats[1:, -1]
        components = components - 1
        min_size = int(self.arr_2d.shape[0] / 3.4)
        img2 = np.zeros((matrix.shape))
        for j in range(0, components):
            if sizes[j] >= min_size:
                img2[matrix == j + 1] = 1
        if option:
            img2 = 1 - img2
        return img2