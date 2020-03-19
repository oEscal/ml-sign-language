import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_file(path_file: str) -> np.ndarray:
   data = pd.read_csv(path_file, header=None, skiprows=1)
   return data.values


def represent_data_graphically(data: np.ndarray, file_save: str, rows: int = 10, cols: int = 10):
   data_image_size = int(math.sqrt(len(data[0, 1:])))
   data_len = len(data)

   fig, axis = plt.subplots(rows, cols, figsize=(data_image_size, data_image_size))
   for row in range(rows):
      for col in range(cols):
         example_id = np.random.randint(data_len)
         axis[row, col].imshow(data[example_id, 1:].reshape(data_image_size, data_image_size, order="F"))
   plt.savefig(file_save)
