from constants import CELEBA_DATADIR
import os
import numpy as np

txt_file = os.path.join(CELEBA_DATADIR, 'Anno/list_landmarks_align_celeba.txt')


with open(txt_file) as file:
    for i, line in enumerate(file):
        fields = line.split()
        if len(fields) < 11:
            continue
        image_name = fields[0]
        coords = fields[1:]
        x_coords = coords[0:-1:2]
        y_coords = coords[1:-1:2] + [coords[-1]]

        if i is 3:
            print(image_name)
            print(coords)
            print(x_coords)
            print(y_coords)