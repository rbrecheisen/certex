import h5py
import numpy as np


file_path = "/mnt/localscratch/maastro/Ralph/bodycomposition/data_mega.h5"

with h5py.File(file_path, "r") as f:

    a_group_key2 = list(f.keys())

    images = np.zeros([512, 512, len(a_group_key2)])
    labels = np.zeros([512, 512, len(a_group_key2)]).astype(np.int16)
    for i, name in enumerate(a_group_key2):
        image = f[str(name)]['image'][()]
        label = f[str(name)]['labels'][()]
        print(np.unique(label))
        images[:, :, i] = image[:, :]
        labels[:, :, i] = label
