import glob
import numpy as np
import nibabel as nib
from langdecoder.paths import OUT, LPPC

F5 = OUT / "transfer"
RSA = OUT / "rsa"
NEB = OUT / "nebula"
MODELS = OUT / "models"
FRACTIONS = {"lppc": {30: 2, 50: 4, 70: 6}, "chen": {30: 3, 50: 4, 70: 6}}


def cifti_axis():

    f = sorted(glob.glob(str(LPPC / "sub-*" / "*.dtseries.nii")))[0]
    return nib.load(f).header.get_axis(1)


def thirds(v):

    return np.digitize(v, np.nanquantile(v, [1 / 3, 2 / 3]))
