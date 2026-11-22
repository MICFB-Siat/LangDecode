import os
from pathlib import Path

DATA = Path(os.environ.get("LANGDECODER_DATA", "data"))
OUT = Path(os.environ.get("LANGDECODER_OUT", "outputs"))

LPPC = DATA / "LPPC-fMRI"
CHEN = DATA / "Chen2026"
SMN4LANG = DATA / "SMN4Lang"
NEBULA = DATA / "NEBULA101"
TRANSCRIPTS = DATA / "transcripts"
ATLAS = DATA / "atlas" / "atlas59412.npz"
SPHERES = DATA / "atlas" / "spheres"
G1 = DATA / "atlas" / "margulies2016_g1_fslr32k.npy"
BLOOM = DATA / "models" / "bloom-1b1"
BGE_M3 = DATA / "models" / "bge-m3"

N_CORTEX = 59_412
LANGS = ("CN", "EN", "FR")
