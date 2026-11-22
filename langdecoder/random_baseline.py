import numpy as np

from .beam import main as beam_main


def uniform_scores(seed):
    rng = np.random.default_rng(seed)
    return lambda sim: rng.random(len(sim)).tolist()


if __name__ == "__main__":
    beam_main(score_fn=uniform_scores, tag="random")
