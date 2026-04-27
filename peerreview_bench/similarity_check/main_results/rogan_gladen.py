"""
Rogan-Gladen prevalence correction for similarity judge outputs.

Calibrated on the 164-pair eval set using GPT-5.4 as the binary similarity judge:
  - 70 true-similar pairs: 61 correctly classified → sensitivity = 61/70 = 0.871
  - 94 true-not-similar pairs: 91 correctly classified → specificity = 91/94 = 0.968

The correction formula:
    π_corrected = (π_observed + Spec - 1) / (Sens + Spec - 1)

When used inside a bootstrap loop, sens and spec are themselves resampled
from Binomial distributions to propagate calibration uncertainty:
    Sens ~ Binomial(N_similar=70, p=0.871) / 70
    Spec ~ Binomial(N_not_similar=94, p=0.968) / 94
"""

import numpy as np

# Calibration parameters from the 164-pair eval set
N_SIMILAR = 70          # number of true-similar pairs in eval set
TP = 61                 # correctly classified similar
SENSITIVITY = TP / N_SIMILAR   # 0.871

N_NOT_SIMILAR = 94      # number of true-not-similar pairs in eval set
TN = 91                 # correctly classified not-similar
SPECIFICITY = TN / N_NOT_SIMILAR   # 0.968


def rogan_gladen_correct(observed_prev, sens=SENSITIVITY, spec=SPECIFICITY):
    """Apply Rogan-Gladen correction to an observed binary prevalence.

    Returns corrected prevalence, clipped to [0, 1].
    """
    denom = sens + spec - 1
    if abs(denom) < 1e-10:
        return observed_prev
    corrected = (observed_prev + spec - 1) / denom
    return max(0.0, min(1.0, corrected))


def resample_sens_spec(rng):
    """Resample sensitivity and specificity from their binomial distributions.

    Call this once per bootstrap iteration to propagate calibration uncertainty.
    """
    sens = rng.binomial(N_SIMILAR, SENSITIVITY) / N_SIMILAR
    spec = rng.binomial(N_NOT_SIMILAR, SPECIFICITY) / N_NOT_SIMILAR
    return sens, spec
