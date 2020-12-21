import numpy as np
import math
from sklearn.preprocessing import KBinsDiscretizer

def KBinsDiscretize(data_x, n_bins=0, alpha=3.322, encode="ordinal", strategy="uniform"):
    """
        
    """
   # Makes n_bins optional, calculates optimal n_bins by default
   # Sturges Rule - num_bins = 1 + 3.322 * log_10(num_inputs)
   if n_bins == 0:
      n_bins = math.floor(1 + alpha * math.log10(data_x.shape[0], 2))
      if n_bins > 256:
         n_bins = 256 # cap n_bins at 256
   binner = KBinsDiscretizer(n_bins, encode="ordinal", strategy="uniform")
   binner.fit(data_x)
   binned_x = binner.transform(data_x)
   return binned_x
