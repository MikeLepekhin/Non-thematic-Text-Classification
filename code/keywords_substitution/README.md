## Experiment Results

Without POS tagging.

|    | 0%   | 10%  | 25%  | 50%  | 100% |
|----|------|------|------|------|------|
| ru | 0.81 | 0.73 | 0.66 | 0.53 | 0.08 |
| en | 0.71 | 0.48 | 0.41 | 0.32 | 0.13 |

Key point -- even slight replacement of keywords in the test dataset results in significant degradation of model accuracy.

With taking into account the part of speech of every replaced token:


|            | 0%   | 2%   | 5%   | 10%  | 25%  | 50%  | 100% |
|------------|------|------|------|------|------|------|------|
| ru         | 0.8  | 0.51 | 0.5  | 0.47 | 0.4  | 0.26 | 0.09 |
| ru_natasha | 0.8  | 0.5  | 0.48 | 0.46 | 0.39 | 0.31 | 0.07 |
| en         | 0.7  | 0.41 | 0.39 | 0.36 | 0.31 | 0.2  | 0.12 |
