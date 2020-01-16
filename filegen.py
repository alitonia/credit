import pandas as pd

import math
import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def normalize(s):
    z = []
    for x in s:
        z.append(sigmoid(x))
    return z


df = pd.read_csv(
    '/home/alitonia/developments/credit_score_predict/sample_submission.csv')

mu, sigma = 0.7, 1.1  # mean and standard deviation
s = np.random.normal(mu, sigma, len(df))

s = normalize(s)

for i in range(len(df)):
    df.set_value(i, 'label', s[i])

export_csv = df.to_csv(
    r'/home/alitonia/developments/credit_score_predict/modified_col.csv', index=False, header=['id', 'label'], sep=',')
