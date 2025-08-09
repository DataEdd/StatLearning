import numpy as np
import pandas as pd
from predict import SRegPred 
from ISLP import load_data

Boston = load_data("Boston")

model1 = SRegPred(feature=Boston['lstat'],target=Boston['medv'])
print(model1.predict())