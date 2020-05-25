import requests
import pandas as pd
import numpy as np
import os
height_path, weight_path = [os.path.join(os.getcwd() + f"/{i}") for i in ["height.npy", "weight.npy"]]

if not (os.path.exists(weight_path) and os.path.exists(height_path)):
    website = "http://socr.ucla.edu/docs/resources/SOCR_Data/SOCR_Data_Dinov_020108_HeightsWeights.html"

    page = requests.get(website)
    df = pd.read_html(page.content, index_col=0)
    df = pd.concat(df, axis=1)
    print("Generated DataFrame...")
    df.columns = ["Height", "Weight"]
    df.drop(df.index[0], inplace=True)
    df = df.applymap(float)
    print("Formatted DataFrame...")
    df = (df-df.mean())/df.std()
    print("Normalized DataFrame...")
    heights, weights = df.Height.values, df.Weight.values
    np.save(height_path, heights)
    np.save(weight_path, weights)
    print(f"Saved .npy files at {height_path} and at {weight_path}")
else:
    print(f"Data already collected at {height_path} and at {weight_path}")


