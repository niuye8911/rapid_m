import pandas as pd

BASE_DIR = '/home/liuliu/Research/rapid_m_backend_server/testData/appself/'
APPS = ['ferret','swaptions','bodytrack','facedetect','nn']

HEADER_DONE = False

for app in APPS:
    p_small = pd.read_csv(BASE_DIR+app+"-perf-small.csv")
    p_big = pd.read_csv(BASE_DIR+app+"-perf-big.csv")
    p_combined = pd.concat([p_small,p_big])
    p_combined.to_csv(app+"-perf-combined.csv",index=False)
