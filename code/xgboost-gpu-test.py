# gpu 스펙 보는 커맨드 (터미널)
# nvidia-smi
# 현재 window에서는 gpu 지원되는 xgboost는
# pip를 통해서만 설치가능!
# pip install xgboost

import numpy as np
import xgboost as xgb

xgb_model = xgb.XGBRegressor( # tree_method="gpu_hist" # deprecated
    tree_method="hist",
    device="cuda"
)

X = np.random.rand(50, 2)
y = np.random.randint(2, size=50)

xgb_model.fit(X, y)

xgb_model 