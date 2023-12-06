import pandas as pd
import numpy as np
from example_estimator import get_estimator

pipeline = get_estimator()

X_test = pd.read_parquet("../input/mdsb-2023/final_test.parquet")

y_pred = pipeline.predict(X_test)
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("submission.csv", index=False)
