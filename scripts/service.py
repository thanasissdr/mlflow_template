import bentoml
import numpy as np
import pandas as pd
from bentoml.models import BentoModel

BENTO_MODEL_ID = "my_model:latest"


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class IrisClassifier:
    global BENTO_MODEL_ID
    bento_model = BentoModel(BENTO_MODEL_ID)

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

    @bentoml.api(batchable=False)
    def predict(self, dataframe_split: dict) -> np.ndarray:
        df = pd.DataFrame(
            data=dataframe_split["data"], columns=dataframe_split["columns"]
        )
        return self.model.predict(df)
