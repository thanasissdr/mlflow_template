# GENERAL
This repo serves as a playground for mlflow / bentoml

# INSTRUCTIONS

## Prerequisites
- Docker
- uv 


### Docker
```cmd
docker network create shared_network
```

- Train/Register models through MLFlow
```
docker compose -f docker-compose.main.yml up
./.venv/Scripts/activate

python -m scripts.train_register_model
```
## Serve model

### Mlflow
- Splin up the server for predictions
```
docker compose -f infrastructure/docker-compose.mlflow-server.yml up
```
Open postman and run the following `POST` request: 
http://localhost:5001/invocations with the following payload:

```json
{
  "dataframe_split": {
    "columns": [
      "sepal length (cm)",
      "sepal width (cm)",
      "petal length (cm)",
      "petal width (cm)"
    ],
    "data": [
      [
        5.1,
        3.5,
        1.4,
        0.2
      ]
    ]
  }
}
```


### BentoML
- Make sure that an mlflow model is registered into the BentoML store
```cmd
python -m scripts.import_mlflow_into_bentoml
```


- Spin up the server for predictions
```cmd
docker compose -f infrastructure/bentoml_server/docker-compose.yml up --build
```

Open postman and run the following `POST` request: 
http://localhost:3000/predict with the following payload:
```json
{
  "dataframe_split": {
    "columns": [
      "sepal length (cm)",
      "sepal width (cm)",
      "petal length (cm)",
      "petal width (cm)"
    ],
    "data": [
      [
        5.1,
        3.5,
        1.4,
        0.2
      ]
    ]
  }
}
```