# GENERAL
This repo serves as a playground for mlflow / bentoml

# INSTRUCTIONS

## Prerequisites
- Docker
- uv 


##  Train/Register models through MLFlow
### Spin up the minio/mlflow server services
```
docker compose -f infrastructure/mlflow/docker-compose.yml --profile default up
```
- Go to localhost:9000 and create a bucket called `mlflow`

```cmd
python -m scripts.train_register_model
```
## Serve model [Mlflow, BentoML]
- Spin up the mlflow server for predictions
```
docker compose -f infrastructure/mlflow/docker-compose.yml --profile serving up
```

### Mlflow

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