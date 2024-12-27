# GENERAL
This repo serves as a playground for mlflow

# INSTRUCTIONS

## Prerequisites
- Docker
- uv 


### Docker
```cmd
docker network create shared_network
```

- Train/Register models
```
docker compose -f docker-compose.main.yml up
./.venv/Scripts/activate
python main.py
```
- Serve model
```
docker compose -f docker-compose.server.yml up
```
Open postman and run the following `POST` request: 
http://localhost:5001/invocations
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