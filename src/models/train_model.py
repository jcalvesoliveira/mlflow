import argparse
import math

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost


import mlflow


def parse_args():
    parser = argparse.ArgumentParser(description="House Prices ML")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.3,
        help="learning rate para atualizar o tamanho do passo em cada passo do boosting (default: 0.3)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="profundidade maxima das arvores. (default: 100)",
    )
    return parser.parse_args()


def main():
    # parse command-line arguments
    args = parse_args()

    # prepare train and test data
    df = pd.read_csv('data/processed/casas.csv')
    X = df.drop('preco', axis=1)
    y = df['preco']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # enable auto logging
    mlflow.set_tracking_uri("http://localhost:5000") ## incluir para modelos registrados apenas
    mlflow.set_experiment('house-prices-script')
    mlflow.xgboost.autolog()

    dtrain = xgboost.DMatrix(X_train, label=y_train)
    dtest = xgboost.DMatrix(X_test, label=y_test)

    with mlflow.start_run():
        # ajuste do modelo
        xgb_params = {
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "seed": 42,
        }
        xgb = xgboost.train(xgb_params, dtrain, evals=[(dtrain, "train")])

        # avaliacao do modelo
        xgb_predicted = xgb.predict(dtest)
        mse = mean_squared_error(y_test,xgb_predicted)
        rmse = math.sqrt(mse)
        r2 = r2_score(y_test,xgb_predicted)

        # enviando metricas para MLFlow
        mlflow.log_metrics({"mse": mse, "rmse": rmse, "r2": r2})


if __name__ == "__main__":
    main()