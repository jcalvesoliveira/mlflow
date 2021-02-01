import mlflow
#alterar para modelo em producao ao final
logged_model = 'file:///mnt/d/Google%20Drive/Jobs/Aulas/Alura/curso-mlflow/mlruns/0/51b8a608b9354411877fa4ad850ece8c/artifacts/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
data = pd.read_csv('../../data/processed/casas.csv')
predicted = loaded_model.predict(data)

import numpy as np
np.savetxt('precos.csv',predicted,fmt='%.0i')