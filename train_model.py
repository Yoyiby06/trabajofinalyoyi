import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Datos simulados
data = pd.DataFrame({
    'tipo': ['reclamo', 'licencia', 'emergencia', 'reclamo', 'solicitud'],
    'urgencia': [3, 1, 5, 4, 2],
    'prioridad': [1, 0, 1, 1, 0]
})

data['tipo'] = data['tipo'].astype('category').cat.codes

X = data[['tipo', 'urgencia']]
y = data['prioridad']

clf = RandomForestClassifier()
clf.fit(X, y)

joblib.dump(clf, 'models/prioridad_model.pkl')
