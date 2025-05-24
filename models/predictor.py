import joblib

model = joblib.load('models/prioridad_model.pkl')

def predecir_prioridad(tipo, urgencia):
    tipos = {'reclamo': 0, 'licencia': 1, 'emergencia': 2, 'solicitud': 3}
    tipo_code = tipos.get(tipo, 0)
    resultado = model.predict([[tipo_code, urgencia]])
    return 'Alta' if resultado[0] == 1 else 'Baja'
