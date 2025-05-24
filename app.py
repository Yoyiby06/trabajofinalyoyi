from flask import Flask, render_template, request
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
import csv

from models.predictor import predecir_prioridad


app = Flask(__name__)

MODEL_FILE = "modelo.pkl"
DATA_FILE = "tramites.csv"

# Entrenar modelo si no existe
def entrenar_modelo():
    data = pd.DataFrame({
        'tipo': ['reclamo', 'licencia', 'emergencia', 'solicitud'],
        'urgencia': [2, 3, 5, 1],
        'prioridad': [1, 2, 3, 0]
    })
    data['tipo'] = data['tipo'].map({'reclamo': 0, 'licencia': 1, 'emergencia': 2, 'solicitud': 3})
    X = data[['tipo', 'urgencia']]
    y = data['prioridad']
    modelo = DecisionTreeClassifier()
    modelo.fit(X, y)
    joblib.dump(modelo, MODEL_FILE)

if not os.path.exists(MODEL_FILE):
    entrenar_modelo()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        nombre = request.form["nombre"]
        correo = request.form["correo"]
        tipo = request.form["tipo"]
        urgencia = int(request.form["urgencia"])

        # Predecir prioridad
        prioridad = predecir_prioridad(tipo, urgencia)

        # Guardar en CSV
        campos = ['nombre', 'correo', 'tipo', 'urgencia', 'prioridad']
        nueva_fila = {
            'nombre': nombre,
            'correo': correo,
            'tipo': tipo,
            'urgencia': urgencia,
            'prioridad': prioridad
        }

        archivo_existe = os.path.exists(DATA_FILE)

        with open(DATA_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            if not archivo_existe:
                writer.writeheader()
            writer.writerow(nueva_fila)

        # Mensaje opcional de confirmación (podrías redirigir o mostrar en el template)
        return f"Trámite registrado con prioridad: {prioridad}"

    return render_template("index.html")


@app.route('/estado', methods=['GET', 'POST'])
def estado():
    tramites = []
    mensaje = ''
    if request.method == 'POST':
        correo = request.form['correo']
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for fila in reader:
                    if fila['correo'] == correo:
                        tramites.append(fila)
            if not tramites:
                mensaje = "No se encontraron trámites con ese correo."
        else:
            mensaje = "No hay datos registrados aún."

    return render_template("estado.html", tramites=tramites, mensaje=mensaje)

if __name__ == '__main__':
    app.run(debug=True)
