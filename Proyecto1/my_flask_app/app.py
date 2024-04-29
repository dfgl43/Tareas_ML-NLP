from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


app = Flask(__name__)

# Cargar el modelo de regresión lineal
model = joblib.load('./models/linear_regression_model.joblib')


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recoger datos del formulario
        year = request.form.get('Year')
        mileage = request.form.get('Mileage')
        state = request.form.get('State')
        make = request.form.get('Make')
        model_car = request.form.get('Model')

        # Crear DataFrame con los datos recibidos
        input_data = pd.DataFrame({
            'Year': [year],
            'Mileage': [mileage],
            'State': [state],
            'Make': [make],
            'Model': [model_car]
        })

        # Procesamiento de los datos
        input_data['Year'] = pd.to_numeric(input_data['Year'], errors='coerce')
        input_data['Mileage'] = pd.to_numeric(input_data['Mileage'], errors='coerce')

        current_year = 2024
        
        input_data['Age'] = current_year - input_data['Year']
        
        X_lr = input_data[['Mileage', 'Age']]  

        # Hacer la predicción
        prediction = model.predict(X_lr)

        # Devolver la predicción
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
