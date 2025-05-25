# predictor/views.py
from django.http import JsonResponse
import joblib
import numpy as np
import os

def predict_mood(request):
    if request.method == 'GET':
        try:
            sleep = float(request.GET.get('sleep'))
            screen = float(request.GET.get('screen'))
            social = float(request.GET.get('social'))
            exercise = request.GET.get('exercise')
            stress = request.GET.get('stress')
            diet = request.GET.get('diet')

            encoders = joblib.load('C:\\Users\\danie\\Documents\\script\\Django\\predictor\\encoders.pkl')
            input_vector = [
                sleep,
                screen,
                social,
                encoders['Exercise Level'].transform([exercise])[0],
                encoders['Stress Level'].transform([stress])[0],
                encoders['Diet Type'].transform([diet])[0]
            ]
            for name, encoder in encoders.items():
                print(f"{name}: {list(encoder.classes_)}")

            predictions = {}
            for model_name in os.listdir('predictor/models'):
                if model_name.endswith('.pkl') and model_name not in ['encoders.pkl', 'accuracies.pkl']:
                    model = joblib.load(f'predictor/models/{model_name}')
                    pred = model.predict([input_vector])[0]
                    predictions[model_name.replace('.pkl', '')] = pred

            accuracies = joblib.load('predictor/models/accuracies.pkl')
            formatted_accuracies = {
                model: f"{accuracy * 100:.1f}%" for model, accuracy in accuracies.items()
            }


            return JsonResponse({
                'predictions': predictions,
                'accuracies': formatted_accuracies
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
