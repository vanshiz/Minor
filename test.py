from tensorflow.keras.models import load_model

try:
    model_first = load_model('first.h5')
    model_second = load_model('second.h5')
    print("Models loaded successfully!")
except Exception as e:
    print("Error loading models:", e)
