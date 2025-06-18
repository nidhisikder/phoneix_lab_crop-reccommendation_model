import gradio as gr
import joblib
import os
import numpy as np
model=joblib.load("model.pkl")
# Values taken through real-time input: 
def predict(N, P, K, Temp, Humidity, pH, Rainfall):
    input_data=np.array([[N, P, K, Temp, Humidity, pH, Rainfall]])
    predicted_crop = model.predict(input_data)
    return f"Recommended Crop:{ predicted_crop[0]}"
demo=gr.Interface(fn=predict,inputs=[gr.Number(label="N"),
                                     gr.Number(label="P"),
                                     gr.Number(label="K"),
                                     gr.Number(label="Temp"),
                                     gr.Number(label="Humidity"),
                                     gr.Number(label="pH"),
                                     gr.Number(label="Rainfall")],outputs="text")
if(__name__=="__main__"):
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)
