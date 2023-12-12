import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained deep learning model for plant disease detection
model = load_model('plant_disease_model.h5')

# Function to detect and classify plant diseases in real-time
def detect_and_treat_disease():
    # Open a connection to the video camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the video feed
        ret, frame = cap.read()

        # Preprocess the frame for deep learning model
        resized_frame = cv2.resize(frame, (224, 224))
        normalized_frame = resized_frame / 255.0
        input_frame = np.expand_dims(normalized_frame, axis=0)

        # Predict disease using the pre-trained deep learning model
        predictions = model.predict(input_frame)
        predicted_class = np.argmax(predictions)

        # Administer treatment based on the detected disease
        administer_treatment(predicted_class)

        # Display the original frame with disease classification
        cv2.putText(frame, f"Disease: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Plant Disease Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video feed and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Function to administer treatment based on the detected disease
def administer_treatment(disease_class):
    # In a real-world scenario, implement logic to administer treatment
    # This could involve dispensing a specific pesticide or nutrient solution

    # For this example, print a message indicating the treatment
    treatments = {
        0: "No disease detected.",
        1: "Fungal infection detected. Apply fungicide.",
        2: "Bacterial infection detected. Apply antibacterial solution.",
        # Add more treatments for additional classes
    }

    print("Treatment:", treatments.get(disease_class, "Unknown disease"))

# Run the real-time detection and treatment
if __name__ == "__main__":
    detect_and_treat_disease()

