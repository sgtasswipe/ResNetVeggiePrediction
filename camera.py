import cv2
import numpy as np
import tensorflow as tf
from keras.src.applications.convnext import preprocess_input

# Loading my model trained on google colab
try:
    model = tf.keras.models.load_model("veggies.keras")
    print("✅ Model loaded successfully!")

    # Print model architecture to confirm it works
    model.summary()

except Exception as e:
    print("❌ Error loading model:", e)
# Define class labels. Order should match that of the training data folder structure
class_labels = ["Bean", "Bitter_Gourd", "Bottle_Gourd", "Brinjal", "Broccoli", "Cabbage",
                "Capsicum", "Carrot", "Cauliflower", "Cucumber", "Papaya", "Dansker",
                "Pumpkin", "Radish", "Tomato"]


# Initialize webcam

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    if not ret:
        break

    # Preprocess the frame
    img = cv2.resize(frame, (100, 100))
    img = img.astype("float32")
    img = preprocess_input(img)  # Important: Didnt work properly without, as the image must get
                                 # pre processed in the same way as ResNet
    img = np.expand_dims(img, axis=0)

    # Get model prediction
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100  # Convert to percentage
    predicted_label = class_labels[class_index]


    print(f"Raw predictions: {predictions}")

    # Display result on the frame
    text = f"{predicted_label} ({confidence:.2f}%)"
    cv2.putText(frame, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Vegetable Recognition', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
vid.release()
cv2.destroyAllWindows()
