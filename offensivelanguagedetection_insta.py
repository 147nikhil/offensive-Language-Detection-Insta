
# Offensive Language Detection Model

# Step 1: Install Necessary Libraries
#!pip install pandas numpy scikit-learn tensorflow plotly

# Step 2: Import Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Step 3: Load Dataset
data_url = 'https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv'
df = pd.read_csv(data_url)
df.head()

# Step 4: Data Preprocessing
## Combine 'hate_speech', 'offensive_language' into one 'label' column
def combine_labels(row):
    if row['hate_speech'] > 0:
        return 'Hate Speech'
    elif row['offensive_language'] > 0:
        return 'Offensive'
    else:
        return 'Neutral'

## Apply the function
df['label'] = df.apply(combine_labels, axis=1)

## Tokenize and vectorize text
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    return text

df['processed_text'] = df['tweet'].apply(preprocess_text)

## Encode Labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# Step 5: Split Data
X = df['processed_text']
y = df['label_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

## Tokenization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_WORDS = 10000
MAX_LEN = 100
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN)

# Step 6: Build Model
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.6, recurrent_dropout=0.6)),
    Dropout(0.6),
    Bidirectional(LSTM(32, dropout=0.6, recurrent_dropout=0.6)),
    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(3, activation='softmax') # 3 classes: Hate Speech, Offensive, Neutral
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 7: Train Model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

def lr_scheduler(epoch, lr):
    return float(lr * tf.math.exp(-0.1))

lr_callback = LearningRateScheduler(lr_scheduler)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train_seq, y_train,
                    validation_split=0.2,
                    epochs=20,
                    batch_size=32,
                    callbacks=[early_stop, lr_callback])

# Step 8: Evaluate Model
## Evaluate on test data
eval_result = model.evaluate(X_test_seq, y_test)
print(f"Test Loss: {eval_result[0]}, Test Accuracy: {eval_result[1]}")

## Classification Report
y_pred = model.predict(X_test_seq)
y_pred_classes = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_classes, target_names=le.classes_))

# Step 9: Visualize Results using Plotly
## Plot Accuracy and Loss
fig = go.Figure()
fig.add_trace(go.Scatter(y=history.history['accuracy'], mode='lines', name='Train Accuracy'))
fig.add_trace(go.Scatter(y=history.history['val_accuracy'], mode='lines', name='Validation Accuracy'))
fig.update_layout(title='Model Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy')
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Train Loss'))
fig.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation Loss'))
fig.update_layout(title='Model Loss', xaxis_title='Epoch', yaxis_title='Loss')
fig.show()

## Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)
fig = px.imshow(conf_matrix, text_auto=True, color_continuous_scale='Viridis', title='Confusion Matrix')
fig.show()

# Step 10: Save the model
model.save('offensive_language_detection_model.h5')

# Save the trained model as .keras file
model.save('offensive_language_detection_model.keras')

# User input for text classification
def classify_text(input_text, model, tokenizer, max_len):
    # Preprocess the input text
    input_text_processed = ''.join([char for char in input_text.lower() if char.isalnum() or char.isspace()])
    input_sequence = tokenizer.texts_to_sequences([input_text_processed])
    input_padded = pad_sequences(input_sequence, maxlen=max_len)

    # Predict using the model
    prediction = model.predict(input_padded)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Map predicted class back to label
    label_mapping = {0: 'Hate Speech', 1: 'Neutral', 2: 'Offensive'}
    return label_mapping[predicted_class]

# Input text from user
user_input = input("Enter text to classify: ")
predicted_label = classify_text(user_input, model, tokenizer, MAX_LEN)

print(f"The entered text is classified as: {predicted_label}")



