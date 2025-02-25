import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pickle  # Import pickle for saving the model
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

import spotipy
from spotipy.oauth2 import SpotifyOAuth

MOOD_PLAYLISTS = {
    "angry": "https://open.spotify.com/playlist/37i9dQZF1EIfX5bt1426JC",
    "happy": "https://open.spotify.com/playlist/1nAFuLv3VOaQ85D7BlVJj5",
    "sad": "https://open.spotify.com/playlist/37i9dQZF1DXdFesNN9TzXT",
    "neutral": "https://open.spotify.com/playlist/37i9dQZF1EIdzRg9sDFEY3",
    "surprise": "https://open.spotify.com/playlist/5qX8fjRLUukyxNrBSyDSQU",
    "fear": "https://open.spotify.com/playlist/6CZjc29DnoDGFyO7Z1pBma",
    "disgust": "https://open.spotify.com/playlist/09u0M1Q7YaqUUOG4GwX7nD",
}

# Spotify API credentials
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="6361952255fd417cb63c5eb5fd6c297d",
    client_secret="b14bfb0ff973480baf2974aa513dbda0",
    redirect_uri="http://localhost:8888/callback",
    scope="user-modify-playback-state,user-read-playback-state"
))

def play_playlist(emotion):
    playlist_uri = MOOD_PLAYLISTS.get(emotion)
    if not playlist_uri:
        print("No playlist found for the emotion:", emotion)
        return

    devices = sp.devices()
    if not devices['devices']:
        print("No active device found to play music.")
        return

    device_id = devices['devices'][0]['id']  # Use the first active device
    sp.start_playback(device_id=device_id, context_uri=playlist_uri)
    print(f"Playing {emotion} playlist on Spotify.")

# Define file path for the saved model
MODEL_PATH = 'emotion_model.pth'


class FERDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Map emotions to indices
        self.emotion_map = {
            'angry': 0, 'disgust': 1, 'fear': 2,
            'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6
        }

        # Load images and labels
        for emotion, idx in self.emotion_map.items():
            emotion_dir = os.path.join(data_dir, emotion)
            for file in os.listdir(emotion_dir):
                self.images.append(os.path.join(emotion_dir, file))
                self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))

        if self.transform:
            img = self.transform(img)

        return img, label

# Define transformations
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 7)  # 7 classes for FER2013

        # Activation and pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Convolutional layers with ReLU activation and Max pooling
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv3(x))
        x = self.pool(x)

        # Flatten the tensor and pass it through the fully connected layers
        x = x.view(-1, 256 * 6 * 6)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Specify the paths for training and validation sets
train_data = FERDataset(data_dir="archive (16)/train", transform=transform)
val_data = FERDataset(data_dir="archive (16)/test", transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
model = CNN_Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check if the model file exists and load it
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("Loaded pre-trained model.")
else:
    # Train the model if it doesn't exist
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

    # Save the model after training
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model trained and saved.")

# Evaluate the model on the validation set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Validation Accuracy: {100 * correct / total:.2f}%')


def predict_image(image_path, model):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = transforms.ToTensor()(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    emotion = list(train_data.emotion_map.keys())[predicted.item()]
    return emotion

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
print("Press 'q' to close.")

# Update the predict_image function to process image arrays
def predict_image_array(face_array, model):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    face_resized = cv2.resize(face_array, (48, 48))
    face_tensor = transform(face_resized).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(face_tensor)
        _, predicted = torch.max(output, 1)

    emotion = list(train_data.emotion_map.keys())[predicted.item()]
    return emotion

# Flag to check if a playlist is already playing
playlist_playing = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Detect emotion
        face_array = gray[y:y + h, x:x + w]
        emotion = predict_image_array(face_array, model)

        # Play music based on detected emotion only if a playlist is not already playing
        if not playlist_playing:
            play_playlist(emotion)
            playlist_playing = True  # Set flag to prevent further playlist changes

        # Display the emotion on the video feed
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 182, 193), 2)

    # Show the frame
    cv2.imshow("Live Emotion Recognition", frame)

    # Exit on pressing 'q' or after one playlist starts playing
    if cv2.waitKey(1) & 0xFF == ord('q') or playlist_playing:
        break

cap.release()
cv2.destroyAllWindows()
