import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# --- Spotify Setup ---
MOOD_PLAYLISTS = {
    "angry": "spotify:playlist:37i9dQZF1EIfX5bt1426JC",
    "happy": "spotify:playlist:1nAFuLv3VOaQ85D7BlVJj5",
    "sad": "spotify:playlist:37i9dQZF1DXdFesNN9TzXT",
    "neutral": "spotify:playlist:37i9dQZF1EIdzRg9sDFEY3",
    "surprise": "spotify:playlist:5qX8fjRLUukyxNrBSyDSQU",
    "fear": "spotify:playlist:6CZjc29DnoDGFyO7Z1pBma",
    "disgust": "spotify:playlist:09u0M1Q7YaqUUOG4GwX7nD",
}
#fill on your own

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="",
    client_secret="",
    redirect_uri="",
    scope=""
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

    device_id = devices['devices'][0]['id']
    sp.start_playback(device_id=device_id, context_uri=playlist_uri)
    print(f"Playing {emotion} playlist on Spotify.")

# --- Dataset ---
MODEL_PATH = 'emotion_model.pth'

class FERDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.emotion_map = {
            'angry': 0, 'disgust': 1, 'fear': 2,
            'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6
        }
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

# --- CNN Model ---
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x

# --- Transform ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- Load Data ---
train_data = FERDataset(data_dir="archive (20)/train", transform=transform)
val_data = FERDataset(data_dir="archive (20)/test", transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# --- Train or Load Model ---
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("Loaded pre-trained model.")
else:
    print("Training model...")
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}/10 - Loss: {running_loss:.4f}, Accuracy: {100*correct/total:.2f}%")
    torch.save(model.state_dict(), MODEL_PATH)

# --- Evaluate ---
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Validation Accuracy: {100 * correct / total:.2f}%')

# --- Predict Image from Webcam ---
def predict_image_array(face_array, model):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    face_resized = cv2.resize(face_array, (48, 48))
    face_tensor = transform(face_resized).unsqueeze(0)
    with torch.no_grad():
        output = model(face_tensor)
        _, predicted = torch.max(output, 1)
    emotion = list(train_data.emotion_map.keys())[predicted.item()]
    return emotion

# --- Webcam Emotion Detection ---
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
print("Press 'q' to quit or wait 20 seconds...")

start_time = time.time()
min_duration = 20
playlist_playing = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_array = gray[y:y + h, x:x + w]
        emotion = predict_image_array(face_array, model)

        if not playlist_playing:
            play_playlist(emotion)
            playlist_playing = True

        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 182, 193), 2)

    cv2.imshow("Live Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif time.time() - start_time >= min_duration:
        break

cap.release()
cv2.destroyAllWindows()
