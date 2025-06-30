# 🎵 Emotion-Based Spotify Music Player 🎭

This project detects human emotions in real-time using webcam input, predicts the emotion using a Convolutional Neural Network (CNN) trained on facial expressions, and plays an emotion-specific Spotify playlist automatically.

---

## 📌 Features

- Real-time facial emotion detection using OpenCV  
- Emotion classification using a deep CNN model trained on the FER-2013 dataset  
- Integration with Spotify API to play music based on detected emotion  
- Supports seven emotions: `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, and `Neutral`

---

## 🧠 Emotion to Playlist Mapping

| Emotion   | Spotify Playlist |
|-----------|------------------|
| Angry     | 🎧 `37i9dQZF1EIfX5bt1426JC` |
| Happy     | 🎧 `1nAFuLv3VOaQ85D7BlVJj5` |
| Sad       | 🎧 `37i9dQZF1DXdFesNN9TzXT` |
| Neutral   | 🎧 `37i9dQZF1EIdzRg9sDFEY3` |
| Surprise  | 🎧 `5qX8fjRLUukyxNrBSyDSQU` |
| Fear      | 🎧 `6CZjc29DnoDGFyO7Z1pBma` |
| Disgust   | 🎧 `09u0M1Q7YaqUUOG4GwX7nD` |

> You can modify the `MOOD_PLAYLISTS` dictionary in the code with your own Spotify playlists.

---

Replace these in your code:

python
Copy
Edit
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    redirect_uri="YOUR_REDIRECT_URI",
    scope="user-read-playback-state,user-modify-playback-state"
))
🖼️ Live Demo Instructions
Your webcam opens and detects a face.

Emotion is predicted.

Spotify plays a corresponding playlist based on the detected emotion.

You can press q or wait 20 seconds to terminate.

