# 🎵 Moodify – AI-Powered Mood-Based Music Recommendation 🎧

**Moodify** is an end-to-end emotion-based music recommender system that curates real-time **Spotify playlists** based on your **facial expressions**. Developed as part of the **C4 Hackathon**, this project bridges **deep learning**, **music**, and **mental wellness** through a seamless full-stack solution.

---

## 🚀 Features

- 🔍 **Real-Time Emotion Detection** via webcam
- 🧠 **Swin-Tiny Vision Transformer** fine-tuned on **FER2013**
- 💡 **Gemini API** integration for intelligent interaction
- 🎧 **Spotify SDK** to dynamically update your playlist
- ⚛️ **React.js Frontend** for live video input and user interaction
- 🌐 **Flask Backend** for model inference and API management
- 🎯 Use-case Focus: Mental health support for users with **Anxiety, ADHD**, or emotional regulation needs

---

## 🛠️ Tech Stack

| Layer       | Technology                         |
|------------|-------------------------------------|
| Frontend   | React.js                            |
| Backend    | Flask (Python)                      |
| ML Model   | Swin-Tiny Vision Transformer (ViT)  |
| Dataset    | FER2013                             |
| APIs       | Gemini API, Spotify SDK             |

---

## 🧪 Demo Flow

1. **User accesses the app** – camera opens via the React frontend.
2. **Facial expression captured** – passed to Flask backend.
3. **Emotion inferred** – using fine-tuned Swin-Tiny ViT model.
4. **Playlist updated** – Spotify SDK serves a playlist matching the mood.
5. **User listens** – curated songs for joy, sadness, calm, or energy.
