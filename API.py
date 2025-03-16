import torch
import timm
import torchvision.transforms as transforms
import base64
import io
from PIL import Image
from flask import Flask, request, jsonify
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import google.generativeai as genai
import os
from flask_cors import CORS


app = Flask(__name__)

CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
GEM_API_KEY = os.getenv("GEM_API_KEY")
REDIRECT_URI = os.getenv("REDIRECT_URI")
SCOPE_MODIFY_PLAYLIST = os.getenv("SCOPE_MODIFY_PLAYLIST")
SCOPE_USER_READ = os.getenv("SCOPE_USER_READ")

genai.configure(api_key=GEM_API_KEY)

sp_modify_Playlist = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                               client_secret=CLIENT_SECRET,
                                               redirect_uri=REDIRECT_URI,
                                               scope=SCOPE_MODIFY_PLAYLIST
                                               ))

sp_user_read = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                               client_secret=CLIENT_SECRET,
                                               redirect_uri=REDIRECT_URI,
                                               scope=SCOPE_USER_READ))

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=7)
model.load_state_dict(torch.load("swin_tiny_fer2013.pth", map_location=device))
model.to(device)
model.eval()

# Define Emotion Labels (FER2013 Classes)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Define Image Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to RGB
    transforms.Resize((224, 224)),  # Resize for Swin Transformer
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize
])

playlist_recommendations = {
    "Happy": "Good Vibes Only",
    "Angry": "No Chill",
    "Disgust": "Cringe Control",
    "Fear": "Heart Racing Hits",
    "Neutral": "Just Vibing",
    "Sad": "Lost In My Feels",
    "Surprise": "Plot Twist!",
}

def get_song_recommendations(mood, artists):
    model = genai.GenerativeModel("gemini-1.5-pro")

    prompt = f"These are my favourite artists: {artists}. " \
             f"I want you to recommend five songs that are {mood} from these 10 artists. " \
             f"Please respond with a list of only song names and artist names in the format 'song name - artist'. " \
             f"I don't want any extra information"

    response = model.generate_content(prompt)

    if response and response.text:
        song_artist_list = response.text.strip().split("\n")

        song_artist_dict = {}
        for song_artist in song_artist_list:
            song_artist = song_artist.lstrip('*').strip()
            song_artist = song_artist.lstrip(' 0123456789').strip()

            try:
                song, artist = song_artist.split(" - ")
                song_artist_dict[song.strip()] = artist.strip()
            except ValueError:
                continue

        return song_artist_dict
    return {"No recommendations found.": "No artist found."}

def search_songs(song_artist_dict):
    track_uris = []
    for song, artist in song_artist_dict.items():
        query = f"{song} {artist}"

        result = sp_modify_Playlist.search(q=query, limit=1)
        if result['tracks']['items']:
            track_uris.append(result['tracks']['items'][0]['uri'])
        else:
            print(f"Song '{song}' by {artist} not found.")
    return track_uris


def create_playlist(name, description="My Spotify Playlist"):

    user_id = sp_modify_Playlist.me()['id']
    playlist = sp_modify_Playlist.user_playlist_create(user=user_id, name=name, public=False, description=description)
    return playlist['id']

def add_songs_to_playlist(playlist_id, track_uris):

    sp_modify_Playlist.playlist_add_items(playlist_id=playlist_id, items=track_uris)

# API Endpoint
@app.route("/predict", methods=["POST","OPTIONS"])
def predict():

    if request.method == "OPTIONS":
        response = jsonify({"message": "CORS Preflight OK"})
        response.headers.add("Access-Control-Allow-Origin", "http://localhost:3000")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        return response, 200

    try:
        #Step 1: Receive Image from Frontend
        data = request.json["image"]

        #Step 2: Decode Base64 Image
        image_data = base64.b64decode(data.split(",")[1])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")  # Convert to RGB

        #Step 3: Preprocess Image
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to GPU/CPU

        #Step 4: Model Inference
        with torch.no_grad():
            outputs = model(image)
            predicted_class = torch.argmax(outputs, dim=1).item()

        #Step 5: Get Predicted Mood Label
        predicted_mood = emotion_labels[predicted_class]

        followed_artists = sp_user_read.current_user_followed_artists(limit=10)
        artist_names = []
        for artist in followed_artists['artists']['items']:
            artist_names.append(artist['name'])

        print(artist_names)

        mood = predicted_mood

        print(mood)

        track_uris = search_songs(get_song_recommendations(mood, artist_names))

        if mood in playlist_recommendations:

            playlist_name = playlist_recommendations[mood]
        else:

            playlist_name = "i dont even know tbh"

        if track_uris:
            playlist_id = create_playlist(playlist_name)
            add_songs_to_playlist(playlist_id, track_uris)
            print("Added songs to playlist")
            return jsonify("Success", f"Playlist '{playlist_name}' created and songs added!")

        else:
            print("Nothing found")
            return jsonify("Success", f"No valid songs found.")

    except Exception as e:
        print("Error")
        return jsonify({"error": str(e)}), 500

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


if __name__ == "__main__":
    app.run(debug=True)
    CORS(app)