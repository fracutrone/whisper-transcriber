import os
import tempfile
from flask import Flask, request, jsonify, send_from_directory
from groq import Groq

app = Flask(__name__, static_folder="static")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

ALLOWED_EXTENSIONS = {"mp3", "mp4", "wav", "m4a", "ogg", "webm", "flac"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "Nessun file caricato"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Formato file non supportato"}), 400

    suffix = "." + file.filename.rsplit(".", 1)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(file.filename, audio_file),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
            )
        return jsonify({
            "text": transcription.text,
            "language": getattr(transcription, "language", None),
            "duration": getattr(transcription, "duration", None),
            "filename": file.filename,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
