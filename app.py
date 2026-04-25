from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import librosa
import numpy as np
import os
import uuid
import random
import matplotlib.pyplot as plt
import pandas as pd
import speech_recognition as sr
from textblob import TextBlob
import sys  # 🔥 ADDED

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

SESSIONS_DIR = "sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

session_store = {}

# ---------------- QUESTIONS ----------------
@app.route("/api/questions", methods=["GET"])
def get_questions():
    hr = [
        "Tell me about yourself",
        "What are your strengths?",
        "How do you handle stress?",
        "Why do you want to work here?",
        "How do you deal with criticism?"
    ]

    tech = [
        "Explain your project",
        "What is overfitting?",
        "Difference between frontend and backend?",
        "What is REST API?",
        "Explain neural networks briefly"
    ]

    personal = [
        "Describe a failure",
        "What motivates you?",
        "Describe leadership experience.",
        "How do you manage time?",
        "Long term goals?"
    ]

    selected = random.sample(hr, 2) + random.sample(tech, 2) + random.sample(personal, 1)
    random.shuffle(selected)
    return jsonify(selected)

# ---------------- START ----------------
@app.route("/api/start_interview", methods=["POST"])
def start_interview():
    session_id = uuid.uuid4().hex
    session_store[session_id] = {
        "answers": [],
        "baseline_pitch": None
    }

    questions = get_questions().get_json()

    return jsonify({
        "session_id": session_id,
        "questions": questions
    })

# ---------------- EVALUATE ----------------
@app.route("/api/evaluate", methods=["POST"])
def evaluate_answer():

    print("🔥 EVALUATE API HIT", flush=True)  # 🔥 ADDED

    if "file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["file"]
    session_id = request.form.get("session_id")
    question = request.form.get("question", "")

    if not session_id or session_id not in session_store:
        return jsonify({"error": "Invalid session ID"}), 400

    tmp_webm = f"tmp_{uuid.uuid4().hex}.webm"
    tmp_wav = f"tmp_{uuid.uuid4().hex}.wav"

    audio_file.save(tmp_webm)
    os.system(f'ffmpeg -y -i "{tmp_webm}" -ac 1 -ar 16000 "{tmp_wav}"')

    emotion = "Error"
    confidence = 0
    feedback = "Processing error"
    sentiment = "neutral"

    try:
        y, sample_rate = librosa.load(tmp_wav, sr=16000)
        duration = librosa.get_duration(y=y, sr=sample_rate)

        if duration < 1.0:
            emotion = "No Speech"
            confidence = 0
            feedback = "Audio too short. Please speak clearly."

        else:
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)

            rms = float(np.sqrt(np.mean(y_trimmed ** 2)))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y_trimmed)))
            speech_duration = len(y_trimmed) / sample_rate

            print("RMS:", rms, flush=True)
            print("ZCR:", zcr, flush=True)
            print("Speech Duration:", speech_duration, flush=True)

            if (
                speech_duration < 0.8 or
                rms < 0.015 or
                zcr < 0.01
            ):
                emotion = "No Speech"
                confidence = 0
                feedback = "No clear speech detected. Please speak properly."

                session_store[session_id]["answers"].append({
                    "Question": question,
                    "Emotion": emotion,
                    "Confidence": float(confidence),
                    "Sentiment": "neutral",
                    "Feedback": feedback
                })

                return jsonify({
                    "emotion": str(emotion),
                    "confidence": float(confidence),
                    "sentiment": "neutral",
                    "feedback": str(feedback)
                })

            energy = rms

            pitch = librosa.yin(y_trimmed, fmin=80, fmax=300)
            pitch = pitch[~np.isnan(pitch)]
            pitch_avg = float(np.mean(pitch)) if len(pitch) > 0 else 0

            if session_store[session_id]["baseline_pitch"] is None:
                session_store[session_id]["baseline_pitch"] = pitch_avg

            baseline = session_store[session_id]["baseline_pitch"]
            pitch_diff = pitch_avg - baseline

            print("Energy:", energy, flush=True)
            print("Pitch Diff:", pitch_diff, flush=True)

            # ❗ EMOTION NOT TOUCHED
            if energy > 0.12:
                emotion = "angry"
            elif energy < 0.04:
                emotion = "sad"
            elif pitch_diff > 2:
                emotion = "happy"
            else:
                emotion = "neutral"

            confidence = round(min(95, max(60, energy * 750)), 1)
            feedback = f"You sounded {emotion}. Confidence: {confidence}%"

            # -------- SENTIMENT DEBUG --------
            try:
                print("🚀 SENTIMENT BLOCK STARTED", flush=True)

                recognizer = sr.Recognizer()

                with sr.AudioFile(tmp_wav) as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = recognizer.record(source)

                try:
                    text = recognizer.recognize_google(audio_data, language="en-IN")
                except:
                    try:
                        text = recognizer.recognize_google(audio_data)
                    except:
                        text = ""

                text = text.lower().strip()

                print("Recognized Text:", text, flush=True)

                if text == "":
                    print("Speech not recognized, using audio-based fallback", flush=True)
                    if energry > 0.09:
                        sentiment = "positive"
                    elif energy < 0.04:
                        sentiment = "negative"
                    else:
                        sentiment = "neutral"

                else:
                    blob = TextBlob(text)
                    polarity = blob.sentiment.polarity

                    print("Polarity:", polarity, flush=True)

                    if polarity > 0.05:
                        sentiment = "positive"
                    elif polarity < -0.05:
                        sentiment = "negative"
                    else:
                        sentiment = "neutral"

            except Exception as e:
                print("Speech Error:", e, flush=True)
                sentiment = "neutral"

    finally:
        if os.path.exists(tmp_webm):
            os.remove(tmp_webm)
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)

    session_store[session_id]["answers"].append({
        "Question": question,
        "Emotion": emotion,
        "Confidence": float(confidence),
        "Sentiment": sentiment,
        "Feedback": feedback
    })

    return jsonify({
        "emotion": str(emotion),
        "confidence": float(confidence),
        "sentiment": sentiment,
        "feedback": str(feedback)
    })

# ---------------- FINALIZE ----------------
@app.route("/api/finalize_interview", methods=["POST"])
def finalize_interview():
    data = request.get_json()
    session_id = data.get("session_id")

    if not session_id or session_id not in session_store:
        return jsonify({"error": "Invalid session"}), 400

    return jsonify({"message": "Interview finalized"})

# ---------------- DOWNLOAD ----------------
@app.route("/api/download_report", methods=["GET"])
def download_report():

    session_id = request.args.get("session_id")
    session_data = session_store.get(session_id)

    if not session_data:
        return jsonify({"error": "No session found"}), 400

    answers = session_data["answers"]

    if not answers:
        return jsonify({"error": "No answers found"}), 400

    folder = os.path.join(SESSIONS_DIR, session_id)
    os.makedirs(folder, exist_ok=True)

    pdf_path = os.path.join(folder, "Interview_Report.pdf")

    df = pd.DataFrame(answers)
    df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce").fillna(0)

    chart_path = os.path.join(folder, "confidence_chart.png")

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(df) + 1), df["Confidence"], marker="o")
    plt.title("Confidence Trend")
    plt.xlabel("Question Number")
    plt.ylabel("Confidence (%)")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(
        Paragraph(
            "<para align='center'><font size=20><b>AI Interview Coach Report</b></font></para>",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 25))

    avg_conf = round(np.mean(df["Confidence"]), 2)

    summary_data = [
        ["Total Questions", str(len(df))],
        ["Average Confidence", f"{avg_conf}%"]
    ]

    summary_table = Table(summary_data, colWidths=[250, 200])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
        ("BOX", (0, 0), (-1, -1), 1, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 30))

    table_data = [["Question", "Emotion", "Confidence", "Sentiment"]]

    for _, row in df.iterrows():
        table_data.append([
            row["Question"],
            row["Emotion"],
            f"{row['Confidence']:.1f}%",
            row["Sentiment"]
        ])

    detail_table = Table(table_data, repeatRows=1)
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))

    elements.append(detail_table)
    elements.append(Spacer(1, 30))
    elements.append(Image(chart_path, width=420, height=260))

    doc.build(elements)

    return send_file(pdf_path, as_attachment=True)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)