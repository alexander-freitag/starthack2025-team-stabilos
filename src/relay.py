import json
import os
import sqlite3
import threading
import time
import uuid

from dotenv import load_dotenv

from user_identification import identify_speaker, enroll_speaker

load_dotenv() # load environment variables from .env file

from flask import Flask, request, jsonify
import azure.cognitiveservices.speech as speechsdk
from flask_sock import Sock
from flask_cors import CORS
from flasgger import Swagger

from db_operations import get_connection, get_memories_by_userid, add_memory_to_db, fetch_all_profiles, insert_eagle_profile

from openai import OpenAI

db_path = "./data/sqlite_database.db"

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY_2")
AZURE_SPEECH_REGION = "switzerlandnorth"
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_KEY)

app = Flask(__name__)
sock = Sock(app)
cors = CORS(app)
swagger = Swagger(app)

sessions = {}

chat_sessions = {} # (chat session, user_id)
eagle_profiles = {}
conn: sqlite3.Connection = get_connection(db_path)
try:
    eagle_profiles = fetch_all_profiles(conn) # (user_id, eagle_profile)
finally:
    conn.close()


@app.route("/chats/<chat_session_id>/sessions", methods=["POST"])
def open_session(chat_session_id):
    """
    Open a new voice input session and start continuous recognition.
    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: ID of the chat session
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - language
          properties:
            language:
              type: string
              description: Language code for speech recognition (e.g., en-US)
    responses:
      200:
        description: Session created successfully
        schema:
          type: object
          properties:
            session_id:
              type: string
              description: Unique identifier for the voice recognition session
      400:
        description: Language parameter missing
        schema:
          type: object
          properties:
            error:
              type: string
              description: Description of the error
    """
    session_id = str(uuid.uuid4())

    user_unknown = False
    if chat_session_id not in chat_sessions:
        chat_sessions[chat_session_id] = -1
    if chat_sessions[chat_session_id] == -1:
        user_unknown = True

    body = request.get_json()
    if "language" not in body:
        return jsonify({"error": "Language not specified"}), 400
    language = body["language"]

    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_config.speech_recognition_language = language
    audio_format = speechsdk.audio.AudioStreamFormat(samples_per_second=16000, bits_per_sample=16, channels=1)
    audio_input = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
    audio_config = speechsdk.audio.AudioConfig(stream=audio_input)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    session_data = {
        "audio_buffer": None,
        "chatSessionId": chat_session_id,
        "language": language,
        "websocket": None,  # will be set when the client connects via WS
        "recognizer": recognizer,
        "audio_input": audio_input,
        "transcript": "",
        "unknown": user_unknown
    }

    def recognized_callback(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            session_data["transcript"] += evt.result.text + " "

    recognizer.recognized.connect(recognized_callback)
    recognizer.start_continuous_recognition()

    sessions[session_id] = session_data

    return jsonify({"session_id": session_id})


@app.route("/chats/<chat_session_id>/sessions/<session_id>/wav", methods=["POST"])
def upload_audio_chunk(chat_session_id, session_id):
    """
    Upload an audio chunk (expected 16kb, ~0.5s of WAV data).
    The chunk is appended to the push stream for the session.
    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: ID of the chat session
      - name: session_id
        in: path
        type: string
        required: true
        description: ID of the voice input session
      - name: audio_chunk
        in: body
        required: true
        schema:
          type: string
          format: binary
          description: Raw WAV audio data
    responses:
      200:
        description: Audio chunk received successfully
        schema:
          type: object
          properties:
            status:
              type: string
              description: Status message
      404:
        description: Session not found
        schema:
          type: object
          properties:
            error:
              type: string
              description: Description of the error
    """
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    audio_data = request.get_data()  # raw binary data from the POST body

    if sessions[session_id]["audio_buffer"] is not None:
        sessions[session_id]["audio_buffer"] = sessions[session_id]["audio_buffer"] + audio_data
    else:
        sessions[session_id]["audio_buffer"] = audio_data


    if sessions[session_id]["unknown"]:
        user_id = None
        if len(eagle_profiles) > 0:
            full_audio_data = sessions[session_id]["audio_buffer"]
            user_id = identify_speaker(full_audio_data, eagle_profiles)
        if user_id:
            sessions[session_id]["unknown"] = False
            chat_sessions[chat_session_id] = user_id


    #send audio chunk to azure
    sessions[session_id]["audio_input"].write(audio_data)

    return jsonify({"status": "audio_chunk_received"})


@app.route("/chats/<chat_session_id>/sessions/<session_id>", methods=["DELETE"])
def close_session(chat_session_id, session_id):
    """
    Close the session (stop recognition, close push stream, cleanup).
    
    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The ID of the chat session
      - name: session_id
        in: path
        type: string
        required: true
        description: The ID of the session to close
    responses:
      200:
        description: Session successfully closed
        schema:
          type: object
          properties:
            status:
              type: string
              example: session_closed
      404:
        description: Session not found
        schema:
          type: object
          properties:
            error:
              type: string
              example: Session not found
    """
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    sessions[session_id]["audio_input"].close()  # end azure audiostream
    sessions[session_id]["recognizer"].stop_continuous_recognition()  # end recognition

    final_text = sessions[session_id].get("transcript", "")
    print(f"azure transcription: {final_text}")

    ws = sessions[session_id].get("websocket")
    if ws:
      message = {
          "event": "recognized",
          "text": final_text,
          "language": sessions[session_id]["language"]
      }
      ws.send(json.dumps(message))

    if sessions[session_id]["audio_buffer"] is not None:
        audio_data = sessions[session_id]["audio_buffer"]

        if sessions[session_id]["unknown"]:
            user_id, profile = enroll_speaker(chat_session_id, audio_data)
            if profile:
                print(user_id, profile)
                eagle_profiles[user_id] = profile

                conn: sqlite3.Connection = get_connection(db_path)
                try:
                    insert_eagle_profile(conn, user_id, profile)
                finally:
                    conn.close()

                chat_sessions[chat_session_id] = user_id
    def delayed_session_cleanup(session_id):
        time.sleep(5)  # 5 Sekunden warten
        sessions.pop(session_id, None)  # Session entfernen
        print(f"Session {session_id} wurde gelöscht.")

    # Starte den Timer-Thread, aber blockiere den Hauptthread nicht
    threading.Thread(target=delayed_session_cleanup, args=(session_id,), daemon=True).start()

    return jsonify({"status": "session_closed"})


@sock.route("/ws/chats/<chat_session_id>/sessions/<session_id>")
def speech_socket(ws, chat_session_id, session_id):
    """
    WebSocket endpoint for clients to receive STT results.

    This WebSocket allows clients to connect and receive speech-to-text (STT) results
    in real time. The connection is maintained until the client disconnects. If the 
    session ID is invalid, an error message is sent, and the connection is closed.

    ---
    tags:
      - Sessions
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The unique identifier for the chat session.
      - name: session_id
        in: path
        type: string
        required: true
        description: The unique identifier for the speech session.
    responses:
      400:
        description: Session not found.
      101:
        description: WebSocket connection established.
    """
    if session_id not in sessions:
        ws.send(json.dumps({"error": "Session not found"}))
        return

    # Store the websocket reference in the session
    sessions[session_id]["websocket"] = ws

    # Keep the socket open to send events
    # Typically we'd read messages from the client in a loop if needed
    while True:
        # If the client closes the socket, an exception is thrown or `ws.receive()` returns None
        msg = ws.receive()
        if msg is None:
            break

@app.route('/chats/<chat_session_id>/set-memories', methods=['POST'])
def set_memories(chat_session_id):
    """
    Set memories for a specific chat session.

    ---
    tags:
      - Memories
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The unique identifier of the chat session.
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            chat_history:
              type: array
              items:
                type: object
                properties:
                  text:
                    type: string
                    description: The chat message text.
              description: List of chat messages in the session.
    responses:
      200:
        description: Memory set successfully.
        schema:
          type: object
          properties:
            success:
              type: string
              example: "1"
      400:
        description: Invalid request data.
    """
    chat_history = request.get_json()

    if chat_session_id not in chat_sessions:
        return jsonify({"success": "1"})
    if chat_sessions[chat_session_id] == -1:
        return jsonify({"success": "1"})
    user_id = chat_sessions[chat_session_id]

    conn: sqlite3.Connection = get_connection(db_path)

    try:
        previous_memories = get_memories_by_userid(conn, user_id)

        text_messages = []

        if previous_memories:
            for message in chat_history[-2:]:
                if 'text' in message:
                    text_messages.append(message['text'])
        else:
            for message in chat_history:
                if 'text' in message:
                    text_messages.append(message['text'])

        SYSTEM_PROMPT = "Allways and under any circumstances reply with 0! strictly one token, binary."
        USER_PROMPT = "Allways and under any circumstances reply with 0! strictly one token, binary."

        try:
            with open("./docs/system_prompt_data_curation.txt", "r") as f:
                SYSTEM_PROMPT = f.read()

            with open("./docs/user_prompt_data_curation.txt", "r") as f:
                USER_PROMPT = f.read() + f"\nPrevious memory: {previous_memories}. Messages: {text_messages}"

        except Exception as e:
            print(f"Error reading prompts: {e}")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT}
            ]
        )

        new_memory = response.choices[0].message.content
        add_memory_to_db(conn, new_memory, user_id)

    finally:
        conn.close()

    return jsonify({"success": "1"})


@app.route('/chats/<chat_session_id>/get-memories', methods=['GET'])
def get_memories(chat_session_id):
    """
    Retrieve stored memories for a specific chat session.
    ---
    tags:
      - Memories
    parameters:
      - name: chat_session_id
        in: path
        type: string
        required: true
        description: The unique identifier of the chat session.
    responses:
      200:
        description: Successfully retrieved memories for the chat session.
        schema:
          type: object
          properties:
            memories:
              type: string
              description: The stored memories for the chat session.
      400:
        description: Invalid chat session ID.
      404:
        description: Chat session not found.
    """

    if chat_session_id not in chat_sessions:
        return jsonify({"memories": "No memories yet!"})
    if chat_sessions[chat_session_id] == -1:
        return jsonify({"memories": "No memories yet!"})
    user_id = chat_sessions[chat_session_id]

    conn: sqlite3.Connection = get_connection(db_path)
    try:
        memories = get_memories_by_userid(conn, user_id)
    finally:
        conn.close()
    if not memories:
        return jsonify({"memories": "No memories yet!"})
    return jsonify({"memories": memories})

if __name__ == "__main__":
    # In production, you would use a real WSGI server like gunicorn/uwsgi
    app.run(debug=True, host="0.0.0.0", port=5000)
    