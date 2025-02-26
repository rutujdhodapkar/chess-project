import socket
import threading
import sys
import time
from flask import Flask, render_template, request, jsonify

# ----- Configuration -----
# Default SERVER_IP is set for LAN connectivity.
SERVER_IP = "192.168.0.1"  # server address for LAN
SERVER_PORT = 5000           # port used for client-server communication
CLIENT_LISTEN_PORT = 6000    # port on which this client listens for incoming messages

# If a third command-line argument "local" is provided, switch to local mode.
if len(sys.argv) >= 4 and sys.argv[3].lower() == "local":
    SERVER_IP = "127.0.0.1"   # use localhost for local connection

# ----- Utility Functions -----
def get_own_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

# ----- Client-to-Client Message Listener -----
def listen_for_messages():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", CLIENT_LISTEN_PORT))
    s.listen(5)
    print(f"Listening for incoming messages on port {CLIENT_LISTEN_PORT}...")
    while True:
        conn, addr = s.accept()
        thread = threading.Thread(target=handle_incoming_message, args=(conn, addr))
        thread.daemon = True
        thread.start()

def handle_incoming_message(conn, addr):
    try:
        data = conn.recv(1024).decode()
        print(f"\nMessage from {addr}: {data}\n> ", end="", flush=True)
    except Exception as e:
        print("Error receiving message:", e)
    finally:
        conn.close()

# ----- Server Communication -----
def register_to_server(username):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((SERVER_IP, SERVER_PORT))
    s.send(("REGISTER:" + username).encode())
    response = s.recv(1024).decode()
    print("Server response:", response)
    return s

def query_user(server_socket, target_username):
    server_socket.send(("QUERY:" + target_username).encode())
    response = server_socket.recv(1024).decode()
    print("Query response:", response)
    return response

# ----- Peer-to-Peer Messaging -----
def send_message_to_user(target_ip, message):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((target_ip, CLIENT_LISTEN_PORT))
        s.send(message.encode())
    except Exception as e:
        print(f"Error sending message to {target_ip}: {e}")
    finally:
        s.close()

# ----- Flask Web Interface for Client -----
app = Flask(__name__, template_folder='templates')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/connect", methods=["POST"])
def connect_to_target():
    data = request.get_json()
    target_username = data.get("target")
    if not target_username:
        return jsonify({"error": "No target username provided."}), 400

    response = query_user(server_socket, target_username)
    if response.startswith("ONLINE:"):
        target_ip = response.split(":", 1)[1].strip()
        default_message = f"Hello from {username}"
        send_message_to_user(target_ip, default_message)
        return jsonify({"status": "connected", "target": target_username})
    elif response.startswith("OFFLINE:"):
        return jsonify({"status": "offline", "target": target_username})
    elif response.startswith("ALL_DATA:"):
        return jsonify({"status": "not_found", "details": response[len("ALL_DATA:"):].strip()})
    else:
        return jsonify({"status": "error", "message": "Unknown server response."}), 500

# ----- Main Client Program -----
if __name__ == "__main__":
    # Use command-line arguments if provided.
    if len(sys.argv) >= 3:
        username = sys.argv[1]
        target_username = sys.argv[2]
        print(f"Starting client for user: {username} with target: {target_username}")
    else:
        username = input("Enter your username: ")
        target_username = None

    own_ip = get_own_ip()
    print("Your IP:", own_ip)

    # Start a background thread to listen for incoming messages.
    listener_thread = threading.Thread(target=listen_for_messages)
    listener_thread.daemon = True
    listener_thread.start()

    # Connect to the central server and register this client.
    server_socket = register_to_server(username)

    # If a target username was passed, automatically attempt a connection.
    if target_username:
        time.sleep(2)  # slight delay to ensure registration completes
        response = query_user(server_socket, target_username)
        if response.startswith("ONLINE:"):
            target_ip = response.split(":", 1)[1].strip()
            default_message = f"Hello from {username}"
            send_message_to_user(target_ip, default_message)
            print(f"Connected to target: {target_username}")
        elif response.startswith("OFFLINE:"):
            print(f"Target user is offline: {target_username}")
        elif response.startswith("ALL_DATA:"):
            print(f"Target user not found: {target_username}")
        else:
            print("Unknown server response:", response)

    # Start the Flask web server.
    app.run(host="0.0.0.0", port=6001, debug=True)
