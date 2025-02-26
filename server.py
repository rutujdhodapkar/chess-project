import socket
import threading
import csv
import os

# ----- Configuration -----
CSV_FILE = r"C:\Users\rutuj\OneDrive\Desktop\allip.csv"
SERVER_IP = "192.168.0.1"  # as given
SERVER_PORT = 5000           # arbitrary port for client-server communications

# Global dictionary to keep track of connected clients: {username: (socket, addr)}
clients = {}
clients_lock = threading.Lock()

# A lock to prevent concurrent access to the CSV file
csv_lock = threading.Lock()

# ----- CSV Helper Functions -----
def update_csv(username, ip):
    """Reads the CSV file, updates or adds the given username/ip entry, then writes it back."""
    csv_lock.acquire()
    try:
        rows = []
        exists = False
        if os.path.exists(CSV_FILE):
            with open(CSV_FILE, mode='r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['username'] == username:
                        row['ip'] = ip  # update IP
                        exists = True
                    rows.append(row)
        if not exists:
            # add new row if username not found
            rows.append({'username': username, 'ip': ip})
        # Write updated rows back to the CSV file
        with open(CSV_FILE, mode='w', newline='') as f:
            fieldnames = ['username', 'ip']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    finally:
        csv_lock.release()

def get_csv_contents():
    """Returns the current contents of the CSV file as a list of dictionaries."""
    csv_lock.acquire()
    try:
        rows = []
        if os.path.exists(CSV_FILE):
            with open(CSV_FILE, mode='r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
        return rows
    finally:
        csv_lock.release()

# ----- Client Handling -----
def handle_client(conn, addr):
    """
    Each client should first send a registration message in the form:
       REGISTER:<username>
    After that, clients can send queries like:
       QUERY:<target_username>
    """
    username = None
    try:
        while True:
            data = conn.recv(1024).decode()
            if not data:
                break  # client disconnected

            print(f"Received from {addr}: {data}")
            if data.startswith("REGISTER:"):
                # Client registration message
                username = data.split(":", 1)[1].strip()
                client_ip = addr[0]  # use the IP from the connection
                update_csv(username, client_ip)
                # Save this client as online
                with clients_lock:
                    clients[username] = (conn, addr)
                conn.send("REGISTERED".encode())
            elif data.startswith("QUERY:"):
                # Client query message: look up target username
                target_username = data.split(":", 1)[1].strip()
                rows = get_csv_contents()
                target_ip = None
                for row in rows:
                    if row['username'] == target_username:
                        target_ip = row['ip']
                        break
                if target_ip:
                    # Check if target user is currently connected
                    with clients_lock:
                        if target_username in clients:
                            conn.send(("ONLINE:" + target_ip).encode())
                        else:
                            conn.send(("OFFLINE:" + target_ip).encode())
                else:
                    # If the username isn’t found, return all CSV contents
                    all_data = "ALL_DATA:"
                    for row in rows:
                        all_data += f"{row['username']}:{row['ip']}, "
                    conn.send(all_data.encode())
            else:
                conn.send("UNKNOWN COMMAND".encode())
    except Exception as e:
        print("Error handling client:", e)
    finally:
        # Clean up on disconnect
        if username:
            with clients_lock:
                if username in clients:
                    del clients[username]
        conn.close()
        print("Connection closed:", addr)

def start_server():
    """Starts the server to listen for incoming client connections."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((SERVER_IP, SERVER_PORT))
    s.listen(5)
    print(f"Server listening on {SERVER_IP}:{SERVER_PORT}")
    while True:
        conn, addr = s.accept()
        print("Connected by", addr)
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.daemon = True
        thread.start()

if __name__ == "__main__":
    start_server()