import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
import csv
import chess
from stockfish import Stockfish
import datetime
import subprocess
import model  # Import our ML functions: store_move() and predict_moves()

# Set base directory as the current file's directory to enable cloud deployability.
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(TEMPLATE_DIR, "static")

app = Flask(__name__,
            template_folder=TEMPLATE_DIR,
            static_folder=STATIC_DIR,
            static_url_path='/static')
app.secret_key = os.environ.get('SECRET_KEY', 'replace_with_your_secret_key')

# ---------------------- User Login / Signup -----------------------
DATA_FILE = os.path.join(TEMPLATE_DIR, "data.csv")
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['first_name', 'last_name', 'username', 'email', 'password'])

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/login.html', methods=['GET'])
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')
    authenticated = False
    username = ''
    with open(DATA_FILE, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['email'] == email and row['password'] == password:
                authenticated = True
                username = row['username']
                break
    if authenticated:
        session['username'] = username
        return redirect(url_for('main_page'))
    else:
        flash('Invalid email or password', 'error')
        return redirect(url_for('login_page', status='error'))

@app.route('/signup', methods=['POST'])
def signup():
    first_name = request.form.get('first_n')
    last_name = request.form.get('last_n')
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    user_exists = False
    with open(DATA_FILE, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['email'] == email:
                user_exists = True
                break
    if user_exists:
        flash('User already exists with that email.', 'error')
        return redirect(url_for('login_page', status='error'))
    with open(DATA_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([first_name, last_name, username, email, password])
    flash('Signup successful. Please log in.', 'success')
    return redirect(url_for('login_page'))

@app.route('/index.html')
def main_page():
    username = session.get('username', 'Guest')
    return render_template('index.html', username=username)

@app.route('/contact.html')
def contact():
    desktop_image = url_for('static', filename='desktop.png')
    return render_template('contact.html', desktop_image=desktop_image)

# ---------------------- Stockfish Integration -----------------------
STOCKFISH_PATH = os.environ.get('STOCKFISH_PATH', os.path.join(BASE_DIR, "stockfish", "stockfish"))
stockfish = Stockfish(STOCKFISH_PATH)
stockfish.set_skill_level(20)

@app.route('/review', methods=['POST'])
def review_move():
    data = request.get_json()
    fen = data.get("fen")
    if not fen:
        return jsonify({"error": "FEN not provided"}), 400
    stockfish.set_fen_position(fen)
    best_move = stockfish.get_best_move()
    evaluation = stockfish.get_evaluation()
    return jsonify({
        "best_move": best_move,
        "evaluation": evaluation
    })

# ---------------------- Moves Saving -----------------------
@app.route('/save_moves', methods=['POST'])
def save_moves():
    data = request.get_json()
    username = data.get("username", "Guest")
    moves = data.get("moves")
    if moves is None:
        return jsonify({"error": "Moves data not provided"}), 400
    if isinstance(moves, list):
        moves = ", ".join(moves)
    timestamp = datetime.datetime.now().isoformat()
    MOVE_DATA_FILE = os.path.join(TEMPLATE_DIR, "movedata.csv")
    expected_header = ['username', 'moves', 'timestamp']
    if not os.path.exists(MOVE_DATA_FILE):
        with open(MOVE_DATA_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(expected_header)
    with open(MOVE_DATA_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([username, moves, timestamp])
    return jsonify({"message": "Moves saved successfully."})

# ---------------------- Game Move Analysis & Prediction -----------------------
def analyze_move(board_before, played_move):
    stockfish.set_fen_position(board_before.fen())
    stockfish.update_engine_parameters({"MultiPV": 3})
    top_moves = stockfish.get_top_moves(3)
    if top_moves and len(top_moves) > 0:
        best_move = top_moves[0]["Move"]
    else:
        best_move = stockfish.get_best_move()
    board_best = board_before.copy()
    board_best.push_uci(best_move)
    stockfish.set_fen_position(board_best.fen())
    best_eval = stockfish.get_evaluation()
    board_played = board_before.copy()
    try:
        board_played.push_san(played_move)
    except Exception:
        return "Invalid Move", "N/A", best_move
    stockfish.set_fen_position(board_played.fen())
    played_eval = stockfish.get_evaluation()
    def extract_score(eval_info):
        if eval_info["type"] == "mate":
            return 10000 if eval_info["value"] > 0 else -10000
        return eval_info["value"]
    played_score = extract_score(played_eval)
    best_score = extract_score(best_eval)
    diff = (best_score - played_score) if board_before.turn else (played_score - best_score)
    try:
        played_uci = board_before.parse_san(played_move).uci()
    except Exception:
        played_uci = ""
    if played_uci == best_move:
        verdict = "Excellent Move"
    else:
        if diff < 0:
            verdict = "Brilliant Move"
        elif diff < 20:
            verdict = "Outstanding Move"
        elif diff < 30:
            verdict = "Very Good Move"
        elif diff < 60:
            verdict = "Good Move"
        elif diff < 100:
            verdict = "Slight Inaccuracy"
        elif diff < 150:
            verdict = "Inaccuracy"
        elif diff < 200:
            verdict = "Mistake"
        elif diff < 300:
            verdict = "Serious Mistake"
        else:
            verdict = "Blunder"
    played_eval_str = f"Mate in {played_eval['value']}" if played_eval["type"] == "mate" else f"{played_eval['value']} cp"
    best_eval_str = f"Mate in {best_eval['value']}" if best_eval["type"] == "mate" else f"{best_eval['value']} cp"
    top_moves_info = ""
    if top_moves:
        top_moves_info = " | Top moves: " + ", ".join(
            [f"{tm['Move']} ({'Mate in ' + str(tm['Mate']) if tm.get('Mate') is not None else str(tm.get('Centipawn')) + ' cp'})"
             for tm in top_moves]
        )
    eval_details = f"Played Eval: {played_eval_str} vs Best Eval: {best_eval_str} (Diff: {diff}){top_moves_info}"
    return verdict, eval_details, best_move

@app.route('/share_move', methods=['POST'])
def share_move():
    """
    This endpoint receives move details from the UI.
    It reconstructs the board from the provided move history,
    analyzes the played move against Stockfish,
    logs the move using the model module,
    and then predicts upcoming moves using the backend ML model.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    notation = data.get("san")
    if not notation:
        frm = data.get("from", "")
        to = data.get("to", "")
        if frm and to:
            notation = f"{frm}-{to}"
        else:
            return jsonify({"error": "No move notation provided"}), 400
    new_game_flag = data.get("new_game", False)
    move_history = data.get("moves", [])
    if new_game_flag:
        move_history = [notation]
    else:
        move_history.append(notation)
    board = chess.Board()
    # Reconstruct the board from the move history.
    # If any move is invalid, return an error immediately.
    for move in move_history[:-1]:
        try:
            board.push_san(move)
        except Exception as e:
            return jsonify({"error": f"Invalid move in history: {move}"}), 400
    try:
        verdict, eval_details, best_move = analyze_move(board, notation)
        player = "White" if board.turn else "Black"
        print(f"Player: {player} | Move: {notation} | Verdict: {verdict} | {eval_details} | Best Move: {best_move}")
        analysis = {"verdict": verdict, "evaluation": eval_details, "best_move": best_move}
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    try:
        model.store_move(notation)
    except Exception as e:
        print(f"Error storing move: {e}")
    try:
        predictions = model.predict_moves(move_history)
    except Exception as e:
        predictions = []
        print(f"Error predicting moves: {e}")
    print(f"Model Prediction: {', '.join(predictions) if predictions else 'None'}")
    return jsonify({
        "notation": notation,
        "analysis": analysis,
        "stockfish": "stockfish: " + best_move,
        "model": "model: " + (", ".join(predictions) if predictions else "")
    }), 200

# ---------------------- Additional Endpoints (e.g., online play) -----------------------
@app.route('/online_play', methods=['POST'])
def online_play():
    data = request.get_json(force=True, silent=True) or request.form
    target_username = data.get("target_username")
    if not target_username:
        return jsonify({"error": "Target username not provided"}), 400
    current_username = session.get('username', 'Guest')
    connection_mode = data.get("connection_mode", "lan").lower()
    try:
        if connection_mode == "local":
            subprocess.Popen(["python", "client.py", current_username, target_username])
        else:
            subprocess.Popen(["python", "client.py", current_username, target_username])
        return jsonify({"message": "Online play initiated."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
