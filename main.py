import pickle
import numpy as np
import pandas as pd
import sqlite3
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Change in production
DB_NAME = "users.db"

# ------------------- Database Setup -------------------
def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL
                    )''')
        conn.commit()

init_db()

# ------------------- Auth Decorator -------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# ------------------- Load Model -------------------
with open('./model/features/features.pkl', 'rb') as f:
    features = pickle.load(f)

with open('./model/features/cat_features.pkl', 'rb') as f:
    cat_features = pickle.load(f)

with open('./model/house_price_predict_model.pkl', 'rb') as f:
    house_price_predict_model = pickle.load(f)

cat_models = house_price_predict_model['catboost_models']
lgb_models = house_price_predict_model['lgb_models']

# ------------------- Preprocessing -------------------
def preprocess_input(input_json):
    df = pd.DataFrame([input_json])

    for feat in features:
        if feat not in df.columns:
            if feat in cat_features:
                df[feat] = ''
            else:
                df[feat] = 0

    df = df[features]

    for c in cat_features:
        if c in df.columns:
            df[c] = df[c].astype('category')

    return df

# ------------------- Routes -------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"], method="pbkdf2:sha256")
        try:
            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                conn.commit()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists.", "danger")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute("SELECT id, password FROM users WHERE username = ?", (username,))
            user = c.fetchone()

        if user and check_password_hash(user[1], password):
            session["user_id"] = user[0]
            flash("Login successful!", "success")
            return redirect(url_for("predict_form"))
        else:
            flash("Invalid username or password.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

@app.route("/predict-form", methods=["GET", "POST"])
@login_required
def predict_form():
    if request.method == "POST":
        input_data = {f: request.form.get(f, '') for f in features}
        X = preprocess_input(input_data)

        preds_cb = np.zeros(len(X))
        for model in cat_models:
            preds_cb += np.expm1(model.predict(X)) / len(cat_models)

        preds_lgb = np.zeros(len(X))
        for model in lgb_models:
            preds_lgb += np.expm1(model.predict(X, num_iteration=model.best_iteration)) / len(lgb_models)

        final_preds = 0.6 * preds_cb + 0.4 * preds_lgb
        result = final_preds[0]

        return render_template("predict.html", prediction=result)

    return render_template("predict.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("data -------------")
        print(data)
        X = preprocess_input(data)
        
        # Debug print statements to check dtypes and sample values
        print(X.dtypes)
        print(X.head())

        preds_cb = np.zeros(len(X))
        for model in cat_models:
            preds_cb += np.expm1(model.predict(X)) / len(cat_models)

        preds_lgb = np.zeros(len(X))
        for model in lgb_models:
            preds_lgb += np.expm1(model.predict(X, num_iteration=model.best_iteration)) / len(lgb_models)

        # Weighted blend of predictions
        final_preds = 0.6 * preds_cb + 0.4 * preds_lgb

        result = final_preds[0] if len(final_preds) == 1 else final_preds.tolist()

        return jsonify({'predicted_price': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
