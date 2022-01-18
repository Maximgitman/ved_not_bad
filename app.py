from flask import Flask, request, jsonify, render_template
import pandas as pd
from waitress import serve
import os

from prediction import predict
from preprocessing import preprocess_data

UPLOAD_FOLDER = "static/uploads/"

app = Flask(__name__)

# Ensure templates are auto-reloaded
app.secret_key = "not-so-secret"
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config['DEBUG'] = True
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/")
def index():
    # Index pages
    exctracted_txt = "Will be add some results"
    return render_template("index.html",
                           exctracted_txt=exctracted_txt)


@app.route("/get_ved", methods=["POST"])
def get_ved():
    if request.method == "POST":
        # BLock reading and getting data
        # Getting first table from request (json)
        data = request.get_json(force=True)
        all_bp = pd.DataFrame(data)

        # Getting second table with time
        time_process = pd.read_csv("/static/data/ved_bp_processing_time.csv")

        # Getting third table with skills
        skills = pd.read_csv("/static/data/ved_bp_skills.csv")

        if not all_bp or time_process or skills:  # If there is no name, showing erros page
            return render_template("error.html", message="Missing data, try again")

        # Preprocessing all_bp
        # Preprocessing time_process
        # Preprocessing skills
        # Concatenate pd.concat([all_bp, time_process, skills])
        prepared_data, bp_id_list, ved_list = preprocess_data(all_bp.copy(),
                                                              skills.copy(),
                                                              time_process.copy())
        # Predict
        results = predict(prepared_data, bp_id_list, ved_list)

        # Making Json into variable --> results

        return jsonify(results)


if __name__ == "__main__":
    # app.run(debug=True)
    serve(app, host="0.0.0.0", port=5000)