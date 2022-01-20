from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
from waitress import serve
from prediction import predict
from preprocessing import preprocess_data

UPLOAD_FOLDER = "static/uploads/"
data_path = "static/data"

app = Flask(__name__)

headers = {'Content-Type': 'application/json'}

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


@app.route("/", methods=["GET", "POST"])
def get_ved():
    if request.method == "POST":
        data = request.get_json(force=True)
        all_bp = pd.DataFrame(data)
        skills_path = os.path.join(data_path, 'ved_bp_skills_3.xlsx')
        time_path = os.path.join(data_path, 'ved_bp_processing_time.csv')

        skills = pd.read_excel(skills_path, sheet_name=None, header=[1])
        time = pd.read_csv(time_path)

        prepared_data, bp_id_list, ved_list = preprocess_data(all_bp.copy(),
                                                              skills.copy(),
                                                              time.copy())
        # Predict
        results = predict(prepared_data, bp_id_list, ved_list)

        return jsonify(results)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    # app.run(debug=True)
    serve(app, host="0.0.0.0", port=8080)