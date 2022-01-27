from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
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
        test_data_path = os.path.join(data_path, 'ved_test.xlsx')

        month_kpi_skills = pd.read_excel(test_data_path, sheet_name='Характеристика ВЭД', header=1)
        quarter_kpi_skills = pd.read_excel(test_data_path, sheet_name='Характеристика ВЭД', header=1)
        positions_skills = pd.read_csv(os.path.join(data_path, "latest_positions_skills.csv"))

        prepared_data, bp_id_list, ved_list = preprocess_data(all_bp,
                                                              month_kpi_skills,
                                                              quarter_kpi_skills,
                                                              positions_skills)

        # Predict
        input_size = prepared_data.shape[2]

        result = predict(prepared_data, bp_id_list, ved_list, input_size)

        return jsonify(result)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8080, url_scheme='https')
