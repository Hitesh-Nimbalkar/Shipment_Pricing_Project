from flask import Flask, render_template, request
from Prediction.batch import batch_prediction
from Prediction.instance_prediction import instance_prediction_class

input_file_path = "SCMS_Delivery_History_Dataset.csv"
feature_engineering_file_path = "prediction_files/feat_eng.pkl"
transformer_file_path = "prediction_files/preprocessed.pkl"
model_file_path = "prediction_files/model.pkl"

COUNTRY_MAP = {'Zambia': 0, 'Ethiopia': 1, 'Nigeria': 2, 'Tanzania': 3, "Côte d'Ivoire": 4, 'Mozambique': 5, 'Others': 6, 'Zimbabwe': 7, 'South Africa': 8, 'Rwanda': 9, 'Haiti': 10, 'Vietnam': 11, 'Uganda': 12}
FULFILL_VIA_MAP = {'From RDC': 0, 'Direct Drop': 1}
SHIPMENT_MODE_MAP = {'Truck': 0, 'Air': 1, 'Air Charter': 2, 'Ocean': 3}
SUB_CLASSIFICATION_MAP = {'Adult': 0, 'Pediatric': 1, 'HIV test': 2, 'HIV test - Ancillary': 3, 'Malaria': 4, 'ACT': 5}
BRAND_MAP = {'Generic': 0, 'Others': 1, 'Determine': 2, 'Uni-Gold': 3}
FIRST_LINE_DESIGNATION_MAP = {'Yes': 0, 'No': 1}

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/batch", methods=["POST"])
def perform_batch_prediction():
    batch = batch_prediction(input_file_path, model_file_path, transformer_file_path, feature_engineering_file_path)
    batch.start_batch_prediction()
    output = "Batch Prediction Done Done DONE"
    return render_template("index.html", prediction_result=output, prediction_type='batch')

@app.route("/instance", methods=["POST"])
def perform_instance_prediction():
    pack_price = float(request.form['pack_price'])
    unit_price = float(request.form['unit_price'])
    weight_kg = float(request.form['weight_kg'])
    line_item_quantity = int(request.form['line_item_quantity'])
    fulfill_via = request.form['fulfill_via']
    shipment_mode = request.form['shipment_mode']
    country = request.form['country']
    brand = request.form['brand']
    sub_classification = request.form['sub_classification']
    first_line_designation = request.form['first_line_designation']

    predictor = instance_prediction_class(pack_price, unit_price, weight_kg, line_item_quantity,
                                                            fulfill_via, shipment_mode, country, brand,
                                                            sub_classification, first_line_designation)
    predicted_price = predictor.predict_price_from_input()

    return render_template('index.html', prediction_type='instance', predicted_price=predicted_price)


if __name__ == '__main__':
    host = '0.0.0.0'  # Specify the host address you want to use
    port = 8000  # Specify the port number you want to use
    app.run(debug=True, host=host, port=port)