from flask import Flask, request, jsonify
import os
from flask import send_from_directory
import csv
from datetime import datetime
from werkzeug.utils import secure_filename
from predict import get_prediction
from generate_pipeline import pipeline_generator
import html
import json

UPLOAD_FOLDER = './uploads/'
RESULTS_FOLDER = './entry-results/'

app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def ensure_results_directory():
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
        csv_file = os.path.join(RESULTS_FOLDER, 'results.csv')
        with open(csv_file, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'filePath', 'textData', 'usedModel', 'prediction']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

def save_to_csv(data):
    csv_file = os.path.join(RESULTS_FOLDER, 'results.csv')
    with open(csv_file, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'filePath', 'textData', 'usedModel', 'prediction']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(data)


@app.route('/')
def get_index():
    response = app.send_static_file('index.html')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/workflow/generate')
def get_generate():
    return app.send_static_file('workflow.html')

@app.route('/example/<path:filename>')
def serve_dataflow_image(filename):
    return send_from_directory('example', filename)

@app.route('/api/workflow', methods=['POST'])
def post_workflow():
    try:
        textData = request.form['textData']
        selectedModel = request.form['selectedModel']
        fileName = request.form['fileName']
        
        file = request.files['fileUploaded']
        prediction_result = get_prediction(textData, selectedModel)

        fullPath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(fileName))
        file.save(fullPath)

        data = {
            "data": {
                "filePath": fullPath,
                "textData": textData,
                "usedModel": selectedModel,
                "prediction": prediction_result
            }
        }
        ensure_results_directory()  # Ensure directory and CSV file existence
        save_to_csv({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'filePath': fullPath,
            'textData': textData,
            'usedModel': selectedModel,
            'prediction': prediction_result
        })
        # prediction_str = f"<div style='text-align: left; border-radius: 10px; border: 1px solid; width:40%;  padding: 10px;'><strong>Predicted Analytical Intent: &nbsp; </strong>{data['data']['prediction']} <button style='border-radius: 10px; background-color: blue; color: white; margin-left: 120px; padding: 10px 20px;'>Next</button></div>"
        prediction_str = f"<div style='text-align: left; border-radius: 10px; border: 1px solid; width:30%;  padding: 10px;'><strong>Predicted Analytical Intent: &nbsp; </strong>{data['data']['prediction']}"
        return prediction_str, 200
    except Exception as e:
            return f"Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=os.environ.get("FLASK_SERVER_PORT", 9000), debug=True)
