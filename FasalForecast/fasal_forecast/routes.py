from fasal_forecast import app
from flask import jsonify, request
import os

@app.route('/')
@app.route('/home')
def home_page():
    return "home page"

@app.route('/upload', methods=['POST'])
def upload_image():
    # checking if request has a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    # checking if no file
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        # saving the image file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # return appropriate response if request was successful
        return jsonify({
            'message': 'File successfully uploaded',
            'file_path': file_path
        }), 200


# @app.route('/predict')
# def make_prediction():
#     image_path = os.path.join(UPLOAD_FOLDER, os.path.basename())
#     return jsonify(prediction_model.predict(image_path))

