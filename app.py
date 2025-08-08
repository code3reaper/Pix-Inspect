from flask import Flask, request, render_template, redirect, url_for, jsonify
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
from PIL import Image
import os
import pandas as pd
from colors import colors

app = Flask(__name__)

if not os.path.exists('uploads'):
    os.makedirs('uploads')

def get_dominant_color(image_path, credentials_path):
    client = vision_v1.ImageAnnotatorClient.from_service_account_file(credentials_path)

    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    response = client.image_properties(image=image)
    props = response.image_properties_annotation

    if props.dominant_colors.colors:
        dominant_color = props.dominant_colors.colors[0].color
        return dominant_color

    return None


def check_image_resolution(image_path): 
    img = Image.open(image_path)

    width, height = img.size

    preferred_width = 1600
    preferred_height = 1600

    if width >= preferred_width and height >= preferred_height:
        resolution_quality = 'Good'
    else:
        resolution_quality = 'Okayish'

    return resolution_quality

# Function to get color name
def get_color_name(hex_color, colors):
    closest_color = min(colors, key=lambda x: abs(colors[x]['rgb'][0] - hex_color.red) +
                                                  abs(colors[x]['rgb'][1] - hex_color.green) +
                                                  abs(colors[x]['rgb'][2] - hex_color.blue))
    return closest_color


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['user_id']
        password = request.form['password']

        if user_id == 'seller' and password == 'pass':
            return redirect(url_for('upload'))
        else:
            return render_template('login.html', error='Invalid ID or password.')
    return render_template('login.html')

# Route for upload page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        title = request.form['title']
        cost = request.form['cost']
        color = request.form['color']
        resolution = request.form['resolution']
        image_filename = request.form['image_filename']

        # Save data to CSV
        data = {'Title': title, 'Cost': cost, 'Color': color, 'Resolution': resolution, 'Image': image_filename}
        df = pd.DataFrame([data])
        
        if not os.path.isfile('product_data.csv'):
            df.to_csv('product_data.csv', index=False)
        else:
            df.to_csv('product_data.csv', mode='a', header=False, index=False)

        return jsonify({'success': True})

    return render_template('upload.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    image_file = request.files['image']
    image_path = os.path.join('uploads', image_file.filename)
    image_file.save(image_path)

    # Load the Google Vision API key path from an environment variable
    credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not credentials_path:
        return jsonify({'error': 'Google Vision API key path not set. Please set the GOOGLE_APPLICATION_CREDENTIALS environment variable.'}), 500

    resolution_quality = check_image_resolution(image_path)
    dominant_color = get_dominant_color(image_path, credentials_path)

    if dominant_color:
        color_name = get_color_name(dominant_color, colors)
        result = {
            'color_name': color_name,
            'resolution_quality': resolution_quality,
            'image_filename': image_file.filename
        }
    else:
        result = {
            'error': 'Failed to retrieve color details.'
        }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
