from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Menggunakan backend non-interactive
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = 'static/upload'
RESULT_FOLDER = 'static/result'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No file part'
        file = request.files['image']
        if file.filename == '':
            return 'No selected file'

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        output_filename = process_image(filepath)
        return render_template('result.html', result_image=output_filename)
    
    return render_template('index.html')

def process_image(filepath):
    img = cv2.imread(filepath, 0)  # Baca sebagai grayscale
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=1)
    img_dilation = cv2.dilate(img, kernel, iterations=1)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    ax = axes.ravel()

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Citra Input")
    ax[0].axis('off')

    ax[1].hist(img.ravel(), bins=256)
    ax[1].set_title("Histogram Citra Input")

    ax[2].imshow(img_erosion, cmap='gray')
    ax[2].set_title("Citra Output Erosi")
    ax[2].axis('off')

    ax[3].hist(img_erosion.ravel(), bins=256)
    ax[3].set_title("Histogram Citra Output Erosi")

    ax[4].imshow(img_dilation, cmap='gray')
    ax[4].set_title("Citra Output Dilasi")
    ax[4].axis('off')

    ax[5].hist(img_dilation.ravel(), bins=256)
    ax[5].set_title("Histogram Citra Output Dilasi")

    plt.tight_layout()
    output_path = os.path.join(RESULT_FOLDER, 'output.png')
    plt.savefig(output_path)
    plt.close()
    
    return 'result/output.png'

if __name__ == '__main__':
    app.run(debug=True)
