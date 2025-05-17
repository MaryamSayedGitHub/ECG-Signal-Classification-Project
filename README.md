# 👁️ EOG Signal Predictor

This project is a Streamlit-based web application for classifying Electrooculography (EOG) signals into different eye movement classes using machine learning. It supports two feature extraction methods: **DWT (Discrete Wavelet Transform)** and **TSFEL (Time Series Feature Extraction Library)**. Users upload horizontal and vertical signal files, and the app predicts the corresponding eye movement.


---

## 🌟 Features

- ✅ Upload and classify horizontal (`h.txt`) and vertical (`v.txt`) EOG signal files
- 🎯 Two feature extraction methods:
  - DWT + Random Forest Classifier
  - TSFEL + SVC Classifier
- 📊 Displays prediction results and relevant images
- 🌄 Custom background support for a better UI experience

---

## 🧪 Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```
streamlit==1.34.0
numpy==1.24.4
scipy==1.11.4
Pillow==9.5.0
pandas==2.2.2
scikit-learn==1.3.2
PyWavelets==1.4.1
tsfel==0.1.4

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>EOG Signal Classification</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      line-height: 1.6;
      background-color: #f9f9f9;
      padding: 2rem;
      color: #333;
    }
    h1, h2 {
      color: #2c3e50;
    }
    code {
      background-color: #f4f4f4;
      padding: 2px 6px;
      border-radius: 4px;
    }
    pre {
      background-color: #f4f4f4;
      padding: 1em;
      border-radius: 4px;
      overflow-x: auto;
    }
    ul {
      padding-left: 1.2em;
    }
  </style>
</head>
<body>

  <h1>👁️ EOG Signal Classification</h1>

  <h2>🧠 Model Details</h2>
  <p><strong>DWT + Random Forest:</strong><br>
  Extracts time-frequency domain features from both horizontal and vertical signals using Discrete Wavelet Transform (DWT). Features are concatenated and standardized before prediction using a Random Forest model.</p>

  <p><strong>TSFEL + SVC:</strong><br>
  Extracts statistical time-domain and frequency-domain features using TSFEL (Time Series Feature Extraction Library) from the horizontal signal only. These features are used to predict the class using an SVC model.</p>

  <h2>📤 Input Format</h2>
  <p>The application expects two plain text files:</p>
  <ul>
    <li><strong>Horizontal Signal File:</strong> <code>h.txt</code></li>
    <li><strong>Vertical Signal File:</strong> <code>v.txt</code></li>
  </ul>
  <p>Each file must contain one float amplitude value per line.</p>
  <pre><code>0.15
0.14
0.11
...</code></pre>

  <h2>🔁 Processing Steps</h2>
  <h3>Preprocessing</h3>
  <ul>
    <li>Bandpass filtering (1–20 Hz)</li>
    <li>Downsampling from 250 Hz to 150 Hz</li>
    <li>DC offset removal</li>
  </ul>

  <h3>Feature Extraction</h3>
  <ul>
    <li><strong>DWT:</strong> From both horizontal and vertical signals</li>
    <li><strong>TSFEL:</strong> From horizontal signal only</li>
  </ul>

  <h3>Prediction</h3>
  <ul>
    <li>Model used: <code>Random Forest</code> or <code>SVC</code></li>
    <li>Scikit-learn used for classification</li>
  </ul>

  <h2>🎯 Predicted Classes</h2>
  <table border="1" cellpadding="5" cellspacing="0">
    <thead>
      <tr><th>Label</th><th>Movement</th></tr>
    </thead>
    <tbody>
      <tr><td>0</td><td>Blink</td></tr>
      <tr><td>1</td><td>Down</td></tr>
      <tr><td>2</td><td>Left</td></tr>
      <tr><td>3</td><td>Right</td></tr>
      <tr><td>4</td><td>Up</td></tr>
    </tbody>
  </table>

  <h2>🚀 Running the App</h2>
  <p>Run the app locally:</p>
  <pre><code>streamlit run app.py</code></pre>

  <p>Or on Google Colab:</p>
  <pre><code>!streamlit run app.py &amp;&amp; npx localtunnel --port 8501</code></pre>

 
  ![image](https://github.com/user-attachments/assets/1f7774c9-b8ea-44e2-aafa-4314ffba42f6)
  
  ![image](https://github.com/user-attachments/assets/ea718d95-2426-4609-bbeb-c9fcb394532d)
  
  ![image](https://github.com/user-attachments/assets/3ad77914-4a2f-4cfd-a16e-de6e6d34c842)
  
  ![image](https://github.com/user-attachments/assets/91be3f24-6be9-4d15-bb95-c069c51ef20f)
  
  ![image](https://github.com/user-attachments/assets/88edc428-9049-4f47-994f-23d151a5b47f)
  
  ![image](https://github.com/user-attachments/assets/ca168c18-8816-4b8f-8249-ef087732798b)


  <h2>👤 Author</h2>
  <p>Built with ❤️ for research and educational purposes.<br>
  Feel free to extend and improve!</p>

  <h2>📃 License</h2>
  <p>This project is licensed under the MIT License.</p>

</body>
</html>
