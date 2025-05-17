import numpy as np
import scipy.signal as signal
import streamlit as st
from PIL import Image
import base64
import pickle
import pywt
import pandas as pd
# Removed plotly.express as px since the chart is removed

# --- Set Background Image ---
def set_background(image_file)
    with open(image_file, rb) as image
        encoded = image.read()
    st.markdown(
        f
        style
        .stApp {{
            background-image url(dataimagejpg;base64,{base64.b64encode(encoded).decode()});
            background-size cover;
            background-position center;
            background-repeat no-repeat;
            background-attachment fixed;
        }}
        style
        ,
        unsafe_allow_html=True
    )

set_background(background.jpeg)

# --- Preprocessing and Feature Extraction Functions ---

def bandpass_filter(data, lowcut=1.0, highcut=20.0, fs=176, order=3)
    nyquist = 0.5  fs
    if not (0  lowcut  nyquist and 0  highcut  nyquist and lowcut  highcut)
        return data
    b, a = signal.butter(order, [lowcut  nyquist, highcut  nyquist], btype='band')
    return signal.filtfilt(b, a, data)

def downsample_filtered_data(filtered_signal, original_fs=250, target_fs=150)
    if target_fs = 0 or original_fs = 0
        return filtered_signal
    
    factor = int(original_fs  target_fs)
    
    if factor  1
        return filtered_signal
    
    if len(filtered_signal)  factor
        return filtered_signal
    
    return signal.decimate(filtered_signal, factor, zero_phase=True)

def remove_dc(signal)
    if signal.size == 0
        return signal
    return signal - np.mean(signal)

TARGET_SIGNAL_LENGTH_FOR_DWT = 115

def extract_dwt_features(signal_input, wavelet='db4', level=2)
    try
        if not isinstance(signal_input, np.ndarray) or signal_input.size == 0 or len(signal_input)  (2level)
            return np.zeros(34)
        
        coeffs = pywt.wavedec(signal_input, wavelet, level=level)
        return coeffs[0]
    except ValueError
        return np.zeros(34)

def process_single_signal_for_dwt(raw_signal)
    filtered_signal = bandpass_filter(raw_signal, fs=176)
    downsampled_signal = downsample_filtered_data(filtered_signal, original_fs=250, target_fs=150)
    dc_removed_signal = remove_dc(downsampled_signal)

    current_length = len(dc_removed_signal)
    if current_length  TARGET_SIGNAL_LENGTH_FOR_DWT
        padded_signal = np.pad(dc_removed_signal, (0, TARGET_SIGNAL_LENGTH_FOR_DWT - current_length), 'constant')
    elif current_length  TARGET_SIGNAL_LENGTH_FOR_DWT
        padded_signal = dc_removed_signal[TARGET_SIGNAL_LENGTH_FOR_DWT]
    else
        padded_signal = dc_removed_signal
    
    dwt_features = extract_dwt_features(padded_signal, wavelet='db4', level=2)
    return dwt_features

# --- Load Trained Model ---
try
    with open('svm_model_DWT.pkl', 'rb') as f
        svm_model = pickle.load(f)
    st.sidebar.success(Trained SVM model loaded successfully!)
    scaler = None
except FileNotFoundError
    st.sidebar.error(Error 'svm_model_DWT.pkl' not found. Please ensure it is in the same directory.)
    svm_model = None
    scaler = None
except Exception as e
    st.sidebar.error(fError loading model {e})
    svm_model = None
    scaler = None

# --- Image Loader & Global Image Definitions ---
try
    global_left_image = Image.open(left.jpeg)  # صورة ميسي
    global_right_image = Image.open(right.jpeg) # صورة BMW البيضاء
    global_blink_image = Image.open(blink.jpeg) # صورة BMW M5 السوداء
except FileNotFoundError
    global_left_image = Image.new('RGB', (150, 150), color = 'red')
    global_right_image = Image.new('RGB', (150, 150), color = 'blue')
    global_blink_image = Image.new('RGB', (400, 400), color = 'green')

# Class labels mapping
class_labels = {
    0 Blink,
    1 Down,
    2 Left,
    3 Right,
    4 Up
}

# --- Initialize session state ---
if prediction_made not in st.session_state
    st.session_state.prediction_made = False
if current_prediction_class not in st.session_state
    st.session_state.current_prediction_class = None
if horizontal_file_uploaded not in st.session_state
    st.session_state.horizontal_file_uploaded = None
if vertical_file_uploaded not in st.session_state
    st.session_state.vertical_file_uploaded = None
if image_order not in st.session_state
    st.session_state.image_order = {
        left_col global_left_image,
        center_col global_blink_image, # Default central image
        right_col global_right_image
    }
if last_prediction_direction not in st.session_state
    st.session_state.last_prediction_direction = None # Can be Left, Right, or None

# --- Application Layout ---
st.title(EOG Signal Classification)

col1, col2, col3 = st.columns([1, 2, 1])

with col1
    st.image(st.session_state.image_order[left_col].resize((150, 150)))

with col2
    if st.session_state.prediction_made and st.session_state.current_prediction_class
        st.subheader(fPredicted Class {st.session_state.current_prediction_class})
        
    st.image(st.session_state.image_order[center_col].resize((400, 400)))

    # Determine if navigation buttons should be disabled
    disable_nav_buttons = st.session_state.current_prediction_class in [Blink, Down, Up]

    # Navigation buttons with direction-aware logic
    cols_nav = st.columns(2)
    with cols_nav[0]
        if st.button(Previous, disabled=disable_nav_buttons)
            if st.session_state.last_prediction_direction == Left
                temp_center = st.session_state.image_order[center_col]
                st.session_state.image_order[center_col] = st.session_state.image_order[right_col]
                st.session_state.image_order[right_col] = st.session_state.image_order[left_col]
                st.session_state.image_order[left_col] = temp_center
            elif st.session_state.last_prediction_direction == Right
                temp_center = st.session_state.image_order[center_col]
                st.session_state.image_order[center_col] = st.session_state.image_order[left_col]
                st.session_state.image_order[left_col] = st.session_state.image_order[right_col]
                st.session_state.image_order[right_col] = temp_center
            else # Fallback for None or other unexpected directions
                temp_center = st.session_state.image_order[center_col]
                st.session_state.image_order[center_col] = st.session_state.image_order[left_col]
                st.session_state.image_order[left_col] = st.session_state.image_order[right_col]
                st.session_state.image_order[right_col] = temp_center
            st.rerun()

    with cols_nav[1]
        if st.button(Next, disabled=disable_nav_buttons)
            if st.session_state.last_prediction_direction == Right
                temp_center = st.session_state.image_order[center_col]
                st.session_state.image_order[center_col] = st.session_state.image_order[right_col]
                st.session_state.image_order[right_col] = st.session_state.image_order[left_col]
                st.session_state.image_order[left_col] = temp_center
            elif st.session_state.last_prediction_direction == Left
                temp_center = st.session_state.image_order[center_col]
                st.session_state.image_order[center_col] = st.session_state.image_order[left_col]
                st.session_state.image_order[left_col] = st.session_state.image_order[right_col]
                st.session_state.image_order[right_col] = temp_center
            else # Fallback for None or other unexpected directions
                temp_center = st.session_state.image_order[center_col]
                st.session_state.image_order[center_col] = st.session_state.image_order[right_col]
                st.session_state.image_order[right_col] = st.session_state.image_order[left_col]
                st.session_state.image_order[left_col] = temp_center
            st.rerun()
            
    # Display message if navigation is disabled
    if disable_nav_buttons
        st.warning(Navigation is not available for Down, Up, or Blink classes.)

with col3
    st.image(st.session_state.image_order[right_col].resize((150, 150)))

st.markdown(### Please upload your Horizontal and Vertical Signal Files)

# File Uploaders
horizontal_file = st.file_uploader(Upload Horizontal Signal (h.txt), type=[txt], key=h_signal_uploader)
vertical_file = st.file_uploader(Upload Vertical Signal (v.txt), type=[txt], key=v_signal_uploader)

# Store uploaded file content in session state to persist across reruns
if horizontal_file is not None
    st.session_state.horizontal_file_uploaded = horizontal_file
if vertical_file is not None
    st.session_state.vertical_file_uploaded = vertical_file

# Determine if Predict button should be enabled
predict_button_enabled = (st.session_state.horizontal_file_uploaded is not None and
                          st.session_state.vertical_file_uploaded is not None and
                          svm_model is not None)

if not predict_button_enabled
    st.warning(Please upload both horizontal and vertical signal files AND ensure the model is loaded to enable prediction.)
else
    st.success(Files ready! Model loaded! Click 'Predict' to classify.)

# Predict Button
if st.button(Predict, disabled=not predict_button_enabled)
    if svm_model
        try
            h_signal = np.loadtxt(st.session_state.horizontal_file_uploaded)
            v_signal = np.loadtxt(st.session_state.vertical_file_uploaded)

            features_h = process_single_signal_for_dwt(h_signal)
            features_v = process_single_signal_for_dwt(v_signal)

            if features_h.size == 0 or features_v.size == 0
                pass 
            else
                combined_features = np.concatenate((features_h, features_v)).reshape(1, -1)
                features_for_prediction = combined_features

                predicted_label_idx = svm_model.predict(features_for_prediction)[0]
                st.session_state.current_prediction_class = class_labels.get(predicted_label_idx, Unknown)
                st.session_state.prediction_made = True
                
                # --- Update image order based on prediction and set direction ---
                if st.session_state.current_prediction_class == Left
                    st.session_state.image_order = {
                        left_col global_right_image,  # BMW White to left
                        center_col global_left_image,   # Messi to center
                        right_col global_blink_image   # BMW M5 Black to right
                    }
                    st.session_state.last_prediction_direction = Left
                elif st.session_state.current_prediction_class == Right
                    st.session_state.image_order = {
                        left_col global_blink_image,   # BMW M5 Black to left
                        center_col global_right_image,  # BMW White to center
                        right_col global_left_image    # Messi to right
                    }
                    st.session_state.last_prediction_direction = Right
                elif st.session_state.current_prediction_class in [Blink, Up, Down]
                    st.session_state.image_order = {
                        left_col global_left_image,    # Messi to left
                        center_col global_blink_image,  # BMW M5 Black to center
                        right_col global_right_image   # BMW White to right
                    }
                    st.session_state.last_prediction_direction = None # No specific direction for these
                else
                    st.session_state.image_order = {
                        left_col global_left_image,
                        center_col global_blink_image,
                        right_col global_right_image
                    }
                    st.session_state.last_prediction_direction = None
                st.rerun()
        except Exception as e
            st.error(fAn error occurred during prediction {e})
    else
        st.error(Model not loaded. Cannot perform prediction.)

# --- Removed Model Performance Section ---
# st.markdown(### Model Performance (Precision, Recall, F1-score))
# st.dataframe(performance_df.set_index('Class'))
# melted_df = performance_df.melt(id_vars=['Class'], var_name='Metric', value_name='Score')
# melted_df[['Model', 'Metric_Type']] = melted_df['Metric'].str.split(' ', n=1, expand=True)
# fig = px.bar(melted_df, x='Class', y='Score', color='Metric',
#              barmode='group',
#              title='Precision, Recall, and F1-score for each Model and Class',
#              labels={'Score' 'Score', 'Class' 'Classes'},
#              height=500, width=800)
# st.plotly_chart(fig, use_container_width=True)