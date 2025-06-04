import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ------------------------------
# LSTM Model
# ------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# ------------------------------
# Create Sequences (Multivariate, Multi-step)
# ------------------------------
def create_sequences_multivariate(X_data, y_data, seq_len, pred_len):
    xs, ys = [], []
    for i in range(len(X_data) - seq_len - pred_len + 1):
        xs.append(X_data[i:i+seq_len])
        ys.append(y_data[i+seq_len:i+seq_len+pred_len].flatten())
    return np.array(xs), np.array(ys)

# ------------------------------
# Streamlit App
# ------------------------------
st.set_page_config(layout="wide")
st.title("📊 LSTM Multivariate Time Series Forecaster")

col1, col2 = st.columns([1, 3])

with col1:
    st.header("Configuration")
    mode = st.radio("Mode", ["Train New Model", "Load Pretrained Model"])

    if mode == "Train New Model":
        uploaded_file = st.file_uploader("Upload Training Excel (.xlsx)", type=["xlsx"])
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.write("Columns:", df.columns.tolist())

        sequence_length = st.number_input("Sequence Length", 5, 100, 10)
        forecast_horizon = st.number_input("Forecast Horizon (Steps Ahead)", 1, 20, 1)
        epochs = st.number_input("Epochs", 10, 500, 100)
        lr = st.number_input("Learning Rate", 0.0001, 0.1, 0.01)
        hidden_size = st.number_input("Hidden Size", 10, 200, 50)
        num_layers = st.slider("LSTM Layers", 1, 3, 1)

        optimizer_choice = st.selectbox("Training Algorithm", ["Adam", "SGD", "RMSprop"])
        loss_choice = st.selectbox("Loss Function", ["MSE", "MAE"])

        input_columns = st.multiselect("Input Feature Columns", df.columns.tolist() if uploaded_file else [])
        target_column = st.selectbox("Target Column", df.columns.tolist() if uploaded_file else [])

        if st.button("Train Model") and uploaded_file:
            X_raw = df[input_columns].values
            y_raw = df[[target_column]].values

            x_scaler = MinMaxScaler()
            y_scaler = MinMaxScaler()
            X_scaled = x_scaler.fit_transform(X_raw)
            y_scaled = y_scaler.fit_transform(y_raw)

            X_seq, y_seq = create_sequences_multivariate(X_scaled, y_scaled, sequence_length, forecast_horizon)

            X_tensor = torch.tensor(X_seq, dtype=torch.float32)
            y_tensor = torch.tensor(y_seq, dtype=torch.float32)

            train_size = int(0.8 * len(X_tensor))
            X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
            y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

            model = LSTMModel(len(input_columns), hidden_size, num_layers, forecast_horizon)

            # Loss and optimizer
            criterion = nn.MSELoss() if loss_choice == "MSE" else nn.L1Loss()
            optimizer = {
                "Adam": optim.Adam,
                "SGD": optim.SGD,
                "RMSprop": optim.RMSprop
            }[optimizer_choice](model.parameters(), lr=lr)

            train_losses = []
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                output = model(X_train)
                loss = criterion(output, y_train)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            model.eval()
            with torch.no_grad():
                preds = model(X_test).numpy()

            preds_inv = y_scaler.inverse_transform(preds)
            actual_inv = y_scaler.inverse_transform(y_test.numpy())

            st.session_state.model = model
            st.session_state.x_scaler = x_scaler
            st.session_state.y_scaler = y_scaler
            st.session_state.sequence_length = sequence_length
            st.session_state.forecast_horizon = forecast_horizon
            st.session_state.input_columns = input_columns

            with col2:
                fig1, ax1 = plt.subplots()
                ax1.plot(train_losses)
                ax1.set_title("Training Loss")
                st.pyplot(fig1)

                fig2, ax2 = plt.subplots()
                ax2.plot(actual_inv[:, 0], label="Actual")
                ax2.plot(preds_inv[:, 0], label="Predicted")
                ax2.set_title("Validation (first step only)")
                ax2.legend()
                st.pyplot(fig2)

                st.write(f"**MSE:** {mean_squared_error(actual_inv, preds_inv):.4f}")
                st.write(f"**R² Score:** {r2_score(actual_inv, preds_inv):.4f}")

                # Save logic using trigger and rerun
                if "save_model_trigger" not in st.session_state:
                    st.session_state.save_model_trigger = False

                if st.session_state.get("model") and st.button("Save Trained Model"):
                    st.session_state.save_model_trigger = True
                    st.rerun()

                if st.session_state.save_model_trigger:
                    torch.save(st.session_state.model.state_dict(), "lstm_model.pt")
                    joblib.dump((st.session_state.x_scaler, st.session_state.y_scaler), "scalers.pkl")
                    st.success("✅ Model and scalers saved as 'lstm_model.pt' and 'scalers.pkl'")
                    st.session_state.save_model_trigger = False

    else:
        model_file = st.file_uploader("Upload Model (.pt)", type=["pt"])
        scaler_file = st.file_uploader("Upload Scalers (.pkl)", type=["pkl"])
        input_columns = st.text_input("Input Columns (comma-separated)", value="x1,x2")
        sequence_length = st.number_input("Sequence Length", 5, 100, 10)
        forecast_horizon = st.number_input("Forecast Horizon", 1, 20, 1)

        if model_file and scaler_file:
            input_cols = [col.strip() for col in input_columns.split(",")]
            model = LSTMModel(len(input_cols), 50, 1, forecast_horizon)
            model.load_state_dict(torch.load(model_file))
            model.eval()

            x_scaler, y_scaler = joblib.load(scaler_file)

            st.session_state.model = model
            st.session_state.x_scaler = x_scaler
            st.session_state.y_scaler = y_scaler
            st.session_state.sequence_length = sequence_length
            st.session_state.forecast_horizon = forecast_horizon
            st.session_state.input_columns = input_cols
            st.success("Pretrained model and scalers loaded.")

# ------------------------------
# Validation Panel
# ------------------------------
with col2:
    st.header("🔎 Validate Model")

    if "model" in st.session_state:
        method = st.radio("Validation Input", ["Manual Entry", "Upload Excel"])

        if method == "Manual Entry":
            val_input = st.text_area("Enter values (comma-separated rows):",
                                     placeholder="e.g., 1.0,2.0,3.0\n4.0,5.0,6.0")
            if st.button("Validate (Manual)") and val_input:
                try:
                    lines = val_input.strip().split("\n")
                    values = np.array([[float(val) for val in line.split(",")] for line in lines])
                    if len(values) < st.session_state.sequence_length:
                        st.warning("Not enough rows for the sequence length.")
                    else:
                        seq = values[-st.session_state.sequence_length:]
                        seq_scaled = st.session_state.x_scaler.transform(seq)
                        x_tensor = torch.tensor(seq_scaled.reshape(1, st.session_state.sequence_length, -1), dtype=torch.float32)
                        with torch.no_grad():
                            pred = st.session_state.model(x_tensor).numpy()
                        pred_inv = st.session_state.y_scaler.inverse_transform(pred)
                        st.success(f"Prediction (next {st.session_state.forecast_horizon} steps): {pred_inv.flatten()}")
                except Exception as e:
                    st.error(f"Error: {e}")

        elif method == "Upload Excel":
            val_file = st.file_uploader("Upload Excel for Validation", type=["xlsx"], key="val_upload")
            if val_file:
                df_val = pd.read_excel(val_file)
                try:
                    values = df_val[st.session_state.input_columns].values
                    if len(values) < st.session_state.sequence_length:
                        st.warning("Not enough rows for sequence length.")
                    else:
                        seq = values[-st.session_state.sequence_length:]
                        seq_scaled = st.session_state.x_scaler.transform(seq)
                        x_tensor = torch.tensor(seq_scaled.reshape(1, st.session_state.sequence_length, -1), dtype=torch.float32)
                        with torch.no_grad():
                            pred = st.session_state.model(x_tensor).numpy()
                        pred_inv = st.session_state.y_scaler.inverse_transform(pred)
                        st.success(f"Prediction (next {st.session_state.forecast_horizon} steps): {pred_inv.flatten()}")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Please train or load a model first.")
