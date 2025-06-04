import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ------------------------------
# Define the LSTM model
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
# Create sequences for multivariate time series
# ------------------------------
def create_sequences_multivariate(X_data, y_data, seq_len, pred_len):
    xs, ys = [], []
    for i in range(len(X_data) - seq_len - pred_len + 1):
        xs.append(X_data[i:i+seq_len])
        ys.append(y_data[i+seq_len:i+seq_len+pred_len])
    return np.array(xs), np.array(ys)

# ------------------------------
# Streamlit App Layout
# ------------------------------
st.set_page_config(layout="wide")
st.title("📊 LSTM Multivariate Time Series Forecaster")

col1, col2 = st.columns([1, 3])

with col1:
    st.header("Configuration")
    mode = st.radio("Mode", ["Train New Model", "Load Pretrained Model"])

    if mode == "Train New Model":
        uploaded_file = st.file_uploader("Upload Training Excel (.xlsx)", type=["xlsx"])
        seq_length = st.number_input("Sequence Length", 5, 100, 10)
        forecast_horizon = st.number_input("Forecast Horizon (Steps Ahead)", 1, 20, 1)
        epochs = st.number_input("Epochs", 10, 500, 100)
        lr = st.number_input("Learning Rate", 0.0001, 0.1, 0.01)
        hidden_size = st.number_input("Hidden Size", 10, 200, 50)
        num_layers = st.slider("LSTM Layers", 1, 3, 1)

        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            all_columns = df.columns.tolist()
            input_columns = st.multiselect("Input Feature Columns", all_columns, default=all_columns[:-1])
            target_column = st.selectbox("Target Column", all_columns, index=len(all_columns)-1)

        if st.button("Train Model") and uploaded_file:
            X_raw = df[input_columns].values
            y_raw = df[[target_column]].values

            X_scaler = MinMaxScaler()
            y_scaler = MinMaxScaler()

            X_scaled = X_scaler.fit_transform(X_raw)
            y_scaled = y_scaler.fit_transform(y_raw)

            X_seq, y_seq = create_sequences_multivariate(X_scaled, y_scaled, seq_length, forecast_horizon)

            X_tensor = torch.tensor(X_seq, dtype=torch.float32)
            y_tensor = torch.tensor(y_seq, dtype=torch.float32)

            train_size = int(len(X_tensor) * 0.8)
            X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
            y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

            model = LSTMModel(len(input_columns), hidden_size, num_layers, forecast_horizon)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
                y_pred = model(X_test).detach().numpy()

            y_test_np = y_test.numpy()
            y_pred_inv = y_scaler.inverse_transform(y_pred)
            y_test_inv = y_scaler.inverse_transform(y_test_np)

            st.session_state.model = model
            st.session_state.X_scaler = X_scaler
            st.session_state.y_scaler = y_scaler
            st.session_state.seq_length = seq_length
            st.session_state.input_columns = input_columns
            st.session_state.forecast_horizon = forecast_horizon

            with col2:
                fig1, ax1 = plt.subplots()
                ax1.plot(train_losses)
                ax1.set_title("Training Loss")
                st.pyplot(fig1)

                fig2, ax2 = plt.subplots()
                ax2.plot(y_test_inv[:, 0], label="Actual")
                ax2.plot(y_pred_inv[:, 0], label="Predicted")
                ax2.set_title("Validation (First Forecast Step)")
                ax2.legend()
                st.pyplot(fig2)

                st.write(f"**MSE:** {mean_squared_error(y_test_inv[:, 0], y_pred_inv[:, 0]):.4f}")
                st.write(f"**R² Score:** {r2_score(y_test_inv[:, 0], y_pred_inv[:, 0]):.4f}")

                if st.button("Save Trained Model"):
                    torch.save(model.state_dict(), "lstm_model.pt")
                    joblib.dump(X_scaler, "x_scaler.pkl")
                    joblib.dump(y_scaler, "y_scaler.pkl")
                    st.success("Model and scalers saved.")

    else:
        model_file = st.file_uploader("Upload Model (.pt)", type=["pt"])
        xscaler_file = st.file_uploader("Upload X Scaler (.pkl)", type=["pkl"])
        yscaler_file = st.file_uploader("Upload Y Scaler (.pkl)", type=["pkl"])
        seq_length = st.number_input("Sequence Length Used", 5, 100, 10)
        forecast_horizon = st.number_input("Forecast Horizon Used", 1, 20, 1)
        num_inputs = st.number_input("Number of Input Features", 1, 20, 1)

        if model_file and xscaler_file and yscaler_file:
            model = LSTMModel(num_inputs, 50, 1, forecast_horizon)
            model.load_state_dict(torch.load(model_file))
            model.eval()
            st.session_state.model = model
            st.session_state.X_scaler = joblib.load(xscaler_file)
            st.session_state.y_scaler = joblib.load(yscaler_file)
            st.session_state.seq_length = seq_length
            st.session_state.forecast_horizon = forecast_horizon
            st.success("Model and scalers loaded.")

# ------------------------------
# Validation Section
# ------------------------------
with col2:
    st.header("📊 Forecast Using Trained Model")

    if "model" in st.session_state:
        val_input = st.text_area("Enter comma-separated values for each feature (one row per line)",
                                 placeholder="e.g.\n1.0,2.0\n1.1,2.2\n...")

        if st.button("Forecast"):
            try:
                rows = [list(map(float, line.split(","))) for line in val_input.strip().splitlines() if line]
                val_array = np.array(rows)
                if val_array.shape[0] < st.session_state.seq_length:
                    st.error("Not enough time steps provided.")
                elif val_array.shape[1] != len(st.session_state.input_columns):
                    st.error(f"Expected {len(st.session_state.input_columns)} features per row.")
                else:
                    val_scaled = st.session_state.X_scaler.transform(val_array)
                    input_seq = val_scaled[-st.session_state.seq_length:].reshape(1, st.session_state.seq_length, -1)
                    input_tensor = torch.tensor(input_seq, dtype=torch.float32)
                    pred = st.session_state.model(input_tensor).detach().numpy()
                    pred_inv = st.session_state.y_scaler.inverse_transform(pred)

                    st.success(f"Predicted next {st.session_state.forecast_horizon} values:")
                    st.write(pred_inv.flatten())

            except Exception as e:
                st.error(f"Error parsing input: {e}")
    else:
        st.info("Train or load a model first.")
