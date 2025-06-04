import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- LSTM Model ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# --- Helper Functions ---
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def plot_metrics(train_losses, test_y, pred_y):
    fig1, ax1 = plt.subplots()
    ax1.plot(train_losses)
    ax1.set_title("Training Loss")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(test_y, label="Actual")
    ax2.plot(pred_y, label="Predicted")
    ax2.set_title("Validation")
    ax2.legend()
    st.pyplot(fig2)

    mse = mean_squared_error(test_y, pred_y)
    r2 = r2_score(test_y, pred_y)
    st.write(f"**MSE:** {mse:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("LSTM Time Series Forecasting (PyTorch)")

with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
    target_column = st.text_input("Target Column Name", value="y")
    seq_length = st.number_input("Sequence Length", 5, 100, 10)
    epochs = st.number_input("Epochs", 10, 500, 100)
    lr = st.number_input("Learning Rate", 0.0001, 0.1, 0.01)
    hidden_size = st.number_input("Hidden Size", 10, 200, 50)
    num_layers = st.slider("LSTM Layers", 1, 3, 1)

    if st.button("Train Model"):
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            if target_column not in df.columns:
                st.error(f"Column '{target_column}' not found.")
                st.stop()

            series = df[target_column].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled_series = scaler.fit_transform(series)

            X, y = create_sequences(scaled_series, seq_length)
            X_torch = torch.tensor(X, dtype=torch.float32)
            y_torch = torch.tensor(y, dtype=torch.float32)

            train_size = int(len(X) * 0.8)
            X_train, X_test = X_torch[:train_size], X_torch[train_size:]
            y_train, y_test = y_torch[:train_size], y_torch[train_size:]

            model = LSTMModel(1, hidden_size, num_layers, 1)
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
                pred_y = model(X_test.unsqueeze(-1)).numpy()

            actual_y = y_test.numpy()
            pred_y_rescaled = scaler.inverse_transform(pred_y)
            actual_y_rescaled = scaler.inverse_transform(actual_y.reshape(-1, 1))

            plot_metrics(train_losses, actual_y_rescaled, pred_y_rescaled)

            # Save model and scaler in session
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.seq_length = seq_length

# --- Validation Section ---
st.header("Manual Validation")
if "model" in st.session_state:
    user_input = st.text_area("Enter new data (comma-separated)", placeholder="e.g., 2.3, 2.5, 2.7, 2.9...")
    if st.button("Validate"):
        try:
            vals = np.array([float(i.strip()) for i in user_input.split(",")]).reshape(-1, 1)
            if len(vals) < st.session_state.seq_length:
                st.warning("Not enough values to match sequence length.")
                st.stop()

            vals_scaled = st.session_state.scaler.transform(vals)
            input_seq = vals_scaled[-st.session_state.seq_length:].reshape(1, st.session_state.seq_length, 1)
            input_tensor = torch.tensor(input_seq, dtype=torch.float32)
            st.session_state.model.eval()
            with torch.no_grad():
                pred = st.session_state.model(input_tensor).numpy()

            pred_inverse = st.session_state.scaler.inverse_transform(pred)
            st.success(f"Predicted Next Value: {pred_inverse[0][0]:.4f}")
        except Exception as e:
            st.error(f"Invalid input: {e}")
