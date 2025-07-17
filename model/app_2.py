import pickle
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained model and scaler
model = joblib.load(open("fraud_detection_model.pkl", "rb"))
scaler = joblib.load(open("scaler.pkl", "rb"))

# Define categorical mappings (same as used during training)
payment_mapping = {"debit card": 0, "credit card": 1, "PayPal": 2, "bank transfer": 3}
product_mapping = {"home & garden": 0, "electronics": 1, "toys & games": 2, "clothing": 3, "health & beauty": 4}
device_mapping = {"desktop": 0, "mobile": 1, "tablet": 2}

# Define feature columns
numeric_columns = ["Transaction Amount", "Quantity", "Customer Age", "Account Age Days", "Transaction Hour"]
categorical_columns = ["Payment Method", "Product Category", "Device Used"]

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submit():
    if request.method == "POST":
        # Get form input
        payment_method = request.form["payment_method"]
        product_category = request.form["product_category"]
        device_used = request.form["device_used"]
        transaction_amount = float(request.form["transaction_amount"])
        quantity = int(request.form["quantity"])
        customer_age = int(request.form["customer_age"])
        account_age_days = int(request.form["account_age_days"])
        transaction_hour = int(request.form["transaction_hour"])
        address_match = int(request.form["address_match"])  # 1 if same, 0 if different

        # Convert categorical inputs using predefined mappings
        payment_method = payment_mapping.get(payment_method, -1)
        product_category = product_mapping.get(product_category, -1)
        device_used = device_mapping.get(device_used, -1)

        # Ensure valid category encoding
        if -1 in [payment_method, product_category, device_used]:
            return "Invalid input for categorical values."

        # Create a DataFrame for the input
        input_df = pd.DataFrame([[
            transaction_amount, payment_method, product_category, quantity,
            customer_age, device_used, address_match, account_age_days, transaction_hour
        ]], columns=[
            "Transaction Amount", "Payment Method", "Product Category", "Quantity",
            "Customer Age", "Device Used", "Address Match", "Account Age Days", "Transaction Hour"
        ])

        # Scale numeric columns
        input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])

        # Make prediction
        prediction = model.predict(input_df)

        # Display result
        result_text = "Transaction is Fraudulent!" if prediction[0] == 1 else "Transaction is Not Fraudulent!"
        return render_template("result.html", labels=result_text)

if __name__ == "__main__":
    app.run(debug=True)
