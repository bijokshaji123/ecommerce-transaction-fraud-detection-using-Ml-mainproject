import pickle
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
import joblib
from flask_cors import CORS
model = joblib.load(open("fraud_detection_model.pkl", "rb"))
scaler = joblib.load(open("scaler.pkl", "rb"))
payment_mapping = {"debit card": 0, "credit card": 1, "PayPal": 2, "bank transfer": 3}
product_mapping = {"home & garden": 0, "electronics": 1, "toys & games": 2, "clothing": 3, "health & beauty": 4}
device_mapping = {"desktop": 0, "mobile": 1, "tablet": 2}
numeric_columns = ["Transaction Amount", "Quantity", "Customer Age", "Account Age Days", "Transaction Hour"]
categorical_columns = ["Payment Method", "Product Category", "Device Used"]
app = Flask(__name__)
CORS(app)
@app.route("/submit", methods=["POST"])
def submit():
    if request.method == "POST":
        try:
            data = request.get_json()
            print("Received JSON data:", data)
            payment_method = data["payment_method"]
            product_category = data["product_category"]
            device_used = data["device_used"]
            transaction_amount = float(data["transaction_amount"])
            quantity = int(data["quantity"])
            customer_age = int(data["customer_age"])
            account_age_days = int(data["account_age_days"])
            transaction_hour = int(data["transaction_hour"])
            address_match = int(data["address_match"])
            print("Extracted values:")
            print(f"payment_method: {payment_method}")
            print(f"product_category: {product_category}")
            print(f"device_used: {device_used}")
            print(f"transaction_amount: {transaction_amount}")
            print(f"quantity: {quantity}")
            print(f"customer_age: {customer_age}")
            print(f"account_age_days: {account_age_days}")
            print(f"transaction_hour: {transaction_hour}")
            print(f"address_match: {address_match}")
            payment_method_mapped = payment_mapping.get(payment_method, -1)
            product_category_mapped = product_mapping.get(product_category, -1)
            device_used_mapped = device_mapping.get(device_used, -1)
            print("Mapped categorical values:")
            print(f"payment_method_mapped: {payment_method_mapped}")
            print(f"product_category_mapped: {product_category_mapped}")
            print(f"device_used_mapped: {device_used_mapped}")
            if -1 in [payment_method_mapped, product_category_mapped, device_used_mapped]:
                return jsonify({"error": "Invalid input for categorical values"}), 400
            input_df = pd.DataFrame([[
                transaction_amount, payment_method_mapped, product_category_mapped, quantity,
                customer_age, device_used_mapped, address_match, account_age_days, transaction_hour
            ]], columns=[
                "Transaction Amount", "Payment Method", "Product Category", "Quantity",
                "Customer Age", "Device Used", "Address Match", "Account Age Days", "Transaction Hour"
            ])
            print("Input DataFrame before scaling:")
            print(input_df.to_string())
            input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])
            print("Input DataFrame after scaling:")
            print(input_df.to_string())
            prediction = model.predict(input_df)
            print("Prediction:", prediction)
            result_text = "Fraudulent" if prediction[0] == 1 else "Not Fraudulent"
            print("Result text:", result_text)
            response = {
                "data": {
                    "predicted_fraud_status": result_text
                }
            }
            return jsonify(response), 200
        except KeyError as e:
            print(f"KeyError: {str(e)}")
            return jsonify({"error": f"Missing field: {str(e)}"}), 400
        except ValueError as e:
            print(f"ValueError: {str(e)}")
            return jsonify({"error": f"Invalid value: {str(e)}"}), 400
        except Exception as e:
            print(f"Exception: {str(e)}")
            return jsonify({"error": f"Server error: {str(e)}"}), 500
if __name__ == "__main__":
    app.run(port=5000, debug=True, threaded=True)