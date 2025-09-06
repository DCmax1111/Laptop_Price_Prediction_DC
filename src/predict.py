# predict.py

from streamlit.utils import get_user_inputs, predict_price

if __name__ == "__main__":
    user_input = get_user_inputs()
    price = predict_price(user_input)
    print(f"\n💰 Predicted Laptop Price: €{price}")
