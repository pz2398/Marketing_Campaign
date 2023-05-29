import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

label_legend = {
    "job": ["admin.",  "blue-collar","entrepreneur","housemaid" , "management","retired","self-employed", "services",
           "student", "technician", "unemployed",],
    "marital": ["divorced","married","single"],
    "education": ["unknown", "secondary", "primary", "tertiary"],
    "default": ["yes", "no"],
    "housing": ["yes", "no"],
    "loan":["yes", "no"]
}


categorical_columns = ["job", "marital", "education", "default", "housing","loan"]
numeric_columns = ["age", "balance"]

def collect_user_input():
    user_input = {}

    user_input["age"] = int(st.number_input("Enter age:", value=35))
    user_input["balance"] = int(st.number_input("Enter balance:", value=1342))

    # for column in numeric_columns:
    #     if column != "age":
    #         user_input[column] = st.number_input(f"Enter {column}:")
    
    for column in categorical_columns:
        user_input[column] = st.selectbox(f"Select {column}:", label_legend[column])

    return user_input

def main():
    st.title("Marketing Campaign classifier")
    st.write("Enter the values for each column")

    user_input = collect_user_input()

    if st.button("Make Prediction"):
        df = pd.DataFrame([user_input])
        df=df[["age","education","default","balance","housing","loan","job","marital"]]
        # st.write("Created dataset:")
        # st.write(df)
        binary_mapping = {"yes": 1, "no": 0}
        # for column in df.columns:
        #     if set(df[column].unique()) == {"yes", "no"}:
        #         df[column] = df[column].replace(binary_mapping)
                
        education_mapping = {'primary': 1, 'secondary': 2,'tertiary':3}
        df["education"] = df["education"].replace(education_mapping)
        df["default"] = df["default"].replace(binary_mapping)
        df["housing"] = df["housing"].replace(binary_mapping)
        df["loan"] = df["loan"].replace(binary_mapping)



        # encoded_df = pd.get_dummies(df, columns=label_legend.keys(), drop_first=False)
        df['job'] = pd.Categorical(df['job'], categories=label_legend["job"])
        df['marital'] = pd.Categorical(df['marital'], categories=label_legend["marital"])

        dummies = pd.get_dummies(df)
        model = pickle.load(open("model.pkl", "rb"))
        predictions = model.predict(dummies)
        # print(predictions)
        probabilities = model.predict_proba(dummies)
        # st.write("Predictions:")
        # st.write(predictions)
        # st.write("Probabilities:")
        # st.write(probabilities)

                # Map predictions to "subscribe" or "not subscribe"
        subscription_mapping = {0: "not subscribe", 1: "subscribe"}
        subscription_status = subscription_mapping[predictions[0]] 

        # Display the prediction and probability
        st.write("#### Prediction:")
        st.write(subscription_status)

        # Display the probability of subscription
        st.write("#### Probability of subscription:")
        st.write(f"{probabilities[0][1]:.2f}")



        # st.write(dummies)

if __name__ == "__main__":
    main()

