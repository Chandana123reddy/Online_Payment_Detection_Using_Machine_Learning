# Import necessary libraries
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Assuming you already have X_train and y_train for model training

# Function to preprocess and predict
def predict(user_data, model):
    user_input_array = [[
        user_data['type'],
        user_data['isFlaggedFraud'],
        user_data['scaled_step'],
        user_data['scaled_amount'],
        user_data['scaled_oldbalanceOrg'],
        user_data['scaled_newbalanceOrig'],
        user_data['scaled_oldbalanceDest'],
        user_data['scaled_newbalanceDest']
    ]]
    prediction = model.predict(user_input_array)
    return prediction[0]

# Function to create and train the model
def train_model(X_train, y_train):
    dt = DecisionTreeClassifier()
    scaler = StandardScaler()
    pipeline = Pipeline([('scaler', scaler), ('dt', dt)])

    tree_params = {
        'dt__criterion': ["gini", "entropy"],
        'dt__max_depth': list(range(2, 4, 1)),
        'dt__min_samples_leaf': list(range(5, 7, 1))
    }

    search = GridSearchCV(pipeline, tree_params, cv=5, scoring='accuracy')
    search.fit(X_train, y_train)

    best_params = search.best_params_
    best_params = {key.replace('dt__', ''): value for key, value in best_params.items()}

    best_model = Pipeline([
        ('scaler', StandardScaler()),
        ('dt', DecisionTreeClassifier(**best_params))
    ])

    return best_model

# Main Streamlit app
def main():
    st.title("ML Project Streamlit App")
    st.sidebar.header("User Input")

    # Get user input
    user_data = {}
    user_data['type'] = st.sidebar.slider("Payment Type", 0, 1, 0)
    user_data['isFlaggedFraud'] = st.sidebar.slider("Is Flagged Fraud", 0, 1, 0)
    user_data['scaled_step'] = st.sidebar.slider("Step", 0.0, 1.0, 0.5)
    user_data['scaled_amount'] = st.sidebar.slider("Amount", 0.0, 1.0, 0.5)
    user_data['scaled_oldbalanceOrg'] = st.sidebar.slider("Old Balance Org", 0.0, 1.0, 0.5)
    user_data['scaled_newbalanceOrig'] = st.sidebar.slider("New Balance Orig", 0.0, 1.0, 0.5)
    user_data['scaled_oldbalanceDest'] = st.sidebar.slider("Old Balance Dest", 0.0, 1.0, 0.5)
    user_data['scaled_newbalanceDest'] = st.sidebar.slider("New Balance Dest", 0.0, 1.0, 0.5)

    # Train the model
    model = train_model(X_train, y_train)

    if st.sidebar.button("Make Prediction"):
        # Make a prediction
        prediction = predict(user_data, model)
        st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()