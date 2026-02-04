from training.data_loader import load_data
from training.cleaning import df_cleaning
from training.model_selection import train_and_select_model
from app.utils.preprocessing import preprocess_text
import joblib
from prefect import flow


@flow(name="Sentiment Training Flow")
def training_flow(csv_path):

    df = load_data(csv_path)
    df = df_cleaning(df)

    df["Review text"] = df["Review text"].apply(preprocess_text)

    model = train_and_select_model(df)

    joblib.dump(model, "models/sentiment_model.joblib")
    print("Model saved successfully.")


if __name__ == "__main__":
    training_flow("data/data.csv")