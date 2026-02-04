def df_cleaning(df):
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna(subset=["Ratings", "Review text"]).reset_index(drop=True)

    df["sentiments"] = df["Ratings"].apply(
        lambda x: "positive" if x >= 4 else "negative"
    )
    df["label"] = df["sentiments"].map({"negative": 0, "positive": 1})

    return df