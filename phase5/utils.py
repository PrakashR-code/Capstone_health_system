import pandas as pd

def prepare_features(data, model):

    df = pd.DataFrame([data])

    # One-hot encoding
    df = pd.get_dummies(
        df,
        columns=["department","visit_type"]
    )

    # Align with model features
    model_features = model.feature_names_in_

    df = df.reindex(columns=model_features, fill_value=0)

    return df