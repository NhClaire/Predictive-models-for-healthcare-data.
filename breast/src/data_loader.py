import pandas as pd


def load_data(config: dict) -> tuple:
    """Return (X: DataFrame, y: Series) from config-specified source(s)."""
    if config.get("labels_path"):
        data = pd.read_csv(config["data_path"])
        labels = pd.read_csv(config["labels_path"])
        df = pd.merge(data, labels, on=config["merge_col"])
        drop_cols = [config["merge_col"], config["label_col"]]
        X = df.drop(columns=drop_cols)
        y = df[config["label_col"]]
    else:
        df = pd.read_csv(config["data_path"])
        drop_cols = [config["label_col"]]
        if config.get("sample_col"):
            drop_cols.append(config["sample_col"])
        X = df.drop(columns=drop_cols)
        y = df[config["label_col"]]
    return X, y
