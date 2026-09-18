"""Small shared helpers for the AGS package."""


def safe_index(data, idx):
    """Index a numpy array or a pandas DataFrame/Series the same way."""
    if hasattr(data, "iloc"):
        return data.iloc[idx]
    return data[idx]
