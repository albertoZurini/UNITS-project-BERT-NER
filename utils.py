from typing import List, Dict
from const import START_TAG, STOP_TAG

tag_to_idx = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

def save_to_pickle(data: Dict[str, List[Dict]], filename: str = "inspec_onehot.pkl"):
    """
    Save the processed dataset splits to a pickle file
    """
    print(f"Saving processed dataset to {filename}...")
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Dataset successfully saved to {filename}")


def read_from_pickle(filename: str = "inspec_onehot.pkl") -> Dict[str, List[Dict]]:
    """
    Read the processed dataset splits from a pickle file
    """
    print(f"Loading processed dataset from {filename}...")
    with open(filename, "rb") as f:
        data = pickle.load(f)
    print(f"Dataset successfully loaded from {filename}")
    return data
