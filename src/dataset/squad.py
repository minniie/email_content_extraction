from datasets import load_dataset


class SquadDataset():

    def __init__(
            self
        ):
        print("init")

    def load(
            self
        ):
        dataset = load_dataset('squad')
        return dataset