import numpy as np

class ScenarioWiseSampler:
    def __init__(self, input_path: str, indices):
        data_dict = np.load(input_path, allow_pickle=True).item()
        self.data = data_dict["data"]        # (N, 365, #features)
        self.mapping = data_dict["mapping"]  # (N, 3) => (scenario, point_id, year)
        self.columns = list(data_dict["columns"])
        self.indices = indices

    def __iter__(self):
        for idx in self.indices:
            sid, year, pid = self.mapping[idx]
            yield sid, year, pid

    def __len__(self) -> int:
        return len(self.indices)