from tqdm import tqdm
import torch

class Bar:
    def __init__(self, loaded):
        self.map = {}
        self.loader = loaded
        self.loop = None

    def start(self):
        self.loop = tqdm(self.loader, leave=True)

    def stop(self):
        self.loop = None
        torch.cuda.empty_cache()


    def set(self, key, value):
        self.map[key] = value

    def __iter__(self):
        return self.loop.__iter__()

    def __next__(self):
        return self.loop.__next__()

    def __len__(self):
        return self.loop.__len__()

    def update(self):
        self.loop.set_postfix(self.map)

    def reset(self, total=None):
        self.loop.reset(total)

    def refresh(self):
        self.loop.refresh()
