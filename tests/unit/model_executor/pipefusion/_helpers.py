import torch


class Work:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


class Group:
    dtype = torch.float32

    def __init__(self, incoming=()):
        self.incoming = iter(incoming)
        self.queued = []
        self.receiving = []
        self.sent = []
        self.works = []
        self.events = []

    def add_pipeline_recv_task(self, index, name):
        self.queued.append((name, index))

    def recv_next(self):
        self.events.append("prefetch")
        self.receiving.append(self.queued.pop(0))

    def get_pipeline_recv_data(self, idx, name):
        assert self.receiving.pop(0) == (name, idx)
        return next(self.incoming)

    def pipeline_isend(self, tensor, name, segment_idx):
        self.events.append(f"send-{segment_idx}")
        self.sent.append((name, segment_idx, tensor.clone()))
        work = Work()
        self.works.append(work)
        return work
