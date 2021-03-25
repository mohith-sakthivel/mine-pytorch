from torch.utils.data import Sampler, RandomSampler, BatchSampler


class CustomSampler(Sampler):
    def __init__(self, data_source, secondary_replacement=False):
        self.data_source = data_source
        self.secondary_replacement = secondary_replacement
        self._batch_sampler = RandomSampler(data_source)
        self._marginal_sampler = RandomSampler(data_source,
                                               replacement=secondary_replacement)

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        return zip(self._batch_sampler.__iter__(), self._marginal_sampler.__iter__())


class CustomBatchSampler(BatchSampler):
    def __iter__(self):
        batch = []
        secondary_batch = []
        for idx, secondary_idx in self.sampler:
            batch.append(idx)
            secondary_batch.append(secondary_idx)
            if len(batch) == self.batch_size:
                yield batch + secondary_batch
                batch = []
                secondary_batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch + secondary_batch