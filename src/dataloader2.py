import os
from typing import Iterator, Tuple, Optional
import numpy as np
from collections import deque
from threading import Thread
import queue

# Export the main classes
__all__ = ['DataLoader', 'build_data_loader', 'PyGrainDatasetIterator',
           'DatasetIterator']


class DatasetIterator:
    """Iterator class for the dataset."""

    def __init__(self, dataloader: 'DataLoader'):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            # Reset iterator and try again
            self.iterator = iter(self.dataloader)
            return next(self.iterator)

    def get_state(self):
        """Get iterator state for checkpointing."""
        return {
            'current_index': self.dataloader.current_index,
            'indices': self.dataloader.indices.copy()
        }

    def set_state(self, state):
        """Restore iterator state from checkpoint."""
        self.dataloader.current_index = state['current_index']
        self.dataloader.indices = state['indices']
        self.iterator = iter(self.dataloader)


# Alias for compatibility with existing code
PyGrainDatasetIterator = DatasetIterator


class DataLoader:
    """A custom data loader implementation to replace Grain."""

    def __init__(
            self,
            data_source: 'BagDataSource',
            batch_size: int,
            shuffle: bool = False,
            drop_remainder: bool = True,
            seed: Optional[int] = None,
            prefetch_size: int = 2
    ):
        """Initialize the data loader."""
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_remainder = drop_remainder
        self.rng = np.random.RandomState(seed)
        self.prefetch_size = prefetch_size

        # Create indices array
        self.indices = np.arange(len(self.data_source))
        self.current_index = 0

        # Setup prefetching queue
        self.queue = queue.Queue(maxsize=prefetch_size)
        self.stop_event = False

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Returns iterator over batches of data."""
        if self.shuffle:
            self.rng.shuffle(self.indices)

        self.current_index = 0

        # Start prefetch thread
        self.stop_event = False
        thread = Thread(target=self._prefetch_worker, daemon=True)
        thread.start()

        try:
            while True:
                batch = self.queue.get()
                if batch is None:  # Signal for end of epoch
                    break
                yield batch
        finally:
            self.stop_event = True
            thread.join()

    def _prefetch_worker(self):
        """Background thread to prefetch batches."""
        try:
            while self.current_index < len(self.indices):
                if self.stop_event:
                    break

                # Get batch indices
                end_idx = self.current_index + self.batch_size
                batch_indices = self.indices[self.current_index:end_idx]
                self.current_index = end_idx

                # Skip incomplete final batch if drop_remainder
                if len(batch_indices) < self.batch_size and self.drop_remainder:
                    break

                # Load batch data
                sequences = []
                loss_masks = []
                for idx in batch_indices:
                    seq, mask = self._process_item(self.data_source[idx])
                    sequences.append(seq)
                    loss_masks.append(mask)

                # Convert to arrays and pad if needed
                sequences = np.array(sequences)
                loss_masks = np.array(loss_masks)

                self.queue.put((sequences, loss_masks))

            self.queue.put(None)  # Signal end of epoch

        except Exception as e:
            print(f"Error in prefetch worker: {e}")
            self.queue.put(None)

    def _process_item(self, item: bytes) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single data item into sequence and mask."""
        raise NotImplementedError("Subclasses must implement _process_item")

    def __call__(self):
        """Returns a dataset iterator."""
        return DatasetIterator(self)


def build_data_loader(config) -> DataLoader:
    """Build appropriate data loader based on config."""
    from searchless_chess.src import bagz

    data_source = bagz.BagDataSource(
        os.path.join(
            os.getcwd(),
            f'../data/{config.split}/{config.policy}_data.bag',
        )
    )

    if config.num_records is not None:
        if len(data_source) < config.num_records:
            raise ValueError(
                f'The number of records requested ({config.num_records}) is '
                f'larger than the dataset ({len(data_source)}).'
            )

    loader_cls = {
        'action_value': ActionValueDataLoader,
        'state_value': StateValueDataLoader,
        'behavioral_cloning': BehavioralCloningDataLoader,
    }[config.policy]

    loader = loader_cls(
        num_return_buckets=config.num_return_buckets,
        data_source=data_source,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        drop_remainder=config.drop_remainder,
        seed=config.seed
    )

    return loader
