import numpy as np


def iid_partition(dataset, num_clients: int) -> dict:
    """Randomly partition dataset indices into num_clients IID splits."""
    n = len(dataset)
    indices = np.random.permutation(n)
    splits = np.array_split(indices, num_clients)
    return {i: splits[i].tolist() for i in range(num_clients)}


def non_iid_partition(
    labels, num_clients: int, num_shards_per_client: int = 2
) -> dict:
    """
    Non-IID partition using label-sorted shards.
    Each client receives num_shards_per_client consecutive shards
    of label-sorted data, producing strong label skew.
    """
    labels = np.array(labels)
    n = len(labels)
    num_shards = num_clients * num_shards_per_client
    shard_size = n // num_shards

    sorted_indices = np.argsort(labels)
    shards = [
        sorted_indices[i * shard_size : (i + 1) * shard_size].tolist()
        for i in range(num_shards)
    ]

    shard_order = np.random.permutation(num_shards)
    client_data = {}
    for i in range(num_clients):
        start = i * num_shards_per_client
        end = start + num_shards_per_client
        client_shards = shard_order[start:end]
        client_data[i] = []
        for s in client_shards:
            client_data[i].extend(shards[s])

    return client_data
