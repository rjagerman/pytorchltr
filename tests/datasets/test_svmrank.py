import contextlib
import pickle
import tempfile
from unittest import mock

from pytest import raises
from pytest import approx
from pytorchltr.datasets.svmrank import SVMRankingDataset


def get_sample_dataset(*args, **kwargs):
    """Get sample dataset, uses the same arguments as `svmranking_dataset`."""
    with open("tests/datasets/dataset.txt", "rb") as dataset_file:
        return SVMRankingDataset(dataset_file, *args, **kwargs)


@contextlib.contextmanager
def mock_svmrank_dataset(package_str="pytorchltr.datasets.svmrank"):
    validate_and_download_str = package_str + ".validate_and_download"
    super_str = "pytorchltr.datasets.svmrank.SVMRankingDataset.__init__"
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch(validate_and_download_str) as mock_vali:
            with mock.patch(super_str) as mock_super:
                yield tmpdir, mock_super, mock_vali


def test_basic():

    # Load data set.
    dataset = get_sample_dataset()

    # Check data set size.
    assert len(dataset) == 4

    # Get first sample.
    sample = dataset[0]
    x, y, q = sample['features'], sample['relevance'], sample['qid']
    assert x.shape == (6, 45)
    assert y.shape == (6,)
    assert x[1, 2] == 1.0
    assert y[1] == 2.0
    assert q == 1

    # Get second sample.
    sample = dataset[1]
    x, y, q = sample['features'], sample['relevance'], sample['qid']
    assert x.shape == (9, 45)
    assert y.shape == (9,)
    assert float(x[5, 3]) == approx(0.422507)
    assert y[5] == 1.0
    assert q == 16

    # Get third sample.
    sample = dataset[2]
    x, y, q = sample['features'], sample['relevance'], sample['qid']
    assert x.shape == (14, 45)
    assert y.shape == (14,)
    assert float(x[12, 2]) == approx(0.461538)
    assert y[12] == 0.0
    assert q == 60

    # Get fourth sample.
    sample = dataset[3]
    x, y, q = sample['features'], sample['relevance'], sample['qid']
    assert x.shape == (10, 45)
    assert y.shape == (10,)
    assert float(x[8, 2]) == approx(0.25)
    assert y[8] == 0.0
    assert q == 63


def test_sparse():

    # Load data set.
    dataset_sparse = get_sample_dataset(sparse=True)
    dataset_dense = get_sample_dataset(sparse=False)

    # Check data set size.
    assert len(dataset_dense) == len(dataset_sparse)

    # Check sparse and dense return same samples.
    for i in range(len(dataset_dense)):
        sample_dense = dataset_dense[i]
        sample_sparse = dataset_sparse[i]
        assert sample_sparse['qid'] == sample_dense['qid']
        assert sample_sparse['n'] == sample_dense['n']
        assert sample_sparse['features'].to_dense().numpy() == approx(
            sample_dense['features'].numpy())
        assert sample_sparse['relevance'].numpy() == approx(
            sample_dense['relevance'].numpy())


def test_normalize():

    # Load data set.
    dataset = get_sample_dataset(normalize=True)

    # Check data set size.
    assert len(dataset) == 4

    # Get first sample and assert the contents is as expected.
    sample = dataset[0]
    x, y, q = sample['features'], sample['relevance'], sample['qid']
    assert x.shape == (6, 45)
    assert y.shape == (6,)
    assert q == 1

    assert float(x[0, 1]) == approx(1.0)
    assert float(x[1, 1]) == approx(0.5)
    assert float(x[2, 1]) == approx(0.25)
    assert float(x[3, 1]) == approx(0.0)
    assert float(x[4, 1]) == approx(0.125)
    assert float(x[5, 1]) == approx(0.5)

    assert float(x[0, 0]) == approx(0.24242424242424246)
    assert float(x[1, 0]) == approx(0.12121212121212122)
    assert float(x[2, 0]) == approx(0.060606060606060615)
    assert float(x[3, 0]) == approx(0.0)
    assert float(x[4, 0]) == approx(1.0)
    assert float(x[5, 0]) == approx(0.12121212121212122)


def test_sparse_normalize():

    # This should raise an error as it is not implemented.
    with raises(NotImplementedError):
        get_sample_dataset(sparse=True, normalize=True)


def test_serialize():

    # Load data set.
    dataset = get_sample_dataset(normalize=True)

    # Attempt to serialize and deserialize it.
    serialized = pickle.dumps(dataset)
    deserialized = pickle.loads(serialized)

    # Assert original and deserialized versions are the same.
    assert len(dataset) == len(deserialized)
    for i in range(len(dataset)):
        sample1 = dataset[i]
        x1, y1, q1 = sample1['features'], sample1['relevance'], sample1['qid']
        sample2 = deserialized[i]
        x2, y2, q2 = sample2['features'], sample2['relevance'], sample2['qid']
        assert x1.numpy() == approx(x2.numpy())
        assert y1.numpy() == approx(y2.numpy())
        assert q1 == q2


def test_serialize_sparse():

    # Load data set.
    dataset = get_sample_dataset(sparse=True)

    # Attempt to serialize and deserialize it.
    serialized = pickle.dumps(dataset)
    deserialized = pickle.loads(serialized)

    # Assert original and deserialized versions are the same.
    assert len(dataset) == len(deserialized)
    for i in range(len(dataset)):
        sample1 = dataset[i]
        x1, y1, q1 = sample1['features'], sample1['relevance'], sample1['qid']
        sample2 = deserialized[i]
        x2, y2, q2 = sample2['features'], sample2['relevance'], sample2['qid']
        assert x1.to_dense().numpy() == approx(x2.to_dense().numpy())
        assert y1.numpy() == approx(y2.numpy())
        assert q1 == q2


def test_double_serialize():

    # Load data set
    dataset = get_sample_dataset(normalize=True)

    # Attempt to serialize and deserialize it multiple times.
    s1 = pickle.dumps(dataset)
    d1 = pickle.loads(s1)
    s2 = pickle.dumps(d1)
    deserialized = pickle.loads(s2)

    # Assert original and deserialized versions are the same.
    assert len(dataset) == len(deserialized)
    for i in range(len(dataset)):
        sample1 = dataset[i]
        x1, y1, q1 = sample1['features'], sample1['relevance'], sample1['qid']
        sample2 = deserialized[i]
        x2, y2, q2 = sample2['features'], sample2['relevance'], sample2['qid']
        assert x1.numpy() == approx(x2.numpy())
        assert y1.numpy() == approx(y2.numpy())
        assert q1 == q2


def test_collate_sparse_10():

    # Load data set.
    dataset = get_sample_dataset(sparse=True)

    # Construct a batch of three samples and collate it with a maximum list
    # size of 10.
    batch = [dataset[0], dataset[1], dataset[2]]
    collate_fn = SVMRankingDataset.collate_fn(max_list_size=10)

    # Assert resulting tensor shape is as expected.
    tensor_batch = collate_fn(batch)
    assert tensor_batch["features"].shape == (3, 10, 45)


def test_collate_dense_10():

    # Load data set.
    dataset = get_sample_dataset(sparse=False)

    # Construct a batch of three samples and collate it with a maximum list
    # size of 10.
    batch = [dataset[0], dataset[1], dataset[2]]
    collate_fn = SVMRankingDataset.collate_fn(max_list_size=10)

    # Assert resulting tensor shape is as expected.
    tensor_batch = collate_fn(batch)
    assert tensor_batch["features"].shape == (3, 10, 45)


def test_collate_sparse_3():

    # Load data set.
    dataset = get_sample_dataset(sparse=True)

    # Construct a batch of three samples and collate it with a maximum list
    # size of 3.
    batch = [dataset[0], dataset[1], dataset[2]]
    collate_fn = SVMRankingDataset.collate_fn(max_list_size=3)

    # Assert resulting tensor shape is as expected.
    tensor_batch = collate_fn(batch)
    assert tensor_batch["features"].shape == (3, 3, 45)


def test_collate_dense_3():

    # Load data set.
    dataset = get_sample_dataset(sparse=False)

    # Construct a batch of three samples and collate it with a maximum list
    # size of 3.
    batch = [dataset[0], dataset[1], dataset[2]]
    collate_fn = SVMRankingDataset.collate_fn(max_list_size=3)

    # Assert resulting tensor shape is as expected.
    tensor_batch = collate_fn(batch)
    assert tensor_batch["features"].shape == (3, 3, 45)


def test_collate_sparse_all():

    # Load data set.
    dataset = get_sample_dataset(sparse=True)

    # Construct a batch of three samples and collate it with an unlimited
    # maximum list size.
    batch = [dataset[0], dataset[1], dataset[2]]
    collate_fn = SVMRankingDataset.collate_fn(max_list_size=None)

    # Assert resulting tensor shape is as expected.
    tensor_batch = collate_fn(batch)
    assert tensor_batch["features"].shape == (3, 14, 45)


def test_collate_dense_all():

    # Load data set.
    dataset = get_sample_dataset(sparse=False)

    # Construct a batch of three samples and collate it with an unlimited
    # maximum list size.
    batch = [dataset[0], dataset[1], dataset[2]]
    collate_fn = SVMRankingDataset.collate_fn(max_list_size=None)

    # Assert resulting tensor shape is as expected.
    tensor_batch = collate_fn(batch)
    assert tensor_batch["features"].shape == (3, 14, 45)


def test_filter_queries():
    # Load data set.
    dataset_filtered = get_sample_dataset(filter_queries=True)
    dataset = get_sample_dataset(filter_queries=False)
    assert len(dataset_filtered) != len(dataset)

    assert dataset_filtered[0]["qid"] == dataset[0]["qid"]
    assert dataset_filtered[1]["qid"] == dataset[1]["qid"]
    assert dataset_filtered[2]["qid"] == dataset[3]["qid"]
