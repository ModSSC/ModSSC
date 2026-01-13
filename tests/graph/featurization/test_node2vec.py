import numpy as np

from modssc.graph.featurization import node2vec as n2v


def test_build_adjacency_filters_and_undirected():
    edge_index = np.array([[0, 0, 1, 2, 3, -1, 1], [1, 1, 2, 2, 0, 1, 1]])
    adj = n2v._build_adjacency(edge_index, n_nodes=3, undirected=True)
    assert adj == [[1], [0, 2], [1]]


def test_build_adjacency_directed():
    edge_index = np.array([[0, 1], [1, 2]])
    adj = n2v._build_adjacency(edge_index, n_nodes=3, undirected=False)
    assert adj == [[1], [2], []]


def test_random_walks_node2vec_dead_end():
    adj = [[1], []]
    walks = n2v._random_walks_node2vec(
        adj,
        num_walks=1,
        walk_length=3,
        p=1.0,
        q=1.0,
        seed=0,
    )
    assert len(walks) == 1
    assert walks[0] == [0, 1]


def test_random_walks_node2vec_biased(monkeypatch):
    class DummyRng:
        def __init__(self):
            self.seen_probs = []

        def integers(self, low, high=None):
            return 0

        def choice(self, a, p=None):
            self.seen_probs.append(p)
            return a[0]

    dummy = DummyRng()
    monkeypatch.setattr(np.random, "default_rng", lambda *_args, **_kwargs: dummy)

    adj = [[1, 2], [0, 2, 3], [0, 1], [1]]
    walks = n2v._random_walks_node2vec(
        adj,
        num_walks=1,
        walk_length=3,
        p=0.5,
        q=2.0,
        seed=0,
    )
    assert len(walks) == 4
    assert all(len(walk) == 3 for walk in walks)
    assert dummy.seen_probs
    assert any(len(p) == 3 for p in dummy.seen_probs)


def test_walk_pairs_empty():
    centers, contexts = n2v._walk_pairs([[0]], window_size=2)
    assert centers.size == 0
    assert contexts.size == 0


def test_walk_pairs_populated():
    centers, contexts = n2v._walk_pairs([[0, 1, 2]], window_size=1)
    assert centers.tolist() == [0, 1, 1, 2]
    assert contexts.tolist() == [1, 0, 2, 1]


def test_sample_negatives_shape_and_dtype():
    rng = np.random.default_rng(0)
    dist = np.array([0.7, 0.3], dtype=np.float64)
    out = n2v._sample_negatives(rng, num_nodes=2, batch_size=3, num_neg=4, dist=dist)
    assert out.shape == (3, 4)
    assert out.dtype == np.int64
    flat = [int(x) for x in out.ravel()]
    assert min(flat) >= 0
    assert max(flat) < 2
