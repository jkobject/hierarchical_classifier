import pytest
import torch

from hierarchical_classifier import HierarchicalClassifier


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    # Create random input data
    batch_size = 32
    input_dim = 256

    X = torch.randn(batch_size, input_dim)

    # Create labels with both leaf and parent classes
    # Some samples have leaf classes (0-3)
    # Some samples have parent classes (10, 11)
    # Some samples have unknown classes (-1)
    labels = torch.randint(-1, 12, (batch_size,))

    return X, labels


@pytest.fixture
def hierarchy_config():
    """Basic configuration for hierarchical classifier"""
    return {
        "input_dim": 256,
        "hidden_layers": [128, 64],
        "num_classes": 4,
        "labels_hierarchy": {
            10: [0, 1],  # Parent class 10 includes classes 0 and 1
            11: [2, 3],  # Parent class 11 includes classes 2 and 3
        },
        "dropout": 0.2,
    }


def test_hierarchical_classifier_initialization(hierarchy_config):
    """Test if the classifier initializes correctly"""
    classifier = HierarchicalClassifier(**hierarchy_config)

    # Check if the model has correct number of output classes
    assert classifier.num_classes == hierarchy_config["num_classes"]

    # Check if hierarchy is stored correctly
    assert classifier.labels_hierarchy == hierarchy_config["labels_hierarchy"]

    # Check if the model architecture is correct
    assert isinstance(classifier.feature_extractor, torch.nn.Sequential)
    assert isinstance(classifier.output_layer, torch.nn.Linear)
    assert classifier.output_layer.out_features == hierarchy_config["num_classes"]


def test_forward_pass(hierarchy_config, sample_data):
    """Test if forward pass produces correct output shape"""
    X, _ = sample_data
    classifier = HierarchicalClassifier(**hierarchy_config)

    output = classifier(X)

    # Check output shape
    assert output.shape == (X.shape[0], hierarchy_config["num_classes"])
    assert not torch.isnan(output).any(), "Output contains NaN values"


def test_hierarchical_loss_computation(hierarchy_config, sample_data):
    """Test if hierarchical loss computation works"""
    X, labels = sample_data
    classifier = HierarchicalClassifier(**hierarchy_config)

    # Get predictions
    pred = classifier(X)

    # Compute loss
    loss = classifier.compute_hierarchical_loss(pred, labels)

    # Check if loss is valid
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Loss should be a scalar
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"


def test_training_step(hierarchy_config, sample_data):
    """Test if training step works"""
    X, labels = sample_data
    classifier = HierarchicalClassifier(**hierarchy_config)
    optimizer = torch.optim.Adam(classifier.parameters())

    # Initial loss
    initial_loss = classifier.training_step(X, labels, optimizer)

    # Train for a few steps
    losses = []
    for _ in range(5):
        loss = classifier.training_step(X, labels, optimizer)
        losses.append(loss.item())

    # Check if loss decreases
    assert losses[-1] < initial_loss.item(), "Loss should decrease during training"


def test_predict(hierarchy_config, sample_data):
    """Test if prediction works"""
    X, _ = sample_data
    classifier = HierarchicalClassifier(**hierarchy_config)

    predictions = classifier.predict(X)

    # Check predictions shape and values
    assert predictions.shape == (X.shape[0],)
    assert predictions.min() >= 0
    assert predictions.max() < hierarchy_config["num_classes"]


@pytest.mark.parametrize("batch_size", [1, 16, 32])
def test_different_batch_sizes(hierarchy_config, batch_size):
    """Test if the model works with different batch sizes"""
    input_dim = hierarchy_config["input_dim"]
    X = torch.randn(batch_size, input_dim)
    labels = torch.randint(-1, 12, (batch_size,))

    classifier = HierarchicalClassifier(**hierarchy_config)
    output = classifier(X)

    assert output.shape == (batch_size, hierarchy_config["num_classes"])

    # Test loss computation
    loss = classifier.compute_hierarchical_loss(output, labels)
    assert not torch.isnan(loss), f"Loss is NaN for batch size {batch_size}"


def test_no_hierarchy(sample_data):
    """Test if the classifier works without hierarchy"""
    X, labels = sample_data

    # Configure classifier without hierarchy
    config = {
        "input_dim": 256,
        "hidden_layers": [128, 64],
        "num_classes": 4,
        "labels_hierarchy": None,
        "dropout": 0.2,
    }

    classifier = HierarchicalClassifier(**config)
    output = classifier(X)
    loss = classifier.compute_hierarchical_loss(output, labels)

    assert not torch.isnan(loss), "Loss should be valid even without hierarchy"
