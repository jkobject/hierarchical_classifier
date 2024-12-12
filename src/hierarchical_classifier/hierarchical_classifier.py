import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from torch.nn import functional as F


class HierarchicalClassifier(PyTorchModelHubMixin, nn.Module):
    """
    HierarchicalClassifier for handling a single hierarchical classification task.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        num_classes: int,
        labels_hierarchy: dict[int, list[int]] | None = None,
        dropout: float = 0.2,
    ):
        """
        HierarchicalClassifier for handling a single hierarchical classification task.

        Args:
            input_dim (int): Dimension of the input features
            hidden_layers (List[int]): List of hidden layer sizes
            num_classes (int): Number of classes to predict
            labels_hierarchy (Optional[Dict[int, List[int]]]): Dictionary mapping parent class indices
                to their child class indices. Parent class indices should be offset by num_classes.
                For example: {10: [0, 1], 11: [2, 3]} means parent class 10 includes classes 0 and 1,
                and parent class 11 includes classes 2 and 3. Defaults to None (standard classification).
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super().__init__()
        self.num_classes = num_classes
        self.labels_hierarchy = labels_hierarchy

        # Build the network layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns
        -------
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        features = self.feature_extractor(x)
        return self.output_layer(features)

    def compute_hierarchical_loss(
        self,
        pred: torch.Tensor,
        cl: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute hierarchical classification loss considering the label hierarchy.

        Implementation follows the original classification loss from loss.py.

        Args:
            pred (torch.Tensor): Model predictions of shape (batch_size, num_classes)
            cl (torch.Tensor): Ground truth labels of shape (batch_size,)

        Returns
        -------
            torch.Tensor: The computed loss value
        """
        # Convert labels to one-hot representation
        newcl = torch.zeros((cl.shape[0], self.num_classes), device=cl.device)
        # If we don't know the label we set the weight to 0 else to 1
        valid_indices = (cl != -1) & (cl < self.num_classes)
        valid_cl = cl[valid_indices]
        newcl[valid_indices, valid_cl] = 1

        weight = torch.ones_like(newcl, device=cl.device)
        weight[cl == -1, :] = 0
        inv = cl >= self.num_classes

        # If we have non-leaf values, handle hierarchical relationships
        if inv.any() and self.labels_hierarchy is not None:
            inv_weight = weight[inv]
            # Set weight of non-leaf elements to 0
            for parent_idx, child_indices in self.labels_hierarchy.items():
                mask = cl[inv] == parent_idx
                if mask.any():
                    inv_weight[mask, child_indices] = 0
            weight[inv] = inv_weight

            # Add new labels for hierarchical relationships
            addnewcl = torch.ones(weight.shape[0], device=pred.device)
            addweight = torch.zeros(weight.shape[0], device=pred.device)
            addweight[inv] = 1

            # Computing hierarchical labels
            addpred = pred.clone()
            inv_addpred = addpred[inv]
            inv_addpred[inv_weight.to(bool)] = torch.finfo(pred.dtype).min
            addpred[inv] = inv_addpred

            # Differentiable max
            addpred = torch.logsumexp(addpred, dim=-1)

            # Add the new labels
            newcl = torch.cat([newcl, addnewcl.unsqueeze(1)], dim=1)
            pred = torch.cat([pred, addpred.unsqueeze(1)], dim=1)
            weight = torch.cat([weight, addweight.unsqueeze(1)], dim=1)

        return F.binary_cross_entropy_with_logits(pred, target=newcl, weight=weight)

    def training_step(
        self,
        batch: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        """
        Perform a training step

        Args:
            batch (torch.Tensor): Input batch
            labels (torch.Tensor): Ground truth labels
            optimizer (torch.optim.Optimizer): Optimizer

        Returns
        -------
            torch.Tensor: The computed loss value
        """
        optimizer.zero_grad()
        pred = self(batch)
        loss = self.compute_hierarchical_loss(pred, labels)
        loss.backward()
        optimizer.step()
        return loss

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions

        Args:
            x (torch.Tensor): Input tensor

        Returns
        -------
            torch.Tensor: Class predictions
        """
        with torch.no_grad():
            logits = self(x)
            return torch.argmax(logits, dim=1)
