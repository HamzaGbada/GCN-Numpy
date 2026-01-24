"""
Abstract MPNN (Message Passing Neural Network) Base Layer.

This module defines the abstract base class for all GNN layers using the
Template Method design pattern. The forward() method implements the standard
MPNN framework: message -> aggregate -> update.
"""

from abc import ABC, abstractmethod


class MPNNLayer(ABC):
    """
    Abstract base class for Message Passing Neural Network layers.
    
    Implements the Template Method pattern where forward() orchestrates
    the message passing steps, while subclasses provide specific implementations.
    
    Attributes:
        X: Cached input node features from forward pass
        A: Cached adjacency matrix from forward pass
        messages: Cached messages from message() step
        aggregated: Cached aggregated messages from aggregate() step
    """
    
    def __init__(self):
        self.X = None
        self.A = None
        self.messages = None
        self.aggregated = None
    
    def forward(self, X, A):
        """
        Template Method implementing the MPNN forward pass.
        
        Orchestrates the three-step message passing:
        1. message() - Construct messages from node features
        2. aggregate() - Aggregate messages from neighbors
        3. update() - Update node representations
        
        Args:
            X: Node feature matrix of shape (N, F_in)
            A: Adjacency matrix of shape (N, N)
            
        Returns:
            Updated node representations of shape (N, F_out)
        """
        self.X = X
        self.A = A
        
        self.messages = self.message(X, A)
        self.aggregated = self.aggregate(self.messages, A)
        out = self.update(self.aggregated)
        return out
    
    @abstractmethod
    def message(self, X, A):
        """
        Construct messages from node features.
        
        Args:
            X: Node feature matrix of shape (N, F_in)
            A: Adjacency matrix of shape (N, N)
            
        Returns:
            Messages to be aggregated
        """
        pass
    
    @abstractmethod
    def aggregate(self, messages, A):
        """
        Aggregate messages from neighboring nodes.
        
        Args:
            messages: Messages from message() step
            A: Adjacency matrix of shape (N, N)
            
        Returns:
            Aggregated neighbor information
        """
        pass
    
    @abstractmethod
    def update(self, aggregated):
        """
        Update node representations based on aggregated messages.
        
        Args:
            aggregated: Aggregated information from aggregate() step
            
        Returns:
            Updated node representations of shape (N, F_out)
        """
        pass
    
    @abstractmethod
    def backward(self, error, lr):
        """
        Compute gradients and update parameters.
        
        Must return gradient with respect to input features for
        backpropagation through multiple layers.
        
        Args:
            error: Gradient of loss with respect to output
            lr: Learning rate for parameter updates
            
        Returns:
            Gradient with respect to input features
        """
        pass
