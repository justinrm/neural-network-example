"""
MNIST Classification with Neural Networks
=========================================

This script demonstrates training neural networks with different activation functions
on the MNIST dataset and compares their performance.

Author: Justin R. Merwin
Date: April 2, 2025
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class NeuralNetwork(nn.Module):
    """Base neural network class for MNIST classification with configurable activation function."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, activation_function):
        """
        Initialize the neural network with two linear layers and a specified activation function.
        
        Args:
            input_dim (int): Size of the input features
            hidden_dim (int): Size of the hidden layer
            output_dim (int): Number of output classes
            activation_function (callable): The activation function to use between layers
        """
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.activation = activation_function
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output predictions of shape (batch_size, output_dim)
        """
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return x


def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100, patience=5):
    """
    Train the model and evaluate on validation data with early stopping.
    
    Args:
        model (nn.Module): The neural network model to train
        criterion: Loss function
        train_loader (DataLoader): DataLoader for training data
        validation_loader (DataLoader): DataLoader for validation data
        optimizer: Optimization algorithm
        epochs (int): Maximum number of training epochs
        patience (int): Number of epochs to wait for improvement before early stopping
        
    Returns:
        dict: Dictionary containing training losses and validation accuracies
    """
    metrics = {'training_loss': [], 'validation_accuracy': []}
    best_accuracy = 0
    no_improvement_count = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        for x, y in train_loader:
            # Convert images to flat vectors
            x_flat = x.view(-1, 28 * 28)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(x_flat)
            loss = criterion(outputs, y)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Track batch-level losses for debugging if needed
            epoch_loss += loss.item()
            batch_count += 1
        
        # Record average epoch loss
        avg_epoch_loss = epoch_loss / batch_count
        metrics['training_loss'].append(avg_epoch_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in validation_loader:
                x_flat = x.view(-1, 28 * 28)
                outputs = model(x_flat)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                
        accuracy = 100 * correct / total
        metrics['validation_accuracy'].append(accuracy)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')
        
        # Early stopping logic
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            no_improvement_count = 0
            # Save the best model
            best_model_state = model.state_dict().copy()
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                # Restore best model
                model.load_state_dict(best_model_state)
                break

    return metrics


def load_mnist_data(batch_size, val_batch_size):
    """
    Load and prepare MNIST datasets and dataloaders.
    
    Args:
        batch_size (int): Batch size for training
        val_batch_size (int): Batch size for validation
        
    Returns:
        tuple: (train_loader, validation_loader) containing DataLoader objects
    """
    try:
        # Ensure data directory exists
        Path('./data').mkdir(exist_ok=True)
        
        # Load and prepare datasets
        train_dataset = dsets.MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=transforms.ToTensor()
        )
        
        validation_dataset = dsets.MNIST(
            root='./data', 
            train=False, 
            download=True, 
            transform=transforms.ToTensor()
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        validation_loader = torch.utils.data.DataLoader(
            dataset=validation_dataset,
            batch_size=val_batch_size,
            shuffle=False
        )
        
        print(f"Loaded MNIST: {len(train_dataset)} training, {len(validation_dataset)} validation images")
        return train_loader, validation_loader
        
    except Exception as e:
        print(f"Error loading MNIST data: {e}")
        raise


def visualize_results(results):
    """
    Visualize training results from different models.
    
    Args:
        results (dict): Dictionary with model names as keys and metrics as values
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    for name, metrics in results.items():
        plt.plot(metrics['training_loss'], label=name)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title('Training Loss Comparison')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    for name, metrics in results.items():
        plt.plot(metrics['validation_accuracy'], label=name)
    plt.ylabel('Validation Accuracy (%)')
    plt.xlabel('Epochs')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('activation_function_comparison.png')
    plt.show()
    
    # Print final accuracies
    print("\nFinal Validation Accuracies:")
    for name, metrics in results.items():
        print(f"{name.capitalize()}: {metrics['validation_accuracy'][-1]:.2f}%")


def main():
    """Main function to run the experiment."""
    
    # Configuration
    config = {
        'input_dim': 28 * 28,  # MNIST images are 28x28 pixels
        'hidden_dim': 100,
        'output_dim': 10,      # 10 digits (0-9)
        'learning_rate': 0.01,
        'batch_size': 2000,
        'val_batch_size': 5000,
        'epochs': 30,
        'patience': 5,         # Early stopping patience
        'random_seed': 42
    }
    
    # Set random seed for reproducibility
    torch.manual_seed(config['random_seed'])
    
    # Load data
    try:
        train_loader, validation_loader = load_mnist_data(
            config['batch_size'], 
            config['val_batch_size']
        )
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create models with different activation functions
    models = {
        'sigmoid': NeuralNetwork(
            config['input_dim'], 
            config['hidden_dim'], 
            config['output_dim'], 
            torch.sigmoid
        ),
        'tanh': NeuralNetwork(
            config['input_dim'], 
            config['hidden_dim'], 
            config['output_dim'], 
            torch.tanh
        ),
        'relu': NeuralNetwork(
            config['input_dim'], 
            config['hidden_dim'], 
            config['output_dim'], 
            torch.relu
        )
    }
    
    # Train the models and store results
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining model with {name} activation:")
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])
        
        try:
            results[name] = train(
                model, 
                criterion, 
                train_loader, 
                validation_loader, 
                optimizer, 
                epochs=config['epochs'],
                patience=config['patience']
            )
        except Exception as e:
            print(f"Error training {name} model: {e}")
            continue
    
    # Visualize results
    if results:
        visualize_results(results)
    else:
        print("No training results to visualize.")


if __name__ == "__main__":
    main()
