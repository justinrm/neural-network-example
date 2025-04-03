# MNIST Classification with Neural Networks

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive implementation of neural networks with different activation functions for classifying handwritten digits from the MNIST dataset. This project demonstrates how to build, train, and evaluate simple neural network architectures using PyTorch.

## Overview

This project implements a comparative study of different activation functions (Sigmoid, Tanh, and ReLU) in a simple feedforward neural network for MNIST digit classification. The code is structured following best practices for PyTorch projects and includes proper documentation, error handling, and visualization tools.

![Activation Function Comparison](activation_function_comparison.png)

## Features

- **Unified Neural Network Architecture**: A flexible neural network implementation that supports different activation functions
- **Comparative Analysis**: Direct comparison of Sigmoid, Tanh, and ReLU activation functions
- **Robust Training Pipeline**: Complete with early stopping to prevent overfitting
- **Comprehensive Metrics**: Training loss and validation accuracy tracking
- **Data Visualization**: Plots for comparing performance across different models
- **Clean Code Structure**: Following PEP 8 guidelines and best practices

## Requirements

- Python 3.7+
- PyTorch 1.8+
- torchvision
- matplotlib
- numpy

## Installation

```bash
# Clone the repository
git clone https://github.com/justinrm/neural-network-example
cd neural-network-example

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision matplotlib numpy
```

## Usage

To run the complete experiment with all activation functions:

```bash
python neural-network-example.py
```

This will:
1. Download and prepare the MNIST dataset
2. Train three models with different activation functions
3. Display training loss and validation accuracy plots
4. Save visualization results to `activation_function_comparison.png`

## Code Structure

- `NeuralNetwork`: A PyTorch module implementing a two-layer neural network with configurable activation function
- `train()`: Training function with early stopping support
- `load_mnist_data()`: Data loading and preparation
- `visualize_results()`: Plotting and visualization functionality
- `main()`: Orchestrates the overall experiment workflow

## Neural Network Architecture

The implemented network has a simple architecture:

```
Input Layer (784 neurons) → Hidden Layer (100 neurons) → Activation → Output Layer (10 neurons)
```

Where the activation function is one of:
- Sigmoid
- Hyperbolic Tangent (Tanh)
- Rectified Linear Unit (ReLU)

## Results

The project demonstrates the impact of different activation functions on model performance:

| Activation | Final Validation Accuracy | Training Speed |
|------------|---------------------------|---------------|
| Sigmoid    | ~97%                      | Slow          |
| Tanh       | ~97.5%                    | Medium        |
| ReLU       | ~98%                      | Fast          |

ReLU typically shows the best performance in terms of both training speed and final accuracy, which aligns with broader findings in deep learning research.

## Customization

You can modify the experiment by adjusting the configuration parameters in the `config` dictionary:

- `hidden_dim`: Size of the hidden layer
- `learning_rate`: Learning rate for the optimizer
- `batch_size`: Training batch size
- `epochs`: Maximum number of training epochs
- `patience`: Early stopping patience parameter

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The MNIST dataset is a subset of a larger set available from NIST
- Thanks to the PyTorch team for their excellent deep learning framework
- This project structure follows best practices for PyTorch projects

## Author

[Justin R. Merwin] - [justin.r.merwin@outlook.com]

---

*This project is for educational purposes and demonstrates basic neural network concepts with PyTorch.*
