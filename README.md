# Sigmoid versus ReLU - A Neural Networks Investigation
This project explores how the choice of activation function in vanilla neural networks impacts training stability and performance. I trained a binary classifier on an open-source wine quality data set, and used models from calculus like Hessians and Lipschitz conditions to analyze whether the Sigmoid or ReLU activation functions produce optimal training results.

As my experiments yielded conclusive results, I used this investigation to form the basis of my International Baccalaureate (IB) Extended Essay in mathematics, addressing the research question: "To what extent does the activation function of a neural network impact the smoothness and curvature of its loss function?"

## Key Features
- Setup of a neural network using Sigmoid and ReLU to classify wines as high or low quality
- Performance insights for each activation function via loss tracking and F1 scoring
- Use of linear algebra techniques (Power Iteration, Singular Value Decomposition) to approximate higher-order differential quantities
- Intersection of machine learning and pure mathematics for the IB Extended Essay
- Modular, reproducible workflow via Google Colab notebooks

## Results Summary
- The Sigmoid function flattens loss surfaces, as its vanishing gradient make the model move slowly towards a loss minimum with negligible local curvature
- Conversely, ReLU optimizes by aggressively exploring loss surfaces, as unbounded gradients increase loss curvature and cause large fluctuations
- While ReLU's robustness can lead to training instability, its tendency to explore means that it often finds local minima faster than Sigmoid
- Similar accuracy metrics between both functions, however, suggest that different network architectures can be sufficient for solving similar problems

## Project Structure
```
project-root/
├─ README.md # Overview and documentation
├─ LICENSE # Usage license (MIT)
├─ notebooks/ # Google Colab notebooks by section
│ ├─ 01_network_setup.ipynb
│ ├─ 02_activation_functions.ipynb
│ ├─ 03_loss_surface_curvature.ipynb
│ ├─ 04_data_handling.ipynb
│ ├─ 05_analytics.ipynb
│ ├─ 06_training_loop.ipynb
│ ├─ 07_conclusions.ipynb
│ └─ 08_error_analysis.ipynb
├─ src/ # Shared helper functions
│ ├─ utils.py # Master document of neural network code
│ └─ init.py
├─ extended_essay/ # LaTeX source, figures, and drafts
│ ├─ README.md
│ ├─ Extended_Essay_Final.tex
│ ├─ References.bib
│ ├─ figures/
│   ├─ Sigmoid.png
│   ├─ Sigmoid_plus_derivative.png
│   ├─ ReLU.png
│   ├─ Sigmoid & ReLU Smoothed Training Loss.png
│   ├─ Sigmoid & ReLU Validation Loss.png
│   ├─ Sigmoid & ReLU Output Lipschitz.png
│   ├─ Sigmoid and ReLU Jacobian Lipschitz.png
│   ├─ Sigmoid & ReLU Max Hessian Eigenvalue.png
│   ├─ Sigmoid & ReLU F1 Score.png
│ ├─ Extended_Essay_First_Draft.pdf
│ ├─ Extended_Essay_Progress_Draft.pdf
│ └─ Extended_Essay_Final.pdf
└─ requirements.txt # Python dependencies
```

## Installation and Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/neural-networks-investigation.git
   cd neural-networks-investigation
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebooks in Google Colab or Jupyter:
   ```bash
   jupyter notebook notebooks/01_network_setup.ipynb
   ```
