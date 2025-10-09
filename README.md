# Neural Networks Investigation
This project explores how the choice of activation function in vanilla neural networks impacts training stability and performance. I trained a binary classifier on an open-source wine quality data set, and used models from calculus like Hessians and Lipschitz conditions to analyze whether the Sigmoid or ReLU activation functions produce optimal training results.

As my experiments yielded conclusive results, I used this investigation to form the basis of my International Baccalaureate (IB) Extended Essay in mathematics, addressing the research question: "To what extent does the activation function of a neural network impact the smoothness and curvature of its loss function?"

## Project Structure
'''
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
'''
