# TinyCNN: When Smaller is Smarter - Deep Learning for the Edge

A minimalist Convolutional Neural Network that achieves **~98.5% accuracy** on handwritten digit classification using just **~19K parameters** demonstrating that **you don’t need big models to get big results.**

Most modern neural networks are overparameterized, energy-intensive, and impractical for edge devices.  

TinyCNN challenges that notion by proving:

- A simple architecture can be highly effective on structured tasks like digit recognition.
- Compact models are easier to train, interpret, and deploy — even on microcontrollers.
- It's a perfect stepping stone for students,hardware hobbyists, and embedded AI developers.

---

## Model Architecture

| Layer     | Details                     |
|-----------|-----------------------------|
| Conv1     | 4 filters, 3×3, ReLU        |
| Conv2     | 4 filters, 3×3, ReLU        |
| Flatten   |                             |
| FC1       | 32 neurons, ReLU           |
| FC2       | 10 neurons, Softmax Output |

- **Total trainable parameters**: ~18,982  
- **Test accuracy**: ~98.5%  
- **Optimized for**: 28×28 grayscale digits

---

## Comparison with LeNet-5

| Model       | Parameters | Accuracy (MNIST) | Notes                               |
|-------------|------------|------------------|-------------------------------------|
| LeNet-5     | ~48,000    | ~99.0%           | Classical CNN for digit recognition |
| **TinyCNN** | **18,982** | **~98.0%**       | Extremely lightweight alternative   |

Despite having less than 1/3rd the parameters, TinyCNN reaches comparable accuracy to LeNet-5.

---
## Training Setup

- **Dataset**: MNIST (or similar digit set)
- **Optimizer**: Adam
- **Loss**: CrossEntropy
- **Batch Size**: 16
- **Epochs**: 10–15

---

## Repo Contents

- `model.py` - TinyCNN definition
- `train.py` - Training script
- `predict.py` - Sample inference
- `train_model.pth`- Saved model weights
- `README.md` - You're reading it!
- `.gitignore` - To exclude `__pycache__`, model weights, etc.

---

## Future Goals

- Port to **TensorFlow Lite Micro** for Arduino/Nano 33 BLE
- Integrate with Android app or web frontend
- Visualize intermediate activations for interpretability

---

