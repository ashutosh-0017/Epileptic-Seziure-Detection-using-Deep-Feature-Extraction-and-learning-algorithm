# Epileptic-Seziure-Detection-using-Deep-Feature-Extraction-and-learning-algorithm

## Overview

This project focuses on developing machine learning models for the detection and classification of epileptic seizures using EEG (Electroencephalogram) signals. The system classifies EEG data into three categories: Non-Epileptic, Mild Epileptic, and Severe Epileptic activity.

## Project Structure

```
Epileptic notebook/
├── model1_2_4.ipynb          # Hybrid LSTM + Autoencoder model
├── model2.ipynb              # LSTM + Transformer model
├── model3.ipynb              # InceptionV3 + Transformer model
├── eeg seziure.pptx          # Project presentation
├── imageedit_*.gif           # Visualization files
└── future work/
    ├── model1.ipynb          # Future model implementations
    ├── model2.ipynb
    └── model4.ipynb
```

## Dataset

The project uses two EEG datasets:
- `eeg-predictive_train.npz` - Predictive EEG data
- `eeg-seizure_train.npz` - Seizure EEG data

### Data Characteristics
- **Input Shape**: (samples, 23 channels, 256 time steps)
- **Sampling Rate**: 256 Hz
- **Classes**: 3 (Non-Epileptic, Mild Epileptic, Severe Epileptic)

## Preprocessing Pipeline

1. **Data Loading**: Merging multiple EEG datasets
2. **Class Balancing**: Random sampling to create balanced classes (5000 samples per original class)
3. **Amplitude-based Classification**: 
   - Non-Epileptic: Bottom 50% of mean absolute values
   - Mild Epileptic: 50-90th percentile
   - Severe Epileptic: Top 10%
4. **Normalization**: Min-Max scaling to [0, 1] range
5. **Filtering**: Bandpass filter (0.5-40 Hz) to remove noise
6. **One-Hot Encoding**: Convert labels to categorical format

## Model Architectures

### 1. Hybrid LSTM + Autoencoder (model1_2_4.ipynb)
- **Feature Extraction**: LSTM-based feature extraction
- **Architecture**: Encoder-Decoder with classification branch
- **Loss**: Combined reconstruction (MSE) and classification (categorical crossentropy)
- **Features**: 64-dimensional LSTM features

### 2. LSTM + Transformer (model2.ipynb)
- **Feature Extraction**: LSTM layers
- **Transformer**: Multi-head attention mechanism
- **Regularization**: L2 regularization and dropout
- **Optimizer**: Adam with learning rate scheduling

### 3. InceptionV3 + Transformer (model3.ipynb)
- **Feature Extraction**: InceptionV3 CNN adapted for time series
- **Transformer**: Multi-head attention with layer normalization
- **Input Adaptation**: EEG signals reshaped for CNN processing

## Key Features

- **Multi-class Classification**: Distinguishes between non-epileptic, mild, and severe epileptic activity
- **Advanced Architectures**: Combines traditional and modern deep learning approaches
- **Comprehensive Evaluation**: Includes ROC curves, precision-recall curves, and confusion matrices
- **Data Augmentation**: Through sophisticated preprocessing and balancing techniques

## Performance Metrics

Models are evaluated using:
- Accuracy
- F1-score (macro, micro, weighted)
- ROC AUC scores
- Average Precision scores
- Confusion matrices

## Installation and Requirements

### Python Dependencies
```bash
pip install numpy pandas matplotlib seaborn tensorflow scipy scikit-learn
```

### Required Libraries
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn

## Usage

1. **Data Preparation**: Place EEG datasets in `D:\dataset eeg\` directory
2. **Run Models**: Execute the Jupyter notebooks in sequence:
   - Start with data preprocessing cells
   - Run model training cells
   - Evaluate results

### Example Usage
```python
# Load and preprocess data
from preprocessing import load_eeg_data, preprocess_signals

# Train model
model.fit(X_train, y_train, validation_data=(X_val, y_val))
```

## Results and Visualizations

The project includes:
- Training/validation loss and accuracy plots
- Confusion matrices for model evaluation
- ROC curves for each class
- Precision-Recall curves
- Feature visualizations (GIF files)

## Future Work

The `future work/` directory contains plans for:
- Additional model architectures
- Hyperparameter optimization
- Real-time deployment considerations
- Cross-validation strategies
- Transfer learning approaches

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is for research and educational purposes. Please ensure proper attribution if used in academic or commercial applications.

## References

- EEG seizure detection literature
- Deep learning for biomedical signal processing
- Transformer architectures for time series
- Transfer learning in healthcare applications

## Contact

For questions about this project, please refer to the documentation or contact the development team.

---

**Note**: This project is intended for research purposes. Clinical applications should undergo proper validation and regulatory approval.
