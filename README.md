# EEGNet: LOSO Cross-Validation with Subject-Specific Fine-Tuning

## Overview

This notebook implements **Leave-One-Subject-Out (LOSO) cross-validation** combined with **subject-specific fine-tuning** for EEG-based Motor Imagery classification using the **EEGNet** deep learning architecture. The approach bridges the gap between a generic population-trained model and subject-specific adaptation with minimal calibration data.

### Key Innovation
The notebook evaluates the effectiveness of fine-tuning a pre-trained LOSO model with minimal subject-specific calibration data (50% of one session) to improve classification accuracy without requiring extensive per-subject training.

---

## Architecture: EEGNet

**EEGNet** is a compact convolutional neural network specifically designed for EEG signal processing. It uses depthwise separable convolutions to reduce parameters while maintaining good performance.

### Network Components

1. **Block 1: Temporal + Spatial Convolution**
   - Temporal convolution (1 × kernel_length)
   - Depthwise convolution (channels × 1) for spatial filtering
   - ELU activation + Average pooling + Dropout

2. **Block 2: Separable Convolution**
   - Separable convolution (1 × 16)
   - ELU activation + Average pooling + Dropout

3. **Classifier**
   - Flatten layer
   - Dense layer (nb_classes) with max-norm constraint
   - Softmax activation for 4-class classification

### Network Configuration
- **Channels (EEG)**: 64
- **Time Samples**: 128
- **Classes**: 4 (Left Hand, Right Hand, Both Feet, Tongue)
- **Parameters**: F1=8, D=2, F2=16
- **Dropout**: 0.5
- **Optimizer**: Adam (lr=0.001)

---

## Dataset: BNCI2014001 (BCI IV 2a)

- **Source**: BCI Competition IV, Dataset 2a from MOABB
- **Subjects**: 9 healthy volunteers
- **Sessions per Subject**: 2 sessions × ~6 minutes per session
- **Total Trials per Subject**: ~288 trials (~144 per session)
- **Motor Imagery Tasks**: 4 classes
  - Left Hand (LH)
  - Right Hand (RH)
  - Both Feet (BF)
  - Tongue (TG)
- **Sampling Rate**: 250 Hz
- **Channels**: 22 EEG channels

---

## Methodology: LOSO + Fine-Tuning Pipeline

### Overview
The pipeline consists of two main stages for each fold:

#### **Stage 1: LOSO Training** (Leave-One-Subject-Out)
- **Training Data**: 8 subjects pooled together
- **Validation Data**: 20% of training data (for early stopping)
- **Test Data**: Held-out subject (full session)
- **Epochs**: 100 (with early stopping, patience=15)
- **Batch Size**: 16
- **Purpose**: Train a generic population-level model

#### **Stage 2: Subject-Specific Fine-Tuning**
- **Data Split**: 
  - Calibration Data: 40% of held-out subject
  - Evaluation Data: 60% of held-out subject
- **Fine-Tuning Configuration**:
  - Epochs: 50 (with early stopping, patience=10)
  - Batch Size: 8 (small dataset)
  - Learning Rate: 0.0001 (lower than LOSO)
  - Optimizer: Adam
- **Purpose**: Adapt generic model to individual subject's characteristics

### Cross-Validation
- **Folds**: 9 (one per subject)
- **Training Set**: 8 subjects in each fold
- **Test Set**: 1 subject (held-out)
- **Results**: 9 independent evaluations for robust performance estimation

### Multiple Split Analysis (Experimental)
The notebook also tests different calibration data splits (30%, 40%, 50%, 60%, 70%) to determine the optimal balance between calibration burden and accuracy improvement.

---

## Results & Performance Metrics

### Key Performance Indicators

1. **LOSO Baseline**
   - Generic model trained on population data
   - No subject-specific adaptation
   - Serves as baseline for comparison

2. **Fine-Tuning Results**
   - Accuracy improvement after subject-specific adaptation
   - Typically shows 2-5% improvement across subjects
   - Variable improvement depending on individual differences

3. **Practical Metrics**
   - **Calibration Requirement**: ~144 trials (~1 session)
   - **Training Time**: ~5-15 minutes per fold
   - **Total Experiment Time**: ~2-3 hours for all 9 folds

### Output Visualizations

1. **Accuracy Comparison** (`loso_finetuning_analysis.png`)
   - Bar charts comparing LOSO vs fine-tuning accuracy
   - Improvement per subject visualization
   - Distribution comparison (box plots)
   - Scatter plot showing LOSO vs fine-tuning relationship
   - Training time breakdown

2. **Confusion Matrices** (`loso_finetuning_confusion_matrices.png`)
   - 9×2 grid of confusion matrices (LOSO and fine-tuning per subject)
   - Per-class performance visualization
   - Normalized and raw confusion matrices

3. **Results Summary**
   - CSV export with detailed metrics per fold
   - Per-subject accuracy and improvement statistics
   - Aggregated statistics across all subjects

---

## Implementation Details

### Dependencies
- **TensorFlow/Keras**: Deep learning framework
- **MOABB**: Motor Imagery dataset loading
- **MNE**: EEG signal processing
- **scikit-learn**: Preprocessing and metrics
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

### Key Features
- **Mixed Precision Training**: Enabled for faster computation (FP16 + FP32)
- **Early Stopping**: Prevents overfitting (patience=15 for LOSO, patience=10 for fine-tuning)
- **Memory Management**: Model cleanup after each fold to prevent memory leaks
- **Stratified Splitting**: Maintains class distribution in train/validation/test splits

### Hyperparameter Configuration

| Parameter | LOSO | Fine-Tuning |
|-----------|------|-------------|
| Epochs | 100 | 50 |
| Batch Size | 16 | 8 |
| Learning Rate | 0.001 | 0.0001 |
| Early Stopping Patience | 15 | 10 |
| Train/Val Split | 80/20 | N/A (use all) |

---

## Practical Applications

### Real-World BCI System Deployment

1. **Offline Phase** (Done Once)
   - Collect data from multiple subjects (population)
   - Train LOSO model on population data
   - Results: Generic BCI system

2. **Online Phase** (Per New User)
   - New user performs minimal calibration (~1 session, ~144 trials)
   - Fine-tune generic model on user's calibration data
   - Results: Subject-specific, calibrated BCI system

### Advantages

✓ **Minimal Calibration**: Only requires ~1 session of user data (~15-20 minutes)  
✓ **Personalized Performance**: Achieves near-subject-specific accuracy  
✓ **Practical**: Reduces time overhead compared to full subject-specific training  
✓ **Transferable**: Population-level knowledge transfers to new users  
✓ **Robust**: Benefits from diverse training data across multiple subjects

### Use Cases

- **Medical Applications**: Motor imagery for stroke rehabilitation
- **Assistive Devices**: BCI-controlled prosthetics or wheelchairs
- **Gaming**: Brain-computer interfaces for interactive applications
- **Communication**: Silent speech recognition via motor imagery

---

## Notebook Structure

### Cell Breakdown

1. **Installation & Setup**
   - Install required Python packages
   - Import necessary libraries

2. **Network Definition**
   - EEGNet architecture implementation
   - Model compilation with appropriate loss and metrics

3. **Dataset Loading**
   - Load BNCI2014001 from MOABB
   - Extract and preprocess EEG data
   - Organize by subject

4. **LOSO + Fine-Tuning Pipeline**
   - Main training loop (9 folds)
   - Stage 1: LOSO training on 8 subjects
   - Stage 2: Evaluation on held-out subject
   - Stage 3: Data splitting for fine-tuning
   - Stage 4: Fine-tuning on calibration data
   - Stage 5: Evaluation on test data

5. **Results Aggregation**
   - Collect accuracy, loss, confusion matrices
   - Calculate improvements
   - Store timing information

6. **Analysis & Visualization**
   - Summary statistics
   - Comprehensive comparison plots
   - Confusion matrix visualization per subject
   - Results export to CSV

---

## Expected Results

### Typical Performance Range

- **LOSO Baseline**: 45-65% accuracy (subject-dependent)
- **LOSO + Fine-Tuning**: 50-70% accuracy (with 50% calibration data)
- **Improvement**: +2 to +8 percentage points (subject-dependent)

### Variability Factors

- **Subject Variability**: Individual differences in EEG signal quality
- **Session Variability**: Changes in signal characteristics between sessions
- **Optimal Calibration Size**: Depends on subject-specific adaptation potential

---

## Output Files

The notebook generates the following outputs:

1. **Visualizations**
   - `loso_finetuning_analysis.png` - Comprehensive accuracy and performance analysis
   - `loso_finetuning_confusion_matrices.png` - Per-subject confusion matrices

2. **Data**
   - `loso_finetuning_results.csv` - Detailed results table
   - In-memory dictionary `results` with all metrics

3. **Console Output**
   - Fold-by-fold progress
   - Summary statistics
   - Per-subject breakdown

---

## Running the Notebook

### Prerequisites
- Python 3.8+
- GPU recommended (CUDA/cuDNN for faster training)
- ~10 GB disk space for dataset
- ~4+ GB RAM

### Execution Steps

1. Install required packages (Cell 2)
2. Import libraries (Cell 3)
3. Define EEGNet architecture (Cells 4-5)
4. Load dataset (Cell 6)
5. Run LOSO + fine-tuning pipeline (Cell 7 or 8)
6. Analyze results (Cells 9-11)

### Expected Runtime

- **Total Time**: 1.5-3 hours (depending on hardware)
- **Per Fold**: ~10-20 minutes
- **GPU Acceleration**: ~3-5x faster than CPU

---

## Important Notes

### Data Preprocessing
- Data is automatically preprocessed by MOABB
- Frequency range: Standard EEG range (typically 4-50 Hz)
- Normalization: Implicit in deep learning model

### Memory Considerations
- Models are cleared after each fold (`tf.keras.backend.clear_session()`)
- Prevents memory accumulation across 9 folds
- Important for long-running experiments

### Reproducibility
- Fixed random seeds in train/test splits
- Consistent shuffle order across runs
- Results should be reproducible with same hardware/software versions

---

## References & Citations

### Key Papers

1. **EEGNet Architecture**
   - Lawhern et al. (2018): "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces"

2. **LOSO Cross-Validation**
   - Standard in EEG/BCI literature for evaluating subject independence

3. **BNCI2014001 Dataset**
   - Brunner et al. (2008): "BCI Competition IV, Dataset 2a"
   - Published in Frontiers in Neuroscience

### Dataset Citation

```
Brunner, C., Leeb, R., Müller-Putz, G., Schlögl, A., & Pfurtscheller, G. (2008).
BCI Competition IV, Dataset 2a.
Retrieved from http://www.bbci.de/competition/iv/
```

---

## Future Improvements

- Test with other EEG architectures (ShallowConvNet, DeepConvNet, etc.)
- Implement adaptive learning rate scheduling
- Add regularization techniques (L1/L2, dropout optimization)
- Explore data augmentation strategies
- Test with other motor imagery datasets
- Implement uncertainty quantification
- Multi-task learning across subjects

---

## Troubleshooting

### Common Issues

**Issue**: MOABB dataset download fails
- **Solution**: Check internet connection; manually download from BCI Competition IV website

**Issue**: Out of memory during training
- **Solution**: Reduce batch size; clear session more frequently; use mixed precision

**Issue**: Poor accuracy on specific subjects
- **Solution**: Increase calibration data; adjust learning rate; check data quality

**Issue**: Slow training
- **Solution**: Use GPU acceleration; reduce model size; use mixed precision training

---

## Contact & Support

For questions or issues with this notebook, please refer to the original papers and documentation:

- **EEGNet Paper**: https://arxiv.org/abs/1611.08024
- **MOABB Documentation**: https://moabb.neurotechx.com/
- **TensorFlow Documentation**: https://www.tensorflow.org/

---

## License

This notebook implementation is provided for research and educational purposes. Please refer to the original dataset and papers for their respective licenses.

---

**Last Updated**: November 2025  
**Author**: Capstone Project - EEG Net Implementation  
**Version**: 1.0
