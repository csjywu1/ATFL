```markdown
# ATFL: Adaptive Federated Self-Supervised Learning for Robust Trajectory Anomaly Detection Across Cities

## Project Overview
This project implements the Adaptive Federated Self-Supervised Learning (ATFL) method proposed in the paper "ATFL: Adaptive Federated Self-Supervised Learning for Robust Trajectory Anomaly Detection Across Cities" for cross-city trajectory anomaly detection. The method enhances the robustness and accuracy of trajectory anomaly detection while preserving data privacy through federated learning and self-supervised learning techniques.

## Environment Setup
Before running the code, ensure your environment meets the following requirements:

- **Python Version**: 3.8 or higher
- **Dependencies**:
  - PyTorch (1.10.0 or higher)
  - NumPy (1.20.0 or higher)
  - Pandas (1.3.0 or higher)
  - Scikit-learn (1.0.0 or higher)
  - Matplotlib (3.4.0 or higher)

Install the dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation
This project requires trajectory data as input. Please follow these steps to prepare the data:
1. Download publicly available trajectory datasets (e.g., traffic trajectory data).
2. Preprocess the data into a format suitable for model input.
3. Store the preprocessed data in the `data/` folder.

### Dataset Description
We utilize three real-world taxi trajectory datasets across cities: Porto, Chengdu, and Xian. These datasets can be obtained from the following links:
- [Porto Dataset](https://www.kaggle.com/datasets/crailtap/taxi-trajectory)
- [Chengdu and Xian Datasets](https://github.com/mzy94/JCLRNT)

### Data Preprocessing
The data preprocessing involves the following steps:
1. **Map Matching**: Map GPS points to the road network.
2. **Trajectory Filtering**: Filter out trajectories with two or fewer road segments or a duration of less than one minute.
3. **Data Splitting**: Divide the data into training and testing sets, with the training set covering the first five days and the testing set covering the following two days.


## Running the Code
### 1. Data Preprocessing
Run the following command to preprocess the raw data:
```bash
python preprocess.py --input_path <path_to_raw_data> --output_path data/processed_data
```

### 2. Model Training
Run the following command to train the ATFL model:
```bash
python  ATFL.py
```


## Experimental Results
After running the code, you can find the experimental results, including detected anomalous trajectories and performance metrics, in the `results/` folder.

## Contribution Statement
This project code is an implementation of the paper "ATFL: Adaptive Federated Self-Supervised Learning for Robust Trajectory Anomaly Detection Across Cities." The code does not contain any information that may reveal the authors' identities or research institutions.

## Acknowledgments
We express our gratitude to all teams and resources that provided support for this project.
```
