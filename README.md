# Machine Learning Seminar Roadmap

## Project Title

**Real-Time Face Attribute Classification Using Machine Learning on the UTKFace Dataset**

## Dataset

**Dataset:** UTKFace New Dataset  
**Source:** Kaggle  
**URL:** https://www.kaggle.com/datasets/jangedoo/utkface-new

### ABOUT DATASET
UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. This dataset could be used on a variety of tasks, e.g., face detection, age estimation, age progression/regression, landmark localization, etc.

## Seminar Context

This roadmap is prepared for a Master's degree seminar project in the subject of **Machine Learning**.  
The project uses the public **UTKFace** dataset and focuses on building a real-time face attribute classification system.

The seminar requires:

- A public dataset used in scientific research.
- Dataset preprocessing and analysis.
- Evaluation of how preprocessing affects model performance.
- Suitable machine learning or deep learning algorithms.
- Real-time performance measurement.
- Metrics such as FPS, accuracy, precision, recall, F1-score, and confusion matrix.
- Comparison with scientific papers that used the same dataset.
- Documentation in English.
- A group of 4 students with responsibilities divided from the beginning.

---

# 1. Recommended Project Scope

## Main Objective

Build a machine learning system that predicts facial attributes from images using the UTKFace dataset.

## Recommended Tasks

### Primary Task

**Gender Classification**

Input:

```text
Face image
```

Output:

```text
Male / Female
```

### Secondary Task

**Race / Ethnicity Classification**

Input:

```text
Face image
```

Output:

```text
White / Black / Asian / Indian / Others
```

### Optional Task

**Age Group Classification**

Instead of predicting exact age as regression, convert age into classification groups:

```text
0–12   = Child
13–19  = Teen
20–59  = Adult
60+    = Senior
```

## Why This Scope Is Recommended

Gender and race classification are classification problems, which fit well with the required evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Age prediction is naturally a regression problem, so it is harder to evaluate using confusion matrix and F1-score unless converted into age groups.

---

# 2. Dataset Understanding

## UTKFace Label Format

UTKFace does not provide labels in a separate CSV file.  
The labels are stored directly in the image filename.

Filename format:

```text
age_gender_race_datetime.jpg
```

Example:

```text
25_0_2_20170116174525125.jpg
```

This means:

```text
Age    = 25
Gender = 0
Race   = 2
```

## Gender Labels

```text
0 = Male
1 = Female
```

## Race / Ethnicity Labels

```text
0 = White
1 = Black
2 = Asian
3 = Indian
4 = Others
```

## Recommended Dataset CSV Structure

After parsing the dataset, create a CSV file named:

```text
dataset.csv
```

Recommended columns:

```text
image_path, age, gender, race, age_group
```

---

# 3. Final Documentation Structure

The final seminar report should contain the following chapters:

```text
1. Abstract and Keywords
2. Introduction
3. Related Work
4. Dataset Description
5. Methodology
6. Implementation
7. Results
8. Discussion
9. Conclusion and Future Work
10. References
```

---

# 4. Chapter-by-Chapter Roadmap

## 4.1 Abstract and Keywords

Write this chapter after completing the implementation and results.

The abstract should briefly explain:

- What dataset was used.
- What problem was solved.
- What model was used.
- What preprocessing was applied.
- What metrics were measured.
- What results were achieved.
- Whether real-time performance was possible.

Recommended keywords:

```text
Machine Learning, Deep Learning, CNN, Face Attribute Classification, UTKFace, Gender Classification, Race Classification, Real-Time Inference
```

---

## 4.2 Introduction

The introduction should explain the general problem and motivation.

Include:

- What face attribute recognition is.
- Why it is important in computer vision.
- Where it can be used.
- What the goal of the project is.
- What the main research questions are.

Example research questions:

```text
RQ1: How accurately can a CNN-based model classify gender from face images?

RQ2: How accurately can the model classify race/ethnicity from the UTKFace dataset?

RQ3: How does preprocessing affect model performance?

RQ4: Can the trained model perform inference in real time based on FPS?
```

---

## 4.3 Related Work

This chapter should compare your project with previous scientific work.

Search for papers related to:

- UTKFace dataset
- Age estimation using UTKFace
- Gender classification using UTKFace
- Race classification using UTKFace
- Multi-task learning on UTKFace
- Face attribute prediction using CNNs

For each paper, summarize:

- Dataset used
- Model used
- Task solved
- Metrics reported
- Main result
- Difference compared to your work

Your difference can be framed as:

```text
Our project focuses on a lightweight real-time CNN-based system. 
In addition to classification accuracy, we evaluate FPS to determine whether the model is suitable for real-time inference.
We also compare preprocessing strategies to analyze their effect on performance.
```

---

## 4.4 Dataset Description

This chapter should describe the dataset clearly.

Include:

- Dataset name: UTKFace
- Dataset type: face image dataset
- Number of images: approximately 20,000+
- Labels: age, gender, race
- Label source: image filename
- Image format: JPG
- Task: face attribute classification

Explain filename format:

```text
age_gender_race_datetime.jpg
```

Example:

```text
34_1_0_20170103182726357.jpg
```

Meaning:

```text
Age    = 34
Gender = Female
Race   = White
```

## Required Dataset Analysis

Generate and include:

- Gender distribution chart
- Race distribution chart
- Age distribution chart
- Age group distribution chart
- Sample images from each class
- Invalid or skipped filename count
- Train / validation / test split statistics

---

## 4.5 Preprocessing

This is one of the most important chapters because the seminar instruction requires dataset preprocessing and analysis of how changes affect results.

## Recommended Preprocessing Pipeline

```text
1. Read image files
2. Parse labels from filenames
3. Remove invalid filenames
4. Remove unreadable images
5. Resize images
6. Normalize pixel values
7. Convert labels into classification targets
8. Split dataset into train, validation, and test sets
9. Apply data augmentation
10. Prepare data loaders
```

## Recommended Image Size

Use one of the following:

```text
128x128
224x224
```

For real-time performance, `128x128` is faster.  
For better accuracy, `224x224` may perform better, especially with pretrained models.

## Recommended Dataset Split

```text
70% train
15% validation
15% test
```

## Preprocessing Experiments

Run at least three experiments:

### Experiment 1: Baseline Preprocessing

```text
Resize + normalization
```

### Experiment 2: Data Augmentation

```text
Resize + normalization + augmentation
```

Recommended augmentations:

- Random horizontal flip
- Random rotation
- Random brightness
- Random contrast
- Random zoom

### Experiment 3: Class Balancing

```text
Resize + normalization + augmentation + class balancing
```

Class balancing options:

- Class weights
- Oversampling
- Weighted random sampler

---

## 4.6 Methodology

This chapter explains the full pipeline used to build the system.

## Project Pipeline

```text
UTKFace Dataset
      ↓
Extract labels from filenames
      ↓
Clean invalid samples
      ↓
Analyze dataset distribution
      ↓
Resize and normalize images
      ↓
Apply augmentation
      ↓
Split dataset into train / validation / test
      ↓
Train baseline CNN
      ↓
Train MobileNetV2
      ↓
Evaluate classification metrics
      ↓
Measure FPS
      ↓
Compare with scientific papers
```

## Recommended Models

### Model 1: Simple CNN Baseline

Purpose:

- Easy to understand.
- Good for explaining core CNN concepts.
- Useful as a baseline comparison.

### Model 2: MobileNetV2

Purpose:

- Lightweight.
- Faster inference.
- Suitable for real-time testing.
- Strong enough for image classification.

### Optional Model 3: EfficientNetB0 or ResNet18

Purpose:

- Better accuracy comparison.
- More advanced model.
- May be slower than MobileNetV2.

## Recommended Final Model

```text
MobileNetV2
```

Reason:

```text
MobileNetV2 provides a good balance between accuracy and speed, making it suitable for real-time face attribute classification.
```

---

## 4.7 Implementation

This chapter should explain how the project was implemented.

## Recommended Project Structure

```text
utkface-seminar/
├── data/
│   ├── raw/
│   ├── processed/
│   └── dataset.csv
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_training_experiments.ipynb
│   └── 03_results_visualization.ipynb
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── realtime.py
│   └── utils.py
├── outputs/
│   ├── models/
│   ├── figures/
│   ├── reports/
│   └── metrics/
├── requirements.txt
├── README.md
└── ROADMAP.md
```

## Recommended Training Configuration

```text
Image size: 128x128 or 224x224
Batch size: 32
Epochs: 15–30
Optimizer: Adam
Learning rate: 0.001
Early stopping: enabled
Validation monitoring: validation loss
```

## Loss Functions

For gender classification:

```text
Binary Cross Entropy
```

For race classification:

```text
Categorical Cross Entropy
```

For age group classification:

```text
Categorical Cross Entropy
```

---

## 4.8 Results

This chapter should report model performance.

Include results for:

- Gender classification
- Race classification
- Optional age group classification
- Preprocessing experiments
- FPS / real-time inference

## Required Metrics

For each classification task, report:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Example Results Table

```text
+-----------------------+----------+-----------+--------+----------+------+
| Task                  | Accuracy | Precision | Recall | F1-score | FPS  |
+-----------------------+----------+-----------+--------+----------+------+
| Gender Classification | 91.2%    | 90.8%     | 91.0%  | 90.9%    | 45   |
| Race Classification   | 78.5%    | 77.9%     | 76.8%  | 77.2%    | 43   |
+-----------------------+----------+-----------+--------+----------+------+
```

## Preprocessing Comparison Table

```text
+-------------------------------+----------+-----------+--------+----------+
| Preprocessing Method          | Accuracy | Precision | Recall | F1-score |
+-------------------------------+----------+-----------+--------+----------+
| Resize + Normalize            | XX.X%    | XX.X%     | XX.X%  | XX.X%    |
| Resize + Normalize + Augment  | XX.X%    | XX.X%     | XX.X%  | XX.X%    |
| Augment + Class Balancing     | XX.X%    | XX.X%     | XX.X%  | XX.X%    |
+-------------------------------+----------+-----------+--------+----------+
```

## FPS Calculation

FPS should be calculated as:

```text
FPS = number of processed images / total inference time
```

Example:

```text
If the model processes 500 images in 10 seconds:

FPS = 500 / 10 = 50 FPS
```

Recommended FPS testing options:

- Test on 500 or 1000 images from the test set.
- Test on webcam frames.
- Test on a short video file.

---

## 4.9 Comparison with Scientific Papers

Create a comparison table.

Example:

```text
+----------------+----------+-------------+----------+----------------+
| Study / Paper  | Dataset  | Model       | Task     | Result         |
+----------------+----------+-------------+----------+----------------+
| Paper 1        | UTKFace  | CNN         | Age      | MAE XX         |
| Paper 2        | UTKFace  | ResNet      | Gender   | XX% accuracy   |
| Paper 3        | UTKFace  | Multi-task  | Multiple | XX% accuracy   |
| Our Work       | UTKFace  | MobileNetV2 | Gender   | XX% accuracy   |
+----------------+----------+-------------+----------+----------------+
```

Explain:

- Whether your model is more lightweight.
- Whether your model is faster.
- Whether your result is lower or higher.
- Why the difference may exist.
- Whether the comparison is fair.

Recommended honest conclusion:

```text
Our model may not outperform larger models, but it is lightweight and more suitable for real-time inference.
```

---

## 4.10 Discussion

This chapter should interpret the results.

Discuss:

- Which task performed best.
- Which task was most difficult.
- Which classes were confused most often.
- Whether preprocessing improved results.
- Whether class balancing helped.
- Whether FPS was good enough for real-time use.
- What makes your system different from previous work.
- What limitations exist.

## Possible Observations

```text
Gender classification usually performs better because it is a binary classification task.
Race classification is more difficult because the dataset contains class imbalance and visual similarities between some categories.
Data augmentation can improve generalization, but excessive augmentation may reduce performance.
MobileNetV2 provides better real-time speed compared to heavier models.
```

## Limitations

Mention limitations clearly:

- Dataset imbalance
- Possible label noise
- Lighting variation
- Pose variation
- Occlusion
- Ethical concerns around race classification
- Limited training time
- Lower performance on minority classes
- Hardware dependency of FPS results

---

## 4.11 Conclusion and Future Work

The conclusion should summarize:

- What was built.
- Which dataset was used.
- Which tasks were solved.
- Which models were trained.
- What results were achieved.
- Whether real-time inference was possible.

## Future Work Ideas

- Train on larger image size.
- Use stronger models such as ResNet50 or EfficientNet.
- Use multi-task learning.
- Add face detection before classification.
- Improve class balancing.
- Add fairness analysis.
- Test on live webcam stream.
- Deploy as a small web application.
- Optimize model using ONNX or TensorRT.
- Train for more epochs.
- Use better augmentation strategies.

---

# 5. Group Responsibility Division

The group must contain 4 students.  
Each student should have clear responsibilities from the beginning.

## Student 1: Dataset and Preprocessing

Responsibilities:

- Download the UTKFace dataset.
- Understand filename label structure.
- Parse age, gender, and race labels.
- Remove invalid filenames.
- Remove unreadable images.
- Create `dataset.csv`.
- Analyze class distribution.
- Prepare train / validation / test split.

Deliverables:

- Dataset parser
- Dataset analysis charts
- Clean dataset CSV
- Dataset description section

---

## Student 2: Model Development

Responsibilities:

- Build baseline CNN model.
- Build MobileNetV2 model.
- Train gender classification model.
- Train race classification model.
- Save trained models.
- Track training and validation loss.

Deliverables:

- Model code
- Training scripts
- Saved model files
- Training graphs

---

## Student 3: Evaluation and Real-Time Testing

Responsibilities:

- Evaluate models on the test set.
- Calculate accuracy, precision, recall, and F1-score.
- Generate confusion matrices.
- Measure FPS.
- Compare preprocessing experiments.
- Test real-time inference using images, webcam, or video.

Deliverables:

- Evaluation script
- Metrics tables
- Confusion matrix figures
- FPS results
- Results chapter data

---

## Student 4: Documentation and Related Work

Responsibilities:

- Search scientific papers using UTKFace.
- Write related work summary.
- Compare project results with papers.
- Prepare seminar report.
- Prepare presentation slides.
- Ensure documentation is written in English.
- Ensure AI-generated documentation does not exceed 30%.

Deliverables:

- Related work chapter
- Paper comparison table
- Final report
- Presentation slides
- References

---

# 6. Implementation Phases

## Phase 1: Dataset Preparation

Tasks:

- Download UTKFace from Kaggle.
- Inspect folder structure.
- Parse labels from image filenames.
- Create structured dataset CSV.
- Remove invalid records.

Output:

```text
outputs/metrics/dataset_summary.json
data/processed/dataset.csv
```

---

## Phase 2: Dataset Analysis

Tasks:

- Count gender classes.
- Count race classes.
- Count age groups.
- Plot age distribution.
- Show sample images.
- Identify class imbalance.

Output:

```text
outputs/figures/gender_distribution.png
outputs/figures/race_distribution.png
outputs/figures/age_distribution.png
outputs/figures/sample_images.png
```

---

## Phase 3: Preprocessing Experiments

Tasks:

- Resize images.
- Normalize images.
- Apply data augmentation.
- Apply class balancing if needed.
- Compare preprocessing variants.

Output:

```text
outputs/metrics/preprocessing_comparison.csv
```

---

## Phase 4: Model Training

Tasks:

- Train simple CNN baseline.
- Train MobileNetV2.
- Track train and validation accuracy.
- Track train and validation loss.
- Save best model.

Output:

```text
outputs/models/gender_cnn.pt
outputs/models/gender_mobilenetv2.pt
outputs/models/race_cnn.pt
outputs/models/race_mobilenetv2.pt
```

---

## Phase 5: Model Evaluation

Tasks:

- Load saved models.
- Evaluate on test set.
- Calculate metrics.
- Generate classification report.
- Generate confusion matrix.

Output:

```text
outputs/metrics/gender_report.json
outputs/metrics/race_report.json
outputs/figures/gender_confusion_matrix.png
outputs/figures/race_confusion_matrix.png
```

---

## Phase 6: Real-Time Inference Test

Tasks:

- Run inference on test images.
- Measure total inference time.
- Calculate FPS.
- Optional: test webcam or video stream.
- Display predicted class and FPS.

Output:

```text
outputs/metrics/fps_results.json
```

---

## Phase 7: Paper Comparison

Tasks:

- Select 3 to 5 papers using UTKFace or similar datasets.
- Extract task, model, and result from each paper.
- Compare with your result.
- Explain differences.

Output:

```text
outputs/reports/paper_comparison_table.md
```

---

## Phase 8: Final Report and Presentation

Tasks:

- Write final report in English.
- Add figures and tables.
- Add references.
- Prepare presentation slides.
- Practice explaining the code.

Output:

```text
outputs/reports/final_report.docx
outputs/reports/final_presentation.pptx
```

---

# 7. Recommended Repository Checklist

Before submitting, make sure the project contains:

```text
[ ] Dataset loading code
[ ] Label parsing code
[ ] Dataset analysis charts
[ ] Preprocessing pipeline
[ ] Train / validation / test split
[ ] Baseline CNN model
[ ] MobileNetV2 model
[ ] Training script
[ ] Evaluation script
[ ] Confusion matrices
[ ] Classification reports
[ ] FPS measurement script
[ ] Related work comparison
[ ] Final report
[ ] Presentation slides
[ ] README.md
[ ] ROADMAP.md
```

---

# 8. Recommended README Sections

The repository README should contain:

```text
1. Project Overview
2. Dataset
3. Tasks
4. Installation
5. Project Structure
6. How to Prepare Dataset
7. How to Train
8. How to Evaluate
9. How to Run Real-Time Inference
10. Results
11. References
```

---

# 9. Technical Stack Recommendation

Recommended language and libraries:

```text
Python
PyTorch or TensorFlow/Keras
OpenCV
NumPy
Pandas
Matplotlib
Scikit-learn
Torchvision
```

Recommended environment:

```text
Python 3.10+
CUDA GPU if available
Google Colab if local GPU is not available
```

---

# 10. Final Recommended Scope

The safest and strongest final scope is:

```text
Dataset: UTKFace
Main task: Gender classification
Second task: Race classification
Optional task: Age group classification
Models: Simple CNN + MobileNetV2
Evaluation: Accuracy, Precision, Recall, F1-score, Confusion Matrix
Real-time: FPS measurement on test images or webcam
Comparison: Scientific papers using UTKFace
```

This scope is realistic, measurable, and aligned with the seminar requirements.

---

# 11. Important Notes for Presentation

During the seminar presentation, be ready to explain:

- How labels are extracted from filenames.
- Why age was converted into groups if used.
- Why gender classification is easier than race classification.
- Why MobileNetV2 is suitable for real-time inference.
- How FPS is calculated.
- What preprocessing improved or did not improve.
- What confusion matrix shows.
- What limitations exist.
- How each group member contributed.
- What part of the code you wrote or generated using AI.

---

# 12. Suggested Presentation Flow

```text
1. Problem introduction
2. Dataset explanation
3. Label extraction from filename
4. Dataset analysis
5. Preprocessing pipeline
6. Model architecture
7. Training setup
8. Results and metrics
9. FPS / real-time performance
10. Comparison with papers
11. Discussion
12. Conclusion and future work
13. Group responsibilities
```

---

# 13. Final Success Criteria

The project is considered successful if:

```text
[ ] The dataset is correctly parsed.
[ ] Labels are correctly extracted.
[ ] At least one model is trained successfully.
[ ] Accuracy, precision, recall, F1-score, and confusion matrix are reported.
[ ] FPS is measured.
[ ] Preprocessing experiments are compared.
[ ] Results are compared with scientific papers.
[ ] The report follows the required structure.
[ ] The group responsibilities are clearly defined.
[ ] Every student can explain their part of the code.
```

---

# 14. Recommended Development Order

Follow this exact order:

```text
1. Download dataset
2. Parse labels from filenames
3. Create dataset.csv
4. Analyze class distributions
5. Split dataset
6. Build preprocessing pipeline
7. Train simple CNN for gender
8. Evaluate simple CNN
9. Train MobileNetV2 for gender
10. Evaluate MobileNetV2
11. Train race classification model
12. Evaluate race model
13. Measure FPS
14. Compare preprocessing experiments
15. Search and summarize papers
16. Write report
17. Prepare slides
18. Practice code explanation
```

---

# 15. Risk Management

## Risk: Dataset labels are not visible

Solution:

```text
Labels are stored in filenames using the format age_gender_race_datetime.jpg.
```

## Risk: Race classification accuracy is low

Solution:

```text
Mention class imbalance, visual similarity, and dataset label noise as limitations.
Use class weights or oversampling.
```

## Risk: FPS is low

Solution:

```text
Use MobileNetV2, smaller image size, batch inference, or model optimization.
```

## Risk: Documentation is too AI-generated

Solution:

```text
Use AI only for structure and grammar. Write the final explanation in your own words.
```

## Risk: Team members cannot explain code

Solution:

```text
Each student must understand their assigned part and prepare code explanation notes.
```

---

# 16. Final Recommendation

Start simple and expand only after the baseline works.

Recommended path:

```text
Gender classification first
      ↓
MobileNetV2 model
      ↓
Metrics and confusion matrix
      ↓
FPS test
      ↓
Race classification
      ↓
Paper comparison
      ↓
Final report
```

This gives a complete and realistic Master's seminar project.
