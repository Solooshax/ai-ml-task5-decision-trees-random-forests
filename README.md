# Task 5: Decision Trees and Random Forests

## Objective
Train and compare a Decision Tree Classifier and a Random Forest Classifier, study overfitting, visualize a decision tree, interpret feature importances, and evaluate using cross-validation.

## Dataset
This project uses the Breast Cancer Wisconsin Diagnostic dataset built into scikit-learn. The dataset is loaded programmatically, so no separate download is needed.

- 569 samples
- 30 numeric features
- Binary classification: malignant or benign

## Repository structure
```text
task-5-decision-trees-random-forests/
├── notebooks/
│   └── task5_decision_trees_random_forests.ipynb
├── outputs/
│   └── .gitkeep
├── requirements.txt
└── README.md
```

## Setup and run
Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Activate the environment:

**macOS/Linux**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Open `notebooks/task5_decision_trees_random_forests.ipynb` and run all cells from top to bottom.

## Work completed
1. Loaded the dataset and performed a stratified train/test split.
2. Trained a baseline Decision Tree and evaluated accuracy.
3. Visualized the decision tree using scikit-learn's plotting tools.
4. Compared train and test accuracy to discuss overfitting.
5. Controlled tree complexity with `max_depth` and `min_samples_leaf`.
6. Trained a Random Forest and compared test accuracy.
7. Evaluated both models with stratified 5-fold cross-validation.
8. Plotted Random Forest feature importances.

The notebook prints actual metric values when run. Results can vary slightly across library versions, although fixed random seeds are used.

## Interview questions and answers

### 1. How does a decision tree work?
A decision tree recursively splits data using feature-based rules. Each split aims to make the resulting groups purer. A leaf node produces the final class prediction.

### 2. What are entropy and information gain?
Entropy measures class impurity:
`Entropy = -sum(p_i * log2(p_i))`.
Information gain is the reduction in entropy after a split. A tree can select a split with high information gain.

### 3. How is a random forest better than a single tree?
A random forest averages predictions from many trees trained on bootstrapped samples and random feature subsets. This generally reduces variance and overfitting compared with one deep tree.

### 4. What is overfitting and how can it be prevented?
Overfitting occurs when a model learns training-specific noise and performs worse on unseen data. For trees, use constraints such as `max_depth`, `min_samples_leaf`, and `min_samples_split`, and validate with cross-validation.

### 5. What is bagging?
Bootstrap aggregating trains models on samples drawn with replacement and combines their predictions to reduce variance.

### 6. How do you visualize a decision tree?
Use `sklearn.tree.plot_tree()` with feature names and class names, or export the tree to Graphviz for a more detailed diagram.

### 7. How do you interpret feature importance?
Tree-based feature importance estimates how much each feature reduces impurity across splits. It is useful for exploration, but can be biased toward certain feature types and does not imply causation.

### 8. What are the pros and cons of random forests?
**Pros:** strong baseline performance, handles nonlinear relationships, reduces variance, and provides feature importance.
**Cons:** less interpretable than a single tree, can require more memory and computation, and impurity importance can be misleading for correlated features.

## Submission checklist
- [ ] Run every notebook cell successfully.
- [ ] Review the generated metric values and plots.
- [ ] Ensure the notebook is saved with outputs if your instructor expects screenshots/results.
- [ ] Push this folder to a new GitHub repository.
- [ ] Submit the GitHub repository URL through the task submission form.
