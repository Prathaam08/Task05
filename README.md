# Task05
# Task 05 - Decision Trees and Random Forests

## 📌 Objective
To build and evaluate classification models using Decision Trees and Random Forests on the Heart Disease Dataset.

## 🧰 Tools Used
- Python 3
- Pandas, Scikit-learn, Matplotlib, Seaborn
- Graphviz (for decision tree visualization)

## 📊 Dataset
- Dataset: Heart Disease Dataset (`heart.csv`)
- Features: age, sex, cp, chol, thalach, etc.
- Target: `0` (no disease), `1` (disease)

## 🔍 Steps Performed
1. Loaded and explored the dataset
2. Trained a **Decision Tree Classifier**
   - Visualized using Graphviz
   - Analyzed performance with depth control
3. Trained a **Random Forest Classifier**
   - Compared accuracy
   - Interpreted feature importances
4. Evaluated both models with **cross-validation**
5. Visualized accuracy trends and feature importances

## 🖼️ Screenshots
Include screenshots in the `images/` folder:
- Decision tree visualization
- Accuracy vs depth plot
- Terminal output

## 📁 Files
- `task05.py`: Main script
- `heart.csv`: Dataset
- `decision_tree.pdf`: Tree diagram
- `accuracy_plot.png`: Accuracy plot

---

## Install required packages:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn graphviz
   ```
## 🚀 How to Run
```
python task05.py
