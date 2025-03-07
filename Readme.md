
<img width="472" alt="image" src="https://github.com/user-attachments/assets/410a37ca-9d68-442c-9495-a0216a91fb5d" />


<h1>Kaggle Competition - Titanic Survival Prediction (Score:0.774)</h1>

This project focuses on predicting passenger survival from the Titanic disaster using data from a Kaggle challenge. The objective was to build a model that predicts which types of passengers were more likely to survive based on features like age, gender, and socio-economic class.

## Project Overview

### 1. Exploratory Data Analysis (EDA)
Performed EDA to understand the dataset and identify key trends:

- **Data Cleaning**: Handled missing values, particularly in the `Age` and `Cabin` columns, by imputing median values or dropping uninformative features.
- **Feature Exploration**: Analyzed key variables (`Sex`, `Age`, `Fare`, `Pclass`) to determine their correlation with survival rates.
- **Outlier Detection**: Used visual techniques like boxplots to identify and remove outliers in continuous variables like `Fare`.
- **Visualization**: Created visualizations using `matplotlib` and `seaborn` to analyze relationships between variables (e.g., survival rates against gender, passenger class, and embarked location).

### 2. Feature Engineering
Enhanced model performance by creating new features:

- **Title Extraction**: Extracted titles (e.g., Mr., Mrs., Miss) from passenger names to capture social status and gender information.
- **Family Size**: Created a new feature by combining `SibSp` and `Parch` (siblings/spouse and parents/children).
- **Fare Binning**: Grouped `Fare` into categories for better interpretability by the model.

### 3. Data Preprocessing
Prepared the data for machine learning:

- **Label Encoding**: Converted categorical variables (`Sex`, `Embarked`) into numerical labels.
- **Standardization**: Scaled continuous features (`Age`, `Fare`) using `StandardScaler`.
- **Splitting Data**: Split the dataset into training and testing sets for evaluation.

### 4. Model Building
Implemented different machine learning models to compare performance:

- **Logistic Regression**: Served as a baseline model for classification.
- **Random Forest Classifier**: Improved performance by capturing non-linear relationships and handling missing data.
- **Support Vector Machines (SVM)**: Experimented with SVM to capture more complex decision boundaries.

### 5. Hyperparameter Tuning
Optimized model performance using:

- **GridSearchCV**: Tuned hyperparameters for the Random Forest model (e.g., number of trees, max depth, and minimum samples per leaf), improving accuracy.

### 6. Statistical Testing and Validation
Validated model performance using:

- **Cross-validation**: Used K-fold cross-validation to ensure the model generalized well across different data subsets.
- **Confusion Matrix & ROC Curve**: Measured precision, recall, and overall classification performance.

### 7. Data Visualization
Presented findings and model results using visualizations:

- **Feature Importance Plot**: Showed which features were most influential in predicting survival using the Random Forest model.
- **Survival Distribution Plots**: Plotted survival distributions across different classes and genders to visually convey insights.

### 8. Model Submission
Generated predictions on the test dataset and submitted them to the Kaggle leaderboard. The final model achieved an accuracy of **0.77033**, placing in a respectable position on the leaderboard.

## Key Takeaways
This project provided hands-on experience in:

- Data analysis and visualization using `pandas`, `numpy`, `matplotlib`, and `seaborn`
- Machine learning with `scikit-learn`
- Feature engineering and preprocessing
- Hyperparameter tuning and model evaluation

---

## How to Run the Project

1. Clone this repository:
   ```sh
   git clone https://github.com/goldstring/kaggle-competition-Titanic-Survival-Prediction-Using-Machine-Learning.git
   ```

2. Run the Jupyter Notebook to execute the analysis:
   ```sh
   jupyter notebook titanic-machine-learning-from-disaster.ipynb
   ```

## Dataset
The dataset can be found on [Kaggle's Titanic Challenge](https://www.kaggle.com/c/titanic).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
