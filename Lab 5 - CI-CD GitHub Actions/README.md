# CI/CD Implementation using GitHub Actions

CSE Skill Gap Classification using Random Forest.

This project demonstrates automated testing, model training,
and website deployment using GitHub Actions.



# CI/CD Implementation using GitHub Actions

## Project Title
CSE Curriculum–Industry Skill Gap Classification using Random Forest

## Objective
To implement Continuous Integration (CI) and Continuous Deployment (CD)
using GitHub Actions for a machine learning project.

## Dataset
The CSE Skill Gap dataset contains curriculum-related and
industry-demand skill features. The target column is `GapStatus`.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Random Forest Classifier
- Pytest
- GitHub Actions
- GitHub Pages

## Project Workflow
1. Load the CSE Skill Gap dataset.
2. Preprocess numerical features and handle missing values.
3. Train a Random Forest classification model.
4. Evaluate accuracy, precision, recall and F1-score.
5. Save the trained model and evaluation metrics.
6. Run automated tests using GitHub Actions.
7. Deploy the project status webpage only after CI succeeds.

## Project Files
- `cse_skill_gap.csv` — dataset
- `train_model.py` — model training and evaluation
- `test_train_model.py` — automated tests
- `requirements.txt` — dependencies
- `site/index.html` — deployment webpage

The combined workflow will be stored at the repository-level path
`.github/workflows/cicd.yml`.

## Expected Outcome
The GitHub Actions workflow automatically tests and trains the model.
If the CI job succeeds, the deployment job publishes the webpage using
GitHub Pages.

## Conclusion
This project demonstrates an automated machine learning workflow
integrating model training, testing and deployment using GitHub Actions.
