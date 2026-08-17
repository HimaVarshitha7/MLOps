Name: A.HimaVarshitha
Register Number: 231FA04A55
Section: 15

## Curriculum–Industry Skill Gap Analysis using Feast

## 1. Problem Statement

The project focuses on identifying the gap between the skills developed through a Computer Science and Engineering (CSE) curriculum and the skills required by the software industry. Students may have good academic performance but may still have gaps in technical and professional skills such as Programming, Data Structures, Algorithms, Cloud Computing, AI/ML, Communication, Problem Solving, and Teamwork.

The objective of this project is to use student skill data to analyze these skill gaps and build a machine learning workflow for predicting student placement status. **Feast Feature Store** is used to manage and retrieve the engineered features for both model training and online prediction.

---

## 2. Dataset

### Dataset Size

* **Number of records:** 1,000 synthetic student records
* **Dataset type:** Synthetic dataset
* **Target variable:** `PlacementStatus`

### Dataset Columns

The dataset contains student information and skill-related attributes:

* `StudentName`
* `CGPA`
* `Programming`
* `DataStructures`
* `Algorithms`
* `Database`
* `WebDevelopment`
* `Cloud`
* `AI_ML`
* `Communication`
* `ProblemSolving`
* `Teamwork`
* `Projects`
* `Internship`
* `Certifications`
* `CurriculumAlignment`
* `IndustryRequirement`
* `SkillGap`
* `PlacementStatus`

The dataset entries were synthetically generated to represent different levels of academic performance, technical skills, professional skills, curriculum alignment, industry requirements, and skill gaps among CSE students.

---

## 3. Feature Engineering

The following features are created and managed through Feast:

| Feature                  | Description                                                    |
| ------------------------ | -------------------------------------------------------------- |
| `student_id`             | Unique identifier assigned to each student                     |
| `cgpa`                   | Student's academic CGPA                                        |
| `programming`            | Programming skill score                                        |
| `data_structures`        | Data Structures skill score                                    |
| `algorithms`             | Algorithms skill score                                         |
| `database`               | Database skill score                                           |
| `web_development`        | Web Development skill score                                    |
| `cloud`                  | Cloud Computing skill score                                    |
| `ai_ml`                  | Artificial Intelligence and Machine Learning skill score       |
| `communication`          | Communication skill score                                      |
| `problem_solving`        | Problem-solving skill score                                    |
| `teamwork`               | Teamwork skill score                                           |
| `projects`               | Project experience/score                                       |
| `internship`             | Internship experience/score                                    |
| `certifications`         | Certification-related score                                    |
| `curriculum_alignment`   | Alignment between curriculum and industry requirements         |
| `industry_requirement`   | Industry-required skill level                                  |
| `skill_gap`              | Difference between current skills and required skills          |
| `technical_skill_avg`    | Average of the major technical skill scores                    |
| `professional_skill_avg` | Average of Communication, Problem Solving, and Teamwork scores |

Two additional derived features are created:

### Technical Skill Average

Calculated using:

```text
Programming
Data Structures
Algorithms
Database
Web Development
Cloud
AI/ML
```

### Professional Skill Average

Calculated using:

```text
Communication
Problem Solving
Teamwork
```

Missing numerical values are handled using the median before feature creation.

---

## 4. Feast Architecture

```text
              Original Dataset
                     |
                     v
             Feature Engineering
                     |
                     v
          Parquet Offline Feature Data
                     |
                     v
              Feast FeatureView
                     |
              +------+------+
              |             |
              v             v
    Historical Features   Materialization
              |             |
              v             v
       Model Training    Online Store
                            |
                            v
                    Online Retrieval
                            |
                            v
                        Prediction
```

The original dataset is first transformed into engineered features. These features are stored as Parquet offline data and registered with Feast through a `FeatureView`. Historical features are retrieved for model training, while the features are materialized into an online store for real-time retrieval and prediction.

---

## 5. Implementation

### Entity

The Feast entity is:

```text
student
```

The join key is:

```text
student_id
```

A unique `student_id` is generated for each student and is used to identify students in Feast.

### Data Source

The engineered features are stored in:

```text
skill_gap_features.parquet
```

Feast uses a `FileSource` to read this Parquet file. The `event_timestamp` is used for time-based feature retrieval.

### FeatureView

The FeatureView is named:

```text
skill_gap_features
```

It contains the academic, technical, professional, and skill-gap features required by the machine learning model.

### Historical Feature Retrieval

Historical features are retrieved using Feast:

```python
training_data = store.get_historical_features(
    entity_df=entity_df,
    features=feature_service
).to_df()
```

The retrieved data is then divided into input features `X` and target `y`.

### Machine Learning Model

A **Decision Tree Classifier** is used as the basic machine learning model.

```python
DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)
```

The dataset is divided into:

* **80% training data**
* **20% testing data**

The model predicts the `PlacementStatus` of students.

### Online Retrieval

After training, the Feast features are materialized into the online store.

Features can then be retrieved using a student's `student_id`, for example:

```python
{"student_id": 25}
```

The retrieved online features are passed to the trained model to generate a placement prediction.

---

## 6. Results

### Historical Feature Output

Feast successfully retrieves the engineered historical features required for model training. The output contains the student ID, academic features, technical skills, professional skills, curriculum alignment, industry requirement, skill gap, and placement status.

### Model Accuracy

The model accuracy is calculated using:

```python
accuracy_score(y_test, predictions)
```

**Accuracy obtained:** `________ %`

> Run the accuracy cell in Google Colab and replace the blank with the value displayed by your notebook.

### Online Feature Output

The Feast online store successfully retrieves feature values for individual students using their `student_id`. For example:

```text
Student ID: 25
```

The retrieved features are converted into a DataFrame and used for online prediction.

### Final Prediction

The trained Decision Tree model generates a placement prediction from the retrieved online features.

```text
Predicted Placement Status: ______
```

Replace the blank with the value displayed after running the final prediction cell in Google Colab. The uploaded notebook currently has its execution outputs cleared, so the exact numerical accuracy and final prediction cannot be determined from the notebook file itself.

---

## 7. Technologies Used

* Python
* Pandas
* Scikit-learn
* Feast
* PyArrow
* Google Colab
* Parquet
* SQLite

---

## 8. Conclusion

This project demonstrates how a **Feast Feature Store** can be integrated into a curriculum–industry skill-gap machine learning workflow. Student skill data is transformed into meaningful features, stored as offline feature data, registered using a Feast FeatureView, and retrieved for both historical model training and online prediction.

The system provides a structured approach for managing student skill features and can serve as a foundation for identifying employability skill gaps and supporting data-driven curriculum improvement.
