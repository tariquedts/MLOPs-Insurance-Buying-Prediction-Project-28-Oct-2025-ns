# Vehicle Insurance Purchase Prediction — End-to-End MLOps Project

An end-to-end **Machine Learning and MLOps project** that predicts whether a customer is likely to purchase vehicle insurance based on demographic, vehicle, policy, and customer-history attributes.

The project demonstrates the complete machine learning lifecycle — from data ingestion and validation to model training, evaluation, artifact management, prediction, API serving, Dockerization, and cloud-oriented deployment automation.

---

## Project Overview

Insurance companies need to identify customers who are more likely to purchase vehicle insurance so that marketing and sales teams can focus their efforts on high-potential customers.

This project builds a machine learning classification system that predicts a customer's likelihood of responding positively to a vehicle insurance offer.

Rather than treating this as only a model-building exercise, the project focuses on **production-oriented ML engineering and MLOps practices**, including:

* Modular project architecture
* Data ingestion and validation
* Feature engineering and preprocessing
* Machine learning model training
* Model evaluation
* Model artifact management
* Prediction pipeline
* FastAPI-based application
* Docker containerization
* AWS-oriented deployment workflow
* GitHub Actions CI/CD workflow
* Configuration and logging
* Reusable Python package structure

---

## Business Problem

An insurance company wants to identify existing customers who are likely to purchase a vehicle insurance policy.

Instead of contacting every customer, a predictive model can be used to prioritize customers with a higher probability of responding positively.

### Business objective

Build a machine learning solution that can:

1. Analyze historical customer and vehicle information.
2. Identify patterns associated with insurance purchase decisions.
3. Predict whether a customer is likely to purchase vehicle insurance.
4. Provide predictions through an application interface.
5. Create a reproducible and deployable ML pipeline.

### Potential business value

The solution can help an insurance company:

* Improve customer targeting
* Prioritize sales leads
* Reduce unnecessary marketing efforts
* Improve campaign efficiency
* Support data-driven decision-making
* Automate customer-response prediction

---

## Machine Learning Problem

This is a **supervised binary classification problem**.

### Target

The model predicts the customer's response:

| Response | Meaning                                                       |
| -------- | ------------------------------------------------------------- |
| `1`      | Customer is likely to respond positively / purchase insurance |
| `0`      | Customer is unlikely to respond positively                    |

The prediction pipeline ultimately converts the model output into:

```text
Response-Yes
Response-No
```

---

## Input Features

The application accepts customer and vehicle-related attributes including:

* Gender
* Age
* Driving License
* Region Code
* Previously Insured
* Annual Premium
* Policy Sales Channel
* Customer Vintage
* Vehicle Age
* Vehicle Damage

The deployed application collects these values through a web form and passes them to the prediction pipeline.

---

## End-to-End ML Workflow

```text
                    ┌──────────────────────┐
                    │   Raw Insurance Data │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Data Ingestion     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Validation      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Transformation  │
                    │ & Feature Engineering│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Model Training     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Model Evaluation     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Model Artifacts      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Prediction Pipeline  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     FastAPI App      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Customer Prediction  │
                    └──────────────────────┘
```

---

## Project Architecture

The repository follows a modular architecture instead of keeping the complete workflow inside a single notebook.

```text
MLOPs-Insurance-Buying-Prediction-Project/
│
├── .github/
│   └── workflows/
│       └── aws.yaml
│
├── artifact/
│   └── 10_28_2025_15_27_57/
│
├── config/
│
├── notebook/
│
├── src/
│   ├── cloud_storage/
│   ├── components/
│   ├── configuration/
│   ├── constants/
│   ├── data_access/
│   ├── entity/
│   ├── exception/
│   ├── logger/
│   ├── pipline/
│   └── utils/
│
├── static/
│   └── css/
│
├── templates/
│
├── app.py
├── demo.py
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── setup.py
├── template.py
├── LICENSE
└── README.md
```

The repository currently separates major responsibilities into `cloud_storage`, `components`, `configuration`, `data_access`, `entity`, `logger`, `pipline`, and `utils`, which makes the project easier to maintain and extend.

---

## 🛠️ Technology Stack

### Programming & Data Science

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Plotly

### Machine Learning

* Supervised Machine Learning
* Binary Classification
* Feature Engineering
* Data Preprocessing
* Model Evaluation
* Imbalanced-data handling

### Backend & API

* FastAPI
* Uvicorn
* Jinja2
* HTML/CSS

### MLOps & Engineering

* Modular Python architecture
* Configuration management
* Logging
* Exception handling
* Model artifact management
* Reproducible pipelines
* Git/GitHub

### Cloud & Deployment

* AWS
* Amazon S3 / cloud storage integration
* Boto3
* Docker
* GitHub Actions

The project's dependency configuration includes Pandas, NumPy, Scikit-learn, imbalanced-learn, PyMongo, Boto3, FastAPI, Uvicorn, Jinja2 and related packages.

---

## Data Science Workflow

### 1. Data Ingestion

The project follows a dedicated pipeline architecture for bringing data into the ML workflow rather than coupling ingestion directly to model training.

### 2. Data Validation

The pipeline is structured to validate incoming data before it proceeds through subsequent stages.

### 3. Data Transformation

Customer and vehicle attributes are transformed into a format suitable for machine learning.

The project also includes handling for categorical/binary vehicle-related features such as:

* Vehicle age categories
* Vehicle damage
* Previously insured status

### 4. Model Training

The transformed dataset is used to train classification models and generate model artifacts.

### 5. Model Evaluation

The trained model is evaluated before being used by the prediction pipeline.

### 6. Prediction

The trained model is loaded by the prediction pipeline and used to generate a response for new customer information.

---

## FastAPI Application

The project provides a FastAPI-based web application.

The application:

1. Displays a customer input form.
2. Collects vehicle/customer information.
3. Converts the submitted information into a structured DataFrame.
4. Passes the data to the prediction pipeline.
5. Generates a model prediction.
6. Displays either `Response-Yes` or `Response-No`.

The application also exposes a `/train` endpoint that triggers the training pipeline.

### Application flow

```text
Customer
   │
   ▼
Web Form
   │
   ▼
FastAPI
   │
   ▼
DataForm
   │
   ▼
VehicleData
   │
   ▼
Prediction Pipeline
   │
   ▼
Trained Model
   │
   ▼
Prediction
   │
   ▼
Response-Yes / Response-No
```

---

## Dockerization

The application is containerized using Docker.

The project uses a lightweight Python 3.10 base image, installs the project's dependencies, exposes port `5000`, and starts the FastAPI application through `app.py`.

### Build Docker image

```bash
docker build -t vehicle-insurance-prediction .
```

### Run container

```bash
docker run -p 5000:5000 vehicle-insurance-prediction
```

Then open:

```text
http://localhost:5000
```

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/tariquedts/MLOPs-Insurance-Buying-Prediction-Project-28-Oct-2025-ns.git
```

```bash
cd MLOPs-Insurance-Buying-Prediction-Project-28-Oct-2025-ns
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

The application is configured to start through Uvicorn.

---

## Training the Model

The project exposes a training route that triggers the training pipeline.

```text
GET /train
```

The training pipeline is initialized and executed through the application's `TrainPipeline` implementation.

---

## 📈 Prediction Example

After launching the application, enter customer information such as:

```text
Age: 35
Gender: 1
Driving License: 1
Previously Insured: 0
Annual Premium: 30,000
Vehicle Damage: Yes
Vehicle Age: > 2 Years
```

The application sends the information through the prediction pipeline and returns:

```text
Response-Yes
```

or

```text
Response-No
```

---

## MLOps Components

This project demonstrates several important MLOps concepts:

### Modularization

The ML system is separated into reusable components rather than being implemented as one large script.

### Configuration Management

Project configuration is separated from business logic.

### Logging

A dedicated logging module is included for tracking application and pipeline execution.

### Exception Handling

A dedicated exception structure is included to make pipeline failures easier to identify and debug.

### Artifact Management

Model-related artifacts are maintained separately from the source-code components.

### Cloud Integration

The project includes cloud-storage functionality and AWS-related dependencies.

### CI/CD

A GitHub Actions workflow is included under:

```text
.github/workflows/aws.yaml
```

This provides the foundation for automating the deployment workflow.

---

## ☁️ Cloud & Deployment Architecture

The project is designed with cloud deployment in mind:

```text
                    GitHub Repository
                           │
                           ▼
                    GitHub Actions
                           │
                           ▼
                      Build/Test
                           │
                           ▼
                       Docker
                           │
                           ▼
                     AWS Services
                           │
                           ▼
                    Deployed Model
                           │
                           ▼
                       FastAPI
                           │
                           ▼
                    Prediction API
```

---

## Key Learning Outcomes

This project helped demonstrate practical experience in:

* End-to-end machine learning workflow
* Classification problem solving
* Data preprocessing
* Feature engineering
* ML pipeline development
* Modular Python architecture
* Model artifact management
* REST API development
* FastAPI
* Docker containerization
* AWS integration
* CI/CD concepts
* GitHub Actions
* Production-oriented ML engineering

---

## Future Improvements

Potential improvements for taking the project further toward production include:

* Add automated unit and integration tests
* Add model-performance tracking
* Add experiment tracking with MLflow
* Add data and model versioning with DVC
* Add automated model retraining
* Add model monitoring and data-drift detection
* Add API authentication and authorization
* Add structured application metrics
* Add Prometheus/Grafana monitoring
* Add automated deployment to AWS
* Add model explainability using SHAP
* Add API documentation and example requests

---

## Author

### Tarique Anwar

**Data Scientist | Machine Learning | MLOps**

I am a Data Scientist focused on building practical machine learning solutions and developing skills across the complete ML lifecycle — from data analysis and model development to deployment and MLOps.

### Connect with me

* **GitHub:** [tariquedts](https://github.com/tariquedts)
* **LinkedIn:** [Tarique Anwar](https://www.linkedin.com/in/tarique-anwar-bb8535a1)

---

## 📄 License

This project is licensed under the **MIT License**.

---

## ⭐ If you find this project useful

If this project helps you understand end-to-end machine learning and MLOps, consider giving the repository a ⭐ and exploring the implementation.

**Project Repository:**
https://github.com/tariquedts/MLOPs-Insurance-Buying-Prediction-Project-28-Oct-2025-ns
