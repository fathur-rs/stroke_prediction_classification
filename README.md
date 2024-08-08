# Stroke Prediction CLI Program

## Overview
This Stroke Prediction CLI Program is a Python-based tool designed to predict the likelihood of a stroke based on user-provided health data. Utilizing Decision Tree algorithm, this program offers a quick and user-friendly way to assess stroke risk.

## Input Parameters
- Patient Name
- Age
- Body Weight (kg)
- Height (cm)
- Average blood sugar
- Residential Area
- Gender
- Profession Type
- is Married
- Smoker?
- Patient has hypertensin?
- Patient has heart disease?

## Installation
To set up the Stroke Prediction CLI Program, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/fathur-rs/stroke-prediction-classification.git
   ```
2. Navigate to the program's directory:
   ```bash
   cd stroke-prediction-classification
   ```
3. Install required Python packages:
   ```bash
   pip install -r requirements_package.txt
   ```

## Usage
Run the program with Python from the command line:

```bash
python src/user_interface.py
```

Follow the on-screen prompts to enter the required health parameters. After all inputs are provided, the program will display the stroke risk assessment.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Dataset Source](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

