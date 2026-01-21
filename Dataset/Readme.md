# Social Media Bot Detection (Metadata-based)

This project focuses on detecting automated social media accounts using structured profile and behavioral metadata. 
Instead of relying on tweet content or NLP techniques, the model analyzes account-level and activity-based features 
to identify bot-like patterns.

## Dataset Overview

The dataset consists of user profile and activity metadata collected at the account level. 
Each record represents a user and includes structured numerical and boolean attributes, along with a binary label 
indicating whether the account is automated (bot) or human-operated.

### Example Features Used
- Follower count and following count
- Follower–following ratio
- Posting activity (status count)
- Account age (in days)
- Profile attributes (verified status, default profile settings)

## Modeling Approach

- **Preprocessing:** Cleaned and standardized structured metadata features.
- **Feature Engineering:** Derived behavioral indicators such as follower–following ratio and account age.
- **Modeling:** Trained a Random Forest classifier to distinguish bot and human accounts.
- **Explainability:** Used feature importance to interpret which attributes influence predictions.

## Evaluation

Model evaluation was performed offline using standard classification metrics such as accuracy and recall. 
The Streamlit application focuses on inference and explainability rather than live metric reporting.

## Application Demo

A lightweight Streamlit interface is provided to:
- Input account metadata
- Generate bot or human predictions
- Visualize feature importance for interpretability

## Notes

This project is intended as a prototype to demonstrate machine learning workflows, feature engineering, 
and model interpretability using structured data rather than production-scale deployment.
