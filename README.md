# Custom Decision Tree Classifier

This is a custom-built Decision Tree Classifier in Python, designed to handle numerical features for binary or multiclass classification tasks. It employs the ID3 algorithm and uses entropy as a measure to calculate information gain between parent and child nodes.

Key Features:
- **ID3 Algorithm**: The classifier uses the ID3 algorithm for decision-making.
- **Entropy-Based Information Gain**: It calculates the information gain based on entropy criteria. The feature (node) with the highest information gain is chosen at each depth.
- **Feature Removal**: Once a feature is selected, it is removed from consideration in subsequent depths. This process continues until no features are left at the leaf nodes.
- **Optimal Split Finding**: The classifier identifies the best split for each feature by calculating entropy after splitting at every possible value.

Usage:
- **Constructor**: Use the constructor to create a Decision Tree object. It includes a parameter to specify the maximum depth.
- **Fit Function**: This function is used for training the model.
- **Predict Function**: This function is used for testing the model, similar to sklearn models.
