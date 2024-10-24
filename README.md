# **ClimateWins Weather Prediction Project**


## **Overview**
This project is part of a machine learning course focused on predicting weather variations using advanced algorithms. The aim is to help ClimateWins, an organization committed to forecasting and understanding climate change, achieve its goals. The project spans two achievements, each exploring different machine learning techniques applied to historical weather data to predict atmospheric conditions.

![Screenshot 2024-10-23 194635](https://github.com/user-attachments/assets/7f75d427-5bf6-4806-9d57-ce315a9f6b9f)
Photo by ChatGPT

## **Objectives**
* Identify weather patterns outside the regional norm in Europe.
* Determine if unusual weather patterns are increasing over time.
* Generate possibilities for future weather conditions over the next 25 to 50 years.
* Identify the safest places for people to live in Europe over the next 25 to 50 years.

## **Dataset**

For this project, the following datasets were used:

1. **Weather Conditions Dataset**  
   Downloaded from Kaggle, this dataset includes images of various weather conditions such as sunny, cloudy, rainy, and snowy.  
   [Download the dataset here](https://www.kaggle.com/datasets/USERNAME/weather-dataset)

2. **Handwritten Digits (MNIST)**  
   The MNIST dataset was used to train and test handwriting recognition models. This dataset contains images of handwritten digits from 0 to 9.  
   [Download the dataset here](http://yann.lecun.com/exdb/mnist/)

3. **European Historical Weather Data**  
   This dataset includes historical weather data from various European cities used for training and testing weather prediction models.  
   [Download the dataset here](https://www.kaggle.com/datasets/USERNAME/european-historical-weather-data)

4. **Synthetic Weather Images for GAN Training**  
   A set of synthetic weather images generated for training Generative Adversarial Networks (GANs).  
   [Download the dataset here](https://www.kaggle.com/datasets/USERNAME/synthetic-weather-images)

Please ensure to cite the dataset sources if you use them for further research or projects.

## **Achievements Summary**

### **Achievement 1**: Statistical and Machine Learning Models
**Data Analysis:** Performed exploratory data analysis (EDA) on historical weather data.

**K-Nearest Neighbors (KNN):** Used to classify weather conditions based on temperature and other features.

**Linear and Logistic Regression:** Developed regression models for temperature trends and to classify binary outcomes (e.g., rain/no rain).

**Random Forest:** Applied to enhance prediction accuracy by combining multiple decision trees.
![Screenshot 2024-10-16 185133](https://github.com/user-attachments/assets/4ec75e16-d32e-45a6-bb88-09a7e04921a0)![Screenshot 2024-10-18 211140](https://github.com/user-attachments/assets/bf58d59a-3ba2-4bb3-9f8a-6c6163764f6b)

### **Achievement 2: Deep Learning Techniques**

**Convolutional Neural Network (CNN):** Built a CNN for visual weather pattern recognition using images of different weather conditions (cloudy, sunny, rainy).

**Recurrent Neural Network (RNN):** Applied to predict time-series weather data, leveraging sequential data.

**Hyperparameter Tuning:** Implemented techniques like grid search and random search for optimizing model performance.

**Model Evaluation:** Assessed models using accuracy, loss, and confusion matrices to ensure reliability.


![Screenshot 2024-10-23 152556](https://github.com/user-attachments/assets/d6d96b93-dd24-4934-b9bb-9251ba4e75f6)

## **Project Structure**

**Data/:** Contains datasets used for training and validation.

**Scripts/:** Jupyter notebooks for each task, showcasing EDA, model development, and evaluation.

**Presentation/:** Final presentation summarizing the project's findings and recommendations for ClimateWins.



## Key Results

**CNN Model for Visual Classification:** Achieved a validation accuracy of 75% when classifying weather conditions from images.

**Random Forest Model:** Improved the accuracy to 73% using hyperparameter tuning techniques like grid search.

**Time-Series Analysis:** Demonstrated the capability to predict weather patterns using RNN, focusing on long-term climate trends.


### Thought Experiments for ClimateWins

**GAN for Simulating Future Weather Scenarios:** Use Generative Adversarial Networks (GANs) to create simulated weather maps, helping to visualize potential climate changes.

**Transfer Learning for Rare Weather Events:** Apply pre-trained models on new data to detect and analyze rare weather patterns, such as extreme heatwaves.

**Ensemble Learning for Enhanced Predictions:** Combine CNN, RNN, and Random Forest outputs to form an ensemble model for more accurate weather predictions.


## **Getting Started**
To run the project locally, follow these steps:⋅⋅


1. **Clone the repository**:

    ```bash
    git clone https://github.com/PooryaBehnamie/ClimateWins.git
    cd ClimateWins
    ```

2. **Install required packages**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run Jupyter notebooks**: Launch Jupyter Lab or Notebook to explore the project tasks:

    ```bash
    jupyter lab
    ```

4. **Data Preparation**: Place the datasets in the `data/` folder as mentioned in each notebook.


## **Usage**
Use the notebooks in `notebooks/` to explore individual machine learning models and their performance.

Run the `presentation/` folder to review the final slides for project recommendations.

Experiment with different model parameters to improve accuracy and predictions.



## **Results & Conclusions**
Machine learning models like CNNs are effective for image-based weather classification, with promising accuracy levels.

Random Forest and other ensemble methods provide robust results for structured data.

Deep learning approaches like RNNs are valuable for time-series predictions, especially for long-term trends.


## Next Steps
* Integrate additional data sources such as satellite imagery and real-time weather updates.

* Apply GANs to generate new weather scenarios based on current patterns.

* Enhance model interpretability using SHAP values to understand the impact of individual features.



## Contributing
Contributions are welcome! Feel free to open a pull request to suggest improvements or report issues.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
* Thanks to CareerFoundry for guidance throughout the project.
* Kaggle for providing weather datasets.
* TensorFlow and scikit-learn libraries for enabling model training and evaluation.
* Special thanks to ChatGPT by OpenAI for support and assistance throughout the project.
