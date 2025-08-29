## About Me

Hi, I'm Charan, an AI/ML Bootcamp participant passionate about using technology to solve real-world healthcare problems.

## Motivation

I chose this project to deepen my understanding of machine learning in healthcare and to contribute to early disease detection. Heart disease is a leading cause of death worldwide, and I wanted to explore how AI can help save lives by predicting risks more accurately.

## Learning Outcomes

Through this project, I learned how to:

- Preprocess and analyze real-world health datasets
- Build, train, and optimize multiple ML models
- Evaluate models using key metrics and visualizations
- Document results for a professional portfolio

## Unique Contributions

- Personalized workflow and explanations throughout the notebook
- Compared multiple models and shared insights on their strengths
- Added extra visualizations and reflections to make the project unique

## Demo & Presentation

Presentation: https://gamma.app/docs/Charans-Heart-Disease-Prediction-Toolkit-7vxxkg53iewz660
Demo video will be added soon!

## Portfolio & Contact

- Portfolio: https://charandzinsdev.netlify.app/
- LinkedIn: https://www.linkedin.com/in/sai-charan-77071b281
- GitHub: https://github.com/charan270469

## What is Heart Disease Prediction using Machine Learning?

Today, heart failure diseases affect more people worldwide than other autoimmune conditions. Cardiovascular Diseases (CVDs) affect the heart and obstruct blood flow through the blood vessels. Chronic ailments in CVD include heart disease (heart attack), cerebrovascular diseases (strokes), congestive heart failure, and many more pathologies. Worldwide, CVDs kill around 17 million a year, and death rates due to heart diseases have increased after the COVID-19 pandemic.

CVDs commonly occur when the heart fails to supply blood under normal conditions, leading to high blood pressure, diabetes, chest pain, or other cardiac disorders. Initially, hospitals use regular tools like ECGs to determine and evaluate the stage of the disease. Better technologies like MCGs help in detecting these diseases when they are in an early stage. However, such devices and technologies are not just expensive and unfit for smaller clinics for heart disease diagnosis but also time-consuming and sensitive to tangential causes. Thus, technicians and clinicians would greatly benefit if there was an autonomous system in place that could perform a test for heart disease detection and warn of CVD risk at an early stage. With AI entering every other industry with the advance of computer science, medical science perhaps has the vastest scope of need and use for artificial intelligence solutions -- especially in developing countries.

## Why use Python for Heart Disease Prediction using Machine Learning?

It is well known that the libraries available in Python for data loading, management, and building models, such as Pandas, NumPy, and Scikit-Learn, help build robust data science applications. However, when dealing with medical data in data science, data privacy and protection are important parameters that cannot be ignored. The training data and learning models must comply with HIPAA laws for any AI project to be helpful in the real world.

With Python programming language, cybersecurity professionals can efficiently accomplish these aspects in healthcare projects. Beyond data science, Python is also widely used for decoding and sending packets, network scanning, accessing servers, analyzing web traffic, and port scanning. Python packages are also robust in conducting a series of cybersecurity tasks, such as malware analysis and scanning in real-time.

## Best Machine Learning Approaches

### 1. Random Forest Classifier

The random forest algorithm provides flexibility and robustness for classification tasks using tabular data, which few other standard models can. Given its simplicity and versatility, the random forest classifier is widely used for fraud detection, loan risk prediction, and predicting heart diseases.

With the ensemble learning theorem, the random forest classifier combines results from several decision trees and optimizes training. It aims to utilize different subsets and find the best combinations to increase the dataset’s predictive accuracy. The first step is building, optimizing, mixing, and matching several decision trees. Next, it uses these trees for prediction and ensembles their results to yield the final output prediction.

### 2. K-Nearest Neighbors

As the name says, a k neighbors classifier takes a data point and finds k other data points nearest to it in the vector space. In a supervised fashion, KNN creates clusters of the data samples having the same target value. Whenever a new value needs to be classified, it uses a distance metric to assign it to one of the classes. For heart disease detection, there are only two classes that KNN needs to build. Thus, it is pretty robust and efficient for this task. Euclidean distance is one of the popular distance metrics used by KNN, but there are many more available. However, the metric choice also impacts the classifier's speed For larger datasets, KNN is already relatively slower than its contemporaries.

### 3. Decision Tree classifier

Decision Trees are the individual models that make a random forest after ensembling. Each decision tree classifier uses the dataset's attributes to create a tree. As shown in the image below, the branches end up in the leaves that are made up of target values. Using visual components and an information gain index, the tree identifies the leading features of the labels of each class. Thus, the branches are created that maximize the information gained in each split and lead up to the leaf node of that class. Decision trees are fast and robust for disease prediction if the dataset has powerful features for a simple use-case.

### 4. Support Vector Machines

A Support Vector Machine (SVM) algorithm is a non-probabilistic classifier aiming to generate hyperplanes that divide the data points of two classes in the vector space. For N number of features and M targets, SVM creates M-1 N-dimensional hyperplanes that separate data points of different classes from each other. The image below shows how "support" vectors are calculated such that the margin (or distance) between the vectors of two classes is the most. SVM optimizes this margin metric to find the best hyperplane for all the categories.

Thus, SVMs are popular for disease prediction since they can effectively categorize tabular data into different categories.
![image](https://github.com/shirinshaik/Heart-Disease-Prediction-Project/assets/113626760/16ec1fcc-daff-4f64-ac96-a2372f2da470)

### 5. Artificial Neural Networks

An ANN is perhaps the most popular machine learning model in today's AI landscape, given its wide applications in deep learning in the form of convolutional neural networks. However, a normal ANN comprised of a handful of linear nodes can perform comparable to the best standard ML models. The architecture of a standard ANN is shown in the figure below. As we can see, the hidden layer is the most crucial part of an ANN, and is made up of several linear nodes.
You can wrap several hidden layers in between the input and the output layer to increase the complexity and, thus, the learning ability of the model. Adding more nodes to a layer and more layers to the network would allow the model to learn more non-linear and complex relationships between the categorical variables and input features.

This ability makes the network very capable of capturing relationships between the various biological and personal markers that are already independently affecting the probability of the presence of heart disease.

## Conclusion:

After trying five different machine learning techniques to predict heart disease, it is clear that K-nearest neighbors perform the best on our UCI dataset. However, feature engineering and hyperparameter tuning in other models can also yield comparable results. Moreover, we can extend the dataset, using more samples and more key features such as the one found in the UCI extended and combined dataset and try out these models again. Data availability remains a major problem for modernizing health care systems using artificial intelligence, especially clean, structured datasets for supervised learning.

Nevertheless, after experimentation in Python, we can say that machine learning can significantly help detect the presence and future risk of heart attacks, strokes, coronary artery disease, and other heart diseases.

In addition to our preliminary experimentation, one should also explore other projects that attempt heart disease detection using data science.
