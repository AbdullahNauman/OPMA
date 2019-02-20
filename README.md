# OPMA: It's a new way to measure pain. 

The Objective Pain Measurement Algorithm, or OPMA, uses data science techniques and machine learning to identify which human physiological responses are most indicative of pain and uses them to develop a neural network that can accurately quantify pateint pain levels. 

Previous research has shown that many variables in the human body serve as indicators of pain (See Figure 1.0). Due to the complex nature of the human body, many of these are variables are interdependent and alone are not an accurate representation of patient pain. However, leveraging data science techniques and machine learning algorithms can help us determine the individual factors that will be most universally indicative of pain, and least affected by external variables. 	

```
Possible Indicators of Pain: 
    - Heart Rate Variability 
    - Blood Pressure Changes 
    - Electrodermal Activity 
    - Pupil Dialation 
    - Proccesed electroencephalography (EEG) 
    - Processed electromyography (EMG) 
    - Brain cellular activity (PET) 
    - Biochemical analysis (Serum lipid levels) 
Source: Cowen et al. 2015
Figure 1.0
```
## Setting up our data

Using a set patient data for each of these variables, principal component analysis (PCA) can be used to complete feature extraction and determine which variables are most effective. Furthermore, reducing our data to a lower dimension will allow us to develop a more accurate model for prediction, with a decreased chance of overfitting, and allow for an easier clinical application. 
We begin with a matrix $X$ where each row represents a single patient's data and each column represents a single variable. To prevent biased skewing of the data, as the variables are measured in different units, we must normalize the data to a mean of $0$ and a variance of $1$ by recursively applying the formula 

![Normalizing equation](/Equations/normalize.jpg)

to each data point in the existing matrix $X$ to form a standardized matrix *X'*.  This standardized matrix *X'* is composed of z-scores that allows us to compare values across different variables. 

As our variables are inter-dependent, we can better examine this relationship by applying the formula
![Covariance equation](/Equations/covariance_equation.jpg)

to each element of *X'* in order to form its covariance matrix. This symmetric matrix describes the variance of the data and the covariance among individual variables,  the measure of how two variables change with respect to each other. The covariance is positive when variables show similar behavior and negative otherwise. PCA attempts to form a Gaussian linear regression through two variables that determines whether two variables change together or if there is a third variables affecting both.

## Extracting Indicators of Pain

Finally, PCA orthogonally transforms the original data onto a direction that reduces the dimension and maximizes variance. The procedure allows us to maintain the model’s accuracy and further explore these interdependencies (See Figure 1.1). The orthogonal transformation is accomplished by performing eigen decomposition on our covariance matrix. Given the resultant eigenvalues, we can rank the data from lowest to highest. As the lowest values bare the least information about the distribution of the data, they can be rejected as less reliable indicators of pain. The elements with higher eigenvalues that demonstrate a larger variance across our randomized set of pain and non-pain patients will then be selected as better indicators of pain.

![PCA dimensionality reduction](https://static1.squarespace.com/static/5a316dfecf81e0076f50dae2/t/5ac35d702b6a284b3fde6131/1522753187751/PCA.png)

Figure 1.1: PCA dimensionality reduction from ℝ<sup>n</sup> to ℝ<sup>2</sup> based on variance.

In order to objectively quantify pain, we can develop a multilayered neural network comprised of nodes. In the first layer, each node is connected to selected indicators of pain chosen by the PCA. The node takes in the input of the chosen variables and computes an output using a function

![f(x) = Wx + b](/Equations/linear_model.jpg) ,

where *w* represents the weight, *b* represents the bias, and *x* represents the input. The sum of these outputs quantitatively predicts the individual’s pain level.

The values of *w* and b are set to an initial value and adjusted as the model trains on a set of patient data. The pain levels in this data are assigned by a single physician to maintain relative consistency. The neural network then uses a cost function to self adjust the values of *w* and *b* in order to optimize the prediction model. 


![Neural network diagram](https://www.dtreg.com/uploaded/pageimg/MLFNwithWeights.jpg)

Figure 1.2: A graphical representation of a neural network. The *x* values represent the human variables selected by the PCA and *y* represents the patient's predicted pain level. 


The training process consists of running several trials and compares the model’s results to the expected output using a cost function. This cost function is defined as

![cost function](/Equations/cost_function.jpg) , 

where *y* represents the output of a given node in the final layer, and yexpected represents the expected output present in the training dataset. The cost function tells us how close we are to accurately predicting pain. In order to get an accurate representation of how far off we are from accurately modeling the data we can take the average cost function as

![average cost equation](/Equations/avg_cost_functions.jpg) ,

where y<sub>i</sub> represents the output on the *i-th* trial and *n* represents the total number of trials. We can define yn as composed of all outputs y<sub>i</sub> for trials *n*. To develop an optimal model for predicting levels of pain, we want to find the minimum value for the function *C'(y)* of several variables. 

To determine the minimum value for *C'(y)*, we simply need to shift our current random values for **W** and **b** in the direction of greatest descent or *-∇C'(y)*(See Figure 2.3), where *∇C'* is a vector valued function composed of the partial derivatives of *C'(y)* with respect to all elements of **y** - the independent variables selected by the Principal Component Analysis. 


Figure 2.3: A graphical representation of gradient descent using the average cost function in ℝ<sup>3</sup>. The blue represents the minimum. By shifting the weights and biases for our human variables in the direction of the minimum, we can minimize the difference between the predicted and expected pain value. This allows our model to train to a point where it accurately and objectively quantifies a patient's pain levels. 


By repeatedly shifting the values for Wand b in the direction of greatest descent, we train the algorithm to find the optimal function *f(x) = Wx+b* that best minimizes the cost function and trains the algorithm to accurately predict a measure of pain based on human responses we identified using the PCA. 

By repeatedly applying this re-adjustment model we define a prediction model in the form: 
![pain equation](/Equations/pain_equation.jpg) ,

where x represents the collected patient data and Wrepresents the calibrated weight for each of the selected patient variables. 
In order to standardize this data to our expected values between 0-10 , we can configure the final layer of the neural network to model a composition between P(x)and a transformed sigmoid function (x) =101+e-x10-5which has the range 0<(x) <10for all values of x >0. 
Therefore: 
![pain model](/Equations/pain_model.jpg)

This technology will provide an objective measurement of pain that can be calculated using fewer patient variables and is more accurate than the existing system of pain measurement scales.
