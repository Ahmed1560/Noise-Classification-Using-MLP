# Noise-Classification-Using-MLP
Based on nature, neural networks are the normal representation I construct of the brain: neurons interconnected to other neurons which makes a network. Basic information transits in a multitude of them before becoming a real item, like “move the hand to take up this pencil”. The functioning of a full neural network is simple: the variables are entered (e.g. an image if a neural network should tell what is in an image) and the output is provided following certain computations (after the first example, a cat picture should return the word "cat"). Now, you should know the normal column-based artificial neural network, such that only neurons from columns n-1 and n+1 may be linked to a neuron in column n-1. Some sorts of networks are using a different design, but for the time being, I will focus on the simplest. 
# How does a neural network learn?
generating and interacting with variables is excellent, but that is not sufficient to teach the entire neural network. A lot of data must be ready for our network. These data contain the predicted neural network inputs and outputs. First, remember that it returns the result when input is supplied to the neural network. In the initial try, the correct output cannot be obtained on its own (excepting chance) and hence each input comes with its label during the learning phase, explaining the neural network output. If the option is right, the parameters will be retained and the next input will be supplied. However, weights will be altered if the results achieved do not match the label. These are the only variables to be modified throughout the learning period. This technique may be thought of as several buttons which every time an input is not properly conjectured into distinct options. A specialized procedure known as "backpropagation" is carried out to discover which weight is preferable to change. I am not going to slow down since I will not utilize this precise procedure for the neural network I design. But it consists of going back to the neural network and inspecting every link so I can see how the weight changes the output. Finally, there is the last parameter to regulate how the neural network learns: the "learning rate." The name states all this, this new value controls how quickly, little by little, or more exactly how the neural network changes weight. 1 is an excellent value in general. All right, I know the fundamentals, let's verify the neural network I am going to develop. This is the first neural network that has ever been developed, dubbed a Perceptron. There are 2 neurons in the column input and 1 neuron in the column in the output. This arrangement offers a simple classification to differentiate between two groups. See a brief example (which doesn't have much of interest except to comprehend), to better understand the potential and limitations.
## KANSAI
Kansai dataset contains 2 files. The First one is a data contains the heartbeat reads of a human body (Person) during time from 09:09:00:718 till 10:10:18:973 as format order (Hour : Minutes : Seconds : Milliseconds). It consists of 3572 records and 7 main columns excluding the time. The features are heartbeat, and this is the most important. Relative Mood Deviation that measure the scale of the person mood between a scale of -1 to 1 (-1 so bad mood and 1 for very good mood). Absolute Psychic Activity (Absolute Deviation) and this feature (Column) contains the absolute deviation of the psychic activity of a person. The Absolute Psychic Activity Rolling Cumulative 10 seconds. Relative Mood Rolling cumulative 10 seconds. The Mood Rolling cumulative 10 seconds header consists of two features. The positive and negative. Those are the feature in the first data. Those reads are having two actions effected on them. The first is the made the person watch two different movies and they take reads before and after watching movies. The second actions are walking (Go and Return) and they take the reads the same as they made in the first actions. This data shows the difference heartbeat, Psychic activity, and relative mood before and after the actions and visualize it to make analysis on this read. 
The second Data contains the reads of like, Interest, Concentration, Calmness, Stress, Noise and two empty columns during a timestamp of date and time feature. The timestamp is from 2021-0324 17:09:00.418 to 2021-0324 18:11:35.160 in formatting of (Year – Month Day Hour : Minute : Seconds . Milliseconds).
# Data Preprocess 
First, read the data using pandas library to import it to the python environment. Put headers name to the data. Then it’s found the there’s a lot of missing data (NaN) in the dataset so it’s need to drop the missing data because it confuses the model in training and it’s a the most reason that make the model reduce accuracy or underfit. The python read the data at it is all objects so its need to convert the data to its suitable type. So, the time is a timestamp type and the date is datetime type and all the other types of data is float or integer depending on its type.
##### Data Analysis
![Heartbeat](https://user-images.githubusercontent.com/96385070/147859028-7221d147-814c-4b6a-9ccf-f4a8ea663186.png)

As it’s shown the heartbeat get affected and have direct correlation by the psychic activity and relative mood because the data has token in milliseconds and this proof that heartbeat immediately change by changing the mood and the psychic activity.

![2nd Movie Mood](https://user-images.githubusercontent.com/96385070/147859049-9912957a-246d-45f7-b40a-b676f5b55015.JPG)

![P_Activity Movie](https://user-images.githubusercontent.com/96385070/147859058-6b30a846-2852-4573-b1b0-fce3e44c6aa0.JPG)

As shown here that the second movie is a Horror movie and the person that watch the movie doesn’t like this kinds of movies so the heartbeat increase and the negativity of mood increase and psychic activity increase that proof the correlation. 
# It’s found that when there’s noise the concentration decrease, and calmness decrease so that proof the noise has direct correlation between concentration and calmness.

# Model

Before training the data it’s found that there’s really bad distribution in noise that I need to classify.

![Noise Distribution](https://user-images.githubusercontent.com/96385070/147859091-ba758ea0-2575-426f-a48f-b4250e42d39b.png)

So, it’s needs to resampling and redistribute the data. It can’t be done by undersampling because there’s very high variance between the negative and positive. So, it needs to be done by oversampling.  

First, I create a model of dense, dropout and change the optimizers, regularization methods and scaling the data and as shown in the next image that the scaling and regularization the data is very important. 

# Before Scaling
![Before scaling](https://user-images.githubusercontent.com/96385070/147859100-79711e58-45e9-4be9-9b3e-08deac40d491.JPG)

# After Scaling
![After scaling](https://user-images.githubusercontent.com/96385070/147859103-99c01812-d695-4d84-a60c-cbc33bbefb5a.JPG)

# To Conclude 
the best accuracy achieved without overfitting or underfitting is 91.2% by using for sure binary cross-entropy because it’s a binary classification problem to the nose (0 or 1) and Adam Optimizer and in the hidden layers (Bottleneck) relu activation function and in the output layer is with sigmoid activation function because it’s the most propriate for classification problems. 
# Accuracy

![Accuracy](https://user-images.githubusercontent.com/96385070/147859113-d88c2d8d-014b-4a2d-8b4a-6e47b795c163.png)

# loss

![Loss](https://user-images.githubusercontent.com/96385070/147859123-2cd35a72-58c4-4875-ae64-a2dd05fe3b94.png)
