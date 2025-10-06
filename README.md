# MNIST_and_Fashion-MNIST_Image-Classification
initial computer vision project
pre_processing_data : This function in train.py performs the following tasks:-
Changes the shape of input data and ground truth.
Converts the output data into one hot vector form.
Creates a validation dataset.
Normalizes data.

data_augmentation : This function in train.py adds a gaussian noise to the train data.
mean of gaussian random noise:1
variance :0.01

Activation functions and its derivatives :

a) sigmoid_function = 1/((1+exp(-z)) sigmoid_function_dash : This function is the derivative of sigmoid_function = sigmoid_function(z)x(1-sigmoid_function(z))

b) relu_function(z) = max(0,z)
Its derivative is given as relu_function_dash(z)

c) identity_function(z) = z
Its derivative is given by identity_function_dash(z)

d) tanh_function(z) : This denotes the tanh activation function
Its derivative is given by tanh_function_dash(z)

e) softmax_function(z) : Since the given problem is a multi-class classification(with 10 labels), we use softmax function for the output layer.
Its derivative is given by softmax_function_dash(z)

Mostly, these activation functions are used in forward propagation and its derivative functions are used in back propagation.

Loss functions :

a) cross_entropy_function : For multi-label problems, cross entropy function is the recommended loss function, which gives a higher penalty in case of misclassifications and therefore, minimizes better.

b) mean_squared_error_function : Here, Mean squared error loss is used to compare the results obtained with cross entropy loss.

Weight initialization :

a) random_initialization : Here , weights are initialized randomly as a uniform distribution between limits -1 and 1 where biases are either initialized to zero or a positive value according to the sweeps.

b) xavier_initialization : Here, weights are initialized as a uniform distribution between the specified limits and bias to zero.

forward_propagation : Forward propagation is used to find the predicted value of output and cost function by going forward,starting from input layer until the output layer.We calculate the pre-activation and activation values and returns the latter after each layer. The input parameters taken are input data set,weights and bias value,and activation function to be used where the default is set as sigmoid function. Softmax function is used to find the values at the output layer. Here,z is the linear part and is the non-linear part(activation function) of a neuron.
z_l = w_l * a_l-1 + b_l, where z_l is the pre-activation part
a_l = g(z_l), where a_l is the activation part

back_propagation : Back propagation is used to find the derivatives of each weights and biases at each layers by starting from the output layer and travelling backwards.We find the derivatives wrto ouput layer,wrto hidden layer and eventuallywrto weights and biases.
dw=dJ/dw,db=dJ/db,dz=dJ/dz

Gradient_descent_variations :

a) gradient_descent : This function is used to implement vanilla gradient descent and stochastic gradient descent
b) momentum_gradient_descent : This function written is used to Momentum gradient descent and Nesterov accelarated gradient descent
c) adam: This function is used to implement Adam(Adaptive moments)
d) nadam : This function is used to implement Nesterov Adam
e) rms_prop : This function is used to implement RMS-Prop Algorithm

accuracy : Function to find accuracy of given data

plot_error : Function used to plot train and validation error

plot_accuracy : Function used to plot train and validation accuracy

predict : Function used to predict output value of test data and find the test accuracy and test error in feed_forward_NN

neural_network : This is the main function where the main function call is performed and also the sweep.

HYPERPARAMETERS :
a. learning_rate(Learning rate)
b. lambd(Regularization parameter)
c. activation(Activation function)
d. loss(Loss function)
e. epsilon(used in gradient_descent algorithms to avoid zero division error)
f. momentum(Momentum used in Momentum gradient descent and NAG)
g. beta(used in rms_prop)
h. beta1(used in adam and nadam)
i. beta2(used in adam and nadam)
j. wandb_project(project_name in wandb)
k. neurons(Number of neurons in hidden layers)
l. num_hidden_layers(number of hidden layers)
m. batch_size(step size in gradient descent)
n. epochs(Number of epochs) o. optimizer(gradient descent algorithm used)

OUTCOMES:-
