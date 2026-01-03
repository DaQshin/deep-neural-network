import numpy as np 

def dictionary_to_vector(parameters):
    theta = []
    L = len(parameters) // 2
    for l in range(1, L + 1):
        w_param = parameters['W' + str(l)].flatten().reshape(-1, 1)
        theta.append(w_param)
        b_param = parameters['b' + str(l)].reshape(-1, 1)  
        theta.append(b_param)

    theta = np.concatenate(theta, axis=0)

    return theta

def gradients_to_vector(gradients):
    dtheta = []
    L = len(gradients) // 2
    for l in range(1, L + 1):
        w_param = gradients['dW' + str(l)].flatten().reshape(-1, 1)
        dtheta.append(w_param)
        b_param = gradients['db' + str(l)].reshape(-1, 1)  
        dtheta.append(b_param)

    dtheta = np.concatenate(dtheta, axis=0)

    return dtheta


def vector_to_dictionary(vector, layers_dims):
    L = len(layers_dims)
    idx = 0
    parameters = {}

    for l in range(1, L):
        w_size = layers_dims[l] * layers_dims[l - 1]
        parameters['W' + str(l)] = vector[idx: idx + w_size].reshape(layers_dims[l - 1], layers_dims[l])
        
        idx += w_size

        b_size = layers_dims[l]
        parameters['b' + str(l)] = vector[idx: idx + b_size].reshape(-1, 1)

        idx += b_size

    return parameters


def gradient_check(X, Y, parameters, gradients, 
                   layer_dims, forward_pass, eps=1e-7, print_msg=False):

    param_values = dictionary_to_vector(parameters)
    grads = gradients_to_vector(gradients)

    num_parameters = param_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    for i in range(num_parameters):

        theta_plus = np.copy(param_values)
        theta_plus[i] += eps
        J_plus[i], _ = forward_pass(X, Y, vector_to_dictionary(theta_plus, layer_dims))


        theta_minus = np.copy(param_values)
        theta_minus[i] += eps
        J_minus[i], _ = forward_pass(X, Y, vector_to_dictionary(theta_minus, layer_dims))

        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * eps)

    
    numerator = np.linalg.norm(grads - gradapprox)
    denominator = np.linalg.norm(grads) + np.linalg.norm(gradapprox)

    difference = numerator / denominator

    if print_msg:
        if difference > 2e-7:
            print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
        else:
            print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

    return difference

