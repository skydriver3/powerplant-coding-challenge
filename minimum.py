import numpy as np 


def allowed_domain(power_M, load) : 
    '''
    Returns the different combinations of powerplants that can satisfy the demand. 

    Inputs : 
        - power_M : a numpy array of shape (number of powerplants, 2) where the first column is the pmin and the second is the pmax 
        - load : int, the load the powerplants combined should meet in power. 

    Output : 
    numpy array of shape (valid combinations, number of plants) filled with 1 and 0. 
    1 means the powerplant is used, 0 means it is not used. 
    '''


    def create_mask() : 
        '''
        Returns all the combination of powerplants in the form of a matrix.

        Output : 
        numpy array of shape (all possible combinations, number of plants)
        '''        

        l = power_M.shape[0] 
        binary_list = []
        for i in range((2 ** l) - 1 ): 
            # Use the binary representation of numbers to generate the masks 
            binary_short = np.array( [int(c) for c in list(bin(i)[2:])] )
            binary_list.append(np.pad(binary_short, (l - binary_short.size, 0), "constant", constant_values = 0 )) 

        return np.vstack(binary_list)

    # vectors of maximum and minimum power output for the different powerplants
    pmax_v = power_M[:, 1] 
    pmin_v = power_M[:, 0] 

    mask = create_mask() 

    # maximum and minimum power that this combination of powerplant can produce
    max_power = mask @ pmax_v 
    min_power = mask @ pmin_v 

    # Only keep the combination of powerplants that will be able to meet the demands 
    valid_mask = mask[(max_power >= load) * (min_power <= load), : ] 
    

    return valid_mask




def find_minimum(mask, power_M, cost_rates, load) : 
    ''' 
    Use gradient descent along the hyperplane created by the constraint given by the load 
    on an initial point of this hyperplane within the bound of the domain given by the 
    max and min power that each powerplant can output, to find the point in the space 
    of MWh produced by each powerplant which minimize the cost.

    Inputs : 
        - mask : numpy array of shape (number of powerplants) filled with 1 and 0, where 1 means the powerplant is active and 0 means it's inactive. 
        - power_M : a numpy array of shape (number of powerplants, 2) where the first column is the pmin and the second is the pmax 
        - cost_rates : a numpy array of shape (number of powerplants) filled with the coefficient of euros per MWh produced for each powerplant. 
        - load : int, the load the powerplants combined should meet in power. 

    Output : 
    a numpy array of shape (number of powerplants) filled with the values of the power each 
    powerplant should produced to minimize the cost.
    ''' 

    # The gradient is the opposite of the cost rates since the cost is a linear function of 
    # each of the powers. We take the opposite since we want to minimize.
    gradient = - cost_rates 

    # Vectors with the max, min power produced by each powerplant. If the powerplant is not
    # used in this combination then its max, min power is set to 0. 
    max_v = power_M[:, 1] * mask 
    min_v = power_M[:, 0] * mask 

    diff_v = (max_v - min_v)

    # The power produced by the wind turbine in this combination of powerplants. 
    wind_turbine_output = max_v @ (diff_v == 0)

    updated_load = load - wind_turbine_output

    # Now that the load takes into account that the turbine produces energy, 
    # we don't need to take them into account in the activated powerplants.
    mask *= (diff_v != 0) 

    # The gradient should not affect the turbine, since their power is set. 
    gradient *= mask  

    # Normalize the gradient
    gradient /= np.linalg.norm(gradient) 

    # Normalize the normal vector of the load hyperplane
    norm_mask = (mask / np.linalg.norm(mask))

    # Get rid of the part of the gradient that's not in the hyperplane since 
    # the minimum point has to be in it. 
    gradient -= ((norm_mask @ gradient) * norm_mask) 

    # Get the initial guess by taking the intersection between the load hyperplane and 
    # the line going from the point with all the minimum power and the point with all maximum power. 
    initial_guess_x =  (updated_load - (min_v @ mask)) / (diff_v @ mask)

    initial_guess = min_v + (diff_v * initial_guess_x)

    guess_load = initial_guess @ mask 
    if guess_load - updated_load  > 1 : 
        print("error") 

    # See how far in the direction of the gradient we can go until we reach a power limit. 
    x_min = (min_v - initial_guess) / gradient

    x_max = (max_v - initial_guess) / gradient

    np.nan_to_num(x_min,copy=False, nan = 0 )
    np.nan_to_num(x_max,copy=False, nan = 0)

    pos_x = (x_min * (x_min > 0)) + (x_max * (x_max > 0))
    x = np.amin(pos_x[pos_x > 0])  

    # Update the initial guess with the gradient until we max/min out one the parameters.
    minimum = initial_guess + (gradient * x)  

    minimum_load = minimum @ mask 
    if minimum_load - updated_load > 1: 
        print("error")
    return minimum


def find_global_minimum(power_M, cost_rates, load) : 
    '''
    Find the power each powerplant should produce to minimize the cost. 

    Inputs : 
        - power_M : a numpy array of shape (number of powerplants, 2) where the first column is the pmin and the second is the pmax 
        - cost_rates : a numpy array of shape (number of powerplants) filled with the coefficient of euros per MWh produced for each powerplant. 
        - load : int, the load the powerplants combined should meet in power. 

    Output : 
    numpy array of shape (number of powerplants) filled with the powers each powerplants should produce 
    to mimize the cost. 
    '''
    mask_M = allowed_domain(power_M, load)
    minimums = np.apply_along_axis(lambda m : find_minimum(m, power_M, cost_rates, load), 1, mask_M)
    costs = minimums @ cost_rates

    return minimums[np.argmin(costs)]

    