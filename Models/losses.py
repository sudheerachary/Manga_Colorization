from keras import losses

def custom_loss(y_true,y_pred):
    
    cosine = losses.cosine_proximity(y_true,y_pred)
    mle = losses.mean_absolute_error(y_true, y_pred)
    l = (cosine)+mle
    
    return l

def custom_loss_2(y_true,y_pred):

    cosine = losses.cosine_proximity(y_true,y_pred)
    mse = losses.mean_squared_error(y_true, y_pred)
    mle = losses.mean_absolute_error(y_true, y_pred)
    l = (cosine+1)*mse+mle
    
    return l