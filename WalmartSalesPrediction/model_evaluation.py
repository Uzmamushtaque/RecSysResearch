from sklearn.metrics import mean_absolute_error

# Function to evaluate model by using accuracy measures

def evaluate_model(y_test, y_pred, method):
    if method =='mae':
        score= mean_absolute_error(y_test, y_pred)
    else: 
        print("Only available acuuracy measures is mae.")
    return score
