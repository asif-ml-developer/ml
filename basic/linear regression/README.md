# Introduction
Linear Regression is a statistical method used to model and analyze the relationship between a dependent variable (target) and one or more independent variables (features). The goal is to find the best-fitting line through the data points.

![image](https://github.com/user-attachments/assets/4a739357-e520-4d15-90e9-1b0b45cb6cda)
https://excalidraw.com/#json=-q1pWlBdOc_SF8opXNBt0,NW_bgUwVzgUNtfKyQaynww

![image](https://github.com/user-attachments/assets/0262ea95-9173-4209-a78c-52b81487af04)
https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html



###  Example in Python

Here’s a simple example using Python and the scikit-learn library:

                import numpy as np
                import matplotlib.pyplot as plt
                from sklearn.linear_model import LinearRegression
                
                # Sample data
                x = np.array([[1], [2], [3], [4], [5]])  # Feature
                y = np.array([1, 2, 2.5, 4, 5])            # Target
                
                # Create and fit the model
                model = LinearRegression()
                model.fit(x, y)
                
                # Predict
                x_pred = np.array([[6]])
                y_pred = model.predict(x_pred)
                
                # Plot
                plt.scatter(x, y, color='blue', label='Data Points')
                plt.plot(x, model.predict(x), color='red', label='Fitted Line')
                plt.scatter(x_pred, y_pred, color='green', label='Prediction for x=6')
                plt.xlabel('x (independent variable)')
                plt.ylabel('y (dependent variable)')
                plt.legend()
                plt.show()

        
![image](https://github.com/user-attachments/assets/adb3841f-4eba-48ba-95ce-215a9d52530e)



https://colab.research.google.com/drive/1wNZx19sUn6H_4uPoWXVoGaM3zLSehVTi#scrollTo=6tW219ht9rh5


## Formula

![image](https://github.com/user-attachments/assets/f3f78179-f8a0-4a65-9bb3-332c086fa05d)

![image](https://github.com/user-attachments/assets/c27db33d-3a44-44af-ba48-88f682a108b4)

![image](https://github.com/user-attachments/assets/7b7c5a13-89c8-4d02-a4ed-81719d6e8ec5)

![image](https://github.com/user-attachments/assets/59fe4dbf-08d5-412e-8dfa-dbb277e53f23)


## Simple Linear Regression full cycle of codes (prediction, Cost, Gradient, Training functions)

![image](https://github.com/user-attachments/assets/9cb4c673-fd48-4c46-8148-4b548c9bb29d)


### Prediction Function:
Calculates the predicted sales using the formula Sales = Weight × Radio + Bias
and returns the result.

      def predict_sales(radio, weight, bias):
          return weight * radio + bias
          
### Cost Function: Mean Squared Error (MSE): 

    def cost_function(radio, sales, weight, bias):
        companies = len(radio)
        total_error = 0.0
        for i in range(companies):
            total_error += (sales[i] - (weight * radio[i] + bias)) ** 2
        return total_error / companies
        
### Gradient Descent: Adjust weights and bias to minimize the MSE:

        def update_weights(radio, sales, weight, bias, learning_rate):
            weight_deriv = 0
            bias_deriv = 0
            companies = len(radio)
        
            for i in range(companies):
                weight_deriv += -2 * radio[i] * (sales[i] - (weight * radio[i] + bias))
                bias_deriv += -2 * (sales[i] - (weight * radio[i] + bias))
        
            weight -= (weight_deriv / companies) * learning_rate
            bias -= (bias_deriv / companies) * learning_rate
        
            return weight, bias

### Training: Iteratively update weights and bias to minimize cost.

          def train(radio, sales, weight, bias, learning_rate, iters):
              cost_history = []
          
              for i in range(iters):
                  weight, bias = update_weights(radio, sales, weight, bias, learning_rate)
                  cost = cost_function(radio, sales, weight, bias)
                  cost_history.append(cost)
          
                  if i % 10 == 0:
                      print(f"iter={i}    weight={weight:.2f}    bias={bias:.4f}    cost={cost:.2f}")
          
              return weight, bias, cost_history

### Simple Regression Theory with examples 
https://colab.research.google.com/drive/1wNZx19sUn6H_4uPoWXVoGaM3zLSehVTi#scrollTo=Kz2w0RAfmMKb

### Simple Regression Codes with out put
https://colab.research.google.com/drive/1_bsDyyUzkl_qBe4Pf9abBIFzP8MsBbWt#scrollTo=eJDiEr1QjrcR

![image](https://github.com/user-attachments/assets/b7d26c13-7d36-4bd0-8f6b-aed185847568)




