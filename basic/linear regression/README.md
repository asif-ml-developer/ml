# Introduction
Linear Regression is a statistical method used to model and analyze the relationship between a dependent variable (target) and one or more independent variables (features). The goal is to find the best-fitting line through the data points.

![image](https://github.com/user-attachments/assets/4a739357-e520-4d15-90e9-1b0b45cb6cda)
https://excalidraw.com/#json=-q1pWlBdOc_SF8opXNBt0,NW_bgUwVzgUNtfKyQaynww


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
