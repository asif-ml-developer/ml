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
        plt.xlabel('x side')
        plt.ylabel('y side')
        plt.legend()
        plt.show()
        
![image](https://github.com/user-attachments/assets/d63ae189-f713-4415-8b6e-f9a42d818525)


https://colab.research.google.com/drive/1wNZx19sUn6H_4uPoWXVoGaM3zLSehVTi#scrollTo=6tW219ht9rh5
