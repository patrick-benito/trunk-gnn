import torch
import numpy as np

class LinearizedModel():
    def __init__(self, model, data_template):
        self.model = model
        self.model.eval()
        self.prediction = None
        self.data_template = data_template
        
        self.dims_out = int(np.prod(self.data_template.x_new.shape))
        self.dims_in_x = int(np.prod(self.data_template.x.shape))
        self.dims_in_u = int(np.prod(self.data_template.u.shape))

    def predict(self, data):
        self.prediction = self.model(data)
        return self.prediction.x_new.reshape(self.dims_out)
    
    def A(self, data):
        data = self.data.clone()
        def f(x):
            data.x = x
            return self.predict(data)

        return torch.autograd.functional.jacobian(f, data.x).reshape(self.dims_out, self.dims_in_x)

    def B(self, data):
        data = self.data.clone()
        print(data.u.shape)
        def f(u):
            data.u = u
            return self.predict(data)
        
        return torch.autograd.functional.jacobian(f, data.u).reshape(self.dims_out, self.dims_in_u)
    
    def __call__(self, x, u):
        self.data = self.data_template.clone()
        self.data.x = x.reshape(self.data_template.x.shape)
        self.data.u = u.reshape(self.data_template.u.shape)
        pred = self.predict(self.data)
        A = self.A(self.data)
        B = self.B(self.data)

        return pred, A, B