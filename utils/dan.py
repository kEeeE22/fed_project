# import torch
# import numpy
# from scipy.stats import chi2

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def pinverse(difference, num_random_features):
#     num_samples, _ = difference.shape
#     sigma = torch.cov(difference.T)
#     mu = torch.mean(difference, 0)
#     if num_random_features == 1:
#         stat = float(num_samples * torch.pow(mu, 2)) / float(sigma)
#     else:
#         sigma = torch.pinverse(sigma)
#         right_side = torch.matmul(mu, torch.matmul(sigma, mu.T))
#         stat = num_samples * right_side
#     return chi2.sf(stat.detach().cpu(), num_random_features)

# def inverse(difference, num_random_features):
#     num_samples, _ = difference.shape
#     sigma = torch.cov(difference.T)
#     mu = torch.mean(difference, 0)
#     if num_random_features == 1:
#         stat = float(num_samples * torch.pow(mu, 2)) / float(sigma)
#     else:
#         print(sigma.shape)
#         print(sigma)
#         sigma = torch.inverse(sigma)
#         right_side = torch.matmul(mu, torch.matmul(sigma, mu.T))
#         stat = num_samples * right_side
#     return chi2.sf(stat.detach().cpu(), num_random_features)

# def unnorm(difference, num_random_features):
#     num_samples, _ = difference.shape
#     sigma = torch.cov(difference.T)
#     mu = torch.mean(difference, 0)
#     if num_random_features == 1:
#         stat = float(num_samples * torch.pow(mu, 2)) / float(sigma)
#     else:
#         right_side = torch.matmul(mu, mu.T)
#         stat = num_samples * right_side
#     return chi2.sf(stat.detach().cpu(), num_random_features)

# def smooth(data):
#     w = torch.linalg.norm(data, dim=1)
#     w = torch.exp(-w ** 2 / 2.0)
#     return w[:, numpy.newaxis]

# def smooth_cf(data, w, random_frequencies):
#     n, _ = data.shape
#     _, d = random_frequencies.shape
#     mat = torch.matmul(data, random_frequencies)
#     arr = torch.cat((torch.sin(mat) * w, torch.cos(mat) * w), dim=1)
#     n1, d1 = arr.shape
#     assert n1 == n and d1 == 2 * d and w.shape == (n, 1)
#     return arr

# def smooth_difference(random_frequencies, X, Y):
#     x_smooth = smooth(X)
#     y_smooth = smooth(Y)
#     characteristic_function_x = smooth_cf(X, x_smooth, random_frequencies)
#     characteristic_function_y = smooth_cf(Y, y_smooth, random_frequencies)
#     return characteristic_function_x - characteristic_function_y

# class MeanEmbeddingTest:

#     def __init__(self, data_x, data_y, scale, number_of_random_frequencies, method, device=DEVICE):
#         self.device = device
#         self.data_x = scale * data_x.to(device)
#         self.data_y = scale * data_y.to(device)
#         self.number_of_frequencies = number_of_random_frequencies
#         self.scale = scale
#         self.method = method

#     def get_estimate(self, data, point):
#         z = data - self.scale * point
#         z2 = torch.norm(z, p=2, dim=1)**2
#         return torch.exp(-z2/2.0)

#     def get_difference(self, point):
#         return self.get_estimate(self.data_x, point) - self.get_estimate(self.data_y, point)

#     def vector_of_differences(self, dim):
#         points = torch.tensor(numpy.random.randn(
#             self.number_of_frequencies, dim)).to(self.device)
#         a = [self.get_difference(point) for point in points]
#         return torch.stack(a).T

#     def compute_pvalue(self):

#         _, dimension = self.data_x.size()
#         obs = self.vector_of_differences(dimension)
#         if self.method == "unnorm":
#             return unnorm(obs, self.number_of_frequencies)
#         else:
#             return pinverse(obs, self.number_of_frequencies)
        
# class SmoothCFTest:

#     def _gen_random(self, dimension):
#         return torch.tensor(numpy.random.randn(dimension, self.num_random_features).astype(numpy.float32)).to(self.device)

#     def __init__(self, data_x, data_y, scale, num_random_features, method, device=DEVICE):
#         self.device = device
#         self.method = method
#         self.data_x = scale*data_x.to(self.device)
#         self.data_y = scale*data_y.to(self.device)
#         self.num_random_features = num_random_features

#         _, dimension_x = numpy.shape(self.data_x)
#         _, dimension_y = numpy.shape(self.data_y)
#         assert dimension_x == dimension_y
#         self.random_frequencies = self._gen_random(dimension_x)

#     def compute_pvalue(self):
#         difference = smooth_difference(
#             self.random_frequencies, self.data_x, self.data_y)
#         if self.method == "unnorm":
#             return unnorm(difference, self.num_random_features)
#         return inverse(difference, self.num_random_features)