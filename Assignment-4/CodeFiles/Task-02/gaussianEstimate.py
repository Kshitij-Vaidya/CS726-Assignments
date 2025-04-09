import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, erf
import numpy.linalg as LA

def branin_hoo(x):
    """
    Calculate the Branin-Hoo function value for given input.
    x : List or array with two elements [x1, x2]
    """
    x1, x2 = x[0], x[1]
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s
    

# Kernel Functions (Students implement)
def rbf_kernel(x1, x2, length_scale=1.0, sigma_f=1.0):
    """Compute the RBF kernel."""
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    squareDist = np.sum(x1**2, axis=1).reshape(-1, 1) + np.sum(x2**2, axis=1).reshape(-1, 1) - 2 * np.dot(x1, x2.T)
    return sigma_f**2 * np.exp(-0.5 * squareDist / (length_scale**2))

def matern_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, nu=1.5):
    """Compute the Matérn kernel (nu=1.5)."""
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    d = np.sqrt(np.sum((x1[:, None, :] - x2[None, :, :])**2, axis=2))
    return sigma_f**2 * (1 + sqrt(3) * d / length_scale) * np.exp(-sqrt(3) * d / length_scale)

def rational_quadratic_kernel(x1, x2, length_scale=1.0, sigma_f=1.0, alpha=1.0):
    """Compute the Rational Quadratic kernel."""
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    d = np.sum((x1[:, None, :] - x2[None, :, :])**2, axis=2)
    return sigma_f**2 * (1 + (d / (2 * alpha * length_scale)))**(-alpha)

def log_marginal_likelihood(x_train, y_train, kernel_func, length_scale, sigma_f, noise=1e-4):
    """Compute the log-marginal likelihood."""
    K = kernel_func(x_train, x_train, length_scale, sigma_f) + noise * np.eye(len(x_train))
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return -np.inf
    alpha = LA.solve(L.T, LA.solve(L, y_train))
    logDet = 2 * np.sum(np.log(np.diag(L)))
    n = len(x_train)
    return -0.5 * np.dot(y_train.T, alpha) - 0.5 * logDet - 0.5 * n * np.log(2 * np.pi)


def optimize_hyperparameters(x_train, y_train, kernel_func, noise=1e-4):
    """Optimize hyperparameters using grid search."""
    bestLL = -np.inf
    bestParams = (1.0, 1.0, noise)
    for length_scale in [0.1, 1.0, 10.0]:
        for sigma_f in [0.1, 1.0, 10.0]:
            for n in [1e-4, 1e-3, 1e-2]:
                LL = log_marginal_likelihood(x_train, y_train, kernel_func, length_scale, sigma_f, noise=n)
                if LL > bestLL:
                    bestLL = LL
                    bestParams = (length_scale, sigma_f, n)
    print(f"Optimised Parameters : Length Scale = {bestParams[0]} Sigma_f = {bestParams[1]} Noise = {bestParams[2]}")
    return bestParams

def gaussian_process_predict(x_train, y_train, x_test, kernel_func, length_scale=1.0, sigma_f=1.0, noise=1e-4):
    """Perform GP prediction."""
    K = kernel_func(x_train, x_train, length_scale, sigma_f) + noise * np.eye(len(x_train))
    Ks = kernel_func(x_train, x_test, length_scale, sigma_f)
    Kss = kernel_func(x_train, x_test, length_scale, sigma_f) + 1e-8 * np.eye(len(x_test))

    L = np.linalg.cholesky(K)
    alpha = LA.solve(L.T, LA.solve(L, y_train))
    # Predictive mean
    yMean = np.dot(Ks.T, alpha)
    # Solve for v : L v = Ks
    v = LA.solve(L, Ks)
    yVar = np.diag(Kss) - np.sum(v**2, axis=0)
    yStd = np.sqrt(np.maximum(yVar, 0))
    return yMean, yStd


def phi(z):
    '''
    Standard Normal PDF
    '''
    return np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)


def Phi(z):
    '''
    Standard Normal CDF using an Approximation
    '''
    return 1 / (1 + np.exp(-1.702 * x))


# Acquisition Functions (Simplified, no erf)
def expected_improvement(mu, sigma, y_best, xi=0.01):
    """Compute Expected Improvement acquisition function."""
    # Approximate Phi(z) = 1 / (1 + exp(-1.702 * z))
    sigma = np.maximum(sigma, 1e-8)
    improvement = y_best - mu * xi
    z = improvement / sigma
    ei = improvement * np.vectorize(Phi)(z) + sigma * phi(z)
    return np.maximum(ei, 0)

def probability_of_improvement(mu, sigma, y_best, xi=0.01):
    """Compute Probability of Improvement acquisition function."""
    # Approximate Phi(z) = 1 / (1 + exp(-1.702 * z))
    sigma = np.maximum(sigma, 1e-8)
    z = (y_best - mu * xi) / sigma
    return np.vectorize(Phi)(z)

def plot_graph(x1_grid, x2_grid, z_values, x_train, title, filename):
    """Create and save a contour plot."""
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(x1_grid, x2_grid, z_values, cmap='viridis', levels=50)
    plt.colorbar(contour)
    plt.scatter(x_train[:, 0], x_train[:, 1], color='red', edgecolor='white', s=50)
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(filename)
    plt.close()

def main():
    """Main function to run GP with kernels, sample sizes, and acquisition functions."""
    np.random.seed(0)
    n_samples_list = [10, 20, 50, 100]
    kernels = {
        'rbf': (rbf_kernel, 'RBF'),
        'matern': (matern_kernel, 'Matern (nu=1.5)'),
        'rational_quadratic': (rational_quadratic_kernel, 'Rational Quadratic')
    }
    acquisition_strategies = {
        'EI': expected_improvement,
        'PI': probability_of_improvement
    }
    
    x1_test = np.linspace(-5, 10, 100)
    x2_test = np.linspace(0, 15, 100)
    x1_grid, x2_grid = np.meshgrid(x1_test, x2_test)
    x_test = np.c_[x1_grid.ravel(), x2_grid.ravel()]
    true_values = np.array([branin_hoo([x1, x2]) for x1, x2 in x_test]).reshape(x1_grid.shape)
    
    for kernel_name, (kernel_func, kernel_label) in kernels.items():
        for n_samples in n_samples_list:
            x_train = np.random.uniform(low=[-5, 0], high=[10, 15], size=(n_samples, 2))
            y_train = np.array([branin_hoo(x) for x in x_train])
            
            print(f"\nKernel: {kernel_label}, n_samples = {n_samples}")
            length_scale, sigma_f, noise = optimize_hyperparameters(x_train, y_train, kernel_func)
            
            for acq_name, acq_func in acquisition_strategies.items():
                x_train_current = x_train.copy()
                y_train_current = y_train.copy()
                
                y_mean, y_std = gaussian_process_predict(x_train_current, y_train_current, x_test, 
                                                        kernel_func, length_scale, sigma_f, noise)
                y_mean_grid = y_mean.reshape(x1_grid.shape)
                y_std_grid = y_std.reshape(x1_grid.shape)
                
                if acq_func is not None:
                    # Hint: Find y_best, apply acq_func, select new point, update training set, recompute GP
                    pass
                
                acq_label = '' if acq_name == 'None' else f', Acq={acq_name}'
                plot_graph(x1_grid, x2_grid, true_values, x_train_current,
                          f'True Branin-Hoo Function (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'true_function_{kernel_name}_n{n_samples}_{acq_name}.png')
                plot_graph(x1_grid, x2_grid, y_mean_grid, x_train_current,
                          f'GP Predicted Mean (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'gp_mean_{kernel_name}_n{n_samples}_{acq_name}.png')
                plot_graph(x1_grid, x2_grid, y_std_grid, x_train_current,
                          f'GP Predicted Std Dev (n={n_samples}, Kernel={kernel_label}{acq_label})',
                          f'gp_std_{kernel_name}_n{n_samples}_{acq_name}.png')

if __name__ == "__main__":
    main()