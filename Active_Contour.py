import numpy as np
from scipy.interpolate import RectBivariateSpline
from skimage.util import img_as_float
from scipy import signal



def sobel_edge(image):
    img = img_as_float(image)
    kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])/4.0
    ky = (kx.transpose())
    Ix = signal.convolve(img, kx)
    Iy = signal.convolve(img, ky)
    result = np.sqrt(Ix ** 2 + Iy ** 2)
    result /= np.sqrt(2)

    return result[1:-1, 1:-1]

def active_contour(image, snake, alpha=0.01, beta=0.1,
                   w_line=0, w_edge=1, gamma=0.01,
                   max_px_move=1.0,
                   max_iterations=2500, convergence=0.1):

    img = img_as_float(image)

    convergence_order =10

    # Find edges using sobel:
    edge = [sobel_edge(img)]

    # Superimpose intensity and edge images:
    img = w_line * img + w_edge * edge[0]

    # Interpolate for smoothness:
    intp = RectBivariateSpline(np.arange(img.shape[1]),
                               np.arange(img.shape[0]),
                               img.T, kx=2, ky=2, s=0)

    snake_xy = snake[:, ::-1]
    x, y = snake_xy[:, 0].astype(float), snake_xy[:, 1].astype(float)
    n = len(x)
    xsave = np.empty((convergence_order, n))
    ysave = np.empty((convergence_order, n))

    # Build snake shape matrix for Euler equation
    a = np.roll(np.eye(n), -1, axis=0) + \
        np.roll(np.eye(n), -1, axis=1) - \
        2*np.eye(n)  # second order derivative, central difference
    b = np.roll(np.eye(n), -2, axis=0) + \
        np.roll(np.eye(n), -2, axis=1) - \
        4*np.roll(np.eye(n), -1, axis=0) - \
        4*np.roll(np.eye(n), -1, axis=1) + \
        6*np.eye(n)  # fourth order derivative, central difference
    A = -alpha*a + beta*b


    # Only one inversion is needed for implicit spline energy minimization:
    inv = np.linalg.inv(A + gamma*np.eye(n))

    # Explicit time stepping for image energy minimization:
    for i in range(max_iterations):
        fx = intp(x, y, dx=1, grid=False)
        fy = intp(x, y, dy=1, grid=False)
        xn = inv @ (gamma*x + fx)
        yn = inv @ (gamma*y + fy)

        # Movements are capped to max_px_move per iteration:
        dx = max_px_move*np.tanh(xn-x)
        dy = max_px_move*np.tanh(yn-y)
        x += dx
        y += dy

        # Convergence criteria needs to compare to a number of previous
        # configurations since oscillations can occur.
        j = i % (convergence_order+1)
        if j < convergence_order:
            xsave[j, :] = x
            ysave[j, :] = y
        else:
            dist = np.min(np.max(np.abs(xsave-x[None, :]) +
                                 np.abs(ysave-y[None, :]), 1))
            if dist < convergence:
                break

    return np.stack([y, x], axis=1)