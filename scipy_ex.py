import matplotlib.pyplot as plt
import scipy as sp
from PIL import Image


# Question 4
def interpolate(x, y, x_new):
    f = sp.interpolate.interp1d(x, y)
    f2 = sp.interpolate.interp1d(x, y, kind='cubic')
    plt.plot(x, y, 'o', x_new, f(x_new), '-', x_new, f2(x_new), '--')
    plt.legend(['data points', 'linear interpolation', 'cubic interpolation'], loc='best')
    plt.show()


# Question 5
def read_original_image():
    with Image.open('sunset.jpg') as image:
        image.show()


# Question 5
def resize_image():
    with Image.open('sunset.jpg') as img:
        new_resized_image = img.resize((int(img.width / 2), int(img.height / 2)))
        new_resized_image.save('sunset_resized.jpg')
        new_resized_image.show()


# Question 6
def other_scipy_functionnalities():
    # TODO
    x = sp.linspace(0, 2 * sp.pi, 100)
    plt.plot(x, sp.sin(x), 'r', x, sp.cos(x), 'b')
    plt.show()


if __name__ == '__main__':
    # Question 1
    # x = np.linspace(0, 10, num=15, endpoint=True)
    # y = np.cos(-x**2/9.0)
    # xnew = np.linspace(0, 10, num=40, endpoint=True)
    # interpolate(x, y, xnew)

    # Question 2
    # read_original_image()
    resize_image()
