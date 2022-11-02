from dataclasses import dataclass

import numpy as np


@dataclass
class Matrix:
    shape: tuple
    matrix: np.ndarray = None

    def dispaly_matrix(self):
        print(self.matrix)

    def create_matrix_with_random_float_values(self):
        self.matrix = np.random.random(self.shape)
        return self.matrix

    def create_matrix_with_random_int_values(self, min_value, max_value):
        self.matrix = np.random.randint(min_value, max_value, self.shape)
        return self.matrix

    def product(self, matrix):
        return self.matrix * matrix.matrix

    def dot_product(self, matrix):
        return self.matrix.dot(matrix.matrix)

    def transpose(self):
        return self.matrix.transpose()

    def determinant(self):
        return np.linalg.det(self.matrix)

    def inverse(self):
        return np.linalg.inv(self.matrix)

    def eigen_values(self):
        return np.linalg.eig(self.matrix)

    def display_matrix_properties(self):
        print(f"Shape: {self.matrix.shape}")
        print(f"Size: {self.matrix.size}")
        print(f"Dimension: {self.matrix.ndim}")
        print(f"Data type: {self.matrix.dtype}")
        print(f"Item size: {self.matrix.itemsize}")
        print(f"Total size: {self.matrix.nbytes}")
        self.dispaly_matrix()


if __name__ == '__main__':
    # 1
    m = Matrix((4, 3, 2))
    # m.create_matrix_with_random_values()
    # m.display_matrix_properties()

    # 2
    m_1 = Matrix((3, 3))
    m_2 = Matrix((3, 3))
    m_1.create_matrix_with_random_int_values(0, 8)
    m_2.create_matrix_with_random_int_values(2, 10)
    # Diplay
    m_1.dispaly_matrix()
    m_2.dispaly_matrix()
    # Products
    # print(m_1.product(m_2))
    # print(m_1.dot_product(m_2))
    # Transpose
    print(m_1.transpose())
    print(m_2.transpose())

    # 3