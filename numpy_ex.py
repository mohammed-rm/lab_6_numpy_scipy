from dataclasses import dataclass

import numpy as np


@dataclass
class Matrix:
    shape: tuple
    matrix: np.ndarray = None

    # Question 1
    def dispaly_matrix(self):
        print(self.matrix)

    # Question 1
    def create_matrix_with_random_float_values(self):
        self.matrix = np.random.random(self.shape)
        return self.matrix

    # Question 1
    def display_matrix_properties(self):
        print(f"Shape: {self.matrix.shape}")
        print(f"Size: {self.matrix.size}")
        print(f"Dimension: {self.matrix.ndim}")
        print(f"Data type: {self.matrix.dtype}")
        print(f"Item size: {self.matrix.itemsize}")
        print(f"Total size: {self.matrix.nbytes}")
        self.dispaly_matrix()

    # Question 2
    def create_matrix_with_random_int_values(self, min_value, max_value):
        self.matrix = np.random.randint(min_value, max_value, self.shape)
        return self.matrix

    # Question 2
    def product(self, matrix):
        return self.matrix * matrix.matrix

    # Question 2
    def dot_product(self, matrix):
        return self.matrix.dot(matrix.matrix)

    # Question 2
    def transpose(self):
        return self.matrix.transpose()

    # Question 3
    def determinant(self):
        return np.linalg.det(self.matrix)

    # Question 3
    def inverse(self):
        return np.linalg.inv(self.matrix) if self.determinant() != 0 else None

    # Question 3
    def eigen_values(self):
        return np.linalg.eig(self.matrix)[0]

    # Question 3
    def eigen_vectors(self):
        return np.linalg.eig(self.matrix)[1]

    # Question 3
    @staticmethod
    def solve_linear_equation(linear_system: list):
        matrix = Matrix((3, 3), np.array([lines[:-1] for lines in linear_system]))
        vector = Matrix((3, 1), np.array([lines[-1] for lines in linear_system]))
        return np.linalg.solve(matrix.matrix, vector.matrix) if matrix.determinant() != 0 else None


if __name__ == '__main__':
    # Question 1
    # m = Matrix((4, 3, 2))
    # m.create_matrix_with_random_float_values()
    # m.display_matrix_properties()

    # Question 2
    # m_1 = Matrix((3, 3))
    # m_2 = Matrix((3, 3))
    # m_1.create_matrix_with_random_int_values(0, 8)
    # m_2.create_matrix_with_random_int_values(2, 10)

    # Diplay
    # print(f"Matrix 1:\n {m_1.matrix}\n")
    # print(f"Matrix 2:\n {m_2.matrix}\n")

    # Products
    # print(f"Product *:\n{m_1.product(m_2)}\n")
    # print(f"Dot product:\n{m_1.dot_product(m_2)}\n")

    # Transpose
    # print(f"Transpose Matrix 1:\n{m_1.transpose()}\n")
    # print(f"Transpose Matrix 2:\n{m_2.transpose()}\n")

    # Question 3
    # m_1 = Matrix((3, 3))
    # m_2 = Matrix((3, 3))
    # m_1.create_matrix_with_random_int_values(0, 8)
    # m_2.create_matrix_with_random_int_values(2, 10)
    #
    # # Diplay
    # print(f"Matrix 1:\n {m_1.matrix}\n")
    # print(f"Matrix 2:\n {m_2.matrix}\n")
    #
    # # Determinant
    # print(f"Determinant Matrix 1:\n {m_1.determinant():.4f}\n")
    # print(f"Determinant Matrix 2:\n {m_2.determinant():.4f}\n")
    #
    # # Inverse
    # print(f"Inverse Matrix 1:\n {m_1.inverse()}\n")
    # print(f"Inverse Matrix 2:\n {m_2.inverse()}\n")
    #
    # # Valeurs propres
    # print(f"Eigen values Matrix 1:\n {m_1.eigen_values()}\n")
    # print(f"Eigen values Matrix 2:\n {m_2.eigen_values()}\n")
    #
    # # Vecteurs propres
    # print(f"Eigen vectors Matrix 1:\n {m_1.eigen_vectors()}\n")
    # print(f"Eigen vectors Matrix 2:\n {m_2.eigen_vectors()}\n")

    # Systeme d'équations
    # 2x + 3y + 4z = 1
    # 3x + 2y + 1z = 2
    # 4x + 1y + 2z = 3
    linear_system_with_null_determinant = [[0, 0, 0, 1], [3, 2, 10, 2], [4, 1, -2, 9]]
    linear_system = [[2, 3, 13, 1], [3, 2, 1, 2], [16, 9, 11, 33]]
    print(
        f"Solution to system with a null determinant:\n {Matrix.solve_linear_equation(linear_system_with_null_determinant)}\n")
    print(f"Solution to system:\n"
          f"x = {Matrix.solve_linear_equation(linear_system)[0]}\n"
          f"y = {Matrix.solve_linear_equation(linear_system)[1]}\n"
          f"z = {Matrix.solve_linear_equation(linear_system)[2]}\n")
