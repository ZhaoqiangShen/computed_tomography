import unittest
import matrixT
from phantoms import circle
import numpy as np
from topolar import topolar



class TestOperators(unittest.TestCase):
    def setUp(self):
        self.size = 8
        
        self.T = matrixT.create_T(self.size)
        self.Dx = matrixT.create_Dx(self.size)
        self.keep_rows = self.size // 2
        self.Mask = matrixT.create_Mask(self.size, keep_rows=self.keep_rows)

    def test_topolar_T(self):
        pass
        #test = circle(self.size, 4, wall_thickness = 1)
        #test_matrix = np.reshape(self.T@test.ravel(), (self.size,self.size))
        #test_topolar_true = topolar(test)[0]


        #self.assertTrue(np.allclose(test_matrix, test_topolar_true))

    def manual_dx(self, A):
        # Caluclate gradient along rows in a 2d image.
        return A - np.roll(A, 1, axis=1)

    def test_dx_constant_lines(self):

        # Dx calculates the gradient along the x axis.

        # Create matrix that is constant along x axis (rows)
        constant_lines = np.zeros((self.size, self.size))
        for i in range(self.size):
            constant_lines[i,:] = i

        # For the above matrix the gradient along x should be zero.
        zero_matrix = np.zeros_like(constant_lines)

        self.assertTrue(np.allclose(zero_matrix, self.manual_dx(constant_lines)))
        
        constant_lines_gradient = self.Dx @ constant_lines.ravel()
        constant_lines_gradient = constant_lines_gradient.reshape(self.size, self.size)
        self.assertTrue(np.allclose(zero_matrix, constant_lines_gradient))

    def test_dx_circle_phantom(self):

        circle_phantom = circle(self.size, 4, wall_thickness = 1)
        manual_dx_calc = self.manual_dx(circle_phantom)
        dx_from_matrix_op = (self.Dx @ circle_phantom.ravel()).reshape(self.size, self.size)

        self.assertTrue(np.allclose(manual_dx_calc, dx_from_matrix_op))


    def test_topolar_dx(self):

        circle_phantom = circle(self.size, 4, wall_thickness = 1)
        circle_polar = topolar(circle_phantom)[0]
        manual_circle_polar_dx = self.manual_dx(circle_polar)

        # Same calculcation but now with linear operators
        operator_circle_polar = self.T @ circle_phantom.ravel()
        operator_circle_polar_dx = self.Dx @ operator_circle_polar
        operator_circle_polar_dx = operator_circle_polar_dx.reshape(self.size, self.size)

        self.assertTrue(np.allclose(manual_circle_polar_dx, operator_circle_polar_dx))


    def test_mask(self):

        image = np.ones((self.size, self.size))

        masked_image = np.zeros_like(image)
        masked_image[0:self.keep_rows] = image[0:self.keep_rows]

        masked_image_by_operator = (self.Mask @ image.ravel()).reshape((self.size, self.size))

        print(image)
        print(masked_image)
        print(masked_image_by_operator)
        self.assertTrue(np.allclose(masked_image_by_operator, masked_image))
        
        

if __name__ == '__main__':
    unittest.main()
