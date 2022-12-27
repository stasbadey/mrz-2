import numpy as np


class Hopfield:

    def __init__(self, images, test_image, view="short"):

        self.print_option = view
        self.set_print_options()

        self.x_vectors = []
        self.xt_multiply_x = []
        self.w = []
        self.zeroed_w = []
        self.y_vector = []
        self.w_multiply_y = []
        self.tanh_w_multiply_y = []
        self.recognized_image = 0

        self.create_x_vectors(images)
        self.calculate_xt_multiply_x()
        self.calculate_w()
        self.zero_out_x()
        self.create_y_vector(test_image)
        self.calculate_zeroed_w_multiply_y()
        self.recognize()

    def set_print_options(self):
        if self.print_option == "full":
            np.set_printoptions(threshold=np.inf)

    def create_x_vectors(self, vectors):
        for vector in vectors:
            numpy_array = np.array(vector)
            self.x_vectors.append(numpy_array)

    def calculate_xt_multiply_x(self):
        for numpy_array in self.x_vectors:
            xt = numpy_array.reshape(numpy_array.size, 1)
            xt_multiply_x = xt * numpy_array
            self.xt_multiply_x.append(xt_multiply_x)

    def calculate_w(self):
        w = 0
        for matrix in self.xt_multiply_x:
            w = w + matrix
        self.w = w

    def zero_out_x(self):
        self.zeroed_w = self.w * (
                np.ones(self.x_vectors[0].size, int) - np.identity(self.x_vectors[0].size, int)
        )

    def create_y_vector(self, vector):
        numpy_array = np.array(vector)
        self.y_vector.append(numpy_array)

    def calculate_zeroed_w_multiply_y(self):
        for numpy_array in self.y_vector:
            y = numpy_array.reshape(numpy_array.size, 1)
            zeroed_w_multiply_y = np.matmul(self.zeroed_w, y)
            self.w_multiply_y.append(zeroed_w_multiply_y)
            tanh_zeroed_w_multiply_y = np.sign(zeroed_w_multiply_y)
            self.tanh_w_multiply_y.append(tanh_zeroed_w_multiply_y)
            temporary_variable = np.zeros(1)

            while True:
                zeroed_w_multiply_y = np.matmul(self.zeroed_w, tanh_zeroed_w_multiply_y)
                tanh_zeroed_w_multiply_y = np.sign(zeroed_w_multiply_y)
                requirement = (
                        np.around(temporary_variable, 4) == np.around(tanh_zeroed_w_multiply_y, 4)
                )
                counter = 0
                for element in requirement:
                    if element == [False]:
                        counter = counter + 1

                if counter == 0:
                    break

                temporary_variable = tanh_zeroed_w_multiply_y
                self.w_multiply_y.append(zeroed_w_multiply_y)
                self.tanh_w_multiply_y.append(tanh_zeroed_w_multiply_y)
                print(tanh_zeroed_w_multiply_y)

    def recognize(self):
        recognized_image = np.sign(self.tanh_w_multiply_y[-1])
        recognized_image = recognized_image.reshape(1, recognized_image.size)[0]
        recognized_image = recognized_image.astype(int)
        count = 1

        for numpy_array in self.x_vectors:
            numpy_array = np.sign(numpy_array)
            numpy_array = np.around(numpy_array, 1)
            numpy_array = numpy_array.astype(int)

            temporary_counter_1 = 0
            temporary_counter_2 = 0
            negative_numpy_array = numpy_array * -1

            for i in range(recognized_image.size):

                if recognized_image[i] == numpy_array[i]:
                    temporary_counter_1 = temporary_counter_1 + 1
                    if temporary_counter_1 >= recognized_image.size - 1:
                        self.recognized_image = count
                        break

                if recognized_image[i] == negative_numpy_array[i]:
                    temporary_counter_2 = temporary_counter_2 + 1
                    if temporary_counter_2 >= recognized_image.size - 1:
                        self.recognized_image = - count
                        break

            count = count + 1



    def print_result(self):
        print(f"Итераций выполнено: {len(self.tanh_w_multiply_y) + 1}")
        print(f"Последняя итерация tanh(W * y): ")
        for element in self.tanh_w_multiply_y[-1]:
            print(" ", element)
        if self.recognized_image > 0:
            print(f"Тестовый образ распознан как образ №{self.recognized_image}")
        else:
            print(f"Тестовый образ не распознан")


def main():
    net = Hopfield(
        images=[[  # Образы
            -1, 1, 1, 1, -1,
            1, -1, -1, -1, 1,
            1, -1, -1, -1, 1,
            1, -1, -1, -1, 1,
            -1, 1, 1, 1, -1
        ], [
            -1, -1, 1, -1, -1,
            -1, -1, 1, -1, -1,
            -1, -1, 1, -1, -1,
            -1, -1, 1, -1, -1,
            -1, -1, 1, -1, -1
        ], [
            1, 1, 1, 1, 1,
            -1, -1, -1, -1, 1,
            1, 1, 1, 1, 1,
            1, -1, -1, -1, -1,
            1, 1, 1, 1, 1
        ]],
        test_image=[
            -1, -1, 1, -1, -1,
            -1, 1, 1, -1, -1,
            -1, -1, 1, -1, -1,
            -1, -1, -1, -1, -1,
            -1, -1, 1, -1, -1
        ]
    )

    net.print_result()


if __name__ == "__main__":
    main()
