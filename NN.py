from PIL import Image
from json import dump, load
from random import uniform
from typing import Tuple, Any, Union
import pickle


def make_weights(height_of_block: int, width_of_block: int, neurons: int = 3) -> Tuple[
    Tuple[Tuple[Any, ...], ...], Tuple[Tuple[Any, ...], ...]]:
    """function for making matrix of weights. height of block and width of block means same parameters of block.
    neurons is a number of neurons on first layer
    :return tuple of two weight matrix(first and second layer)"""
    weight1 = tuple(tuple(uniform(-1, 1) for _ in range(neurons)) for __ in range(height_of_block * width_of_block * 3))
    weight2 = tuple(tuple(uniform(-1, 1) for _ in range(height_of_block * width_of_block * 3)) for __ in range(neurons))
    return weight1, weight2


def transpose(matrix: list) -> Tuple[Tuple[Any, ...], ...]:
    """get transpose matrix of input matrix
    :return transpose matrix"""
    return tuple(tuple(matrix[j][i] for j in range(len(matrix))) for i in range(len(matrix[0])))


def multiplication(matrix1: list, matrix2: list) -> Tuple[Tuple[Union[int, Any], ...], ...]:
    """get multiplication of two matrix
    :return matrix"""
    return tuple(
        tuple(sum([matrix1[i][k] * matrix2[k][j] for k in range(len(matrix2))]) for j in range(len(matrix2[0]))) for i
        in range(len(matrix1)))


def get_vector(matrix: Tuple[Tuple[list]], height_of_block: int, width_of_block: int) -> Tuple[Tuple[Any, ...], ...]:
    """beat the image into blocks (height)x(width*3)
    :return blocks of image"""
    y = 0
    result = []
    width = len(matrix[0])
    height = len(matrix)
    while y + width_of_block < height:
        x = 0
        while x + height_of_block < width:
            result.append(tuple(matrix[y + i][x + j] for i in range(width_of_block) for j in range(height_of_block)))
            x += height_of_block
        if x + height_of_block >= width:
            result.append(tuple(matrix[y + i][width - height_of_block + j] for i in range(width_of_block) for j in
                                range(height_of_block)))
        y += width_of_block
    if y + width_of_block >= height:
        x = 0
        while x + height_of_block < width:
            result.append(tuple(matrix[height - width_of_block + i][x + j] for i in range(width_of_block) for j in
                                range(height_of_block)))
            x += height_of_block
        if x + height_of_block >= width:
            result.append(tuple(
                matrix[height - width_of_block + i][width - height_of_block + j] for i in range(width_of_block) for j in
                range(height_of_block)))
    return tuple(result)


def convert_to_float(matrix: Tuple[Tuple[Any, ...], ...]) -> Tuple[Tuple[Any, ...], ...]:
    """covert input image colors to numbers between (-1, 1)
    :return changed view of picture colors"""
    return tuple(tuple(color * 2 / 255 - 1 for j in i for color in j) for i in matrix)


def convert_to_color(matrix: Tuple[Tuple[Any, ...], ...]) -> Tuple[Tuple[Any, ...], ...]:
    """convert neural net output into image format
    :return neural net output converted in picture view"""
    return tuple(tuple(int((j + 1) / 2 * 255) for j in i) for i in matrix)


def convert_into_tuples(vector: Tuple[Tuple[Any, ...], ...]) -> Tuple[Tuple[Tuple[Any, ...], ...], ...]:
    """convert neural net output into image format
    :return neural net output converted in picture view"""
    return tuple(
        tuple(tuple((vector[i][j * 3 + color] for color in range(3))) for j in range(len(vector[i]) // 3)) for i in
        range(len(vector)))


def convert_into_pixels(matrix, height, width, n, m):
    """convert neural net output into pixels row for image:
    :return row of pixels"""
    buffer = []
    for i in range(height // m):
        for l in range(m):
            string = []
            for j in range(width // n):
                for k in range(n):
                    string.append(matrix[i * width // m + j][k + l * m])
            buffer.append(string)
    result = []
    for i in buffer:
        result += i
    return tuple(result)


def alpha(matrix: Tuple[Tuple[Any, ...], ...]) -> float:
    """function for finding adaptive step of learning
    :return adaptive step of learning"""
    return 1 / sum([j ** 2 for i in matrix for j in i])


def errors(matrix: Tuple[Tuple[Any, ...], ...]) -> float:
    """function for calculate the sum of errors in a picture
    :return number of errors"""
    return sum([j ** 2 for i in matrix for j in i])


def subtraction_of_matrix(matrix1: Tuple[Tuple[Any, ...], ...], matrix2: Tuple[Tuple[Any, ...], ...]) -> Tuple[
    Tuple[Any, ...], ...]:
    """function for substracting between two matrix
    :return result matrix"""
    result = [[0 for j in range(len(matrix1[0]))] for i in range(len(matrix1))]
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            result[i][j] = matrix1[i][j] - matrix2[i][j]
    result = tuple(tuple(j for j in i) for i in result)
    return result


def multiply_matrix_on_number(number, matrix):
    """function for multiplying number on matrix
    :return result matrix"""
    return tuple(tuple(matrix[i][j] * number for j in range(len(matrix[i]))) for i in range(len(matrix)))


def learning(height, width, n, m, W1, W2, image, E):
    pixels = tuple(image.getdata())
    matrix = tuple(tuple(pixels[i * width + j] for j in range(width)) for i in range(height))
    if n < width or m < height:
        # row of blocks
        squares = get_vector(matrix, n, m)
        # X matrix (input)
        X = convert_to_float(squares)
        # Y matrix (X * W1)
        Y = multiplication(X, W1)
        # X' matrix (output) (Y * W2)
        nX = multiplication(Y, W2)
        # delta X matrix (X` - X)    (actual - expected)
        delta_X = subtraction_of_matrix(nX, X)
        # W2(t+1) = W2 - a` * Y^T * delta X
        W2 = subtraction_of_matrix(W2,
                                   multiplication(multiply_matrix_on_number(alpha(Y), transpose(Y)), delta_X))
        # W1(t+1) = W1 - a * X^T * delta X * W2 ^ T
        W1 = subtraction_of_matrix(W1, multiplication(
            multiplication(multiply_matrix_on_number(alpha(Y), transpose(X)), delta_X), transpose(W2)))
        # count of attempts
        counter = 1
        E1 = errors(delta_X)
        # main loop of learning
        while E < E1:
            E2 = E1
            print(counter, errors(delta_X))
            Y = multiplication(X, W1)
            nX = multiplication(Y, W2)
            delta_X = subtraction_of_matrix(nX, X)
            W2 = subtraction_of_matrix(W2,
                                       multiplication(multiply_matrix_on_number(alpha(Y), transpose(Y)),
                                                      delta_X))
            W1 = subtraction_of_matrix(W1, multiplication(
                multiplication(multiply_matrix_on_number(alpha(Y), transpose(X)), delta_X),
                transpose(W2)))
            counter += 1
            E1 = errors(delta_X)

            # strings for  slide show of learning(interactive learning) for watch learning progress
            # pixels = convert_to_color(nX)
            # pixels = convert_into_tuples(pixels)
            # pixels = convert_into_pixels(pixels, height, width, n, m)
            # out_image = Image.new(image.mode, image.size)
            # out_image.putdata(pixels)
            # out_image.show()

            # end of learning if dE < 0.01
            # additional condition of ending learning
            if abs(E2 - E1) < E2 * 0.001:
                break
    else:
        print("doesn't work")
    return nX, Y, X, W1, W2


def saving(nX, Y, height, width, n, m, image, neurons, image_name, W1, W2):
    pixels = convert_to_color(nX)
    pixels = convert_into_tuples(pixels)
    pixels = convert_into_pixels(pixels, height, width, n, m)
    out_image = Image.new(image.mode, image.size)
    out_image.putdata(pixels)
    out_image.show()
    out_image.save(f'resulted pictures/+{image_name}')
    # archieve after learning
    with open(f"archieve/{n}x{m}x{neurons}.json", 'w+') as json_file:
        dump({"n": n, "m": m, "neurons": neurons, "weight1": W1, "weight2": W2}, json_file)
    with open(f"Y/{image_name}.json", 'w+') as json_file:
        dump({"Y": Y, "n": n, "m": m, "neurons": neurons, "weight1": W1, "weight2": W2, "width": width,
              "height": height}, json_file)


def from_archieve():
    E = int(input("error "))  # between c 1 for 256x256 picture
    image_name = input("image name ")
    image = Image.open("images/" + image_name)
    image = image.convert('RGB')
    width, height = image.size
    n = int(input("width of block "))  # width of block
    m = int(input("height of block "))  # height of block
    neurons = int(input("neurons number "))
    with open(f"archieve/{n}x{m}x{neurons}.json", "r") as json_file:
        start_parametrs = load(json_file)
        W1 = tuple(tuple(i) for i in start_parametrs["weight1"])
        W2 = tuple(tuple(i) for i in start_parametrs["weight2"])
    nX, Y, X, W1, W2 = learning(height, width, n, m, W1, W2, image, E)
    saving(nX, Y, height, width, n, m, image, neurons, image_name, W1, W2)


def new_weights():
    E = int(input("error "))  # between 1 for 256x256 picture
    image_name = input("image name ")
    image = Image.open("images/" + image_name)
    image = image.convert('RGB')
    width, height = image.size
    n = int(input("width of block "))  # width of block
    m = int(input("height of block "))  # height of block
    neurons = int(input("neurons number (usually 3) "))
    W1, W2 = make_weights(n, m, neurons)
    nX, Y, X, W1, W2 = learning(height, width, n, m, W1, W2, image, E)
    saving(nX, Y, height, width, n, m, image, neurons, image_name, W1, W2)


def from_compressed_image():
    # E = int(input("error "))  # between 1 for 256x256 picture
    E = int(input("error "))  # between 1 for 256x256 picture
    image_name = input("image name ")
    image = Image.open("images/" + image_name)
    image = image.convert('RGB')
    # width, height = image.size
    with open(f"Y/{image_name}.json", "r") as json_file:
        start_parametrs = load(json_file)
        W1 = tuple(tuple(i) for i in start_parametrs["weight1"])
        W2 = tuple(tuple(i) for i in start_parametrs["weight2"])
        Y = tuple(tuple(i) for i in start_parametrs["Y"])
        m = start_parametrs["n"]
        n = start_parametrs["m"]
        width = start_parametrs["width"]
        height = start_parametrs["height"]
        neurons = start_parametrs["neurons"]
    nX = multiplication(Y, W2)
    nX, Y, X, W1, W2 = learning(height, width, n, m, W1, W2, image, E)
    saving(nX, Y, height, width, n, m, image, neurons, image_name, W1, W2)
