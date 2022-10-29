from NN import *
from json import dump, load

if __name__ == '__main__':
    while True:
        match = int(input("0 - from archieve, 1 - new weights, 2 - from compressed image, 3 - exit"))
        if match == 0:
            E = int(input("error "))  # between 1 for 256x256 picture
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
            nX, Y = learning(height, width, n, m, W1, W2, image, E)
            saving(nX, Y, height, width, n, m, image, neurons, image_name, W1, W2)
        elif match == 1:
            E = int(input("error "))  # between 1 for 256x256 picture
            image_name = input("image name ")
            image = Image.open("images/" + image_name)
            image = image.convert('RGB')
            width, height = image.size
            n = int(input("width of block "))  # width of block
            m = int(input("height of block "))  # height of block
            neurons = int(input("neurons number (usually 3) "))
            W1, W2 = make_weights(n, m, neurons)
            nX, Y = learning(height, width, n, m, W1, W2, image, E)
            saving(nX, Y, height, width, n, m, image, neurons, image_name, W1, W2)
        elif match == 2:
            # E = int(input("error "))  # between 1 for 256x256 picture
            image_name = input("image name ")
            # # image = Image.open("Y/" + image_name)
            # image = image.convert('RGB')
            # width, height = image.size
            with open(f"Y/{image_name}", "r") as json_file:
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
            pixels = convert_to_color(nX)
            pixels = convert_into_tuples(pixels)
            pixels = convert_into_pixels(pixels, height, width, n, m)
            out_image = Image.new("RGB", (width, height))
            out_image.putdata(pixels)
            out_image.show()

            # saving(nX, Y, height, width, n, m, image, neurons, image_name, W1, W2)
        elif match == 3:
            break
        else:
            print("wrong answer")

