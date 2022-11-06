from NN import from_archieve, new_weights, from_compressed_image


if __name__ == '__main__':
    while True:
        match = int(input("0 - from archieve, 1 - new weights, 2 - from compressed image, 3 - exit"))
        if match == 0:
            from_archieve()
        elif match == 1:
            new_weights()
        elif match == 2:
            from_compressed_image()
        elif match == 3:
            break
        else:
            print("wrong answer")

