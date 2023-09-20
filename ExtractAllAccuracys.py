import os
import plot

modelname = "vgg16_new_clear_clear"
folder_path = f"E:\\Carla\\_{modelname}_textfiles"



def extract_accuracy(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        accuracy_start = content.find("accuracy") + len("accuracy")
        accuracy_end = content.find("'", accuracy_start)
        accuracy = float(content[accuracy_start:accuracy_end])
        accuracy = int(accuracy * (10**3))/(10**3)
        return accuracy


def insert_data(word, test):
    accuracy_list = []
    tt = ""
    if test:
        tt = "test"
    else:
        tt = "train"

    for i in range (0, 50):
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(f"{word}_epoch{i}_{tt}.txt"):
                file_path = os.path.join(folder_path, filename)
                accuracy = extract_accuracy(file_path)
                accuracy_list.append(accuracy)

    print(f"{modelname}: accuracy_list")
    return accuracy_list


if __name__ == '__main__':


    test = insert_data(modelname, True)
    train = insert_data(modelname, False)

    print(f"best accuracy {max(test)} on testdata with index {test.index(max(test))}")

    plot.plotLoss(train, test, "vgg16_clear_clear")
