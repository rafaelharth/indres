"""
Contains various methods for printing information on the console and also saving a version ordered by neuron, provided that
settings.PRINT_INFORMATION is set to True.
"""

import os
import numpy as np

import settings

index = 0

def show(information, history=[]):
    if settings.PRINT_INFORMATION:
        print('\n\n' + information)
        history.append('\n\n' + information)
        dashes = "-" * len(information)
        print(dashes)
        history.append(dashes)
        return dashes

def show_simple(information, history=[]):
    if settings.PRINT_INFORMATION:
        print(information)
        history.append(information)

def show_single(information, history=[]):
    if settings.PRINT_INFORMATION:
        dashes = "-" * (len(information) + 4)
        print('\n\n' + dashes)
        history.append('\n\n' + dashes)
        print('- ' + information + ' -')
        history.append('- ' + information + ' -')
        print(dashes)
        history.append(dashes)

def aftermath(dashes, history=[]):
    if settings.PRINT_INFORMATION:
        print(dashes + '\n')
        history.append(dashes + '\n')

def show_neuron_hits_record(get_neuron_hits_record, threshold):
    def number_array(arr):
        output = ""
        for row in arr:
            for elem in row:
                output += '{:01X}'.format(int(elem / 10))
            output += "\n"
        return output

    if settings.PRINT_INFORMATION:
        # First, clear directory
        file_list = [f for f in os.listdir('neuron_hits_images/')]
        for f in file_list:
            os.remove(os.path.join('neuron_hits_images/', f))


        # Limit ourselves to only a fixed number of images
        get_neuron_hits_record = get_neuron_hits_record[0:6]


        dashes = show(f"\n\nInformation on neurons during get_neuron_hits (settings.py):")
        print("Neuron # / Image #\n")

        image_console_outputs = []

        for (n_i, im_i_total, im_i_filtered, old_im, new_im) in get_neuron_hits_record:
            image_console_outputs.append(build_ss_image(old_im, threshold[n_i], n_i, im_i_total))
            filename = f"neuron_hits_images/Neuron_{n_i}_Image_{im_i_total}"
            new_im.save(f"{filename}.png", "PNG")

            file = open(f"{filename}.txt", 'w')
            file.write(number_array(np.array(new_im)))
            file.close()


        image_console_outputs = np.array(image_console_outputs)

        for row_i in range(len(image_console_outputs[0])):
            line = image_console_outputs[:,row_i]
            for col in line:
                print(col, end='')
            print("")

        _, _, _, old, new = get_neuron_hits_record[-1]


        number_array(np.array(new))

        aftermath(dashes)

def show_list(list_of_items, max_length=130, history=[]):
    if settings.PRINT_INFORMATION:
        assert isinstance(list_of_items, list)

        # turn the list into a list of strings
        L = []
        for i, item in enumerate(list_of_items):
            L.append(str(item))

        i = 0

        while i < len(L):
            line = ""

            if len(L[i]) > max_length:
                print(L[i] + ', ')
                history.append(L[i] + ', ')
                i += 1
            else:
                while i < len(L) and len(line) + len(L[i]) <= max_length:
                    line += L[i] + ', '
                    i += 1
                print(line)
                history.append(line)

def build_ss_image(image, threshold, n_i, im_i):
    mask = image > threshold
    n_rows = []

    for y in mask:
        n_row = ""
        for elem in y:
            if elem:
                n_row += "X "
            else:
                n_row += "- "
        n_row += "  "
        n_rows.append(n_row)

    output = []
    header = f"{n_i} / {im_i}"
    padding = (len(n_rows[0]) - len(header))
    pad_left = int(padding/2)-1
    pad_right = padding - pad_left
    header = pad_left * ' ' + header + pad_right * ' '

    output.append(header)
    for n_row in n_rows:
        output.append(n_row)

    return output