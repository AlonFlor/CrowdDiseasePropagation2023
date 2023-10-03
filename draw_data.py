import os
import file_handling
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

scale = 40.

def map_to_drawing(input_val):
    return scale*input_val+640

def plot_data_two_curves(x_axis_data, y_axis_data1, y_axis_data2):
    plt.plot(x_axis_data, y_axis_data1, 'b', x_axis_data, y_axis_data2, 'r')
    plt.show()


def plot_data_three_curves(x_axis_data, y_axis_data1, y_axis_data2, y_axis_data3):
    plt.plot(x_axis_data, y_axis_data1, 'b', x_axis_data, y_axis_data2, 'g', x_axis_data, y_axis_data3, 'r')
    plt.show()

def draw_data(circle_radius_input, input_dir, number_of_time_steps, time_step_skip):
    output_dir = 'images'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    circle_radius = scale * circle_radius_input

    frame_count = 0
    for i in range(0, number_of_time_steps, time_step_skip):
        full_file = os.path.join(input_dir, str(i)+".csv")
        image_name = str(frame_count).zfill(4) + ".png"
        image1 = Image.new("RGB", (1280, 1280), (255, 255, 255))
        draw1 = ImageDraw.Draw(image1)
        data = file_handling.read_numerical_csv_file(full_file)
        for row in data:
            x = row[0]
            y = row[1]
            x_draw = map_to_drawing(x)
            y_draw = map_to_drawing(y)

            circle_bound = [x_draw - circle_radius, y_draw - circle_radius, x_draw + circle_radius, y_draw + circle_radius]
            draw1.ellipse(circle_bound, (0,0,0))#(int(groupC[0] * 255), int(groupC[1] * 255), int(groupC[2] * 255)))
        image1.save(os.path.join(output_dir, image_name))
        frame_count += 1

