import os
import file_handling
import numpy as np
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


#TODO y is upside-down for both draw_crowd and draw_disease
#TODO consider drawing arrows in each crowd agent pointing in the agent's velocity

def draw_crowd(draw, data, circle_radius):
    for row in data:
        x = row[0]
        y = row[1]
        x_draw = map_to_drawing(x)
        y_draw = map_to_drawing(y)

        circle_bound = [x_draw - circle_radius, y_draw - circle_radius, x_draw + circle_radius, y_draw + circle_radius]
        draw.ellipse(circle_bound, (0, 0, 0))  # (int(groupC[0] * 255), int(groupC[1] * 255), int(groupC[2] * 255)))
def draw_disease(draw, data, circle_radius):
    for row in data:
        x = row[0]
        y = row[1]
        disease_amount = row[2]
        color = int(256*(1.-disease_amount*4))
        x_draw = map_to_drawing(x)
        y_draw = map_to_drawing(y)

        circle_bound = [x_draw - circle_radius, y_draw - circle_radius, x_draw + circle_radius, y_draw + circle_radius]
        draw.ellipse(circle_bound, (color, color, color))  # (int(groupC[0] * 255), int(groupC[1] * 255), int(groupC[2] * 255)))

def draw_single_image(frame_count, output_dir, circle_radius, data):
    image_name = str(frame_count).zfill(4) + ".png"
    image1 = Image.new("RGB", (1280, 1280), (255, 255, 255))
    draw1 = ImageDraw.Draw(image1)
    draw_crowd(draw1, data, circle_radius)
    #draw_disease(draw1, data, circle_radius)
    image1.save(os.path.join(output_dir, image_name))

def draw_data(input_dir, output_dir, circle_radius_input, number_of_time_steps, time_steps_per_frame):
    circle_radius = scale * circle_radius_input
    frame_count = 0

    total_number_of_frames = int(number_of_time_steps / time_steps_per_frame)
    #print("total_number_of_frames",total_number_of_frames)

    #interpolate time steps
    data = []
    for i in range(0, number_of_time_steps+1):
        file_name = os.path.join(input_dir, str(i) + ".csv")
        data_time_step_i = file_handling.read_numerical_csv_file(file_name)
        data.append(data_time_step_i)
    #print(len(data))
    for i in np.linspace(0, number_of_time_steps, total_number_of_frames, endpoint=False):
        start_time_step = int(i)
        end_time_step = int(i+1)
        #print(i,start_time_step, end_time_step)

        data_to_use = (end_time_step-i)*data[start_time_step] + (i-start_time_step)*data[end_time_step]
        draw_single_image(frame_count, output_dir, circle_radius, data_to_use)
        frame_count += 1
    #draw_single_image(frame_count, output_dir, circle_radius, data[len(data)-1])
