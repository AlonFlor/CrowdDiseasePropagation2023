import os
import file_handling
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import subprocess

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
def draw_fluid_for_single_image(draw, data, circle_radius, data_index, normalization_r, normalization_g, normalization_b):
    for row in data:
        x = row[0]
        y = row[1]
        amount = row[data_index]
        color_r = int(256*(normalization_r(amount)))
        color_g = int(256*(normalization_g(amount)))
        color_b = int(256*(normalization_b(amount)))
        x_draw = map_to_drawing(x)
        y_draw = map_to_drawing(y)

        circle_bound = [x_draw - circle_radius, y_draw - circle_radius, x_draw + circle_radius, y_draw + circle_radius]
        draw.ellipse(circle_bound, (color_r, color_g, color_b))  # (int(groupC[0] * 255), int(groupC[1] * 255), int(groupC[2] * 255)))

def draw_images_for_given_time(frame_count, output_dir, circle_radius, data):
    #TODO more orderly and systematic version of these so they can be consolidated, plus systematically define the normalization functions
    image_name_base = str(frame_count).zfill(4)
    image1_name = image_name_base + "_disease" + ".png"
    image1 = Image.new("RGB", (1280, 1280), (255, 255, 255))
    draw1 = ImageDraw.Draw(image1)
    disease_normalization = lambda a : 1.-a*4
    draw_fluid_for_single_image(draw1, data, circle_radius, 4, disease_normalization, disease_normalization, disease_normalization)

    image1.save(os.path.join(output_dir, image1_name))
    image2_name = image_name_base + "_velocity_x" + ".png"
    image2 = Image.new("RGB", (1280, 1280), (255, 255, 255))
    draw2 = ImageDraw.Draw(image2)
    velocity_limit = 1.5    #plus or minus this value
    velocity_positive_normalization = lambda a : 1. - max(-a/velocity_limit, 0.)
    velocity_neutral_normalization = lambda a : 1. - abs(a/velocity_limit)
    velocity_negative_normalization = lambda a : 1. - max(a/velocity_limit, 0.)
    draw_fluid_for_single_image(draw2, data, circle_radius, 2, velocity_positive_normalization, velocity_neutral_normalization, velocity_negative_normalization)

    image2.save(os.path.join(output_dir, image2_name))
    image3_name = image_name_base + "_velocity_y" + ".png"
    image3 = Image.new("RGB", (1280, 1280), (255, 255, 255))
    draw3 = ImageDraw.Draw(image3)
    draw_fluid_for_single_image(draw3, data, circle_radius, 3, velocity_positive_normalization, velocity_neutral_normalization, velocity_negative_normalization)

    image3.save(os.path.join(output_dir, image3_name))
    #draw_crowd(draw1, data, circle_radius)

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

        len_data = min(data[end_time_step].shape[0], data[start_time_step].shape[0]) #TODO this is a method of interpolating for the adaptive grid.
                                                                                     # Please find a different way than trauncation
        data_to_use = (end_time_step-i)*data[start_time_step][:len_data,:] + (i-start_time_step)*data[end_time_step][:len_data,:]
        draw_images_for_given_time(frame_count, output_dir, circle_radius, data_to_use)
        frame_count += 1
    #draw_single_image(frame_count, output_dir, circle_radius, data[len(data)-1])


def make_video(frame_rate, scenario_folder, video_base_name="video", image_type_name=""):
    command_str = f"ffmpeg -framerate {frame_rate} -i {scenario_folder}/images/%04d{image_type_name}.png" \
                  f" -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {scenario_folder}/{video_base_name}{image_type_name}.mp4"
    process = subprocess.Popen(command_str, shell=True, stdout=subprocess.PIPE)
    process.wait()
