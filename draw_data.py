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


#TODO consider drawing arrows in each crowd agent pointing in the agent's velocity


def overlay_images(image_1_path, image_2_path, new_image_path):
    image_1 = np.array(Image.open(image_1_path))
    image_2 = np.array(Image.open(image_2_path))
    new_image_array = np.minimum(image_1,image_2).astype("uint8")
    new_image = Image.fromarray(new_image_array,"RGB")
    new_image.save(new_image_path)

def overlay_disease_and_crowd(number_of_time_steps, output_dir):
    for i in np.arange(number_of_time_steps):
        image_name_base = str(i).zfill(4)
        image1_path = os.path.join(output_dir, image_name_base + "_crowd" + ".png")
        image2_path = os.path.join(output_dir, image_name_base + "_disease" + ".png")
        image3_path = os.path.join(output_dir, image_name_base + "_combined" + ".png")

        overlay_images(image1_path, image2_path, image3_path)

def draw_crowd(draw, data, circle_radius):
    #TODO draw disease in people, and draw a border for infectious people
    for row in data:
        x = row[0]
        y = row[1]
        x_draw = map_to_drawing(x)
        y_draw = map_to_drawing(-y) #y-axis is upside-down in drawings

        disease = row[7]

        circle_bound = [x_draw - circle_radius, y_draw - circle_radius, x_draw + circle_radius, y_draw + circle_radius]
        draw.ellipse(circle_bound, (int(255.*disease), 0, 0))  # (int(groupC[0] * 255), int(groupC[1] * 255), int(groupC[2] * 255)))
def draw_fluid_density_for_single_image(draw, data, circle_radius, data_index, normalization_r, normalization_g, normalization_b):
    for row in data:
        x = row[0]
        y = row[1]
        amount = row[data_index]
        color_r = int(256*(normalization_r(amount)))
        color_g = int(256*(normalization_g(amount)))
        color_b = int(256*(normalization_b(amount)))
        x_draw = map_to_drawing(x)
        y_draw = map_to_drawing(-y) #y-axis is upside-down in drawings

        circle_bound = [x_draw - circle_radius, y_draw - circle_radius, x_draw + circle_radius, y_draw + circle_radius]
        draw.ellipse(circle_bound, (color_r, color_g, color_b))  # (int(groupC[0] * 255), int(groupC[1] * 255), int(groupC[2] * 255)))

def draw_fluid_field_for_single_image(name, output_dir, data, data_index_1, data_index_2):
    skip = 4
    X = data[::skip,0]
    Y = data[::skip,1]
    field_1 = data[::skip,data_index_1]
    field_2 = data[::skip,data_index_2]

    #test to see if y is upside-down. It is not.
    #field_2 = np.array([0.5 * i - Y[0] for i in Y])


    plt.figure(figsize=(20,10))
    plt.quiver(X, Y, field_1, field_2, color="black", angles='xy')
    #plt.xlim((0, 1))
    #plt.ylim((0, 1))
    #plt.show()
    plt.savefig(os.path.join(output_dir, name))
    plt.close()


def draw_fluid_images_for_given_time(frame_count, output_dir, circle_radius, data):
    image_name_base = str(frame_count).zfill(4)
    image1_name = image_name_base + "_disease" + ".png"
    image1 = Image.new("RGB", (1280, 1280), (255, 255, 255))
    draw1 = ImageDraw.Draw(image1)
    disease_normalization = lambda a : 1.-a
    draw_fluid_density_for_single_image(draw1, data, circle_radius, 4, disease_normalization, disease_normalization, disease_normalization)
    image1.save(os.path.join(output_dir, image1_name))

    image2_name = image_name_base + "_velocity" + ".png"
    draw_fluid_field_for_single_image(image2_name, output_dir, data, 2, 3)

def draw_crowd_for_given_time(frame_count, output_dir, circle_radius, data):
    image_name_base = str(frame_count).zfill(4)
    image_name = image_name_base + "_crowd" + ".png"
    image = Image.new("RGB", (1280, 1280), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw_crowd(draw, data, circle_radius)
    image.save(os.path.join(output_dir, image_name))

def draw_data(input_dir, output_dir, circle_radius_input, number_of_time_steps, time_steps_per_frame, data_type):
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
        if data_type=="crowd":
            draw_crowd_for_given_time(frame_count, output_dir, circle_radius, data_to_use)
        if data_type=="air":
            draw_fluid_images_for_given_time(frame_count, output_dir, circle_radius, data_to_use)

        frame_count += 1
    #draw_single_image(frame_count, output_dir, circle_radius, data[len(data)-1])


def make_video(frame_rate, scenario_folder, video_base_name="video", image_type_name=""):
    command_str = f"ffmpeg -framerate {frame_rate} -i {scenario_folder}/images/%04d{image_type_name}.png" \
                  f" -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {scenario_folder}/{video_base_name}{image_type_name}.mp4"
    process = subprocess.Popen(command_str, shell=True, stdout=subprocess.PIPE)
    process.wait()
