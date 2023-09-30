import numpy as np
import os
import draw_data
import file_handling
from scipy import optimize
import random
import subprocess
import time

agent_radius = 0.35
interaction_radius = 2.
social_force_strength = 1.
dt = 1./24.

time_amount = 30.



class Person:
    def __init__(self, position, velocity, destination, desired_speed, disease, immunity):
        self.position = position
        self.velocity = velocity

        self.destination = destination
        self.desired_speed = desired_speed

        self.disease = disease
        self.immunity = immunity

        self.disease_change = 0.0

    def get_desired_velocity(self):
        displacement = self.destination - self.position
        distance = np.linalg.norm(displacement)
        if distance > 1.:
            self.desired_velocity =  self.desired_speed * displacement / distance
        else:
            self.desired_velocity = self.desired_speed * displacement * distance

    def update_position(self):
        self.position += self.velocity * dt


#TODO: add symptomaticity interactions
def pairwise_interactions(p1, p2):
    '''Get social force between two agents'''
    relative_displacement = p2.position - p1.position
    distance = np.linalg.norm(relative_displacement)
    if distance > interaction_radius:
        return np.array([0.,0.])
    relative_displacement_vector = relative_displacement / distance

    #relative_velocity = p2.v - p1.v
    #relative_speed = np.linalg.norm(relative_velocity)

    social_force = social_force_strength * relative_displacement_vector / (distance * distance)
    return -1*social_force, social_force


def get_and_apply_forces(agents):
    '''Explicit solve (desired velocity force and social force)'''
    num_agents = len(agents)

    #desired velocity force
    desired_velocity_forces = np.zeros((num_agents, 2))
    for i in np.arange(num_agents):
        agents[i].get_desired_velocity()
        desired_velocity_forces[i] = agents[i].desired_velocity - agents[i].velocity

    #social force
    social_forces = np.zeros((num_agents, 2))
    for i in np.arange(num_agents):
        for j in np.arange(i+1,num_agents):
            social_force_i, social_force_j = pairwise_interactions(agents[i], agents[j])
            social_forces[i] += social_force_i
            social_forces[j] += social_force_j

    for i in np.arange(num_agents):
        agents[i].velocity += (desired_velocity_forces[i] + social_forces[i]) * dt


def potential_energy_value_and_gradient(velocity_vector, agents):
    '''sum of momentum potential, desired velocity potential, social force potential'''

    value = 0.
    gradient = np.zeros(velocity_vector.shape)
    num_agents = len(agents)

    #add momentum potential and desired velocity potential
    for i in np.arange(num_agents):
        velocity = velocity_vector[2*i:2*i+2]
        velocity_diff = velocity - agents[i].velocity
        goal_velocity_diff = velocity - agents[i].desired_velocity
        value += 0.5 * dt * np.dot(velocity_diff,velocity_diff) + 0.5 * dt * np.dot(goal_velocity_diff,goal_velocity_diff)
        gradient[2*i:2*i+2] += dt * (velocity_diff + goal_velocity_diff)

    #add social force potential
    social_force_gradient = np.zeros(velocity_vector.shape)
    for i in np.arange(num_agents):
        velocity_i = velocity_vector[2*i:2*i+2]
        for j in np.arange(i+1, num_agents):
            #check to make sure the agents are within each others' range. TODO: consider replacing with more efficient code, like the locality query used in the Implicit Crowds code.
            if np.linalg.norm(agents[i].position - agents[j].position) < interaction_radius:
                #print(f"here {i} {j}")
                velocity_j = velocity_vector[2*j:2*j+2]

                #use distance after a time step
                displacement_after = agents[j].position + velocity_j*dt - (agents[i].position + velocity_i*dt)
                dist_after = np.linalg.norm(displacement_after)
                value += social_force_strength / dist_after
                local_gradient = -1*social_force_strength * displacement_after * dt / np.power(dist_after, 3)
                social_force_gradient[2 * i:2 * i + 2] -= local_gradient
                social_force_gradient[2 * j:2 * j + 2] += local_gradient

                #use minimum distance between current and after a time step (helps with tunneling)
                '''current_displacement = agents[j].position - agents[i].position
                displacement_after = agents[j].position + velocity_j*dt - (agents[i].position + velocity_i*dt)
                displacement_diff = displacement_after - current_displacement
                numerator = np.dot(displacement_after, displacement_diff)
                denominator = np.dot(displacement_diff, displacement_diff) + 0.0001
                min_dist_factor = numerator / denominator
                #print("min_dist_factor", min_dist_factor)
                if min_dist_factor <= 0.:
                    min_dist_factor = 0.
                    min_dist_factor_gradient = 0.
                elif min_dist_factor >= 1.:
                    min_dist_factor = 1.
                    min_dist_factor_gradient = 0.
                else:
                    min_dist_factor_gradient = 1.#(2 * displacement_after * dt - current_displacement * dt - 2 * current_displacement * denominator * dt) / (denominator * denominator)
                    print("min_dist_factor",min_dist_factor)
                    print("min_dist_factor_gradient",min_dist_factor_gradient)
                min_displacement = (1. - min_dist_factor) * displacement_after + min_dist_factor * current_displacement
                min_displacement_gradient = dt - dt*min_dist_factor - displacement_after * min_dist_factor_gradient + current_displacement * min_dist_factor_gradient
                min_dist = np.linalg.norm(min_displacement)
                value += social_force_strength / min_dist
                local_gradient = -1*social_force_strength * min_displacement * min_displacement_gradient / np.power(min_dist, 3)

                social_force_gradient[2*i:2*i+2] -= local_gradient
                social_force_gradient[2*j:2*j+2] += local_gradient'''

                #use the distance energy found in the Implicit Crowds code (also helps with tunneling, but not in the paper)
                '''current_displacement = agents[j].position - agents[i].position
                velocity_diff = velocity_j - velocity_i
                speed_of_collision = velocity_diff[0]*velocity_diff[0] + velocity_diff[1]*velocity_diff[1]
                time_to_impact = -1*np.dot(current_displacement, velocity_diff) / (speed_of_collision + 0.0001)
                time_to_impact = max(min(time_to_impact, dt), 0.) #<0 = agents are diverging, >0 and <dt = potential to tunnel, >dt: use dt because this is implicit
                print("time_to_impact",time_to_impact)

                #get minimum distance between the agents within this time step
                min_displacement = velocity_diff * time_to_impact - current_displacement
                min_distance = np.linalg.norm(min_displacement)

                ##collision or tunneling. Either way, this is not an applicable velocity.
                #if min_distance <= agent_radius:
                #    got_infinity = True
                #    break

                #distance = min_distance - agent_radius
                value += social_force_strength / min_distance#distance

                #get the gradient
                time_to_impact_derivatives = np.array([0., 0.])
                if time_to_impact >0 and time_to_impact < dt:
                    time_to_impact_derivatives = (current_displacement + 2*time_to_impact*velocity_diff) / speed_of_collision
                distance_derivative = np.array([
                    min_displacement[0] * (time_to_impact + velocity_diff[0] * time_to_impact_derivatives[0]) + min_displacement[1] * velocity_diff[1] * time_to_impact_derivatives[1],
                    min_displacement[1] * (time_to_impact + velocity_diff[1] * time_to_impact_derivatives[1]) + min_displacement[0] * velocity_diff[0] * time_to_impact_derivatives[0]
                ])
                distance_derivative *= - social_force_strength / (min_distance * min_distance * min_distance)#(min_distance * distance * distance)
                print("distance_derivative", distance_derivative)

                social_force_gradient[2*i:2*i+2] -= distance_derivative
                social_force_gradient[2*j:2*j+2] += distance_derivative
                print("social_force_gradient", social_force_gradient)'''
    #print(np.linalg.norm(gradient))
    #print(social_force_gradient)
    #print()

    gradient += social_force_gradient
    return value, gradient



def velocities_implicit_solve(agents):
    '''Implicit solve for velocities of agents'''
    num_agents = len(agents)

    #calculate desired velocity for agents
    for i in np.arange(num_agents):
        agents[i].get_desired_velocity()

    velocity_vector = np.zeros((num_agents*2))
    '''potential_energy_value_and_gradient(velocity_vector, agents)
    print("That's all there is for now")
    exit()'''
    result = optimize.minimize(potential_energy_value_and_gradient, velocity_vector, args=(agents), jac=True)
    print(f"{result.nit} iterations\tresult={result.fun}, gradient size = {np.linalg.norm(result.jac)}\t\tsuccess={result.success}\tmessage={result.message}\n")
    new_velocity_vector = result.x

    if not result.success:
        #debug this
        velocity_vector = np.zeros((num_agents * 2))
        x_speeds = []
        values = []
        actual_gradients = []
        gradients = []
        dv = 0.01
        for second_agent_x_speed in np.arange(-5.,5.,dv):
            velocity_vector[2] = second_agent_x_speed
            value, gradient = potential_energy_value_and_gradient(velocity_vector, agents)
            x_speeds.append(second_agent_x_speed)
            values.append(value)
            gradients.append(gradient[2])
        for i in np.arange(len(x_speeds)):
            if i<2 or i>len(x_speeds)-3:
                actual_gradients.append(np.nan)
            else:
                actual_gradients.append((-values[i+2] + 8*values[i+1] - 8*values[i-1] + values[i-2])/(12.*dv))
        draw_data.plot_data_three_curves(x_speeds, values, actual_gradients, gradients)
        #draw_data.plot_data_two_curves(x_speeds, values, actual_gradients)
        exit()


    for i in np.arange(num_agents):
        agents[i].velocity = new_velocity_vector[2*i:2*i+2]


def generate_point(x_range, y_range):
    return np.array([random.uniform(x_range[0], x_range[1]), random.uniform(y_range[0], y_range[1])])

def save_crowd_data(number, agents, folder, extra_info=""):
    num_agents = len(agents)
    data = np.zeros((num_agents, 9))
    for i in np.arange(num_agents):
        agent = agents[i]
        data[i][0] = agent.position[0]
        data[i][1] = agent.position[1]
        data[i][2] = agent.velocity[0]
        data[i][3] = agent.velocity[1]
        data[i][4] = agent.destination[0]
        data[i][5] = agent.destination[1]
        data[i][6] = agent.desired_speed
        data[i][7] = agent.disease
        data[i][8] = agent.immunity
    header = "pos_x,pos_y,vel_x,vel_y,dest_x,dest_y,desired_speed,disease,immunity"
    file_handling.write_csv_file(os.path.join(folder, extra_info+str(number)+".csv"), header, data)

#TODO: scenario code will need to include disease, immunity, and contagiousness info, as well as obstacles info. Maybe agents will be split into groups.
# Change write_csv_file in file_handling to use lists (so int and float can be written together in data)
def generate_scenario(total_pop, scenario_number, folder, extra_info=""):
    '''Generate a scenario'''
    agents = []
    x_range = (-5, 5)
    y_range = (-5, 5)
    for i in np.arange(total_pop):
        position = None
        destination = None
        bad_pos = True
        bad_dest = True
        while bad_pos:
            position = generate_point(x_range, y_range)
            bad_pos = False
            for agent in agents:
                if np.linalg.norm(position - agent.position) < 1.2*agent_radius:
                    bad_pos = True
                    break
        while bad_dest:
            destination = generate_point(x_range, y_range)
            bad_dest = False
            for agent in agents:
                if np.linalg.norm(destination - agent.destination) < 1.2*agent_radius:
                    bad_dest = True
                    break

        velocity = np.array([0., 0.])
        desired_speed = 1.
        disease = 0.
        immunity = 0.
        agents.append(Person(position, velocity, destination, desired_speed, disease, immunity))

    save_crowd_data(scenario_number, agents, folder, extra_info)

#TODO: all changes to generate_scenario also need to be done here
def open_scenario(scenario_number, folder, extra_info=""):
    data = file_handling.read_numerical_csv_file(os.path.join(folder, extra_info+str(scenario_number)+".csv"))

    agents = []
    for i in np.arange(len(data)):
        position = np.array([data[i][0], data[i][1]])
        velocity = np.array([data[i][2], data[i][3]])
        destination = np.array([data[i][4], data[i][5]])
        desired_speed = data[i][6]
        disease = data[i][7]
        immunity = data[i][8]
        agents.append(Person(position, velocity, destination, desired_speed, disease, immunity))

    return agents


#generate_scenario(15, 0, "scenarios")


sim_start = time.perf_counter_ns()

agents = open_scenario(0, "scenarios")
pop_num = len(agents)
number_of_time_steps = int(np.ceil(time_amount / dt))

time_steps_dir = "time_steps"
if not os.path.isdir(time_steps_dir):
    os.mkdir(time_steps_dir)

for i in np.arange(number_of_time_steps):
    save_crowd_data(i, agents, time_steps_dir)
    #get_and_apply_forces(agents) #explicit solve for agent velocities
    velocities_implicit_solve(agents) #implicit solve for agent velocities
    for j in np.arange(pop_num):
        agents[j].update_position()
save_crowd_data(number_of_time_steps, agents, time_steps_dir)

sim_end = time.perf_counter_ns()
time_to_run_sims = (sim_end - sim_start) / 1e9
print('Time to run simulations:', time_to_run_sims, 's\t\t(', time_to_run_sims/3600., 'h)')
print("Finished simulation. Moving to drawing.")

draw_start = time.perf_counter_ns()
time_steps_per_frame = 1
#TODO: interpolation for smooth animation when time_steps_per_frame < 1.
draw_data.draw_data(agent_radius, time_steps_dir, number_of_time_steps, time_steps_per_frame)
draw_end = time.perf_counter_ns()
time_to_run_draw = (draw_end - draw_start) / 1e9
print('Time to draw:', time_to_run_draw, 's\t\t(', time_to_run_draw/3600., 'h)')
print("Finished drawing. Making a video.")


#make the video
def do_thing(command_str):
    process = subprocess.Popen(command_str, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print("Done")
frame_rate = 1/(dt * time_steps_per_frame)
command_str = f"ffmpeg -framerate {frame_rate} -i images/%04d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4"
do_thing(command_str)

