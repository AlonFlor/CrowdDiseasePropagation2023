import numpy as np
import os
import draw_data
import file_handling
from scipy import optimize
import random
import time
import PDE

agent_radius = 0.25
interaction_radius = 10.
social_force_strength = 1.
dt = 1./24.
frame_rate = 24.
time_amount = 30.

scenario_name = "bus"#"choir_practice"#"implicit_crowds_8_agents"#"2_agents"#

ttc_smoothing_eps = 0.2
ttc_constant = 1.5
ttc_power = 3.25
goal_velocity_coefficient = 2.
ttc_0 = 3

maximum_disease = 1.
#disease_person_to_air_coefficient = 1.
disease_air_to_person_coefficient = 1.

MAC_cell_width = 0.1  # each MAC cell is 0.1 meters on each side
density_threshold = 0.003
number_of_buffer_layers = 3
disease_diffusivity_constant = 0.05    #TODO Different size time step = more diffusion, but loss from one big time step >> loss from several small time steps.
disease_die_off_rate = 0.



class Person:
    def __init__(self, position, velocity, destination, desired_speed, disease, immunity, is_infectious):
        self.position = position
        self.velocity = velocity

        self.destination = destination
        self.desired_speed = desired_speed

        self.disease = disease
        self.immunity = immunity
        self.is_infectious = True if is_infectious==1.0 else False
        if self.is_infectious:
            self.disease = maximum_disease

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
def explicit_solve_pairwise_interactions(p1, p2):
    '''Get social force between two agents'''
    relative_displacement = p2.position - p1.position
    distance = np.linalg.norm(relative_displacement)
    if distance > interaction_radius:
        return np.array([0.,0.])
    relative_displacement_vector = relative_displacement / (distance - 2.* agent_radius)

    #relative_velocity = p2.v - p1.v
    #relative_speed = np.linalg.norm(relative_velocity)

    social_force = social_force_strength * relative_displacement_vector / (distance * distance)
    return -1*social_force, social_force


def ttci_energy(relative_positions, relative_velocities, combined_radius):
    '''inverse time to contact'''

    relative_positions_magn = np.linalg.norm(relative_positions)
    relative_positions_dir = relative_positions / relative_positions_magn
    relative_velocities_projected = np.dot(-relative_velocities, relative_positions_dir)
    relative_velocities_tangential = relative_velocities - relative_velocities_projected * relative_positions_dir
    relative_velocities_tangential_magn = np.linalg.norm(relative_velocities_tangential)

    r_sq = combined_radius*combined_radius
    denominator = relative_positions_magn*relative_positions_magn - r_sq

    v_t_max = combined_radius * relative_velocities_projected / np.sqrt(denominator)
    cutoff = np.sqrt(1.-ttc_smoothing_eps*ttc_smoothing_eps)
    v_t_star = cutoff*v_t_max

    if relative_velocities_tangential_magn < v_t_star:
        #compute inverse time to contact as usual
        vt_sq = relative_velocities_tangential_magn * relative_velocities_tangential_magn
        discriminant = np.sqrt(r_sq * relative_velocities_projected * relative_velocities_projected - denominator * denominator * vt_sq)
        numerator = relative_velocities_projected * relative_positions_magn + discriminant
        ttci = numerator / denominator
        if ttci > 0.:
            exp_val = 1. / (ttci * ttc_0)
            energy_mult = ttc_constant * np.power(ttci, ttc_power-1) * np.exp(-1*exp_val)
            energy = energy_mult * ttci

            a = -relative_positions + relative_velocities*dt - relative_velocities_projected*dt*relative_positions_dir
            b = ((dt*relative_velocities_projected + relative_positions_magn)*relative_velocities_tangential * denominator / relative_positions_magn +
                -relative_positions*dt*vt_sq + r_sq * relative_velocities_projected*a / relative_positions_dir) / discriminant + dt*relative_velocities_projected
            gradient = -energy_mult / denominator * ((a+b)*(ttc_power+exp_val) - 2.*dt*(1./ttc_0 + ttc_power*ttci)*relative_positions)

            return energy, gradient
        return None
    else:
        sqrt_denominator = np.sqrt(denominator)
        ttci = (relative_positions_magn + ttc_smoothing_eps*combined_radius)*relative_velocities_projected / denominator - \
               cutoff  / ttc_smoothing_eps * (relative_velocities_tangential_magn - v_t_star) / sqrt_denominator
        if ttci > 0.:
            exp_val = 1. / (ttci * ttc_0)
            energy_mult = ttc_constant * np.power(ttci, ttc_power-1) * np.exp(-1*exp_val)
            energy = energy_mult * ttci

            a = (-relative_positions + relative_velocities*dt - relative_velocities_projected*dt*relative_positions_dir) / relative_positions_magn
            b = ((ttc_smoothing_eps*combined_radius + relative_positions_magn)*a) / denominator + \
                (cutoff*((relative_velocities_tangential*dt*relative_velocities_projected / relative_positions_magn + relative_velocities_tangential) /
                         relative_velocities_tangential_magn + combined_radius*cutoff / sqrt_denominator*(a - dt*relative_velocities_projected*relative_positions / denominator))) / \
                (ttc_smoothing_eps*sqrt_denominator) - dt*relative_positions / denominator* \
                (relative_velocities_projected*(ttc_smoothing_eps*combined_radius + relative_positions_magn) / denominator - relative_velocities_projected / relative_positions_magn + ttci)
            gradient = energy_mult * np.power(ttci, ttc_power-1)*(ttc_power+exp_val) * b

            return energy, gradient
        return None

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
        value += 0.5 * dt * np.dot(velocity_diff,velocity_diff) + 0.5 * dt * goal_velocity_coefficient * np.dot(goal_velocity_diff,goal_velocity_diff)
        gradient[2*i:2*i+2] += dt * (velocity_diff + goal_velocity_coefficient * goal_velocity_diff)

    #add social force and time to contact potential
    social_force_gradient = np.zeros(velocity_vector.shape)
    ttci_gradient = np.zeros(velocity_vector.shape)
    for i in np.arange(num_agents):
        velocity_i = velocity_vector[2*i:2*i+2]
        position_i_new = agents[i].position + velocity_i * dt
        for j in np.arange(i+1, num_agents):
            #check to make sure the agents are within each others' range. TODO: consider replacing with more efficient code, like the locality query used in the Implicit Crowds code.
            if np.linalg.norm(agents[i].position - agents[j].position) < interaction_radius:
                #print(f"here {i} {j}")
                velocity_j = velocity_vector[2*j:2*j+2]
                position_j_new = agents[j].position + velocity_j*dt

                #distance potential
                displacement_after = position_j_new - position_i_new
                dist_after = np.linalg.norm(displacement_after)
                modified_dist_after = dist_after - 2.*agent_radius
                value += social_force_strength / modified_dist_after
                local_gradient = -1*social_force_strength * displacement_after * dt / (modified_dist_after * modified_dist_after * dist_after)
                social_force_gradient[2 * i:2 * i + 2] -= local_gradient
                social_force_gradient[2 * j:2 * j + 2] += local_gradient

                #TODO: consider anti-tunneling code

                #time to contact potential
                relative_positions = position_j_new - position_i_new
                relative_velocities = velocity_j - velocity_i
                ttci_energy_result = ttci_energy(relative_positions, relative_velocities, 2.*agent_radius)
                if ttci_energy_result is not None:
                    ttci_energy_value, ttci_energy_gradient = ttci_energy_result
                    value += ttci_energy_value
                    ttci_gradient[2 * i:2 * i + 2] -= ttci_energy_gradient
                    ttci_gradient[2 * j:2 * j + 2] += ttci_energy_gradient

        #distance and TTC potentials with respect to obstacles
        for j in np.arange(len(obstacles)):
            obstacle = obstacles[j]
            current_signed_distance, _ = obstacle_signed_distance_and_gradient(obstacle, agents[i].position)
            obstacle_signed_distance, obstacle_gradient = obstacle_signed_distance_and_gradient(obstacle, position_i_new)
            if current_signed_distance < interaction_radius:
                modified_dist = obstacle_signed_distance - agent_radius
                #distance potential
                value += social_force_strength / modified_dist
                local_gradient = -1 * social_force_strength * obstacle_gradient * dt / (modified_dist * modified_dist)
                social_force_gradient[2 * i:2 * i + 2] += local_gradient

                #time to contact potential
                relative_positions = np.array([0., 0.])
                if obstacle_gradient[0] > 0.:
                    #obstacle right - agent pos x
                    relative_positions[0] = obstacle[1] - position_i_new[0]
                elif obstacle_gradient[0] < 0.:
                    #obstacle left - agent pos x
                    relative_positions[0] = obstacle[0] - position_i_new[0]
                if obstacle_gradient[1] > 0.:
                    #obstalce top - agent pos y
                    relative_positions[1] = obstacle[3] - position_i_new[1]
                elif obstacle_gradient[1] < 0.:
                    #obstacle bottom - agent pos y
                    relative_positions[1] = obstacle[2] - position_i_new[1]
                relative_velocities = - velocity_i
                ttci_energy_result = ttci_energy(relative_positions, relative_velocities, agent_radius)
                if ttci_energy_result is not None:
                    ttci_energy_value, ttci_energy_gradient = ttci_energy_result
                    value += ttci_energy_value
                    ttci_gradient[2 * i:2 * i + 2] -= ttci_energy_gradient

    #print(np.linalg.norm(gradient))
    #print(social_force_gradient)
    #print()

    gradient += social_force_gradient
    gradient += ttci_gradient
    return value, gradient



def velocities_implicit_solve(agents):
    '''Implicit solve for velocities of agents'''
    num_agents = len(agents)

    #calculate desired velocity for agents
    for i in np.arange(num_agents):
        agents[i].get_desired_velocity()

    velocity_vector = np.zeros((num_agents*2))
    result = optimize.minimize(potential_energy_value_and_gradient, velocity_vector, args=(agents), jac=True)
    print(f"{result.nit} iterations\tresult={result.fun}, gradient size = {np.linalg.norm(result.jac)}\t\tsuccess={result.success}\tmessage={result.message}\n")
    new_velocity_vector = result.x

    if not result.success:
        #debug this
        #print(potential_energy_value_and_gradient(velocity_vector, agents),"\n")
        orig_velocity_vector = np.zeros((num_agents * 2))
        for idx in np.arange(orig_velocity_vector.shape[0]):
            orig_velocity_vector[idx] = velocity_vector[idx]
        for idx in np.arange(velocity_vector.shape[0]):
            print(idx)
            velocity_vector = np.zeros((num_agents * 2))
            for idx_2 in np.arange(velocity_vector.shape[0]):
                velocity_vector[idx_2] = orig_velocity_vector[idx_2]
            x_speeds = []
            values = []
            actual_gradients = []
            gradients = []
            dv = 0.01
            for chosen_agent_coord_speed in np.arange(-5.,5.,dv):
                velocity_vector[idx] = chosen_agent_coord_speed
                value, gradient = potential_energy_value_and_gradient(velocity_vector, agents)
                x_speeds.append(chosen_agent_coord_speed)
                values.append(value)
                gradients.append(gradient[idx])
            for i in np.arange(len(x_speeds)):
                if i<2 or i>len(x_speeds)-3:
                    actual_gradients.append(np.nan)
                else:
                    actual_gradients.append((-values[i+2] + 8*values[i+1] - 8*values[i-1] + values[i-2])/(12.*dv))
            draw_data.plot_data_three_curves(x_speeds, values, actual_gradients, gradients)
            draw_data.plt.close()
            #draw_data.plot_data_two_curves(x_speeds, values, actual_gradients)
        exit(1)


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

def infect_people(agents, grid):
    #TODO consider an implicit version of this, perhaps updating agents' disease value for the previous time step using the current time step
    for agent in agents:
        disease_at_agent = grid.interpolate_value(agent.position, "disease_concentration")
        agent.disease += dt * disease_air_to_person_coefficient * disease_at_agent * (1.-agent.immunity)
        if agent.disease > maximum_disease:
            agent.disease = maximum_disease

def spread_infection_to_air(agents, grid):
    for i,agent in enumerate(agents):
        if agent.is_infectious:
            grid.distribute_value(agent.position, "disease_concentration", maximum_disease)
            grid.fixed_airflow_data[i] = (agent.position, 1.5*agent.velocity) #TODO expelled air velocity needs a minimum value and known agent direction for cases of motionless infectious people.


def obstacle_signed_distance_and_gradient(obstacle, point):
    # inside obstacle, distance value does not matter, but distance sign does matter
    left,right,bottom,top = obstacle
    dist_list = np.array([left - point[0], point[0] - right, bottom - point[1], point[1] - top])
    grad_list = [np.array([-1.,0.]), np.array([1.,0.]), np.array([0.,-1.]), np.array([0.,1.])]
    dist_arg = np.argmax(dist_list)

    return dist_list[dist_arg], grad_list[dist_arg]




#TODO: generate_scenario code will need to include disease, immunity, and contagiousness info, and will need to generate an obstacles file. Maybe agents will be split into groups.
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
                if np.linalg.norm(position - agent.position) < 2.2*agent_radius:
                    bad_pos = True
                    break
        while bad_dest:
            destination = generate_point(x_range, y_range)
            bad_dest = False
            for agent in agents:
                if np.linalg.norm(destination - agent.destination) < 2.2*agent_radius:
                    bad_dest = True
                    break

        velocity = np.array([0., 0.])
        desired_speed = 1.
        disease = 0.
        immunity = 0.
        is_infectious = 0.
        agents.append(Person(position, velocity, destination, desired_speed, disease, immunity, is_infectious))

    save_crowd_data(scenario_number, agents, folder, extra_info)

#TODO: all changes to generate_scenario also need to be done here
def open_scenario(folder,scenario_name):
    data = file_handling.read_numerical_csv_file(os.path.join(folder, scenario_name+".csv"))

    agents = []
    for i in np.arange(len(data)):
        position = np.array([data[i][0], data[i][1]])
        velocity = np.array([data[i][2], data[i][3]])
        destination = np.array([data[i][4], data[i][5]])
        desired_speed = data[i][6]
        disease = data[i][7]
        immunity = data[i][8]
        is_infectious = data[i][9]
        agents.append(Person(position, velocity, destination, desired_speed, disease, immunity, is_infectious))

    obstacles_file_path = os.path.join(folder, scenario_name + "_obstacles.csv")
    if os.path.isfile(obstacles_file_path):
        obstacles_data = file_handling.read_numerical_csv_file(obstacles_file_path)
        #for i in np.arange(obstacles_data.shape[0]):
        #    row = obstacles_data[i]
        #    obstacles.append([np.array([row[0], row[2]]), np.array([row[0], row[3]]), np.array(row[1], row[2]), np.array(row[1], row[3])])
        return agents, obstacles_data

    return agents, None

def get_world_dimensions_from_walls(obstacles):
    min_x = max_x = min_y = max_y = 0.0
    for obstacle in obstacles:
        candidate_min_x, candidate_max_x, candidate_min_y, candidate_max_y = obstacle
        if candidate_min_x < min_x:
            min_x = candidate_min_x
        if candidate_max_x > max_x:
            max_x = candidate_max_x
        if candidate_min_y < min_y:
            min_y = candidate_min_y
        if candidate_max_y > max_y:
            max_y = candidate_max_y

    return max_x-min_x, max_y-min_y

#generate_scenario(15, 0, "scenarios")


sim_start = time.perf_counter_ns()

#make folder with scenario name
scenario_output_folder = scenario_name + "_dt=" + ("%.5f" % (dt))
if not os.path.isdir((scenario_output_folder)):
    os.mkdir(scenario_output_folder)

#open the scenario
agents, obstacles = open_scenario("scenarios",scenario_name)
pop_num = len(agents)
number_of_time_steps = int(np.ceil(time_amount / dt))

world_x_length = 18
world_y_length = 18
if obstacles is not None:
    world_x_length, world_y_length = get_world_dimensions_from_walls(obstacles)
grid_shape = (int(np.ceil(world_x_length/MAC_cell_width)),int(np.ceil(world_y_length/MAC_cell_width)))

#set up data output
crowd_data_dir = os.path.join(scenario_output_folder, "crowd_data")
if not os.path.isdir(crowd_data_dir):
    os.mkdir(crowd_data_dir)
air_data_dir = os.path.join(scenario_output_folder, "air_data")
if not os.path.isdir(air_data_dir):
    os.mkdir(air_data_dir)

#set up air and disease grid
air_and_disease_grid = PDE.grid(grid_shape, MAC_cell_width, density_threshold, number_of_buffer_layers, disease_diffusivity_constant, disease_die_off_rate)

#set up the solid cells in the air and disease grid
for cell_list in air_and_disease_grid.cell_table.values():
    for cell in cell_list:
        cell_pos = np.array([cell.x, cell.y])
        for obstacle in obstacles:
            sd, _ = obstacle_signed_distance_and_gradient(obstacle, cell_pos)
            if sd <= 0.:
                cell.type=2

#main loop
for i in np.arange(number_of_time_steps):
    print("Step:",i,"out of",number_of_time_steps)
    save_crowd_data(i, agents, crowd_data_dir)
    air_and_disease_grid.save_data(i, air_data_dir)

    #update crowd velocities
    #TODO restore agent velocities implicit solve
    #velocities_implicit_solve(agents) #implicit solve for agent velocities

    #infect people
    infect_people(agents, air_and_disease_grid)

    #TODO: restore airflow solve
    '''#update airflow velocities
    air_and_disease_grid.backwards_velocity_trace(dt)
    air_and_disease_grid.set_airflow_in_cells() #this applies forces to the airflow from talking, singing, coughing, sneezing, breezes, and HVAC
    air_and_disease_grid.enforce_velocity_boundary_conditions()
    air_and_disease_grid.velocity_divergence_check("before pressure", dt)#TODO: delete this
    assignment_list = air_and_disease_grid.matrix_solve("pressure", dt)
    air_and_disease_grid.apply_pressures(assignment_list, dt)
    air_and_disease_grid.velocity_divergence_check("after pressure", dt)#TODO: delete this
    air_and_disease_grid.enforce_velocity_boundary_conditions()
    air_and_disease_grid.velocity_divergence_check("after second enforcement", dt)#TODO: delete this'''

    #update disease concentration
    air_and_disease_grid.advect_disease_densities(dt)
    air_and_disease_grid.reset_cell_types()
    if len(air_and_disease_grid.active_disease_and_buffer_cells.values()) > 0:
        air_and_disease_grid.matrix_solve("disease_concentration", dt)
    spread_infection_to_air(agents, air_and_disease_grid)
    air_and_disease_grid.reset_cell_types()
    air_and_disease_grid.row_assign_reset()

    #update crowd positions
    for j in np.arange(pop_num):
        agents[j].update_position()
save_crowd_data(number_of_time_steps, agents, crowd_data_dir)
air_and_disease_grid.save_data(number_of_time_steps, air_data_dir)

sim_end = time.perf_counter_ns()
time_to_run_sims = (sim_end - sim_start) / 1e9
print('Time to run simulations:', time_to_run_sims, 's\t\t(', time_to_run_sims/60., 'm)\t\t(', time_to_run_sims/3600., 'h)')
print("Finished simulation. Moving to drawing.")

draw_start = time.perf_counter_ns()
time_steps_per_frame = 1./(dt*frame_rate)
images_dir = os.path.join(scenario_output_folder, "images")
if not os.path.isdir(images_dir):
    os.mkdir(images_dir)
#TODO: simulate up to frame instead of interpolating?
draw_data.draw_data(crowd_data_dir, images_dir, agent_radius, number_of_time_steps, time_steps_per_frame, "crowd", obstacles)
draw_data.draw_data(air_data_dir, images_dir, MAC_cell_width/2, number_of_time_steps, time_steps_per_frame, "air", obstacles)
draw_data.overlay_disease_and_crowd(number_of_time_steps, images_dir)
draw_end = time.perf_counter_ns()
time_to_run_draw = draw_end - draw_start
print('Time to draw:', time_to_run_draw, 's\t\t(', time_to_run_draw/60., 'm)\t\t(', time_to_run_draw/3600., 'h)')
print("Finished drawing. Making a video.")


#make the videos
draw_data.make_video(frame_rate, scenario_output_folder, image_type_name="_disease")
draw_data.make_video(frame_rate, scenario_output_folder, image_type_name="_velocity")
draw_data.make_video(frame_rate, scenario_output_folder, image_type_name="_crowd")
draw_data.make_video(frame_rate, scenario_output_folder, image_type_name="_combined")
print("Done")


