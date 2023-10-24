import numpy as np
import os
import file_handling
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import cg




src_x,src_y = 0,0#-35,0 #temp coords of source TODO: delete when replacing with better sources


class grid_cell:
    def __init__(self, i, j, MAC_cell_width):
        self.i = i
        self.j = j

        self.x = i * MAC_cell_width
        self.y = j * MAC_cell_width

        self.layer = -1
        self.type = 0   #three types: clean air=0, diseased air=1, and solid=2
        self.bound_type = 0    #three types: interior cell = 0, boundary cell = 1

        #values in reused functions
        self.vals = dict()
        self.vals["velocity_x"] = 0.        #x coord of velocity located at (self.x-0.5*MAC_cell_width, self.y)
        self.vals["velocity_y"] = 0.        #y coord of velocity located at (self.x, self.y-0.5*MAC_cell_width)
        self.vals["temp_velocity_x"] = 0.   #for updating the velocity
        self.vals["temp_velocity_y"] = 0.   #for updating the velocity
        self.vals["disease_concentration"] = 0.
        self.vals["temp_disease_concentration"] = np.array([0., 0.]) #for updating the disease density
        #self.vals["curl"] = np.array([0., 0.]) #for calculating the vorticity confinement force
        self.vals["pressure"] = 0.
        self.vals["place_in_disease_concentration_matrix"] = None
        self.vals["place_in_pressure_matrix"] = None

        #pointers to neighboring cells. b=below, a=above
        self.neighbor_id = None
        self.neighbor_iu = None
        self.neighbor_jd = None
        self.neighbor_ju = None
        self.neighbor_ijd = None
        self.neighbor_iju = None


def hash_grid_coords(i, j):
    return 541*i + 79*j

class grid:
    def __init__(self, grid_shape, MAC_cell_width, density_threshold, number_of_buffer_layers, disease_diffusivity_constant, disease_die_off_rate):
        self.MAC_cell_width = MAC_cell_width
        self.density_threshold = density_threshold
        self.number_of_buffer_layers = number_of_buffer_layers
        self.disease_diffusivity_constant = disease_diffusivity_constant
        self.disease_die_off_rate = disease_die_off_rate

        self.min_i = -grid_shape[0] / 2
        self.max_i = grid_shape[0] / 2 - 1
        self.min_j = -grid_shape[1] / 2
        self.max_j = grid_shape[1] / 2 - 1

        self.cell_table = dict()
        size_side1, size_side2 = grid_shape
        half_way1 = size_side1/2
        half_way2 = size_side2/2
        for i_raw in np.arange(size_side1):
            i = i_raw-half_way1
            for j_raw in np.arange(size_side2):
                j = j_raw-half_way2
                cell = grid_cell(i,j, self.MAC_cell_width)
                cell.vals["velocity_x"] = 0.
                cell.vals["velocity_y"] = 0.
                hash_val = hash_grid_coords(i, j)
                if hash_val in self.cell_table:
                    self.cell_table[hash_val].append(cell)
                else:
                    self.cell_table[hash_val] = [cell]

                # configure the new cell's neighbors:
                # neighbor in the -x direction
                neighbor_id = self.get_cell(i - 1, j)
                if neighbor_id is not None:
                    cell.neighbor_id = neighbor_id
                    neighbor_id.neighbor_iu = cell
                # neighbor in the +x direction
                neighbor_iu = self.get_cell(i + 1, j)
                if neighbor_iu is not None:
                    cell.neighbor_iu = neighbor_iu
                    neighbor_iu.neighbor_id = cell
                # neighbor in the -y direction
                neighbor_jd = self.get_cell(i, j - 1)
                if neighbor_jd is not None:
                    cell.neighbor_jd = neighbor_jd
                    neighbor_jd.neighbor_ju = cell
                # neighbor in the +y direction
                neighbor_ju = self.get_cell(i, j + 1)
                if neighbor_ju is not None:
                    cell.neighbor_ju = neighbor_ju
                    neighbor_ju.neighbor_jd = cell
                # neighbor in the -x and -y direction
                neighbor_ijd = self.get_cell(i - 1, j - 1)
                if neighbor_ijd is not None:
                    cell.neighbor_ijd = neighbor_ijd
                    neighbor_ijd.neighbor_iju = cell
                # neighbor in the +x and +y direction
                neighbor_iju = self.get_cell(i + 1, j + 1)
                if neighbor_iju is not None:
                    cell.neighbor_iju = neighbor_iju
                    neighbor_iju.neighbor_ijd = cell

        #set solid boundary
        for cell_list in self.cell_table.values():
            for cell in cell_list:
                if cell.j == self.min_j or cell.j == self.max_j:
                    cell.type=2
                elif cell.i == self.min_i:
                    if cell.j < 0.5*(self.min_j+self.max_j):
                        #cell.type=2#
                        cell.bound_type=1
                    else:
                        cell.type=2
                elif cell.i == self.max_i:
                    if cell.j > 0.5*(self.min_j+self.max_j):
                        #cell.type=2#
                        cell.bound_type=1
                    else:
                        cell.type=2


        #set temporary source   #TODO get rid of this as part of coordinating with main simulation
        origin_cell = self.get_cell(src_x,src_y)
        origin_cell.vals["disease_concentration"]=0.25
        origin_cell.type=1

        self.active_disease_and_buffer_cells = dict()
        hash_val = hash_grid_coords(origin_cell.i, origin_cell.j)
        self.active_disease_and_buffer_cells[hash_val] = [origin_cell]
        self.reset_cell_types()


    def get_cell(self, i, j):
        hash_val = hash_grid_coords(i, j)
        if hash_val not in self.cell_table:
            return None
        cells = self.cell_table[hash_val]
        for cell in cells:
            if cell.i == i and cell.j == j:
                return cell
        return None


    def reset_cell_types(self):
        #reset all cell layers
        cells_in_current_layer = []
        for cell_list in self.active_disease_and_buffer_cells.values():
            for cell in cell_list:
                if cell.vals["disease_concentration"] < self.density_threshold:
                    cell.type=0
                    cell.layer = -1
                else:
                    cell.type=1
                    cell.layer = 0
                    cells_in_current_layer.append(cell)
        #add buffer cells by layer
        for layer in np.arange(self.number_of_buffer_layers):
            cells_in_next_layer = []
            for cell in cells_in_current_layer:
                neighbor_list = [cell.neighbor_id, cell.neighbor_iu, cell.neighbor_jd, cell.neighbor_ju]
                for neighbor_cell in neighbor_list:
                    if neighbor_cell is not None:
                        if neighbor_cell.layer == -1 and neighbor_cell.type != 2:
                            hash_val = hash_grid_coords(neighbor_cell.i, neighbor_cell.j)
                            if hash_val not in self.active_disease_and_buffer_cells:
                                self.active_disease_and_buffer_cells[hash_val] = [neighbor_cell]
                            elif neighbor_cell not in self.active_disease_and_buffer_cells[hash_val]:
                                self.active_disease_and_buffer_cells[hash_val].append(neighbor_cell)
                            neighbor_cell.layer = layer + 1
                            cells_in_next_layer.append(neighbor_cell)
                        cells_in_next_layer.append(neighbor_cell)
            cells_in_current_layer = cells_in_next_layer

        #delete cells from active if they are beyond the buffer
        cells_to_remove = []
        for cell_list in self.active_disease_and_buffer_cells.values():
            for cell in cell_list:
                if cell.layer==-1:
                    cells_to_remove.append(cell)
        for cell in cells_to_remove:
            hash_val = hash_grid_coords(cell.i, cell.j)
            self.active_disease_and_buffer_cells[hash_val].remove(cell)

    def interpolate_value(self, coords, value_name):
        x = coords[0] / self.MAC_cell_width
        y = coords[1] / self.MAC_cell_width
        i = int(np.floor(x))
        j = int(np.floor(y))

        result = 0.
        weight = 0.
        cell_00 = self.get_cell(i, j)
        if cell_00 is not None:
            cell_00_weight = (i+1-x) * (j+1-y)
            cell_00_term =  cell_00_weight* cell_00.vals[value_name]
            #print(cell_00.vals[value_name],cell_00_weight,i,x,j,y)
            result += cell_00_term
            weight += cell_00_weight
        cell_01 = self.get_cell(i+1, j)
        if cell_01 is not None:
            cell_01_weight = (x-i) * (j+1-y)
            cell_01_term =  cell_01_weight* cell_01.vals[value_name]
            #print(cell_01.vals[value_name],cell_01_weight)
            result += cell_01_term
            weight += cell_01_weight
        cell_10 = self.get_cell(i, j+1)
        if cell_10 is not None:
            cell_10_weight = (i+1-x) * (y-j)
            cell_10_term =  cell_10_weight* cell_10.vals[value_name]
            #print(cell_10.vals[value_name],cell_10_weight)
            result += cell_10_term
            weight += cell_10_weight
        cell_11 = self.get_cell(i+1, j+1)
        if cell_11 is not None:
            cell_11_weight = (x-i) * (y-j)
            cell_11_term =  cell_11_weight* cell_11.vals[value_name]
            #print(cell_11.vals[value_name],cell_11_weight)
            result += cell_11_term
            weight += cell_11_weight

        if weight != 0:
            result /= weight
        #print(result)
        #print()

        return result


    def update_velocities_from_temp_velocities(self):
        '''update the velocities'''
        for cell_list in self.cell_table.values():
            for cell in cell_list:
                cell.vals["velocity_x"] = cell.vals["temp_velocity_x"]
                cell.vals["velocity_y"] = cell.vals["temp_velocity_y"]
                cell.vals["temp_velocity_x"] = 0.
                cell.vals["temp_velocity_y"] = 0.

    def backwards_velocity_trace(self, dt):
        #advect
        coord_names = ["velocity_x","velocity_y"]
        for cell_list in self.cell_table.values():
            for cell in cell_list:
                for coord_i in np.arange(2):
                    coord_name = coord_names[coord_i]
                    velocity_location = np.array([cell.x, cell.y])
                    velocity_location[coord_i] = velocity_location[coord_i] - 0.5 * self.MAC_cell_width

                    current_location_velocity = np.array([self.interpolate_value(velocity_location, "velocity_x"), self.interpolate_value(velocity_location, "velocity_y")])

                    temp_location = velocity_location - 0.5*dt*current_location_velocity
                    temp_location_velocity = np.array([self.interpolate_value(temp_location, "velocity_x"), self.interpolate_value(temp_location, "velocity_y")])

                    new_velocity_location = velocity_location - dt*temp_location_velocity
                    cell.vals["temp_"+coord_name] = self.interpolate_value(new_velocity_location, coord_name)

        #do the update
        self.update_velocities_from_temp_velocities()

    def advect_disease_densities(self, dt):
        #advect
        for cell_list in self.active_disease_and_buffer_cells.values():
            for cell in cell_list:
                location = np.array([cell.x, cell.y])
                velocity = np.array([self.interpolate_value(location, "velocity_x"), self.interpolate_value(location, "velocity_y")])
                temp_location = location - 0.5 * dt * velocity
                temp_location_velocity = np.array([self.interpolate_value(temp_location, "velocity_x"), self.interpolate_value(temp_location, "velocity_y")])
                new_disease_concentration_location = location - dt * temp_location_velocity
                cell.vals["temp_disease_concentration"] = self.interpolate_value(new_disease_concentration_location, "disease_concentration")

        #do the update
        for cell_list in self.active_disease_and_buffer_cells.values():
            for cell in cell_list:
                cell.vals["disease_concentration"] = cell.vals["temp_disease_concentration"]
                cell.vals["temp_disease_concentration"] = 0.

                #dissipation - in this case disease die-off outside of humans
                cell.vals["disease_concentration"] = max(0., cell.vals["disease_concentration"] - self.disease_die_off_rate * dt)

        self.get_cell(src_x,src_y).vals["disease_concentration"]=0.25 #for temporary source cell  #TODO please delete once no longer necessary, and replace with a different mechanism for sources.

    def find_gradient(self, cell, value_name):
        '''Find gradient for the value in the given cell.
         Done using backward difference.
         Note: returned gradient values are not divided by cell width.'''
        id_val = 0.#cell.vals[value_name]
        if cell.neighbor_id is not None:
            id_val = cell.neighbor_id.vals[value_name]
            if cell.neighbor_id.type == 2:
                id_val = cell.vals[value_name]      #counter from solid neighbor
        jd_val = 0.#cell.vals[value_name]
        if cell.neighbor_jd is not None:
            jd_val = cell.neighbor_jd.vals[value_name]
            if cell.neighbor_jd.type == 2:
                jd_val = cell.vals[value_name]      #counter from solid neighbor
        return np.array([
            cell.vals[value_name] - id_val,
            cell.vals[value_name] - jd_val
        ])

    def find_velocity_divergence(self, cell):
        '''Find divergence for the value in the given cell.
         Done using forward difference.
         Note: returned divergence values are not divided by cell width.'''
        iu_velocity = 0.#cell.vals["velocity_x"]
        if cell.neighbor_iu is not None:
            iu_velocity = cell.neighbor_iu.vals["velocity_x"]
            #if cell.neighbor_iu.type==2 and iu_velocity!=0.:
            #    print("solid-fluid velocity iu:",iu_velocity)
        ju_velocity = 0.#cell.vals["velocity_y"]
        if cell.neighbor_ju is not None:
            ju_velocity = cell.neighbor_ju.vals["velocity_y"]
            #if cell.neighbor_iu.type==2 and ju_velocity!=0:
            #    print("solid-fluid velocity ju:",ju_velocity)
        #if cell.neighbor_id is not None:
        #    if cell.neighbor_id.type==2 and cell.vals["velocity_x"]!=0:
        #        print("solid-fluid velocity id:",cell.vals["velocity_x"])
        #if cell.neighbor_jd is not None:
        #    if cell.neighbor_jd.type==2 and cell.vals["velocity_y"]!=0:
        #        print("solid-fluid velocity jd:",cell.vals["velocity_y"])
        return iu_velocity + ju_velocity - cell.vals["velocity_x"] - cell.vals["velocity_y"]


    def find_Laplacian(self, cell, value_name):
        '''Find Laplacian for the value in the given cell.
         Note: returned Laplacian values are not divided by cell width^2.'''
        iu_value = 0.#cell.vals[value_name]
        if cell.neighbor_iu is not None:
            iu_value = cell.neighbor_iu.vals[value_name][0]
        ju_value = 0.#cell.vals[value_name]
        if cell.neighbor_ju is not None:
            ju_value = cell.neighbor_ju.vals[value_name][1]
        id_value = 0.#cell.vals[value_name]
        if cell.neighbor_id is not None:
            id_value = cell.neighbor_id.vals[value_name][0]
        jd_value = 0.#cell.vals[value_name]
        if cell.neighbor_jd is not None:
            jd_value = cell.neighbor_jd.vals[value_name][1]
        return iu_value + id_value + ju_value + jd_value - 4 * cell.vals[value_name]

    def row_assign_for_matrix_solve(self, type_name):
        assignment_list = []
        if type_name == "disease_concentration":
            cell_list_list = self.active_disease_and_buffer_cells.values()
        else:
            cell_list_list = self.cell_table.values()
        for cell_list in cell_list_list:
            for cell in cell_list:
                if cell.type !=2 and cell.bound_type != 1:
                    assignment_list.append(cell)
                    cell.vals[f"place_in_{type_name}_matrix"] = len(assignment_list) - 1
        num_cells = len(assignment_list)
        return assignment_list, num_cells

    def row_assign_reset(self):
        for cell_list in self.cell_table.values():
            for cell in cell_list:
                cell.vals["place_in_disease_concentration_matrix"] = None
                cell.vals["place_in_pressure_matrix"] = None


    def matrix_solve(self, type_name, dt):
        #type_name is "disease_concentration" or "pressure"

        #assign a row to each cell
        assignment_list, num_cells = self.row_assign_for_matrix_solve(type_name)

        A = np.zeros((num_cells, num_cells))
        b = np.zeros((num_cells))

        # add rows to A and b
        for i in np.arange(num_cells):
            cell = assignment_list[i]
            neighbors = [cell.neighbor_id, cell.neighbor_iu, cell.neighbor_jd, cell.neighbor_ju]
            neighbor_mask = [1,1,1,1]

            for j in np.arange(4):
                if neighbors[j] is None:
                    neighbor_mask[j] = 0
                elif neighbors[j].type==2:#neighbors[j].vals[f"place_in_{type_name}_matrix"] is None:
                    neighbor_mask[j] = 0

            A[i][i] = -sum(neighbor_mask)
            for j in np.arange(4):
                if neighbors[j] is not None:
                    if neighbors[j].vals[f"place_in_{type_name}_matrix"] is not None:
                        A[i][neighbors[j].vals[f"place_in_{type_name}_matrix"]] = 1.

            #set up b
            if type_name=="disease_concentration":
                b[i] = -1*cell.vals["disease_concentration"]
            if type_name=="pressure":
                # calculate cell's divergence and add it to b
                divergence = self.find_velocity_divergence(cell)
                b[i] = self.MAC_cell_width * divergence / dt

        if type_name == "disease_concentration":
            A *= dt * self.disease_diffusivity_constant / (self.MAC_cell_width * self.MAC_cell_width)
            A -= np.identity(A.shape[0])

        #do matrix solve
        #print(A)
        #print("det A:", np.linalg.det(A))
        if sum(sum(A-A.T)) != 0.:
            print(f"{type_name} solve matrix not symmetric")
            exit(1)
        x, result = cg(csc_matrix(A), b, tol=1e-8)
        if result != 0:
            print("CG failed to converge. Result =",result)
            exit(1)
        if np.linalg.norm(A.dot(x) - b) > 0.000001*num_cells:
            print("CG result is not precise (disease solve).")
            print("\tnorm =",np.linalg.norm(A.dot(x) - b))
            print("\tthreshold =",0.0000001*num_cells)
            #print(A,"\n",x,"\n",b)
            #print(np.linalg.norm(x))
            #exit(1)

        if type_name == "disease_concentration":
            print("x:",sum(x),"\t\t-b:",sum(-b),"\t\tloss:",sum(-b)-sum(x),"\t\t% loss:",100.*(sum(-b)-sum(x))/sum(-b))
            #TODO with large time steps, disease dissipates too quickly

        if type_name == "pressure":
            #print(A, "\n", b, "\n", x)
            print("laplacian of pressure - adjusted divergences:\t", np.linalg.norm(A.dot(x) - b))
            print("pressure size:",np.linalg.norm(x))

        for i in np.arange(num_cells):
            cell = assignment_list[i]
            cell.vals[type_name] = x[i]

        return assignment_list


    def apply_pressures(self, pressure_solve_cell_list, dt):
        #apply pressures: update cells' velocities
        for i in np.arange(len(pressure_solve_cell_list)):
            cell = pressure_solve_cell_list[i]
            change_val = dt / self.MAC_cell_width * self.find_gradient(cell, "pressure")
            cell.vals["velocity_x"] = cell.vals["velocity_x"] - change_val[0]  #NOTE: using -= causes all cells to update simultaneously. I don't know why this happens. Beware.
            cell.vals["velocity_y"] = cell.vals["velocity_y"] - change_val[1]  #NOTE: using -= causes all cells to update simultaneously. I don't know why this happens. Beware.
            #if np.linalg.norm(-change_val) > 3.:
            #    print("change > 3 at",cell.i, cell.j)
            #    exit()


    def enforce_velocity_boundary_conditions(self):
        for cell_row in self.cell_table.values():
            for cell in cell_row:
                #zero velocity on boundaries of all solid cells
                if cell.type==2:
                    cell.vals["velocity_x"]= 0.
                    cell.vals["velocity_y"]= 0.
                    if cell.neighbor_iu is not None:
                        cell.neighbor_iu.vals["velocity_x"]=0.
                    if cell.neighbor_ju is not None:
                        cell.neighbor_ju.vals["velocity_y"]=0.

                #boundary cells have no velocity where they do not border interior cells
                if cell.bound_type==1:
                    if cell.neighbor_id is None:
                        cell.vals["velocity_x"] = 0.
                    if cell.neighbor_jd is None:
                        cell.vals["velocity_y"] = 0.
    def set_airflow_in_cells(self, time_step, number_of_time_steps):
        #TODO please delete once no longer necessary, and replace with a different mechanism for sources.
        # Specifically, add a velocity to cells where someone is coughing/sneezing/talking/singing.
        #for area around and including temporary source cell
        if time_step < 0.2*number_of_time_steps:
            print("SETTING VELOCITY IN AREA")
            source_cell = self.get_cell(src_x,src_y)
            source_coords = np.array([source_cell.x, source_cell.y])
            new_velocity = np.array([3., -1.])

            #source_cell.vals["velocity_x"] = new_velocity[0]
            #source_cell.vals["velocity_y"] = new_velocity[1]
            for cell_list in self.cell_table.values():
                for cell in cell_list:
                    cell_coords = np.array([cell.x, cell.y])
                    dist = np.linalg.norm(cell_coords-source_coords)
                    if dist < 0.3:
                        cell.vals["velocity_x"] = new_velocity[0]# * (1.-dist/0.3)
                        cell.vals["velocity_y"] = new_velocity[1]# * (1.-dist/0.3)



    def save_data(self, number, folder, extra_info=""):
        data = np.zeros((len(self.cell_table.values()), 5))
        cell_index = 0
        for cell_list in self.cell_table.values():
            for cell in cell_list:
                data[cell_index][0] = cell.x
                data[cell_index][1] = cell.y
                data[cell_index][2] = cell.vals["velocity_x"]
                data[cell_index][3] = cell.vals["velocity_y"]
                data[cell_index][4] = cell.vals["disease_concentration"]
                cell_index += 1
        header = "pos_x,pos_y,vel_x,vel_y,disease_concentration"
        file_handling.write_csv_file(os.path.join(folder, extra_info+str(number)+".csv"), header, data)


    def velocity_divergence_check(self,message,dt):
        print(message)
        velocity_sum = 0.
        divergence_sum = 0.
        abs_divergence_sum = 0.
        divergences = []
        velocities_x = []
        velocities_y = []
        near_boundary_divergences = []
        near_boundary_divergences_sum = 0.
        abs_near_boundary_divergences_sum = 0.
        max_vel = 0.
        for cell_row in self.cell_table.values():
            for cell in cell_row:
                if cell.type!=2 and cell.bound_type!=1:
                    vel = np.linalg.norm(np.array([cell.vals["velocity_x"], cell.vals["velocity_y"]]))
                    velocity_sum += vel
                    if vel > max_vel:
                        max_vel = vel
                    divergence = self.find_velocity_divergence(cell)
                    divergence_sum += divergence
                    abs_divergence_sum += abs(divergence)
                    divergences.append(divergence)
                    velocities_x.append(cell.vals["velocity_x"])
                    velocities_y.append(cell.vals["velocity_y"])
                    if cell.i == self.max_i-1 or cell.i==self.min_i+1 or cell.j == self.min_j+1 or cell.j == self.max_j-1:
                        near_boundary_divergences.append(divergence)
                        near_boundary_divergences_sum += divergence
                        abs_near_boundary_divergences_sum += abs(divergence)

        print("velocity sum:",velocity_sum,"\t\tdivergence sum:",divergence_sum,
              "\t\tabs divergence sum:",abs_divergence_sum,"\t\taverage abs_divergence_sum",abs_divergence_sum/len(divergences),
              "\nabs divergence sum / velocity sum:",abs_divergence_sum/velocity_sum,
              "\nmax_vel:",max_vel
              #"\nstd dev abs_divergence_sum",np.std(np.abs(np.array(divergences))),
              #"\t\tworst abs_divergence", np.max(np.abs(np.array(divergences))),
              #"\nnear boundary divergence sum:",near_boundary_divergences_sum,
              #"\t\tnb abs divergence sum:",abs_near_boundary_divergences_sum,"\t\taverage nb abs_divergence_sum",abs_near_boundary_divergences_sum/len(near_boundary_divergences)
              )
        if max_vel > 3.*self.MAC_cell_width/dt:
            print("Possible CFL violation!!!")

