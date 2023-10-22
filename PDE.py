import numpy as np
import os
import file_handling
import draw_data
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import cg
#import PCG

#TODO: maybe remove all physical units from simulation solve steps, and add them in afterwards?

width = 0.1     #each MAC cell is 0.1 meters on each side
density_threshold = 0.003
number_of_buffer_layers = 3
atmospheric_pressure = 1.

baseline_velocity = np.array([0.,0.])#np.array([3.,-1.])#np.array([1.,-1.])#np.array([3.,0.])# #TODO replace with airflow
disease_diffusivity_constant = 0.07


#TODO: coordinate these with main simulation
dt = 0.01
number_of_time_steps = 300#1000
#vorticity_confinement_force_strength = 1.
disease_die_off_rate = 0.
grid_shape = (80,80)#(8,8)#(256,256)#
min_i = -grid_shape[0] / 2
max_i = grid_shape[0] / 2 - 1
min_j = -grid_shape[1] / 2
max_j = grid_shape[1] / 2 - 1
src_x,src_y = 0,0#-35,0 #temp coords of source TODO: delete when replacing with better sources


#TODO: handling boundary conditions. This includes enforcement with respect to solids, and ???


class grid_cell:
    def __init__(self, i, j):
        self.i = i
        self.j = j

        self.x = i*width
        self.y = j*width

        self.layer = -1
        self.type = 0   #three types: clean air=0, diseased air=1, and solid=2
        self.bound_type = 0    #three types: ordinary cell = 0, airflow source = 1, airflow sink = 2

        #values in reused functions
        self.vals = dict()
        self.vals["velocity_x"] = 0.        #x coord of velocity located at (self.x-0.5*width, self.y)
        self.vals["velocity_y"] = 0.        #y coord of velocity located at (self.x, self.y-0.5*width)
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
    def __init__(self, grid_shape):
        #TODO need to initiate cell_table with cells for sources. Sinks are gas cells right outside the bounds of the simulation.
        #   To be clear, a source cell is a regular cell except for the fact that it is close to a contagious person, and is needed to track that person's disease output
        #       Perhaps we can have an external force at the source location, to push the disease out of the person's mouth.
        self.cell_table = dict()
        size_side1, size_side2 = grid_shape
        half_way1 = size_side1/2
        half_way2 = size_side2/2
        for i_raw in np.arange(size_side1):
            i = i_raw-half_way1
            for j_raw in np.arange(size_side2):
                j = j_raw-half_way2
                cell = grid_cell(i,j)
                cell.vals["velocity_x"] = baseline_velocity[0] #initialize uniform velocities for all cells. TODO: allow alternate velocity field initilization options.
                cell.vals["velocity_y"] = baseline_velocity[1]
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
                if cell.j == min_j or cell.j == max_j:
                    cell.type=2
                elif cell.i == min_i:
                    if cell.j < 0.5*(min_j+max_j):
                        #cell.type=2#
                        cell.bound_type=2
                    else:
                        cell.type=2
                elif cell.i == max_i:
                    if cell.j > 0.5*(min_j+max_j):
                        #cell.type=2#
                        cell.bound_type=2
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
        # TODO: elsewhere, make sure that solid cells have density of 0. Assuming active_disease_and_buffer_cells has no solid cells.
        #reset all cell layers
        cells_in_current_layer = []
        for cell_list in self.active_disease_and_buffer_cells.values():
            for cell in cell_list:
                if cell.vals["disease_concentration"] < density_threshold:
                    cell.type=0
                    cell.layer = -1
                else:
                    cell.type=1
                    cell.layer = 0
                    cells_in_current_layer.append(cell)
        #add buffer cells by layer
        for layer in np.arange(number_of_buffer_layers):
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
        #TODO: consider switching to cubic interpolation (pages 39-42 in the Bridson book)
        x = coords[0] / width
        y = coords[1] / width
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

    def backwards_velocity_trace(self):
        #TODO: check to make sure not doing this for solids.
        #advect
        coord_names = ["velocity_x","velocity_y"]
        for cell_list in self.cell_table.values():
            for cell in cell_list:
                for coord_i in np.arange(2):
                    coord_name = coord_names[coord_i]
                    velocity_location = np.array([cell.x, cell.y])
                    velocity_location[coord_i] = velocity_location[coord_i] - 0.5*width

                    current_location_velocity = np.array([self.interpolate_value(velocity_location, "velocity_x"), self.interpolate_value(velocity_location, "velocity_y")])

                    temp_location = velocity_location - 0.5*dt*current_location_velocity
                    temp_location_velocity = np.array([self.interpolate_value(temp_location, "velocity_x"), self.interpolate_value(temp_location, "velocity_y")])

                    new_velocity_location = velocity_location - dt*temp_location_velocity
                    cell.vals["temp_"+coord_name] = self.interpolate_value(new_velocity_location, coord_name)

        #do the update
        self.update_velocities_from_temp_velocities()

    def advect_disease_densities(self):
        #TODO: not in solids
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
                cell.vals["disease_concentration"] = max(0., cell.vals["disease_concentration"] - disease_die_off_rate * dt)

        self.get_cell(src_x,src_y).vals["disease_concentration"]=0.25 #for temporary source cell  #TODO please delete once no longer necessary, and replace with a different mechanism for sources.

        '''#sum density
        sum_density = 0.
        for cell_list in self.cell_table.values():
            for cell in cell_list:
                sum_density += cell.vals["disease_concentration"]
        print(sum_density)'''

    #TODO consider adding this in the non-barebones version
    '''def vorticity_confinement_force(self):
        x_shift = np.array([0.5*width,0.])
        y_shift = np.array([0.,0.5*width])
        for cell_list in self.cell_table.values():
            for cell in cell_list:
                cell_center = np.array([cell.x, cell.y])
                if cell.type != 2:
                    vel_x_on_y_0 = self.interpolate_value(cell_center-y_shift,"velocity_x")
                    vel_x_on_y_1 = self.interpolate_value(cell_center+y_shift,"velocity_x")
                    vel_y_on_x_0 = self.interpolate_value(cell_center-x_shift,"velocity_y")
                    vel_y_on_x_1 = self.interpolate_value(cell_center+x_shift,"velocity_y")
                    cell.vals["curl"] = ((vel_y_on_x_1 - vel_y_on_x_0) - (vel_x_on_y_1 - vel_x_on_y_0)) / (0.5*width)'''

    def find_gradient(self, cell, value_name):
        '''Find gradient for the value in the given cell.
         Done using backward difference.
         Note: returned gradient values are not divided by width.'''
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
         Note: returned divergence values are not divided by width.'''
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
         Note: returned Laplacian values are not divided by width^2.'''
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

    '''def set_pressure_to_boundary_conditions(self):
        for cell_list in self.cell_table.values():
            for cell in cell_list:
                if cell.i==min_i:
                    cell.vals["pressure"] = self.get_cell(cell.i+1,cell.j).vals["pressure"]
                if cell.i==max_i:
                    cell.vals["pressure"] = self.get_cell(cell.i-1,cell.j).vals["pressure"]
                if cell.j==min_j:
                    cell.vals["pressure"] = 0.#self.get_cell(cell.i,cell.j+1).vals["pressure"]
                if cell.j==max_j:
                    cell.vals["pressure"] = self.get_cell(cell.i,cell.j-1).vals["pressure"]'''

    def row_assign_for_matrix_solve(self, type_name):
        assignment_list = []
        if type_name == "disease_concentration":
            cell_list_list = self.active_disease_and_buffer_cells.values()
        else:
            cell_list_list = self.cell_table.values()
        for cell_list in cell_list_list:
            for cell in cell_list:
                if cell.type !=2 and cell.bound_type != 2:
                    assignment_list.append(cell)
                    cell.vals[f"place_in_{type_name}_matrix"] = len(assignment_list) - 1
        num_cells = len(assignment_list)
        return assignment_list, num_cells

    def row_assign_reset(self):
        for cell_list in self.cell_table.values():
            for cell in cell_list:
                cell.vals["place_in_disease_concentration_matrix"] = None
                cell.vals["place_in_pressure_matrix"] = None

    '''def setup_matrix_equation(self, type_name):
        #type_name is "disease_concentration" or "pressure"
        assignment_list, num_cells = self.row_assign_for_matrix_solve(type_name)

        A_diag = np.zeros((num_cells))
        A_x = np.zeros((num_cells))
        A_y = np.zeros((num_cells))
        b = np.zeros((num_cells))

        #set up A and b
        for i in np.arange(num_cells):
            cell = assignment_list[i]
            neighbors = [cell.neighbor_id, cell.neighbor_iu, cell.neighbor_jd, cell.neighbor_ju]
            neighbor_mask = [1, 1, 1, 1]

            for j in np.arange(4):
                if neighbors[j] is None:
                    neighbor_mask[j] = 0
                elif neighbors[j].vals[f"place_in_{type_name}_matrix"] is None:
                    neighbor_mask[j] = 0
                elif neighbors[j].type==2:
                    print("solid neighbor")
                    exit()
            A_diag[i] = sum(neighbor_mask)
            if neighbor_mask[1] > 0:
                A_x[neighbors[1].vals[f"place_in_{type_name}_matrix"]] = -1
            if neighbor_mask[3] > 0:
                A_y[neighbors[3].vals[f"place_in_{type_name}_matrix"]] = -1

            #set up b
            if type_name=="disease_concentration":
                b[i] = cell.vals["disease_concentration"]
            if type_name=="pressure":
                # calculate cell's divergence and subtract it from b
                divergence = -1*self.find_velocity_divergence(cell)
                b[i] = width * divergence / dt

        #make changes to A
        if type_name == "disease_concentration":
            scale = dt * disease_diffusivity_constant / (width * width)
            A_diag *= scale
            A_x *= scale
            A_y *= scale
            A_diag += np.ones((num_cells))  #np.identity(A.shape[0])

        return assignment_list, num_cells, A_diag, A_x, A_y, b

    def matrix_solve(self, type_name):
        assignment_list, num_cells, A_diag, A_x, A_y, b = self.setup_matrix_equation(type_name)
        tol = 1e-6
        x = PCG.PCG(tol, A_diag, A_x, A_y, b, assignment_list, type_name)

        for i in np.arange(num_cells):
            cell = assignment_list[i]
            cell.vals[type_name] = x[i]'''


    def matrix_solve(self, type_name):
        #type_name is "disease_concentration" or "pressure"
        #TODO consider replacing with a vector-based or grid-based solver. See pages 77-78 of Bridson book for the start of a section describing this.

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
                #for j in np.arange(4):
                #    if neighbors[j] is not None:
                #        if neighbors[j].bound_type==2:
                #            divergence -= atmospheric_pressure
                #if abs(divergence) > 0.:
                    #print(cell.i, cell.j, divergence)
                b[i] = width * divergence / dt

        if type_name == "disease_concentration":
            A *= dt * disease_diffusivity_constant / (width * width)
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

        if type_name == "pressure":
            #print(A, "\n", b, "\n", x)
            print("laplacian of pressure - adjusted divergences:\t", np.linalg.norm(A.dot(x) - b))
            print("pressure size:",np.linalg.norm(x))
            #self.set_pressure_to_boundary_conditions()

        for i in np.arange(num_cells):
            cell = assignment_list[i]
            cell.vals[type_name] = x[i]

        return assignment_list


    def apply_pressures(self, pressure_solve_cell_list):
        #apply pressures: update cells' velocities
        for i in np.arange(len(pressure_solve_cell_list)):
            cell = pressure_solve_cell_list[i]
            change_val = dt / width * self.find_gradient(cell, "pressure")
            cell.vals["velocity_x"] = cell.vals["velocity_x"] - change_val[0]  #NOTE: using -= causes all cells to update simultaneously. I don't know why this happens. Beware.
            cell.vals["velocity_y"] = cell.vals["velocity_y"] - change_val[1]  #NOTE: using -= causes all cells to update simultaneously. I don't know why this happens. Beware.
            #if np.linalg.norm(-change_val) > 3.:
            #    print("change > 3 at",cell.i, cell.j)
            #    exit()


    def enforce_velocity_boundary_conditions(self):
        #TODO: I think here is where the air-solid boundary conditions should be enforced. Enforce disease-solid boundary conditions in a separate function.
        #zero velocity on boundaries of all solid cells
        for cell_row in self.cell_table.values():
            for cell in cell_row:
                if cell.type==2:
                    cell.vals["velocity_x"]= 0.
                    cell.vals["velocity_y"]= 0.
                    if cell.neighbor_iu is not None:
                        cell.neighbor_iu.vals["velocity_x"]=0.
                    if cell.neighbor_ju is not None:
                        cell.neighbor_ju.vals["velocity_y"]=0.

                if cell.bound_type==2:
                    if cell.neighbor_id is None:
                        cell.vals["velocity_x"] = 0.
                    if cell.neighbor_jd is None:
                        cell.vals["velocity_y"] = 0.
    def set_airflow_in_cells(self, time_step):
        #TODO please delete once no longer necessary, and replace with a different mechanism for sources.
        #for area around and including temporary source cell
        if time_step < 0.2*number_of_time_steps:
            print("SETTING VELOCITY IN AREA")
            source_cell = self.get_cell(src_x,src_y)
            source_coords = np.array([source_cell.x, source_cell.y])
            new_baseline_velocity = np.array([3., -1.])

            #source_cell.vals["velocity_x"] = new_baseline_velocity[0]
            #source_cell.vals["velocity_y"] = new_baseline_velocity[1]
            for cell_list in self.cell_table.values():
                for cell in cell_list:
                    cell_coords = np.array([cell.x, cell.y])
                    dist = np.linalg.norm(cell_coords-source_coords)
                    if dist < 0.3:
                        cell.vals["velocity_x"] = new_baseline_velocity[0]# * (1.-dist/0.3)
                        cell.vals["velocity_y"] = new_baseline_velocity[1]# * (1.-dist/0.3)



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


def velocity_divergence_check(grid,message):
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
    for cell_row in grid.cell_table.values():
        for cell in cell_row:
            if cell.type!=2 and cell.bound_type!=2:
                vel = np.linalg.norm(np.array([cell.vals["velocity_x"], cell.vals["velocity_y"]]))
                velocity_sum += vel
                if vel > max_vel:
                    max_vel = vel
                divergence = grid.find_velocity_divergence(cell)
                divergence_sum += divergence
                abs_divergence_sum += abs(divergence)
                divergences.append(divergence)
                velocities_x.append(cell.vals["velocity_x"])
                velocities_y.append(cell.vals["velocity_y"])
                if cell.i == max_i-1 or cell.i==min_i+1 or cell.j == min_j+1 or cell.j == max_j-1:
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
    if max_vel > 3.*width/dt:
        print("Possible CFL violation!!!")

    #print("divergences:\n",np.round(np.array(divergences),4).reshape((grid_shape[0]-2,grid_shape[1]-2)))
    #print("velocities_x:\n",np.round(np.array(velocities_x),4).reshape((grid_shape[0]-2,grid_shape[1]-2)))
    #print("velocities_y:\n",np.round(np.array(velocities_y),4).reshape((grid_shape[0]-2,grid_shape[1]-2)))
    #print("velocity of cell -40, 40:",grid.get_cell(-40,-40).vals["velocity_x"],grid.get_cell(-40,-40).vals["velocity_y"])

#TODO: coordinate these with main simulation
time_steps_dir = "disease_sim_time_steps"

if not os.path.isdir(time_steps_dir):
    os.mkdir(time_steps_dir)

air_and_disease_grid = grid(grid_shape)

for i in np.arange(number_of_time_steps):
    air_and_disease_grid.save_data(i, time_steps_dir)

    #handle airflow
    air_and_disease_grid.backwards_velocity_trace()
    air_and_disease_grid.set_airflow_in_cells(i) #this applies forces to the airflow
    air_and_disease_grid.enforce_velocity_boundary_conditions()
    velocity_divergence_check(air_and_disease_grid,"before pressure")#TODO: delete this
    assignment_list = air_and_disease_grid.matrix_solve("pressure")
    air_and_disease_grid.apply_pressures(assignment_list)
    velocity_divergence_check(air_and_disease_grid,"after pressure")#TODO: delete this
    air_and_disease_grid.enforce_velocity_boundary_conditions()
    velocity_divergence_check(air_and_disease_grid,"after second enforcement")#TODO: delete this

    #handle disease concentration
    air_and_disease_grid.advect_disease_densities()
    air_and_disease_grid.reset_cell_types()
    air_and_disease_grid.matrix_solve("disease_concentration")
    air_and_disease_grid.reset_cell_types()

    air_and_disease_grid.row_assign_reset()

air_and_disease_grid.save_data(number_of_time_steps, time_steps_dir)


images_dir = os.path.join("images")
if not os.path.isdir(images_dir):
    os.mkdir(images_dir)

time_steps_per_frame = 1
draw_data.draw_data(time_steps_dir, images_dir, width/2, number_of_time_steps, time_steps_per_frame)

frame_rate = 1./(dt * time_steps_per_frame)
draw_data.make_video(frame_rate, ".", image_type_name="_disease")
draw_data.make_video(frame_rate, ".", image_type_name="_velocity")
print("Done")

