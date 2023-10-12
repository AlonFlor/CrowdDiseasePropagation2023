import numpy as np
import os
import file_handling
import draw_data
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg

'''
It is 1280 px / (40 px/m) = 32 m on one side.
that is 1024 m^2. If each MAC cell is 0.25 m on each side, there are 16 cells in each m^2, so there are 16384 cells in all.
Changing it so each MAC cell is 0.1 m on each side, there are 100 cells in each m^2, so there are 102400 cells in all.


Plan:
I am doing the smoke simulation instead of the liquid simulation.
    No particles; grid only
Air will be simulated as stated in the document, in the form of wind and pressure from the air cells.
'''

width = 0.1
density_threshold = 0.01
number_of_buffer_layers = 3
atmospheric_pressure = 1

#TODO: coordinate these with main simulation
dt = 0.01
vorticity_confinement_force_strength = 1.
disease_die_off_rate = 0.

class grid_cell:
    def __init__(self, i, j):
        self.i = i
        self.j = j

        self.x = i*width
        self.y = j*width

        self.layer = -1
        self.type = 0   #three types: clean air=0, diseased air=1, and solid=2
        self.bound_type = 0    #three types: ordinary cell = 0, source = 1, sink = 2

        self.disease_density = 0.
        self.pressure = 0.

        self.velocity = np.array([0., 0.])
        #x coord of velocity located at (self.x-0.5*width, self.y)
        #y coord of velocity located at (self.x, self.y-0.5*width)

        self.temp_velocity = np.array([0., 0.]) #for updating the velocity
        self.temp_disease_density = np.array([0., 0.]) #for updating the disease density
        #self.curl = np.array([0., 0.]) #for calculating the vorticity confinement force
        self.place_in_pressure_matrix = 0

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
    def __init__(self):
        #TODO need to initiate cell_table with cells for sources. Sinks are gas cells right outside the bounds of the simulation.
        #   To be clear, a source cell is a regular cell except for the fact that it is close to a contagious person, and is needed (ie do not delete) to track that person's disease output
        #       Perhaps we can have an external force at the source location, to push the disease out of the person's mouth.
        #TODO For now, since I just want somthing out, I will set a source at the origin.
        self.cell_table = dict()
        self.add_cell(0, 0)
        self.get_cell(0,0).disease_density=0.25
        self.get_cell(0,0).type=1
        self.get_cell(0,0).bound_type=1 #does nothing so far
        self.get_cell(0,0).velocity += np.array([0.4, 0.2])

    def get_cell(self, i, j):
        hash_val = hash_grid_coords(i, j)
        if hash_val not in self.cell_table:
            return None
        cells = self.cell_table[hash_val]
        for cell in cells:
            if cell.i == i and cell.j == j:
                return cell
        return None
    def delete_cell(self, cell):
        hash_val = hash_grid_coords(cell.i, cell.j)
        self.cell_table[hash_val].pop(self.cell_table[hash_val].index(cell))

    def add_cell(self, i, j):
        cell = grid_cell(i, j)
        hash_val = hash_grid_coords(cell.i, cell.j)
        if hash_val in self.cell_table:
            self.cell_table[hash_val].append(cell)
        else:
            self.cell_table[hash_val] = [cell]
        # TODO: check if neighbor_cell is in a wall. If so, set its type to 2 (solid).

        #configure the new cell's neighbors:
        #neighbor in the -x direction
        neighbor_id = self.get_cell(i-1, j)
        if neighbor_id is not None:
            cell.neighbor_id = neighbor_id
            neighbor_id.neighbor_iu = cell
        #neighbor in the +x direction
        neighbor_iu = self.get_cell(i+1, j)
        if neighbor_iu is not None:
            cell.neighbor_iu = neighbor_iu
            neighbor_iu.neighbor_id = cell
        #neighbor in the -y direction
        neighbor_jd = self.get_cell(i, j-1)
        if neighbor_jd is not None:
            cell.neighbor_jd = neighbor_jd
            neighbor_jd.neighbor_ju = cell
        #neighbor in the +y direction
        neighbor_ju = self.get_cell(i, j+1)
        if neighbor_ju is not None:
            cell.neighbor_ju = neighbor_ju
            neighbor_ju.neighbor_jd = cell
        #neighbor in the -x and -y direction
        neighbor_ijd = self.get_cell(i-1, j-1)
        if neighbor_ijd is not None:
            cell.neighbor_ijd = neighbor_ijd
            neighbor_ijd.neighbor_iju = cell
        #neighbor in the +x and +y direction
        neighbor_iju = self.get_cell(i+1, j+1)
        if neighbor_iju is not None:
            cell.neighbor_iju = neighbor_iju
            neighbor_iju.neighbor_ijd = cell

        return cell

    def update_grid(self):
        '''Update the grid based on disease densities'''
        cells_in_current_layer = []
        for cell_list in self.cell_table.values():
            for cell in cell_list:
                cell.layer = -1
                if cell.disease_density > density_threshold:
                    cell.type=1 #diseased air
                    cell.layer=0
                    cells_in_current_layer.append(cell)
        for index in np.arange(number_of_buffer_layers):
            cells_in_next_layer = []
            for cell in cells_in_current_layer:
                neighbor_list = [(cell.neighbor_id, (-1,0)), (cell.neighbor_iu, (1,0)), (cell.neighbor_jd, (0,-1)), (cell.neighbor_ju, (0,1))]
                for neighbor_cell, up_down_tuple in neighbor_list:
                    if neighbor_cell is not None:
                        if neighbor_cell.layer==-1 and neighbor_cell.type!=2:
                            neighbor_cell.layer = index+1
                            cells_in_next_layer.append(neighbor_cell)
                    else:
                        i = up_down_tuple[0] + cell.i
                        j = up_down_tuple[1] + cell.j
                        neighbor_cell = self.add_cell(i,j)
                        neighbor_cell.velocity = np.array([-3.,1.])
                        neighbor_cell.layer = index+1
                        #TODO: need something for sink cells, which are non-solid cells at the border of the simulation.
                        cells_in_next_layer.append(neighbor_cell)
            cells_in_current_layer = cells_in_next_layer

        #delete empty non-buffer cells
        for cell_list in self.cell_table.values():
            for cell in cell_list:
                if cell.layer == -1:
                    self.delete_cell(cell)

    def update_velocities_from_temp_velocities(self):
        '''update the velocities'''
        for cell_list in self.cell_table.values():
            for cell in cell_list:
                cell.velocity[:] = cell.temp_velocity[:]
                cell.temp_velocity *= 0.


    #TODO maybe make sure that interpolate_disease and interpolate_velocity produce the correct values.
    #   Maybe also find a way to merge them for brevity.
    #   Use cell neighbors stored in cell(i,j)'s list

    def interpolate_disease(self, coords):
        x = coords[0] / width
        y = coords[1] / width
        i = int(x)
        j = int(y)

        result = 0.
        weight = 0.
        cell_00 = self.get_cell(i, j)
        if cell_00 is not None:
            cell_00_weight = (i+1-x) * (j+1-y)
            cell_00_term =  cell_00_weight* cell_00.disease_density
            result += cell_00_term
            weight += cell_00_weight
        cell_01 = self.get_cell(i+1, j)
        if cell_01 is not None:
            cell_01_weight = (x-i) * (j+1-y)
            cell_01_term =  cell_01_weight* cell_01.disease_density
            result += cell_01_term
            weight += cell_01_weight
        cell_10 = self.get_cell(i, j+1)
        if cell_10 is not None:
            cell_10_weight = (i+1-x) * (y-j)
            cell_10_term =  cell_10_weight* cell_10.disease_density
            result += cell_10_term
            weight += cell_10_weight
        cell_11 = self.get_cell(i+1, j+1)
        if cell_11 is not None:
            cell_11_weight = (x-i) * (y-j)
            cell_11_term =  cell_11_weight* cell_11.disease_density
            result += cell_11_term
            weight += cell_11_weight

        if weight != 0:
            result /= weight

        return result

    def interpolate_velocity(self, coords, index):
        x = coords[0] / width
        y = coords[1] / width
        i = int(x)
        j = int(y)

        result = 0.
        weight = 0.
        cell_00 = self.get_cell(i, j)
        if cell_00 is not None:
            cell_00_weight = (i+1-x) * (j+1-y)
            cell_00_term =  cell_00_weight* cell_00.velocity[index]
            result += cell_00_term
            weight += cell_00_weight
        cell_01 = self.get_cell(i + 1, j)
        if cell_01 is not None:
            cell_01_weight = (x-i) * (j+1-y)
            cell_01_term = cell_01_weight * cell_01.velocity[index]
            result += cell_01_term
            weight += cell_01_weight
        cell_10 = self.get_cell(i, j + 1)
        if cell_10 is not None:
            cell_10_weight = (i+1-x) * (y-j)
            cell_10_term = cell_10_weight * cell_10.velocity[index]
            result += cell_10_term
            weight += cell_10_weight
        cell_11 = self.get_cell(i + 1, j + 1)
        if cell_11 is not None:
            cell_11_weight = (x-i) * (y-j)
            cell_11_term = cell_11_weight * cell_11.velocity[index]
            result += cell_11_term
            weight += cell_11_weight

        if weight != 0:
            result /= weight

        return result


    def backwards_velocity_trace(self):
        #TODO: check to make sure not doing this for solids and for clean air.
        #advect
        for cell_list in self.cell_table.values():
            for cell in cell_list:
                for coord_i in np.arange(2):
                    velocity_location = np.array([cell.x, cell.y])
                    velocity_location[coord_i] =- 0.5*width

                    #RK2. TODO: confirm I don't need to replace explicit RK2 here with another scheme. Note the time is not incremented here.
                    new_velocity_location = velocity_location - dt*self.interpolate_velocity(velocity_location - 0.5*dt*self.interpolate_velocity(velocity_location, coord_i),coord_i)
                    cell.temp_velocity[coord_i] += self.interpolate_velocity(new_velocity_location, coord_i)

        #do the update
        self.update_velocities_from_temp_velocities()


    #TODO consider adding this in the non-barebones version
    '''def vorticity_confinement_force(self):
        x_shift = np.array([0.5*width,0.])
        y_shift = np.array([0.,0.5*width])
        for cell_list in self.cell_table.values():
            for cell in cell_list:
                cell_center = np.array([cell.x, cell.y])
                if cell.type != 2:
                    vel_x_on_y_0 = self.interpolate_velocity(cell_center-y_shift,0)
                    vel_x_on_y_1 = self.interpolate_velocity(cell_center+y_shift,0)
                    vel_y_on_x_0 = self.interpolate_velocity(cell_center-x_shift,1)
                    vel_y_on_x_1 = self.interpolate_velocity(cell_center+x_shift,1)
                    cell.curl = ((vel_y_on_x_1 - vel_y_on_x_0) - (vel_x_on_y_1 - vel_x_on_y_0)) / (0.5*width)'''

    def find_and_apply_pressures(self):
        #assign a row to each non-solid cell
        pressure_solve_cell_list = []
        for cell_list in self.cell_table.values():
            for cell in cell_list:
                if cell.type != 2:
                    pressure_solve_cell_list.append(cell)
                cell.place_in_pressure_matrix = len(pressure_solve_cell_list)-1
        num_non_solid_cells = len(pressure_solve_cell_list)

        A = np.zeros((num_non_solid_cells, num_non_solid_cells))
        b = np.zeros((num_non_solid_cells))
        #add rows to A and b
        for i in np.arange(num_non_solid_cells):
            cell = pressure_solve_cell_list[i]
            neighbors = [cell.neighbor_id, cell.neighbor_iu, cell.neighbor_jd, cell.neighbor_ju]
            neighbor_iu_exists_and_is_non_solid = False
            neighbor_ju_exists_and_is_non_solid = False

            #calculate number of non-solid neighbors and save some of these values for divergence calculation
            num_non_existing_neighbors = 0
            num_non_solid_neighbors = 0
            for j in np.arange(4):
                if neighbors[j] is not None:
                    if neighbors[j].type!=2:
                        num_non_solid_neighbors += 1
                        if neighbors[j].type==1:
                            neighbor_place_in_pressure_matrix = neighbors[j].place_in_pressure_matrix
                            A[i][neighbor_place_in_pressure_matrix] = 1     #set this neighbor's place in the current row to 1
                            #TODO this is non-symmetric, but making it symmetric makes it singular.
                        if j==1:
                            neighbor_iu_exists_and_is_non_solid=True
                        elif j==3:
                            neighbor_ju_exists_and_is_non_solid=True
                else:
                    num_non_existing_neighbors += 1

            #calculate divergence
            divergence = 0.
            if neighbor_iu_exists_and_is_non_solid:
                divergence += cell.neighbor_iu.velocity[0] - cell.velocity[0]
            else:
                divergence += -1*cell.velocity[0]
            if neighbor_ju_exists_and_is_non_solid:
                divergence += cell.neighbor_ju.velocity[1] - cell.velocity[1]
            else:
                divergence += -1*cell.velocity[1]

            A[i][i] = -1*num_non_solid_neighbors    #set the cell's place in the matrix A
            #assuming that non-existent cells outside the air buffer do not have divergences with respect to
            b[i] = width/dt * divergence - num_non_existing_neighbors * atmospheric_pressure    #set the cell's place in the vector b
            #b[i] = width/dt * divergence    #set the cell's place in the vector b

        #solve the matrix equation
        print("A")
        for i in np.arange(A.shape[0]):
            for j in np.arange(A.shape[1]):
                print(A[i][j], end="")
                if j < A.shape[1]-1:
                    print(", ", end="")
            print()
        print("b\n",b)
        #A_inv = np.linalg.inv(A)
        #print("A_inv\n",A_inv)
        #result = np.matmul(A_inv,b)
        #print("result\n",result)
        #print("det(A)",np.linalg.det(A))
        A = csr_matrix(A)
        result = cg(A, b)
        pressures = result[0]
        if result[1] !=0:
            print(f"Failed to solve matrix equation. number of iterations={result[1]}\t\t(if <0, illegal input or breakdown)")
            #exit(1)

        print("pressures\n",pressures)
        print("result\n",A.dot(pressures))
        print("reasonably converged?",np.allclose(A.dot(pressures), b))
        exit()

        #apply pressures: update cells' velocities
        #TODO: don't use the pressures directly. First need to find gradient of pressures using eq. 4 from the document.
        for i in np.arange(len(pressure_solve_cell_list)):
            cell = pressure_solve_cell_list[i]
            cell.velocity -= dt/width * pressures[i]


    '''def update_disease_densities(self):
        #TODO: not in solids
        #advect
        for cell_list in self.cell_table.values():
            for cell in cell_list:
                #TODO this is an explicit RK2, it needs to be replaced by an implicit solve.
                location = np.array([cell.x, cell.y])
                velocity = np.array([self.interpolate_velocity(location, 0), self.interpolate_velocity(location, 1)])
                temp_location = location - 0.5 * dt * velocity
                temp_location_velocity = np.array([self.interpolate_velocity(temp_location, 0), self.interpolate_velocity(temp_location, 1)])
                new_disease_density_location = location - dt * temp_location_velocity
                cell.temp_disease_density = self.interpolate_disease(new_disease_density_location)

        #do the update
        for cell_list in self.cell_table.values():
            for cell in cell_list:
                cell.disease_density = cell.temp_disease_density
                cell.temp_disease_density *= 0.

                #dissipation - in this case disease die-off outside of humans
                cell.disease_density = max(0., cell.disease_density - disease_die_off_rate * dt) #TODO replace with implicit solve


        self.get_cell(0,0).disease_density=0.25 #for temporary source cell  #TODO please delete once no longer necessary, and replace with a different mechanism for sources.'''

    def diffuse_disease(self):
        '''What equation should I use for disease diffusion?????

        Use heat equation. Find the Laplacian at each cell, that is the force.
        Problem: Implicit mean I need the force in the next time step.

        u_new = u +dt * f(t_new, u_new)

        '''
        pass

    def save_disease_data(self, number, folder, extra_info=""):
        num_diseased_air_cells = 0
        for cell_list in self.cell_table.values():
            for cell in cell_list:
                if cell.type==1:
                    num_diseased_air_cells += 1
        data = np.zeros((num_diseased_air_cells, 3))
        cell_index = 0
        for cell_list in self.cell_table.values():
            for cell in cell_list:
                if cell.type==1:
                    data[cell_index][0] = cell.x
                    data[cell_index][1] = cell.y
                    data[cell_index][2] = cell.disease_density
                    cell_index += 1
        header = "pos_x,pos_y,disease_density"
        file_handling.write_csv_file(os.path.join(folder, extra_info+str(number)+".csv"), header, data)



#TODO: coordinate these with main simulation
time_steps_dir = "disease_sim_time_steps"
number_of_time_steps = 10

if not os.path.isdir(time_steps_dir):
    os.mkdir(time_steps_dir)

disease_grid = grid()
for i in np.arange(number_of_time_steps):
    disease_grid.save_disease_data(i, time_steps_dir)

    disease_grid.update_grid()
    disease_grid.backwards_velocity_trace()
    #TODO apply forces (wind, solids, etc.)
    disease_grid.find_and_apply_pressures()
    #TODO: velocities pointing from non-solid to solid cells should be set to 0
    disease_grid.update_disease_densities()

disease_grid.save_disease_data(number_of_time_steps, time_steps_dir)

time_steps_per_frame = 1
draw_data.draw_data(width/2, time_steps_dir, number_of_time_steps, time_steps_per_frame)



