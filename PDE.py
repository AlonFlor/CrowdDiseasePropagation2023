import numpy as np
import os
import file_handling
import draw_data
from scipy.sparse import csc_matrix
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
density_threshold = 0.003
number_of_buffer_layers = 3
#atmospheric_pressure = 1

baseline_velocity = np.array([3.,-1.])#np.array([1.,-1.])#np.array([3.,0.])# #TODO replace with airflow
disease_diffusivity_constant = 0.07

#TODO: coordinate these with main simulation
dt = 0.01
number_of_time_steps = 300#1000
#vorticity_confinement_force_strength = 1.
disease_die_off_rate = 0.
grid_shape = (256,256)
src_x,src_y = -127,0 #temp coords of source TODO: delete when replacing with better sources




class grid_cell:
    def __init__(self, i, j):
        self.i = i
        self.j = j

        self.x = i*width
        self.y = j*width

        self.layer = -1
        self.type = 0   #three types: clean air=0, diseased air=1, and solid=2
        self.bound_type = 0    #three types: ordinary cell = 0, source = 1, sink = 2

        self.disease_concentration = 0.
        self.pressure = 0.

        self.velocity = np.array([0., 0.])
        #x coord of velocity located at (self.x-0.5*width, self.y)
        #y coord of velocity located at (self.x, self.y-0.5*width)

        self.temp_velocity = np.array([0., 0.]) #for updating the velocity
        self.temp_disease_concentration = np.array([0., 0.]) #for updating the disease density
        #self.curl = np.array([0., 0.]) #for calculating the vorticity confinement force
        self.place_in_disease_matrix = None
        self.place_in_pressure_matrix = None

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
        #   To be clear, a source cell is a regular cell except for the fact that it is close to a contagious person, and is needed (ie do not delete) to track that person's disease output
        #       Perhaps we can have an external force at the source location, to push the disease out of the person's mouth.
        #TODO For now, since I just want somthing out, I will set a source at the origin.
        self.cell_table = dict()
        size_side1, size_side2 = grid_shape
        half_way1 = size_side1/2
        half_way2 = size_side2/2
        for i_raw in np.arange(size_side1):
            i = i_raw-half_way1
            for j_raw in np.arange(size_side2):
                j = j_raw-half_way2
                cell = grid_cell(i,j)
                cell.velocity = baseline_velocity #initialize uniform velocities for all cells. TODO: allow alternate velocity field initilization options.
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

        #set temporary source at origin
        origin_cell = self.get_cell(src_x,src_y)
        origin_cell.disease_concentration=0.25
        origin_cell.type=1
        origin_cell.bound_type=1 #does nothing so far

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
                if cell.disease_concentration < density_threshold:
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
        i = int(np.floor(x))
        j = int(np.floor(y))

        result = 0.
        weight = 0.
        cell_00 = self.get_cell(i, j)
        if cell_00 is not None:
            cell_00_weight = (i+1-x) * (j+1-y)
            cell_00_term =  cell_00_weight* cell_00.disease_concentration
            #print(cell_00.disease_concentration,cell_00_weight,i,x,j,y)
            result += cell_00_term
            weight += cell_00_weight
        cell_01 = self.get_cell(i+1, j)
        if cell_01 is not None:
            cell_01_weight = (x-i) * (j+1-y)
            cell_01_term =  cell_01_weight* cell_01.disease_concentration
            #print(cell_01.disease_concentration,cell_01_weight)
            result += cell_01_term
            weight += cell_01_weight
        cell_10 = self.get_cell(i, j+1)
        if cell_10 is not None:
            cell_10_weight = (i+1-x) * (y-j)
            cell_10_term =  cell_10_weight* cell_10.disease_concentration
            #print(cell_10.disease_concentration,cell_10_weight)
            result += cell_10_term
            weight += cell_10_weight
        cell_11 = self.get_cell(i+1, j+1)
        if cell_11 is not None:
            cell_11_weight = (x-i) * (y-j)
            cell_11_term =  cell_11_weight* cell_11.disease_concentration
            #print(cell_11.disease_concentration,cell_11_weight)
            result += cell_11_term
            weight += cell_11_weight

        if weight != 0:
            result /= weight
        #print(result)
        #print()

        return result

    def interpolate_velocity(self, coords, index):
        x = coords[0] / width
        y = coords[1] / width
        i = int(np.floor(x))
        j = int(np.floor(y))

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

    def find_disease_gradient(self, cell):
        '''Find gradient for the disease concentration in the given cell. Assume all neighbors exist, disease concentration in solid cells is 0,
         and that divergence is not calculated in the boundary cells.
         Done using backwards difference.
         Note: returned gradient values are not divided by width.'''
        return np.array([
            cell.disease_concentration - cell.neighbor_id.disease_concentration,
            cell.disease_concentration - cell.neighbor_jd.disease_concentration
        ])

    def find_velocity_divergence(self, cell):
        '''Find divergence for the velocity field in the given cell. Assume all neighbors exist, velocities at solid-fluid boundaries are 0,
         and that divergence is not calculated in the boundary cells.
         Done using forward difference.
         Note: returned divergence values are not divided by width.'''
        return cell.neighbor_iu.velocity[0] - cell.velocity[0] + cell.neighbor_ju.velocity[1] - cell.velocity[1]

    def find_disease_Laplacian(self, cell):
        '''Find Laplacian for the disease concentration for given cell. Assume all neighbors exist, disease concentration in solid cells is 0,
         and that the Laplacian is not calculated in the boundary cells.
         Note: returned Laplacian values are not divided by width^2.'''
        return cell.neighbor_iu.disease_concentration + cell.neighbor_id.disease_concentration + \
               cell.neighbor_ju.disease_concentration + cell.neighbor_jd.disease_concentration - 4 * cell.disease_concentration

    def find_velocity_Laplacian(self, cell):
        '''Find Laplacian for the velocity field for given cell. Assume all neighbors exist, velocities at solid-fluid boundaries are 0,
         and that the Laplacian is not calculated in the boundary cells.
         Note: returned Laplacian values are not divided by width^2.'''
        return cell.neighbor_iu.velocity + cell.neighbor_id.velocity + cell.neighbor_ju.velocity + cell.neighbor_jd.velocity - 4*cell.velocity

    def row_assign_disease_matrix_solve(self):
        assignment_list = []
        for cell_list in self.active_disease_and_buffer_cells.values():
            for cell in cell_list:
                assignment_list.append(cell)
                cell.place_in_disease_matrix = len(assignment_list) - 1
        num_cells = len(assignment_list)
        return  assignment_list, num_cells

    def disease_diffusion_matrix_solve(self):
        #assign a row to each cell
        assignment_list, num_cells = self.row_assign_disease_matrix_solve()

        A_diffusion = np.zeros((num_cells, num_cells))
        b = np.zeros((num_cells))

        # add rows to A and b
        for i in np.arange(num_cells):
            cell = assignment_list[i]
            neighbors = [cell.neighbor_id, cell.neighbor_iu, cell.neighbor_jd, cell.neighbor_ju]
            neighbor_mask = [1,1,1,1]

            for j in np.arange(4):
                if neighbors[j] is None:
                    neighbor_mask[j] = 0
                elif neighbors[j].place_in_disease_matrix is None:
                    neighbor_mask[j] = 0

            A_diffusion[i][i] = -4.
            for j in np.arange(4):
                if neighbor_mask[j] > 0:
                    A_diffusion[i][neighbors[j].place_in_disease_matrix] = 1.

            b[i] = -1*cell.disease_concentration

        #print(sum(A_diffusion-A_diffusion.T))
        A_diffusion *= dt * disease_diffusivity_constant / (width * width)
        A_diffusion -= np.identity(A_diffusion.shape[0])

        #do matrix solve
        x, result = cg(csc_matrix(A_diffusion), b)
        if result != 0:
            print("CG failed to converge. Result =",result)
            exit(1)
        if np.linalg.norm(A_diffusion.dot(x) - b) > 0.00001:
            print("CG result is not precise.")
            print(A_diffusion.dot(x))
            print(b)
        #TODO: sum of disease concentration is not conserved.
        #print(x,sum(x))
        #print(-b,sum(-b))
        print("x:",sum(x),"\t\t-b:",sum(-b),"\t\tloss:",sum(-b)-sum(x),"\t\t% loss:",100.*(sum(-b)-sum(x))/sum(-b))

        for i in np.arange(num_cells):
            cell = assignment_list[i]
            cell.disease_concentration = x[i]

        #exit()


    '''def find_and_apply_pressures(self):
        #assign a row to each non-solid cell
        pressure_solve_cell_list, num_non_solid_cells = self.row_assign()

        A = np.zeros((num_non_solid_cells, num_non_solid_cells))
        b = np.zeros((num_non_solid_cells))
        #add rows to A and b
        for i in np.arange(num_non_solid_cells):
            cell = pressure_solve_cell_list[i]
            neighbors = [cell.neighbor_id, cell.neighbor_iu, cell.neighbor_jd, cell.neighbor_ju]

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
                else:
                    num_non_existing_neighbors += 1

            #calculate divergence
            divergence = self.find_velocity_divergence(cell)

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
            cell.velocity -= dt/width * pressures[i]'''


    def advect_disease_densities(self):
        #TODO: not in solids
        #advect
        for cell_list in self.active_disease_and_buffer_cells.values():
            for cell in cell_list:
                location = np.array([cell.x, cell.y])
                velocity = np.array([self.interpolate_velocity(location, 0), self.interpolate_velocity(location, 1)])
                temp_location = location - 0.5 * dt * velocity
                temp_location_velocity = np.array([self.interpolate_velocity(temp_location, 0), self.interpolate_velocity(temp_location, 1)])
                new_disease_concentration_location = location - dt * temp_location_velocity
                cell.temp_disease_concentration = self.interpolate_disease(new_disease_concentration_location)

        #do the update
        for cell_list in self.active_disease_and_buffer_cells.values():
            for cell in cell_list:
                cell.disease_concentration = cell.temp_disease_concentration
                cell.temp_disease_concentration *= 0.

                #dissipation - in this case disease die-off outside of humans
                cell.disease_concentration = max(0., cell.disease_concentration - disease_die_off_rate * dt) #TODO replace with implicit solve

        self.get_cell(src_x,src_y).disease_concentration=0.25 #for temporary source cell  #TODO please delete once no longer necessary, and replace with a different mechanism for sources.

        '''#sum density
        sum_density = 0.
        for cell_list in self.cell_table.values():
            for cell in cell_list:
                sum_density += cell.disease_concentration
        print(sum_density)'''

    def save_disease_data(self, number, folder, extra_info=""):
        num_diseased_air_cells = 0
        for cell_list in self.cell_table.values():
            for cell in cell_list:
                if cell.disease_concentration>density_threshold:#cell.type==1:
                    num_diseased_air_cells += 1
        data = np.zeros((num_diseased_air_cells, 3))
        cell_index = 0
        for cell_list in self.cell_table.values():
            for cell in cell_list:
                if cell.disease_concentration>density_threshold:#cell.type==1:
                    data[cell_index][0] = cell.x
                    data[cell_index][1] = cell.y
                    data[cell_index][2] = cell.disease_concentration
                    cell_index += 1
        header = "pos_x,pos_y,disease_concentration"
        file_handling.write_csv_file(os.path.join(folder, extra_info+str(number)+".csv"), header, data)



#TODO: coordinate these with main simulation
time_steps_dir = "disease_sim_time_steps"

if not os.path.isdir(time_steps_dir):
    os.mkdir(time_steps_dir)

air_and_disease_grid = grid(grid_shape)
for i in np.arange(number_of_time_steps):
    air_and_disease_grid.save_disease_data(i, time_steps_dir)

    #air_and_disease_grid.update_grid()
    #air_and_disease_grid.backwards_velocity_trace() #TODO: fix bias that turns off x-axis velocity
    #TODO apply forces (wind, solids, etc.) to the airflow???
    air_and_disease_grid.advect_disease_densities()
    air_and_disease_grid.reset_cell_types()
    #air_and_disease_grid.find_and_apply_pressures()
    air_and_disease_grid.disease_diffusion_matrix_solve()
    air_and_disease_grid.reset_cell_types()
    #TODO: velocities pointing from non-solid to solid cells should be set to 0

air_and_disease_grid.save_disease_data(number_of_time_steps, time_steps_dir)

images_dir = os.path.join("images")
if not os.path.isdir(images_dir):
    os.mkdir(images_dir)

time_steps_per_frame = 1
draw_data.draw_data(time_steps_dir, images_dir, width/2, number_of_time_steps, time_steps_per_frame)

frame_rate = 1/(dt * time_steps_per_frame)
draw_data.make_video(frame_rate, ".", "video")
print("Done")

