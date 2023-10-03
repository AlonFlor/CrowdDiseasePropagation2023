import numpy as np

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

#TODO: coordinate these with main simulation
origin_offset_from_lower_left = -16.
dt = 0.01
vorticity_confinement_force_strength = 1.

def grid_index_to_center_of_cell_coord(grid_index):
    return (grid_index+0.5)*width + origin_offset_from_lower_left
def coord_to_grid_index(coord):
    return int((coord - origin_offset_from_lower_left) / width)

class grid_cell:
    def __init__(self, i, j):
        self.i = i
        self.j = j

        self.x = grid_index_to_center_of_cell_coord(i)
        self.y = grid_index_to_center_of_cell_coord(j)

        self.layer = -1
        self.type = 0   #three types: clean air=0, diseased air=1, and solid=2
        self.bound_type = 0    #three types: ordinary cell = 0, source = 1, sink = 2

        self.disease_density = 0.
        self.pressure = 0.

        self.velocity = np.array([0., 0.])
        #x coord of velocity located at (self.x-0.5*width, self.y)
        #y coord of velocity located at (self.x, self.y-0.5*width)
        self.temp_velocity = np.array([0., 0.]) #for updating the velocity
        self.curl = np.array([0., 0.]) #for calculating the vorticity confinement force


def hash_grid_coords(i, j):
    return 541*i + 79*j

class grid:
    def __init__(self):
        #TODO need to initiate cell_table with cells for sources. Sinks are gas cells right outside the bounds of the simulation.
        #   To be clear, a source cell is a regular cell except for the fact that it is close to a contagious person, and is needed (ie do not delete) to track that person's disease output
        self.cell_table = dict()

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
        return cell

    def update_grid(self, density_threshold = 0.01, number_of_buffer_layers=3):
        '''Update the grid based on disease densities'''
        cells_in_current_layer = []
        for cell_list in self.cell_table:
            for cell in cell_list:
                cell.layer = -1
                if cell.disease_density > density_threshold:
                    cell.type=1 #diseased air
                    cell.layer=0
                    cells_in_current_layer.append(cell)
        for index in np.arange(number_of_buffer_layers):
            cells_in_next_layer = []
            for cell in cells_in_current_layer:
                i_j_list = [(cell.i-1, cell.j), (cell.i+1, cell.j), (cell.i, cell.j-1), (cell.i, cell.j+1)]
                for i,j in i_j_list:
                    neighbor_cell = self.get_cell(i,j)
                    if neighbor_cell is not None:
                        if neighbor_cell.layer==-1 and neighbor_cell.type!=2:
                            neighbor_cell.layer = index+1
                            cells_in_next_layer.append(neighbor_cell)
                    else:
                        neighbor_cell = self.add_cell(i,j)
                        neighbor_cell.layer = index+1
                        #TODO: check if neighbor_cell is in a wall. If so, set its type to 2 (solid).
                        #TODO: also, need something for sink cells, which are non-solid cells at the border of the simulation.
                        cells_in_next_layer.append(neighbor_cell)
            cells_in_current_layer = cells_in_next_layer

        #delete empty non-buffer cells
        for cell_list in self.cell_table:
            for cell in cell_list:
                if cell.layer == -1:
                    self.delete_cell(cell)

    def interpolate_velocity(self, coords, index):
        #TODO there might still be errors here, probably due to shifting a coord by 0.5

        i = coord_to_grid_index(coords[0])
        j = coord_to_grid_index(coords[1])

        x = coords[0] - origin_offset_from_lower_left
        y = coords[1] - origin_offset_from_lower_left

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
        for cell_list in self.cell_table:
            for cell in cell_list:
                for coord_i in np.arange(2):
                    velocity_location = np.array([cell.x, cell.y])
                    velocity_location[coord_i] =- 0.5*width

                    #RK2. TODO: confirm I don't need to replace explicit RK2 here with another scheme. Note the time is not incremented here.
                    new_velocity_location = velocity_location - dt*self.interpolate_velocity(velocity_location - 0.5*dt*self.interpolate_velocity(velocity_location, coord_i),coord_i)
                    cell.temp_velocity[coord_i] += self.interpolate_velocity(new_velocity_location, coord_i)



    '''def vorticity_confinement_force(self):
        x_shift = np.array([0.5*width,0.])
        y_shift = np.array([0.,0.5*width])
        for cell_list in self.cell_table:
            for cell in cell_list:
                cell_center = np.array([cell.x, cell.y])
                if cell.type != 2:
                    vel_x_on_y_0 = self.interpolate_velocity(cell_center-y_shift,0)
                    vel_x_on_y_1 = self.interpolate_velocity(cell_center+y_shift,0)
                    vel_y_on_x_0 = self.interpolate_velocity(cell_center-x_shift,1)
                    vel_y_on_x_1 = self.interpolate_velocity(cell_center+x_shift,1)
                    cell.curl = ((vel_y_on_x_1 - vel_y_on_x_0) - (vel_x_on_y_1 - vel_x_on_y_0)) / (0.5*width)'''

    def update_velocities(self):
        '''update the velocities'''
        for cell_list in self.cell_table:
            for cell in cell_list:
                cell.velocity[:] = cell.temp_velocity[:]
                cell.temp_velocity *= 0.






MAC_grid = grid()


#set a source at the origin


