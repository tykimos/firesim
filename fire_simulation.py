import pygame
import numpy as np
import random
import time
import math
import json
import struct
import os

# Color definitions
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (64, 64, 64)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)
BROWN = (139, 69, 19)

# Fire states
NORMAL = 0
PREHEATING = 1
IGNITION = 2
BURNING = 3
FLASHOVER = 4
BURNED_OUT = 5

# Display modes
FIRE_MODE = 0
TEMPERATURE_MODE = 1
SMOKE_MODE = 2
VISIBILITY_MODE = 3
CO_MODE = 4
HCN_MODE = 5
AIR_VELOCITY_MODE = 6
THERMAL_RADIATION_MODE = 7
PRESSURE_MODE = 8

# Fire science constants - more realistic values
AMBIENT_TEMP = 20.0          # Ambient temperature (°C)
PREHEATING_TEMP = 150.0      # Preheating temperature (°C) - increased
IGNITION_TEMP = 400.0        # Ignition temperature (°C) - increased
FLASHOVER_TEMP = 600.0       # Flashover temperature (°C)
MAX_BURN_TEMP = 1200.0       # Maximum burning temperature (°C)
THERMAL_DIFFUSIVITY = 0.08   # Heat diffusion rate - reduced
SMOKE_DIFFUSIVITY = 0.12     # Smoke diffusion rate - reduced
SMOKE_YIELD = 0.25           # Smoke production factor
CO_YIELD = 0.04              # CO production factor
HCN_YIELD = 0.012            # HCN production factor

class FireCell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.state = NORMAL
        
        # Physical properties
        self.temperature = AMBIENT_TEMP + random.uniform(-1, 1)
        self.oxygen = 20.9 + random.uniform(-0.2, 0.2)    # O2 concentration (%)
        self.fuel_load = 40.0 + random.uniform(-8, 12)    # Fuel load (MJ/m²) - reduced
        self.heat_release_rate = 0.0                       # HRR (kW/m²)
        self.thermal_mass = 1500.0 + random.uniform(-400, 400)  # Thermal mass (kJ/m²K) - increased
        
        # Environmental conditions
        self.humidity = 50.0 + random.uniform(-10, 10)
        self.pressure = 101325.0 + random.uniform(-20, 20)
        self.wind_speed = random.uniform(0, 1.0)           # Reduced wind
        self.wind_direction = random.uniform(0, 2*math.pi)
        
        # Fire dynamics
        self.burn_time = 0
        self.ignition_delay = 0
        self.preheating_time = 0
        self.flame_height = 0.0
        self.burning_rate = 0.0                            # kg/m²s
        
        # Smoke and toxicity
        self.smoke_density = 0.0                           # Optical density (1/m)
        self.smoke_production_rate = 0.0                   # kg/s/m²
        self.smoke_layer_height = 0.0                      # Height of smoke layer
        self.visibility_distance = 30.0                    # Visibility (m)
        self.co_concentration = 0.0                        # CO (ppm)
        self.hcn_concentration = 0.0                       # HCN (ppm)
        self.o2_depletion_rate = 0.0                       # O2 consumption rate
        
        # Thermal effects
        self.thermal_radiation = 0.0                       # kW/m²
        self.convective_heat = 0.0                         # kW/m²
        self.heat_flux_received = 0.0                      # Total heat flux (kW/m²)
        
        # Air movement
        self.air_velocity_x = 0.0
        self.air_velocity_y = 0.0
        self.buoyancy_velocity = 0.0
        
    def calculate_fire_spread_probability(self, neighbors):
        """Calculate ignition probability based on realistic fire physics"""
        if self.state not in [NORMAL, PREHEATING]:
            return 0.0
            
        # Heat flux from neighboring fires
        total_heat_flux = 0.0
        burning_neighbors = [n for n in neighbors if n.state in [BURNING, FLASHOVER]]
        
        for neighbor in burning_neighbors:
            # Distance factor (adjacent cells)
            distance = 1.0  # Grid distance
            
            # Radiative heat transfer (Stefan-Boltzmann)
            if neighbor.temperature > AMBIENT_TEMP:
                temp_k = neighbor.temperature + 273.15
                emissivity = 0.8
                stefan_boltzmann = 5.67e-8
                radiative_flux = emissivity * stefan_boltzmann * (temp_k**4 - (AMBIENT_TEMP + 273.15)**4) / 1000
                total_heat_flux += radiative_flux / (distance**2)
            
            # Convective heat transfer
            convective_flux = neighbor.convective_heat / (distance + 1)
            total_heat_flux += convective_flux
        
        # Smoke preheating effect - reduced influence
        smoke_neighbors = [n for n in neighbors if n.smoke_density > 2.0]
        for neighbor in smoke_neighbors:
            smoke_heat = neighbor.smoke_density * 8  # Reduced heat from smoke
            total_heat_flux += smoke_heat
        
        self.heat_flux_received = total_heat_flux
        
        # Critical heat flux for ignition - much higher threshold
        critical_heat_flux = 25.0 * (1 + self.humidity/100) * (self.fuel_load/40)
        
        # Temperature-based ignition - stricter requirements
        if self.state == PREHEATING:
            temp_factor = max(0, (self.temperature - PREHEATING_TEMP) / 300)
        else:
            temp_factor = max(0, (self.temperature - IGNITION_TEMP) / 200)
        
        # Oxygen availability
        oxygen_factor = max(0, (self.oxygen - 16) / 4)  # Need minimum 16% O2
        
        # Fuel availability
        fuel_factor = min(1.0, self.fuel_load / 25)
        
        # Wind effect - reduced
        wind_factor = 1.0 + self.wind_speed * 0.1
        
        # More realistic ignition probability
        if total_heat_flux > critical_heat_flux:
            if self.state == PREHEATING:
                base_prob = min(0.15, total_heat_flux / critical_heat_flux * 0.02)
            else:
                base_prob = min(0.08, total_heat_flux / critical_heat_flux * 0.01)
        else:
            base_prob = 0.001
            
        return base_prob * temp_factor * oxygen_factor * fuel_factor * wind_factor
    
    def update_fire_dynamics(self):
        """Update fire behavior with much more realistic progression"""
        dt = 1.0  # Time step (seconds)
        
        if self.state == NORMAL:
            # Very slow heating from environment
            if self.heat_flux_received > 8.0:  # Higher threshold
                heating_rate = self.heat_flux_received / self.thermal_mass * 600  # Slower heating
                self.temperature += heating_rate * dt
                
                # Transition to preheating state
                if self.temperature > PREHEATING_TEMP:
                    self.state = PREHEATING
                    self.preheating_time = 0
                
        elif self.state == PREHEATING:
            self.preheating_time += dt
            # Very gradual temperature rise during preheating
            heating_rate = self.heat_flux_received / self.thermal_mass * 800
            self.temperature += heating_rate * dt
            
            # Minimal smoke production during preheating
            self.smoke_production_rate = 0.003
            self.smoke_density = min(0.5, self.smoke_density + 0.005 * dt)
                
        elif self.state == IGNITION:
            self.ignition_delay += dt
            self.temperature += 30 * dt  # Slower heating during ignition
            
            # Small amount of initial smoke
            self.smoke_production_rate = 0.02
            self.smoke_density = min(1.5, self.smoke_density + 0.03 * dt)
            
            # Realistic ignition delay
            if self.ignition_delay > 5.0:  # 5 second ignition delay
                self.state = BURNING
                self.burn_time = 0
                
        elif self.state == BURNING:
            self.burn_time += dt
            
            # Realistic heat release rate growth
            max_hrr = min(800.0, self.fuel_load * 15)  # Higher maximum HRR
            growth_rate = 0.05  # Faster t-squared fire growth
            
            # t-squared fire growth: HRR = α * t²
            self.heat_release_rate = min(max_hrr, growth_rate * (self.burn_time ** 2))
            
            # Temperature from HRR
            self.temperature = AMBIENT_TEMP + (self.heat_release_rate * 1.2)
            
            # Realistic fuel consumption
            fuel_consumption_rate = self.heat_release_rate / 2000  # Faster consumption
            self.fuel_load = max(0, self.fuel_load - fuel_consumption_rate * dt)
            
            # Realistic oxygen consumption
            o2_consumption = fuel_consumption_rate * 1.2 * 1.225 / 200  # Faster O2 depletion
            self.oxygen = max(0, self.oxygen - o2_consumption * dt)
            
            # Burning rate
            self.burning_rate = fuel_consumption_rate
            
            # More realistic flashover conditions
            if self.temperature > FLASHOVER_TEMP and self.heat_release_rate > 150:
                self.state = FLASHOVER
                
            # Fire extinguishes when fuel depleted or insufficient oxygen
            if self.fuel_load <= 1.0 or self.oxygen < 12:
                self.state = BURNED_OUT
                
        elif self.state == FLASHOVER:
            self.burn_time += dt
            
            # Very high heat release during flashover
            self.heat_release_rate = min(1500.0, self.fuel_load * 25)
            self.temperature = min(MAX_BURN_TEMP, FLASHOVER_TEMP + self.heat_release_rate * 0.2)
            
            # Much faster fuel consumption during flashover
            fuel_consumption_rate = self.heat_release_rate / 1000
            self.fuel_load = max(0, self.fuel_load - fuel_consumption_rate * dt)
            
            # Much faster oxygen depletion during flashover
            o2_consumption = fuel_consumption_rate * 3.0 * 1.225 / 100
            self.oxygen = max(0, self.oxygen - o2_consumption * dt)
            
            # Transition to burned out
            if self.fuel_load <= 0.5 or self.oxygen < 10:
                self.state = BURNED_OUT
                
        elif self.state == BURNED_OUT:
            # Very slow cooling down
            cooling_rate = (self.temperature - AMBIENT_TEMP) * 0.02
            self.temperature = max(AMBIENT_TEMP, self.temperature - cooling_rate * dt)
            self.heat_release_rate = max(0, self.heat_release_rate - 8 * dt)
            
            # Residual smoke from hot surfaces
            if self.temperature > 80:
                self.smoke_production_rate = max(0, 0.01 - self.burn_time * 0.0005)
            else:
                self.smoke_production_rate = 0
    
    def update_smoke_and_toxicity(self):
        """Enhanced smoke behavior with slower spread"""
        dt = 1.0
        
        if self.state in [PREHEATING, IGNITION, BURNING, FLASHOVER]:
            # Smoke production based on state
            if self.state == PREHEATING:
                base_smoke_rate = 0.003
            elif self.state == IGNITION:
                base_smoke_rate = 0.02
            elif self.state == BURNING:
                base_smoke_rate = self.burning_rate * SMOKE_YIELD
            elif self.state == FLASHOVER:
                base_smoke_rate = self.burning_rate * SMOKE_YIELD * 1.3
            
            self.smoke_production_rate = base_smoke_rate
            
            # Slower smoke density increase
            smoke_generation = self.smoke_production_rate * 150  # Reduced factor
            self.smoke_density = min(6.0, self.smoke_density + smoke_generation * dt)
            
            # CO and HCN production
            if self.state in [BURNING, FLASHOVER]:
                co_factor = max(1.2, (21 - self.oxygen) / 5)
                co_production = self.burning_rate * CO_YIELD * co_factor * 800000
                self.co_concentration = min(40000, self.co_concentration + co_production * dt)
                
                hcn_production = self.burning_rate * HCN_YIELD * 800000
                self.hcn_concentration = min(6000, self.hcn_concentration + hcn_production * dt)
            
        elif self.state == BURNED_OUT:
            # Slow smoke dissipation
            if self.temperature > 60:
                self.smoke_density = max(0, self.smoke_density - 0.03 * dt)
            else:
                self.smoke_density = max(0, self.smoke_density - 0.08 * dt)
            self.co_concentration = max(0, self.co_concentration - 100 * dt)
            self.hcn_concentration = max(0, self.hcn_concentration - 12 * dt)
        else:
            # Normal dissipation
            self.smoke_density = max(0, self.smoke_density - 0.05 * dt)
            self.co_concentration = max(0, self.co_concentration - 60 * dt)
            self.hcn_concentration = max(0, self.hcn_concentration - 6 * dt)
        
        # Visibility calculation
        if self.smoke_density > 0.1:
            optical_density = self.smoke_density * (1 + self.temperature / 300)
            self.visibility_distance = max(0.05, 2.0 / optical_density)
        else:
            self.visibility_distance = 30.0
    
    def update_thermal_effects(self):
        """Update thermal radiation and convective heat transfer"""
        if self.state in [BURNING, FLASHOVER]:
            # Thermal radiation
            temp_k = self.temperature + 273.15
            emissivity = 0.8
            stefan_boltzmann = 5.67e-8
            
            self.thermal_radiation = emissivity * stefan_boltzmann * (temp_k**4) / 1000
            
            # Convective heat transfer
            self.convective_heat = self.heat_release_rate * 0.3
            
            # Flame height
            if self.heat_release_rate > 0:
                self.flame_height = 0.235 * (self.heat_release_rate**0.4) - 1.02
                self.flame_height = max(0, self.flame_height)
        else:
            self.thermal_radiation = max(0, self.thermal_radiation - 1.5)
            self.convective_heat = max(0, self.convective_heat - 6.0)
            self.flame_height = max(0, self.flame_height - 0.1)
    
    def update_airflow(self):
        """Update air movement due to buoyancy and wind"""
        if self.state in [BURNING, FLASHOVER]:
            # Buoyancy-driven flow
            temp_diff = self.temperature - AMBIENT_TEMP
            if temp_diff > 0:
                self.buoyancy_velocity = math.sqrt(2 * 9.81 * temp_diff / (AMBIENT_TEMP + 273.15)) * 0.6
                self.air_velocity_y = min(4.0, self.buoyancy_velocity)
            
            # Wind effects
            self.air_velocity_x = self.wind_speed * math.cos(self.wind_direction) + random.uniform(-0.2, 0.2)
        else:
            self.air_velocity_x = self.wind_speed * 0.3 + random.uniform(-0.05, 0.05)
            self.air_velocity_y = max(0, self.air_velocity_y - 0.1)
            self.buoyancy_velocity = max(0, self.buoyancy_velocity - 0.15)
    
    def update(self, neighbors):
        """Main update function with enhanced physics"""
        # Check for ignition
        if self.state in [NORMAL, PREHEATING]:
            ignition_prob = self.calculate_fire_spread_probability(neighbors)
            if random.random() < ignition_prob:
                if self.state == PREHEATING:
                    self.state = IGNITION
                    self.ignition_delay = 0
                elif self.temperature > PREHEATING_TEMP:
                    self.state = IGNITION
                    self.ignition_delay = 0
                else:
                    self.state = PREHEATING
                    self.preheating_time = 0
        
        # Update all fire dynamics
        self.update_fire_dynamics()
        self.update_smoke_and_toxicity()
        self.update_thermal_effects()
        self.update_airflow()
        
        # Pressure effects from hot gases
        if self.state in [BURNING, FLASHOVER]:
            pressure_increase = (self.temperature - AMBIENT_TEMP) * 1.8
            self.pressure = 101325.0 + pressure_increase
        else:
            self.pressure = max(101325.0, self.pressure - 8.0)
    
    def get_data_vector(self):
        """Return 9-dimensional data vector for export"""
        return [
            self.state,
            self.temperature,
            self.smoke_density,
            self.visibility_distance,
            self.co_concentration,
            self.hcn_concentration,
            math.sqrt(self.air_velocity_x**2 + self.air_velocity_y**2),
            self.thermal_radiation,
            self.pressure
        ]

class FireSimulation:
    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        self.grid = [[FireCell(x, y) for x in range(width)] for y in range(height)]
        self.time_step = 0
        self.selected_x = 18
        self.selected_y = 11
        self.simulation_time = 0.0
        
        # Data storage for export - store in memory until completion
        self.simulation_data = []  # List of timestep data arrays
        self.max_simulation_time = 1800  # 30 minutes maximum
        self.start_time = time.time()
        self.data_exported = False  # Flag to prevent multiple exports
        self.simulation_complete = False  # Flag to indicate simulation is finished
        self.metadata = {
            'grid_width': width,
            'grid_height': height,
            'ignition_point': [18, 11],
            'variables': [
                'fire_state',
                'temperature', 
                'smoke_density',
                'visibility_distance',
                'co_concentration',
                'hcn_concentration', 
                'air_velocity_magnitude',
                'thermal_radiation',
                'pressure'
            ],
            'variable_units': [
                'state_enum',
                'celsius',
                'optical_density_per_meter',
                'meters',
                'ppm',
                'ppm',
                'meters_per_second',
                'kilowatts_per_square_meter',
                'pascals'
            ],
            'fire_states': {
                '0': 'NORMAL',
                '1': 'PREHEATING', 
                '2': 'IGNITION',
                '3': 'BURNING',
                '4': 'FLASHOVER',
                '5': 'BURNED_OUT'
            },
            'data_format': 'float32',
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_timesteps': 0,
            'simulation_duration_seconds': 0
        }
        
        # Set ignition point (18, 11) with high fuel load
        ignition_cell = self.grid[11][18]
        ignition_cell.state = IGNITION
        ignition_cell.fuel_load = 60.0
        ignition_cell.temperature = IGNITION_TEMP
        
    def get_neighbors(self, x, y):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbors.append(self.grid[ny][nx])
        return neighbors
        
    def update(self):
        """Update simulation with data collection"""
        # Update all cells
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                neighbors = self.get_neighbors(x, y)
                cell.update(neighbors)
        
        # Enhanced smoke and heat diffusion
        self.update_diffusion()
        
        # Collect data for export
        self.collect_simulation_data()
        
        self.time_step += 1
        self.simulation_time += 1.0
        
        # Check if simulation should end
        if not self.simulation_complete:
            if self.simulation_time >= self.max_simulation_time:
                if not self.data_exported:
                    print(f"Simulation timeout at {self.max_simulation_time} seconds")
                    self.export_simulation_data()
                    self.data_exported = True
                self.simulation_complete = True
                return True  # Signal completion
            elif self.is_simulation_complete():
                if not self.data_exported:
                    print(f"Simulation complete - Total burnout achieved at {self.simulation_time:.0f} seconds")
                    self.export_simulation_data()
                    self.data_exported = True
                self.simulation_complete = True
                return True  # Signal completion
        
        return self.simulation_complete  # Return completion status
        
    def collect_simulation_data(self):
        """Collect current state data for all cells as numpy array"""
        # Create 3D array: [height, width, 9_variables]
        timestep_data = np.zeros((self.height, self.width, 9), dtype=np.float32)
        
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                data_vector = cell.get_data_vector()
                timestep_data[y, x, :] = data_vector
        
        self.simulation_data.append(timestep_data)
    
    def is_simulation_complete(self):
        """Check if simulation is complete (complete burnout - all fuel consumed)"""
        active_fires = 0
        total_fuel = 0
        burned_out_cells = 0
        
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if cell.state in [IGNITION, BURNING, FLASHOVER]:
                    active_fires += 1
                elif cell.state == BURNED_OUT:
                    burned_out_cells += 1
                total_fuel += cell.fuel_load
        
        total_cells = self.width * self.height
        
        # Simulation complete when:
        # 1. No active fires AND
        # 2. Significant burnout occurred (at least 15% cells burned out) AND
        # 3. Very low total fuel remaining AND
        # 4. Minimum simulation time passed
        burnout_ratio = burned_out_cells / total_cells
        
        return (active_fires == 0 and 
                burnout_ratio > 0.15 and
                total_fuel < 200 and
                self.simulation_time > 60)  # Minimum 1 minute simulation time
    
    def export_simulation_data(self):
        """Export simulation data to binary file with JSON metadata"""
        if not self.simulation_data:
            print("No simulation data to export")
            return
            
        # Ensure dataset directory exists
        dataset_dir = "train_dataset"
        os.makedirs(dataset_dir, exist_ok=True)
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = f"fire_simulation_{timestamp}"
        
        # Convert list of arrays to single 4D numpy array
        # Shape: [timesteps, height, width, variables]
        data_array = np.stack(self.simulation_data, axis=0)
        
        # Update metadata
        self.metadata['total_timesteps'] = len(self.simulation_data)
        self.metadata['simulation_duration_seconds'] = float(self.simulation_time)
        self.metadata['data_shape'] = list(data_array.shape)
        self.metadata['file_size_bytes'] = int(data_array.nbytes)
        self.metadata['export_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Export binary data to dataset directory
        bin_filename = os.path.join(dataset_dir, f"{base_filename}.bin")
        with open(bin_filename, 'wb') as f:
            # Write data as contiguous binary
            f.write(data_array.tobytes())
        
        # Export metadata as JSON to dataset directory
        json_filename = os.path.join(dataset_dir, f"{base_filename}.json")
        with open(json_filename, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Simulation data exported:")
        print(f"  Binary data: {bin_filename} ({data_array.nbytes / 1024 / 1024:.1f} MB)")
        print(f"  Metadata: {json_filename}")
        print(f"  Data shape: {data_array.shape} (timesteps x height x width x variables)")
        
        return bin_filename, json_filename
        
    def update_diffusion(self):
        """Slower smoke and heat diffusion"""
        # Create temporary arrays for diffusion
        new_smoke = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        new_temp = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        new_co = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                neighbors = self.get_neighbors(x, y)
                
                # Slower smoke diffusion
                total_smoke = cell.smoke_density
                for neighbor in neighbors:
                    smoke_diff = (neighbor.smoke_density - cell.smoke_density) * SMOKE_DIFFUSIVITY
                    total_smoke += smoke_diff
                new_smoke[y][x] = max(0, total_smoke)
                
                # Slower heat diffusion
                if cell.state in [NORMAL, PREHEATING]:
                    total_temp = cell.temperature
                    for neighbor in neighbors:
                        temp_diff = (neighbor.temperature - cell.temperature) * THERMAL_DIFFUSIVITY
                        total_temp += temp_diff
                    new_temp[y][x] = max(AMBIENT_TEMP, total_temp)
                else:
                    new_temp[y][x] = cell.temperature
                
                # CO diffusion
                total_co = cell.co_concentration
                for neighbor in neighbors:
                    co_diff = (neighbor.co_concentration - cell.co_concentration) * 0.05
                    total_co += co_diff
                new_co[y][x] = max(0, total_co)
        
        # Apply diffusion results
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x].smoke_density = new_smoke[y][x]
                if self.grid[y][x].state in [NORMAL, PREHEATING]:
                    self.grid[y][x].temperature = new_temp[y][x]
                self.grid[y][x].co_concentration = new_co[y][x]
        
    def move_selection(self, dx, dy):
        self.selected_x = max(0, min(self.width - 1, self.selected_x + dx))
        self.selected_y = max(0, min(self.height - 1, self.selected_y + dy))
        
    def get_statistics(self, variable_name):
        """Calculate min, max, avg for environmental variables"""
        values = []
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if variable_name == 'temperature':
                    values.append(cell.temperature)
                elif variable_name == 'smoke':
                    values.append(cell.smoke_density * 1000)
                elif variable_name == 'visibility':
                    values.append(cell.visibility_distance)
                elif variable_name == 'co':
                    values.append(cell.co_concentration)
                elif variable_name == 'hcn':
                    values.append(cell.hcn_concentration)
                elif variable_name == 'air_velocity':
                    values.append(math.sqrt(cell.air_velocity_x**2 + cell.air_velocity_y**2))
                elif variable_name == 'thermal_radiation':
                    values.append(cell.thermal_radiation)
                elif variable_name == 'pressure':
                    values.append(cell.pressure)
        
        if values:
            return min(values), max(values), sum(values) / len(values)
        return 0, 0, 0
        
    def get_cell_color(self, cell, mode):
        if mode == FIRE_MODE:
            if cell.state == NORMAL:
                # Normal state is always green - no gradients
                return GREEN
            elif cell.state == PREHEATING:
                return (255, 200, 100)  # Light orange
            elif cell.state == IGNITION:
                return YELLOW
            elif cell.state == BURNING:
                return ORANGE
            elif cell.state == FLASHOVER:
                return RED
            elif cell.state == BURNED_OUT:
                return (80, 80, 80)  # Dark gray
                
        elif mode == TEMPERATURE_MODE:
            temp_ratio = min(1.0, max(0.0, (cell.temperature - AMBIENT_TEMP) / 1000.0))
            blue_val = max(0, min(255, int(255 * (1 - temp_ratio))))
            red_val = max(0, min(255, int(255 * temp_ratio)))
            return (red_val, 0, blue_val)
            
        elif mode == SMOKE_MODE:
            smoke_ratio = min(1.0, max(0.0, cell.smoke_density / 4.0))
            val = max(0, min(255, int(255 * (1 - smoke_ratio))))
            return (val, val, val)
            
        elif mode == VISIBILITY_MODE:
            vis_ratio = min(1.0, max(0.0, cell.visibility_distance / 30.0))
            red_val = max(0, min(255, int(255 * (1 - vis_ratio))))
            green_val = max(0, min(255, int(255 * vis_ratio)))
            return (red_val, green_val, 0)
            
        elif mode == CO_MODE:
            co_ratio = min(1.0, max(0.0, cell.co_concentration / 12000.0))
            green_val = max(0, min(255, int(255 * (1 - co_ratio))))
            red_val = max(0, min(255, int(255 * co_ratio)))
            return (red_val, green_val, 0)
            
        elif mode == HCN_MODE:
            hcn_ratio = min(1.0, max(0.0, cell.hcn_concentration / 1200.0))
            cyan_val = max(0, min(255, int(255 * (1 - hcn_ratio))))
            purple_val = max(0, min(255, int(255 * hcn_ratio)))
            return (purple_val, 0, cyan_val)
            
        elif mode == AIR_VELOCITY_MODE:
            velocity_mag = math.sqrt(cell.air_velocity_x**2 + cell.air_velocity_y**2)
            vel_ratio = min(1.0, max(0.0, velocity_mag / 6.0))
            blue_val = max(0, min(255, int(255 * (1 - vel_ratio))))
            red_val = max(0, min(255, int(255 * vel_ratio)))
            return (red_val, 0, blue_val)
            
        elif mode == THERMAL_RADIATION_MODE:
            rad_ratio = min(1.0, max(0.0, cell.thermal_radiation / 80.0))
            orange_r = max(0, min(255, int(255 * rad_ratio)))
            orange_g = max(0, min(165, int(165 * rad_ratio)))
            return (orange_r, orange_g, 0)
            
        elif mode == PRESSURE_MODE:
            pressure_ratio = min(1.0, max(0.0, (cell.pressure - 101000) / 1200.0))
            blue_val = max(0, min(255, int(255 * (1 - pressure_ratio))))
            red_val = max(0, min(255, int(255 * pressure_ratio)))
            return (red_val, 0, blue_val)
            
        return WHITE

def main():
    pygame.init()
    
    # Screen settings
    MAIN_CELL_SIZE = 24
    MINI_CELL_SIZE = 6
    GRID_WIDTH = 20
    GRID_HEIGHT = 20
    SCREEN_WIDTH = 1200
    SCREEN_HEIGHT = 700
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Realistic Fire Simulation - Training Mode with Data Export")
    font = pygame.font.Font(None, 14)
    title_font = pygame.font.Font(None, 16)
    main_title_font = pygame.font.Font(None, 20)
    
    # Simulation initialization
    simulation = FireSimulation()
    clock = pygame.time.Clock()
    
    running = True
    paused = False
    
    # Environmental variable names and configurations
    env_vars = [
        ("Temperature", "temperature", "°C"),
        ("Smoke Density", "smoke", "mg/m³"),
        ("Visibility", "visibility", "m"),
        ("CO Level", "co", "ppm"),
        ("HCN Level", "hcn", "ppm"),
        ("Air Velocity", "air_velocity", "m/s"),
        ("Heat Radiation", "thermal_radiation", "kW/m²"),
        ("Pressure", "pressure", "Pa")
    ]
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                if simulation.simulation_data and not simulation.data_exported:
                    simulation.export_simulation_data()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    simulation = FireSimulation()
                elif event.key == pygame.K_e:  # Export data manually
                    if simulation.simulation_data:
                        if not simulation.data_exported:
                            simulation.export_simulation_data()
                            simulation.data_exported = True
                        else:
                            print("Data already exported")
                    else:
                        print("No data to export yet")
                elif event.key == pygame.K_UP:
                    simulation.move_selection(0, -1)
                elif event.key == pygame.K_DOWN:
                    simulation.move_selection(0, 1)
                elif event.key == pygame.K_LEFT:
                    simulation.move_selection(-1, 0)
                elif event.key == pygame.K_RIGHT:
                    simulation.move_selection(1, 0)
                    
        if not paused and running and not simulation.simulation_complete:
            simulation_complete = simulation.update()
            if simulation_complete:
                if not simulation.data_exported:
                    print("\nSimulation complete - Exporting data...")
                    bin_file, json_file = simulation.export_simulation_data()
                    print(f"\nData exported successfully to train_dataset folder:")
                    print(f"Binary data: {os.path.basename(bin_file)}")
                    print(f"Metadata: {os.path.basename(json_file)}")
                    print("\nExiting program...")
                running = False  # 프로그램 종료
            
        # Clear screen
        screen.fill(BLACK)
        
        # Draw main fire state grid (left side)
        main_start_x = 20
        main_start_y = 40
        
        # Main title
        main_title = main_title_font.render("REALISTIC FIRE SIMULATION - TRAINING MODE", True, YELLOW)
        screen.blit(main_title, (main_start_x, 5))
        
        # Check completion status
        completion_status = ""
        if simulation.is_simulation_complete():
            completion_status = " | STATUS: BURNOUT COMPLETE"
        elif simulation.simulation_time >= simulation.max_simulation_time:
            completion_status = " | STATUS: TIMEOUT (30min)"
        
        # Add time remaining for long simulations
        time_remaining = simulation.max_simulation_time - simulation.simulation_time
        if time_remaining > 0 and simulation.simulation_time > 60:
            minutes_remaining = int(time_remaining // 60)
            completion_status += f" | Remaining: {minutes_remaining}min"
            
        time_info = title_font.render(f"Time: {simulation.simulation_time:.0f}s | Step: {simulation.time_step} | Cell: ({simulation.selected_x}, {simulation.selected_y}){completion_status}", True, WHITE)
        screen.blit(time_info, (main_start_x, 22))
        
        # Draw main grid
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                cell = simulation.grid[y][x]
                color = simulation.get_cell_color(cell, FIRE_MODE)
                rect = pygame.Rect(main_start_x + x * MAIN_CELL_SIZE, main_start_y + y * MAIN_CELL_SIZE, 
                                 MAIN_CELL_SIZE, MAIN_CELL_SIZE)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, DARK_GRAY, rect, 1)
                
                # Highlight selected cell
                if x == simulation.selected_x and y == simulation.selected_y:
                    pygame.draw.rect(screen, WHITE, rect, 2)
        
        # Fire state legend
        legend_y = main_start_y + GRID_HEIGHT * MAIN_CELL_SIZE + 10
        legend_title = title_font.render("Fire States:", True, CYAN)
        screen.blit(legend_title, (main_start_x, legend_y))
        
        fire_states = [
            ("Normal", GREEN),
            ("Preheating", (255, 200, 100)),
            ("Ignition", YELLOW),
            ("Burning", ORANGE),
            ("Flashover", RED),
            ("Burned Out", (80, 80, 80))
        ]
        
        for i, (state, color) in enumerate(fire_states):
            legend_rect = pygame.Rect(main_start_x + i * 80, legend_y + 18, 10, 10)
            pygame.draw.rect(screen, color, legend_rect)
            pygame.draw.rect(screen, WHITE, legend_rect, 1)
            state_text = font.render(state, True, WHITE)
            screen.blit(state_text, (main_start_x + i * 80 + 14, legend_y + 18))
        
        # Draw mini environmental maps (right side)
        mini_start_x = 520
        mini_start_y = 40
        mini_grid_size = GRID_WIDTH * MINI_CELL_SIZE
        
        # Mini maps title
        mini_title = main_title_font.render("ENVIRONMENTAL MONITORING", True, YELLOW)
        screen.blit(mini_title, (mini_start_x, 5))
        
        # Draw 8 mini maps in 4x2 grid
        for i, (var_name, var_key, unit) in enumerate(env_vars):
            col = i % 4
            row = i // 4
            
            map_x = mini_start_x + col * 170
            map_y = mini_start_y + row * 180
            
            # Get statistics
            min_val, max_val, avg_val = simulation.get_statistics(var_key)
            
            # Draw mini map title with unit
            title_with_unit = f"{var_name} ({unit})"
            title_text = title_font.render(title_with_unit, True, WHITE)
            screen.blit(title_text, (map_x, map_y - 18))
            
            # Draw mini grid
            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    cell = simulation.grid[y][x]
                    mode_map = {
                        'temperature': TEMPERATURE_MODE,
                        'smoke': SMOKE_MODE,
                        'visibility': VISIBILITY_MODE,
                        'co': CO_MODE,
                        'hcn': HCN_MODE,
                        'air_velocity': AIR_VELOCITY_MODE,
                        'thermal_radiation': THERMAL_RADIATION_MODE,
                        'pressure': PRESSURE_MODE
                    }
                    color = simulation.get_cell_color(cell, mode_map[var_key])
                    rect = pygame.Rect(map_x + x * MINI_CELL_SIZE, map_y + y * MINI_CELL_SIZE, 
                                     MINI_CELL_SIZE, MINI_CELL_SIZE)
                    pygame.draw.rect(screen, color, rect)
                    
                    # Highlight selected cell
                    if x == simulation.selected_x and y == simulation.selected_y:
                        pygame.draw.rect(screen, WHITE, rect, 1)
            
            # Draw statistics
            stats_y = map_y + mini_grid_size + 3
            
            # Format values
            if var_key == 'pressure':
                min_text = f"Min:{min_val:.0f}"
                max_text = f"Max:{max_val:.0f}"
                avg_text = f"Avg:{avg_val:.0f}"
            elif var_key in ['temperature', 'smoke', 'co']:
                min_text = f"Min:{min_val:.0f}"
                max_text = f"Max:{max_val:.0f}"
                avg_text = f"Avg:{avg_val:.0f}"
            else:
                min_text = f"Min:{min_val:.1f}"
                max_text = f"Max:{max_val:.1f}"
                avg_text = f"Avg:{avg_val:.1f}"
            
            min_surface = font.render(min_text, True, CYAN)
            max_surface = font.render(max_text, True, RED)
            avg_surface = font.render(avg_text, True, WHITE)
            
            screen.blit(min_surface, (map_x, stats_y))
            screen.blit(max_surface, (map_x, stats_y + 12))
            screen.blit(avg_surface, (map_x, stats_y + 24))
        
        # Selected cell detailed information panel
        selected_cell = simulation.grid[simulation.selected_y][simulation.selected_x]
        info_x = mini_start_x
        info_y = mini_start_y + 370
        
        cell_title = title_font.render(f"CELL ({simulation.selected_x}, {simulation.selected_y}) READINGS", True, YELLOW)
        screen.blit(cell_title, (info_x, info_y))
        
        # Key parameters in compact layout
        critical_params = [
            ("Fire State", f"{['Normal', 'Preheating', 'Ignition', 'Burning', 'Flashover', 'Burned Out'][selected_cell.state]}"),
            ("Temperature", f"{selected_cell.temperature:.1f}°C"),
            ("HRR", f"{selected_cell.heat_release_rate:.0f}kW/m²"),
            ("Fuel Load", f"{selected_cell.fuel_load:.1f}MJ/m²"),
            ("Oxygen", f"{selected_cell.oxygen:.1f}%"),
            ("Smoke Density", f"{selected_cell.smoke_density*1000:.0f}mg/m³"),
            ("Visibility", f"{selected_cell.visibility_distance:.1f}m"),
            ("CO", f"{selected_cell.co_concentration:.0f}ppm"),
            ("HCN", f"{selected_cell.hcn_concentration:.1f}ppm"),
        ]
        
        # Display in 3 columns
        for i, (title, value) in enumerate(critical_params):
            col = i % 3
            row = i // 3
            x_pos = info_x + col * 200
            y_pos = info_y + 20 + row * 15
            
            # Color code dangerous values
            color = WHITE
            if "Temperature" in title and selected_cell.temperature > 60:
                color = RED
            elif "CO" in title and selected_cell.co_concentration > 1000:
                color = RED
            elif "HCN" in title and selected_cell.hcn_concentration > 50:
                color = RED
            elif "Visibility" in title and selected_cell.visibility_distance < 3:
                color = RED
            elif "Fire State" in title and selected_cell.state in [FLASHOVER]:
                color = RED
            
            title_surface = font.render(f"{title}:", True, CYAN)
            value_surface = font.render(value, True, color)
            
            screen.blit(title_surface, (x_pos, y_pos))
            screen.blit(value_surface, (x_pos + 80, y_pos))
        
        # Safety warnings and controls - horizontal layout
        warning_y = info_y + 80
        
        # Left side - warnings
        warning_title = title_font.render("SAFETY ALERTS", True, RED)
        screen.blit(warning_title, (info_x, warning_y))
        
        warnings = []
        if selected_cell.temperature > 60:
            warnings.append("⚠ HIGH TEMP")
        if selected_cell.co_concentration > 1000:
            warnings.append("⚠ TOXIC CO")
        if selected_cell.visibility_distance < 3:
            warnings.append("⚠ LOW VISIBILITY")
        if selected_cell.state == FLASHOVER:
            warnings.append("⚠ FLASHOVER!")
        if selected_cell.oxygen < 16:
            warnings.append("⚠ LOW O2")
        
        if not warnings:
            warnings.append("✓ SAFE")
            
        # Display warnings horizontally
        for i, warning in enumerate(warnings[:4]):
            color = RED if "⚠" in warning else GREEN
            warning_surface = font.render(warning, True, color)
            screen.blit(warning_surface, (info_x + i * 120, warning_y + 20))
        
        # Right side - controls
        control_x = info_x + 500
        control_title = title_font.render("CONTROLS", True, YELLOW)
        screen.blit(control_title, (control_x, warning_y))
        
        controls = [
            "↑↓←→: Navigate",
            "SPACE: Pause",
            "R: Reset",
            "E: Export Data"
        ]
        
        for i, text in enumerate(controls):
            control_surface = font.render(text, True, WHITE)
            screen.blit(control_surface, (control_x, warning_y + 20 + i * 12))
        
        # Data export status
        data_info = font.render(f"Data Points: {len(simulation.simulation_data)}", True, LIGHT_GRAY)
        screen.blit(data_info, (control_x, warning_y + 80))
        
        pygame.display.flip()
        clock.tick(10)  # 10 FPS
        
    pygame.quit()
    
    # Force exit after pygame quit
    if simulation.simulation_complete:
        import sys
        sys.exit(0)

if __name__ == "__main__":
    main()