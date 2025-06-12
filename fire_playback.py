import pygame
import json
import math
import os
import glob
import numpy as np
from tkinter import filedialog
import tkinter as tk

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

class FirePlayback:
    def __init__(self, json_file):
        # Load metadata
        with open(json_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Load binary data
        bin_file = json_file.replace('.json', '.bin')
        if not os.path.exists(bin_file):
            raise ValueError(f"Binary data file not found: {bin_file}")
            
        # Read binary data
        with open(bin_file, 'rb') as f:
            data_bytes = f.read()
        
        # Reconstruct numpy array from metadata
        data_shape = tuple(self.metadata['data_shape'])
        self.simulation_data = np.frombuffer(data_bytes, dtype=np.float32).reshape(data_shape)
        
        # Get grid dimensions
        self.height = self.metadata['grid_height']
        self.width = self.metadata['grid_width']
        
        # Playback controls
        self.current_timestep = 0
        self.max_timesteps = self.metadata['total_timesteps']
        self.playing = False
        self.playback_speed = 1.0  # Speed multiplier
        self.selected_x = self.metadata['ignition_point'][0]
        self.selected_y = self.metadata['ignition_point'][1]
        
        print(f"Loaded simulation data: {self.max_timesteps} timesteps, {self.height}x{self.width} grid")
        print(f"Duration: {self.metadata['simulation_duration_seconds']:.1f}s, File size: {self.metadata['file_size_bytes']/1024/1024:.1f} MB")
        
    def get_current_data(self):
        """Get data for current timestep"""
        if 0 <= self.current_timestep < self.max_timesteps:
            return self.simulation_data[self.current_timestep]  # Shape: [height, width, 9]
        return None
        
    def get_cell_data(self, x, y):
        """Get data for specific cell at current timestep"""
        current_data = self.get_current_data()
        if current_data is not None and 0 <= y < self.height and 0 <= x < self.width:
            return current_data[y, x, :]  # Return 9-element array
        return None
        
    def move_selection(self, dx, dy):
        self.selected_x = max(0, min(self.width - 1, self.selected_x + dx))
        self.selected_y = max(0, min(self.height - 1, self.selected_y + dy))
        
    def set_timestep(self, timestep):
        self.current_timestep = max(0, min(self.max_timesteps - 1, timestep))
        
    def next_timestep(self):
        if self.current_timestep < self.max_timesteps - 1:
            self.current_timestep += 1
            return True
        return False
        
    def prev_timestep(self):
        if self.current_timestep > 0:
            self.current_timestep -= 1
            return True
        return False
        
    def get_statistics(self, variable_name):
        """Calculate min, max, avg for environmental variables at current timestep"""
        current_data = self.get_current_data()
        if current_data is None:
            return 0, 0, 0
            
        # Map variable names to indices
        var_indices = {
            'temperature': 1,
            'smoke': 2,
            'visibility': 3, 
            'co': 4,
            'hcn': 5,
            'air_velocity': 6,
            'thermal_radiation': 7,
            'pressure': 8
        }
        
        if variable_name not in var_indices:
            return 0, 0, 0
            
        var_index = var_indices[variable_name]
        values = current_data[:, :, var_index].flatten()
        
        # Apply scaling for smoke (convert to mg/m³)
        if variable_name == 'smoke':
            values = values * 1000
            
        return float(np.min(values)), float(np.max(values)), float(np.mean(values))
        
    def get_cell_color(self, cell_data, mode):
        if cell_data is None:
            return WHITE
            
        state = int(cell_data[0])
        temperature = float(cell_data[1])
        smoke_density = float(cell_data[2])
        visibility = float(cell_data[3])
        co_concentration = float(cell_data[4])
        hcn_concentration = float(cell_data[5])
        air_velocity = float(cell_data[6])
        thermal_radiation = float(cell_data[7])
        pressure = float(cell_data[8])
        
        if mode == FIRE_MODE:
            if state == NORMAL:
                return GREEN
            elif state == PREHEATING:
                return (255, 200, 100)  # Light orange
            elif state == IGNITION:
                return YELLOW
            elif state == BURNING:
                return ORANGE
            elif state == FLASHOVER:
                return RED
            elif state == BURNED_OUT:
                return (80, 80, 80)  # Dark gray
                
        elif mode == TEMPERATURE_MODE:
            temp_ratio = min(1.0, max(0.0, (temperature - 20.0) / 1000.0))
            blue_val = max(0, min(255, int(255 * (1 - temp_ratio))))
            red_val = max(0, min(255, int(255 * temp_ratio)))
            return (red_val, 0, blue_val)
            
        elif mode == SMOKE_MODE:
            smoke_ratio = min(1.0, max(0.0, smoke_density / 4.0))
            val = max(0, min(255, int(255 * (1 - smoke_ratio))))
            return (val, val, val)
            
        elif mode == VISIBILITY_MODE:
            vis_ratio = min(1.0, max(0.0, visibility / 30.0))
            red_val = max(0, min(255, int(255 * (1 - vis_ratio))))
            green_val = max(0, min(255, int(255 * vis_ratio)))
            return (red_val, green_val, 0)
            
        elif mode == CO_MODE:
            co_ratio = min(1.0, max(0.0, co_concentration / 12000.0))
            green_val = max(0, min(255, int(255 * (1 - co_ratio))))
            red_val = max(0, min(255, int(255 * co_ratio)))
            return (red_val, green_val, 0)
            
        elif mode == HCN_MODE:
            hcn_ratio = min(1.0, max(0.0, hcn_concentration / 1200.0))
            cyan_val = max(0, min(255, int(255 * (1 - hcn_ratio))))
            purple_val = max(0, min(255, int(255 * hcn_ratio)))
            return (purple_val, 0, cyan_val)
            
        elif mode == AIR_VELOCITY_MODE:
            vel_ratio = min(1.0, max(0.0, air_velocity / 6.0))
            blue_val = max(0, min(255, int(255 * (1 - vel_ratio))))
            red_val = max(0, min(255, int(255 * vel_ratio)))
            return (red_val, 0, blue_val)
            
        elif mode == THERMAL_RADIATION_MODE:
            rad_ratio = min(1.0, max(0.0, thermal_radiation / 80.0))
            orange_r = max(0, min(255, int(255 * rad_ratio)))
            orange_g = max(0, min(165, int(165 * rad_ratio)))
            return (orange_r, orange_g, 0)
            
        elif mode == PRESSURE_MODE:
            pressure_ratio = min(1.0, max(0.0, (pressure - 101000) / 1200.0))
            blue_val = max(0, min(255, int(255 * (1 - pressure_ratio))))
            red_val = max(0, min(255, int(255 * pressure_ratio)))
            return (red_val, 0, blue_val)
            
        return WHITE

def select_data_file():
    """Open file dialog to select simulation metadata file"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Look for JSON metadata files in train_dataset directory
    json_files = glob.glob("train_dataset/fire_simulation_*.json")
    
    if json_files:
        print("Available simulation files:")
        for i, file in enumerate(json_files):
            # Check if corresponding .bin file exists
            bin_file = file.replace('.json', '.bin')
            if os.path.exists(bin_file):
                # Try to get file info
                try:
                    with open(file, 'r') as f:
                        metadata = json.load(f)
                    duration = metadata.get('simulation_duration_seconds', 0)
                    timesteps = metadata.get('total_timesteps', 0)
                    print(f"{i+1}. {file} ({timesteps} timesteps, {duration:.1f}s)")
                except:
                    print(f"{i+1}. {file}")
            else:
                print(f"{i+1}. {file} (missing .bin file)")
        
        try:
            choice = input(f"Select file (1-{len(json_files)}) or press Enter to browse: ")
            if choice.strip():
                file_index = int(choice) - 1
                if 0 <= file_index < len(json_files):
                    return json_files[file_index]
        except (ValueError, EOFError):
            # If input fails, return the first valid file
            for file in json_files:
                bin_file = file.replace('.json', '.bin')
                if os.path.exists(bin_file):
                    print(f"Auto-selecting: {file}")
                    return file
    
    # Open file dialog in train_dataset directory
    initial_dir = os.path.join(os.getcwd(), "train_dataset")
    os.makedirs(initial_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    file_path = filedialog.askopenfilename(
        title="Select Fire Simulation Metadata File",
        initialdir=initial_dir,
        filetypes=[("JSON metadata files", "*.json"), ("All files", "*.*")]
    )
    
    root.destroy()
    return file_path

def main():
    # Select data file
    data_file = select_data_file()
    if not data_file:
        print("No file selected. Exiting.")
        return
        
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return
    
    try:
        playback = FirePlayback(data_file)
    except Exception as e:
        print(f"Error loading data file: {e}")
        return
    
    pygame.init()
    
    # Screen settings
    MAIN_CELL_SIZE = 24
    MINI_CELL_SIZE = 6
    GRID_WIDTH = playback.width
    GRID_HEIGHT = playback.height
    SCREEN_WIDTH = 1200
    SCREEN_HEIGHT = 700
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"Fire Simulation Playback - {os.path.basename(data_file)}")
    font = pygame.font.Font(None, 14)
    title_font = pygame.font.Font(None, 16)
    main_title_font = pygame.font.Font(None, 20)
    
    clock = pygame.time.Clock()
    
    running = True
    frame_counter = 0
    
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
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    playback.playing = not playback.playing
                elif event.key == pygame.K_r:
                    playback.current_timestep = 0
                elif event.key == pygame.K_LEFT:
                    if pygame.key.get_pressed()[pygame.K_LCTRL] or pygame.key.get_pressed()[pygame.K_RCTRL]:
                        playback.prev_timestep()
                    else:
                        playback.move_selection(-1, 0)
                elif event.key == pygame.K_RIGHT:
                    if pygame.key.get_pressed()[pygame.K_LCTRL] or pygame.key.get_pressed()[pygame.K_RCTRL]:
                        playback.next_timestep()
                    else:
                        playback.move_selection(1, 0)
                elif event.key == pygame.K_UP:
                    if pygame.key.get_pressed()[pygame.K_LCTRL] or pygame.key.get_pressed()[pygame.K_RCTRL]:
                        playback.playback_speed = min(5.0, playback.playback_speed + 0.5)
                    else:
                        playback.move_selection(0, -1)
                elif event.key == pygame.K_DOWN:
                    if pygame.key.get_pressed()[pygame.K_LCTRL] or pygame.key.get_pressed()[pygame.K_RCTRL]:
                        playback.playback_speed = max(0.1, playback.playback_speed - 0.5)
                    else:
                        playback.move_selection(0, 1)
                elif event.key == pygame.K_HOME:
                    playback.current_timestep = 0
                elif event.key == pygame.K_END:
                    playback.current_timestep = playback.max_timesteps - 1
        
        # Auto-advance if playing
        if playback.playing:
            frame_counter += playback.playback_speed
            if frame_counter >= 10:  # Advance every 10 frames (adjusted by speed)
                if not playback.next_timestep():
                    playback.playing = False  # Stop at end
                frame_counter = 0
        
        # Clear screen
        screen.fill(BLACK)
        
        # Get current timestep data
        current_data = playback.get_current_data()
        if current_data is None:
            # Show error message
            error_text = main_title_font.render("No data available", True, RED)
            screen.blit(error_text, (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2))
            pygame.display.flip()
            continue
        
        # Draw main fire state grid (left side)
        main_start_x = 20
        main_start_y = 40
        
        # Main title
        main_title = main_title_font.render("FIRE SIMULATION PLAYBACK", True, CYAN)
        screen.blit(main_title, (main_start_x, 5))
        
        # Playback controls info - calculate time from timestep
        current_time = playback.current_timestep * (playback.metadata['simulation_duration_seconds'] / playback.max_timesteps)
        time_info = title_font.render(f"Time: {current_time:.0f}s | Step: {playback.current_timestep+1}/{playback.max_timesteps} | Speed: {playback.playback_speed:.1f}x", True, WHITE)
        screen.blit(time_info, (main_start_x, 22))
        
        # Playback status
        status = "▶ PLAYING" if playback.playing else "⏸ PAUSED"
        status_color = GREEN if playback.playing else YELLOW
        status_text = title_font.render(status, True, status_color)
        screen.blit(status_text, (main_start_x + 400, 22))
        
        # Draw main grid
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                cell_data = current_data[y, x, :]
                color = playback.get_cell_color(cell_data, FIRE_MODE)
                rect = pygame.Rect(main_start_x + x * MAIN_CELL_SIZE, main_start_y + y * MAIN_CELL_SIZE, 
                                 MAIN_CELL_SIZE, MAIN_CELL_SIZE)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, DARK_GRAY, rect, 1)
                
                # Highlight selected cell
                if x == playback.selected_x and y == playback.selected_y:
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
            min_val, max_val, avg_val = playback.get_statistics(var_key)
            
            # Draw mini map title with unit
            title_with_unit = f"{var_name} ({unit})"
            title_text = title_font.render(title_with_unit, True, WHITE)
            screen.blit(title_text, (map_x, map_y - 18))
            
            # Draw mini grid
            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    cell_data = current_data[y, x, :]
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
                    color = playback.get_cell_color(cell_data, mode_map[var_key])
                    rect = pygame.Rect(map_x + x * MINI_CELL_SIZE, map_y + y * MINI_CELL_SIZE, 
                                     MINI_CELL_SIZE, MINI_CELL_SIZE)
                    pygame.draw.rect(screen, color, rect)
                    
                    # Highlight selected cell
                    if x == playback.selected_x and y == playback.selected_y:
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
        selected_cell_data = playback.get_cell_data(playback.selected_x, playback.selected_y)
        if selected_cell_data is not None:
            info_x = mini_start_x
            info_y = mini_start_y + 370
            
            cell_title = title_font.render(f"CELL ({playback.selected_x}, {playback.selected_y}) READINGS", True, YELLOW)
            screen.blit(cell_title, (info_x, info_y))
            
            # Extract cell data
            state = int(selected_cell_data[0])
            temperature = selected_cell_data[1]
            smoke_density = selected_cell_data[2]
            visibility = selected_cell_data[3]
            co_concentration = selected_cell_data[4]
            hcn_concentration = selected_cell_data[5]
            air_velocity = selected_cell_data[6]
            thermal_radiation = selected_cell_data[7]
            pressure = selected_cell_data[8]
            
            # Key parameters in compact layout
            critical_params = [
                ("Fire State", f"{['Normal', 'Preheating', 'Ignition', 'Burning', 'Flashover', 'Burned Out'][state]}"),
                ("Temperature", f"{temperature:.1f}°C"),
                ("Thermal Rad", f"{thermal_radiation:.0f}kW/m²"),
                ("Smoke Density", f"{smoke_density*1000:.0f}mg/m³"),
                ("Visibility", f"{visibility:.1f}m"),
                ("CO", f"{co_concentration:.0f}ppm"),
                ("HCN", f"{hcn_concentration:.1f}ppm"),
                ("Air Velocity", f"{air_velocity:.1f}m/s"),
                ("Pressure", f"{pressure:.0f}Pa"),
            ]
            
            # Display in 3 columns
            for i, (title, value) in enumerate(critical_params):
                col = i % 3
                row = i // 3
                x_pos = info_x + col * 200
                y_pos = info_y + 20 + row * 15
                
                # Color code dangerous values
                color = WHITE
                if "Temperature" in title and temperature > 60:
                    color = RED
                elif "CO" in title and co_concentration > 1000:
                    color = RED
                elif "HCN" in title and hcn_concentration > 50:
                    color = RED
                elif "Visibility" in title and visibility < 3:
                    color = RED
                elif "Fire State" in title and state == FLASHOVER:
                    color = RED
                
                title_surface = font.render(f"{title}:", True, CYAN)
                value_surface = font.render(value, True, color)
                
                screen.blit(title_surface, (x_pos, y_pos))
                screen.blit(value_surface, (x_pos + 80, y_pos))
        
        # Playback controls - horizontal layout
        control_y = mini_start_y + 480
        
        control_title = title_font.render("PLAYBACK CONTROLS", True, YELLOW)
        screen.blit(control_title, (mini_start_x, control_y))
        
        controls = [
            "SPACE: Play/Pause",
            "↑↓←→: Move Cell Selection", 
            "CTRL+←→: Step Back/Forward",
            "CTRL+↑↓: Speed Up/Down",
            "HOME/END: First/Last Frame",
            "R: Reset to Start"
        ]
        
        for i, text in enumerate(controls):
            control_surface = font.render(text, True, WHITE)
            screen.blit(control_surface, (mini_start_x + (i % 2) * 300, control_y + 20 + (i // 2) * 12))
        
        # Progress bar
        progress_y = control_y + 80
        progress_width = 600
        progress_height = 10
        
        # Background
        progress_bg = pygame.Rect(mini_start_x, progress_y, progress_width, progress_height)
        pygame.draw.rect(screen, DARK_GRAY, progress_bg)
        
        # Progress
        if playback.max_timesteps > 1:
            progress = playback.current_timestep / (playback.max_timesteps - 1)
            progress_fill = pygame.Rect(mini_start_x, progress_y, int(progress_width * progress), progress_height)
            pygame.draw.rect(screen, GREEN, progress_fill)
        
        pygame.draw.rect(screen, WHITE, progress_bg, 1)
        
        pygame.display.flip()
        clock.tick(120)  # 30 FPS for smooth playback
        
    pygame.quit()

if __name__ == "__main__":
    main()