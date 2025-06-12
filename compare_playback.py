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

class ComparisonPlayback:
    def __init__(self, test_file, pred_file):
        """
        Initialize comparison playback for Ground Truth vs AI Prediction
        
        Args:
            test_file: Path to test data (ground truth)
            pred_file: Path to prediction data (AI prediction)
        """
        # Load ground truth data
        self.gt_data, self.gt_metadata = self._load_data(test_file)
        
        # Load prediction data
        self.pred_data, self.pred_metadata = self._load_data(pred_file)
        
        if self.gt_data is None or self.pred_data is None:
            raise ValueError("Failed to load comparison data")
        
        # Verify data compatibility
        if self.gt_data.shape != self.pred_data.shape:
            raise ValueError(f"Data shape mismatch: GT {self.gt_data.shape} vs Pred {self.pred_data.shape}")
        
        # Get grid dimensions
        self.timesteps, self.height, self.width, self.variables = self.gt_data.shape
        
        # Playback controls
        self.current_timestep = 0
        self.playing = False
        self.playback_speed = 1.0
        self.selected_x = self.gt_metadata.get('ignition_point', [18, 11])[0]
        self.selected_y = self.gt_metadata.get('ignition_point', [18, 11])[1]
        
        # Statistics
        self.prediction_start = 25  # AI predictions start from timestep 25
        
        print(f"Loaded comparison data:")
        print(f"  Shape: {self.gt_data.shape}")
        print(f"  Ground Truth: {os.path.basename(test_file)}")
        print(f"  AI Prediction: {os.path.basename(pred_file)}")
        print(f"  Prediction starts from timestep: {self.prediction_start}")
    
    def _load_data(self, json_file):
        """Load simulation data from JSON and binary files"""
        try:
            # Load metadata
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            
            # Load binary data
            bin_file = json_file.replace('.json', '.bin')
            if not os.path.exists(bin_file):
                print(f"Warning: Binary file not found: {bin_file}")
                return None, None
            
            with open(bin_file, 'rb') as f:
                data_bytes = f.read()
            
            # Reconstruct data array
            data_shape = tuple(metadata['data_shape'])
            data = np.frombuffer(data_bytes, dtype=np.float32).reshape(data_shape)
            
            return data, metadata
            
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            return None, None
    
    def get_current_data(self):
        """Get current timestep data for both GT and prediction"""
        if 0 <= self.current_timestep < self.timesteps:
            gt_frame = self.gt_data[self.current_timestep]
            pred_frame = self.pred_data[self.current_timestep]
            return gt_frame, pred_frame
        return None, None
    
    def get_diff_data(self, gt_frame, pred_frame):
        """Calculate difference between ground truth and prediction"""
        # Only calculate diff for timesteps where prediction exists
        if self.current_timestep < self.prediction_start:
            # Before prediction starts, diff is zero
            return np.zeros_like(gt_frame)
        else:
            # Calculate absolute difference
            diff = np.abs(gt_frame - pred_frame)
            return diff
    
    def move_selection(self, dx, dy):
        self.selected_x = max(0, min(self.width - 1, self.selected_x + dx))
        self.selected_y = max(0, min(self.height - 1, self.selected_y + dy))
    
    def set_timestep(self, timestep):
        self.current_timestep = max(0, min(self.timesteps - 1, timestep))
    
    def next_timestep(self):
        if self.current_timestep < self.timesteps - 1:
            self.current_timestep += 1
            return True
        return False
    
    def prev_timestep(self):
        if self.current_timestep > 0:
            self.current_timestep -= 1
            return True
        return False
    
    
    def get_statistics(self, data, variable_name):
        """Calculate min, max, avg for environmental variables"""
        if data is None:
            return 0, 0, 0
            
        # Map variable names to indices
        var_indices = {
            'fire': 0,
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
        values = data[:, :, var_index].flatten()
        
        # Apply scaling for smoke (convert to mg/m³)
        if variable_name == 'smoke':
            values = values * 1000
            
        return float(np.min(values)), float(np.max(values)), float(np.mean(values))
    
    def get_cell_color(self, cell_data, mode):
        """Get color for a cell based on its data and display mode"""
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
                return (255, 200, 100)
            elif state == IGNITION:
                return YELLOW
            elif state == BURNING:
                return ORANGE
            elif state == FLASHOVER:
                return RED
            elif state == BURNED_OUT:
                return (80, 80, 80)
                
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
    
    def get_diff_color(self, diff_value, mode, max_diff=None):
        """Get color for difference visualization"""
        if diff_value == 0:
            return BLACK  # No difference
        
        # Normalize difference value
        if max_diff is None or max_diff == 0:
            diff_ratio = 0
        else:
            diff_ratio = min(1.0, diff_value / max_diff)
        
        # Color scale from green (small diff) to red (large diff)
        if diff_ratio < 0.5:
            # Green to yellow
            green_val = 255
            red_val = int(255 * diff_ratio * 2)
            blue_val = 0
        else:
            # Yellow to red
            red_val = 255
            green_val = int(255 * (1 - (diff_ratio - 0.5) * 2))
            blue_val = 0
        
        return (red_val, green_val, blue_val)


def select_comparison_files():
    """Select test and prediction files for comparison"""
    root = tk.Tk()
    root.withdraw()
    
    # Select test file
    test_files = glob.glob("test_dataset/fire_simulation_*.json")
    pred_files = glob.glob("pred_dataset/fire_simulation_*.json")
    
    if not test_files:
        print("No test files found in test_dataset/")
        return None, None
    
    if not pred_files:
        print("No prediction files found in pred_dataset/")
        print("Please run inference_model.py first to generate predictions.")
        return None, None
    
    print("Available test files:")
    for i, file in enumerate(test_files):
        base_name = os.path.basename(file)
        pred_file = f"pred_dataset/{base_name}"
        exists = "✓" if os.path.exists(pred_file) else "✗"
        print(f"{i+1}. {base_name} {exists}")
    
    try:
        choice = input(f"Select test file (1-{len(test_files)}): ")
        file_index = int(choice) - 1
        if 0 <= file_index < len(test_files):
            test_file = test_files[file_index]
            pred_file = f"pred_dataset/{os.path.basename(test_file)}"
            
            if os.path.exists(pred_file):
                return test_file, pred_file
            else:
                print(f"Prediction file not found: {pred_file}")
                return None, None
    except (ValueError, EOFError):
        # Auto-select first available pair
        for test_file in test_files:
            pred_file = f"pred_dataset/{os.path.basename(test_file)}"
            if os.path.exists(pred_file):
                print(f"Auto-selecting: {os.path.basename(test_file)}")
                return test_file, pred_file
    
    return None, None


def main():
    # Select comparison files
    test_file, pred_file = select_comparison_files()
    if not test_file or not pred_file:
        print("Could not find matching test and prediction files.")
        return
    
    try:
        playback = ComparisonPlayback(test_file, pred_file)
    except Exception as e:
        print(f"Error loading comparison data: {e}")
        return
    
    pygame.init()
    
    # Screen settings - larger for all minimaps
    MINI_CELL_SIZE = 5  # Slightly bigger cells
    GRID_WIDTH = playback.width
    GRID_HEIGHT = playback.height
    SCREEN_WIDTH = 1700
    SCREEN_HEIGHT = 1300
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"Ground Truth vs AI Prediction Comparison")
    font = pygame.font.Font(None, 12)
    title_font = pygame.font.Font(None, 14)
    main_title_font = pygame.font.Font(None, 16)
    
    clock = pygame.time.Clock()
    running = True
    frame_counter = 0
    
    # Variable names
    variable_names = [
        "Fire State", "Temperature", "Smoke Density", "Visibility",
        "CO Level", "HCN Level", "Air Velocity", "Thermal Radiation", "Pressure"
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
                    if pygame.key.get_pressed()[pygame.K_LCTRL]:
                        playback.prev_timestep()
                    else:
                        playback.move_selection(-1, 0)
                elif event.key == pygame.K_RIGHT:
                    if pygame.key.get_pressed()[pygame.K_LCTRL]:
                        playback.next_timestep()
                    else:
                        playback.move_selection(1, 0)
                elif event.key == pygame.K_UP:
                    if pygame.key.get_pressed()[pygame.K_LCTRL]:
                        playback.playback_speed = min(5.0, playback.playback_speed + 0.5)
                    else:
                        playback.move_selection(0, -1)
                elif event.key == pygame.K_DOWN:
                    if pygame.key.get_pressed()[pygame.K_LCTRL]:
                        playback.playback_speed = max(0.1, playback.playback_speed - 0.5)
                    else:
                        playback.move_selection(0, 1)
                elif event.key == pygame.K_HOME:
                    playback.current_timestep = 0
                elif event.key == pygame.K_END:
                    playback.current_timestep = playback.timesteps - 1
        
        # Auto-advance if playing
        if playback.playing:
            frame_counter += playback.playback_speed
            if frame_counter >= 10:
                if not playback.next_timestep():
                    playback.playing = False
                frame_counter = 0
        
        # Clear screen
        screen.fill(BLACK)
        
        # Get current data
        gt_frame, pred_frame = playback.get_current_data()
        if gt_frame is None:
            continue
        
        diff_frame = playback.get_diff_data(gt_frame, pred_frame)
        
        # Main title
        main_title = main_title_font.render("GROUND TRUTH vs AI PREDICTION - ALL VARIABLES COMPARISON", True, CYAN)
        screen.blit(main_title, (20, 5))
        
        # Time and control info
        current_time = playback.current_timestep
        status = "▶ PLAYING" if playback.playing else "⏸ PAUSED"
        status_color = GREEN if playback.playing else YELLOW
        
        pred_status = "AI PREDICTION" if playback.current_timestep >= playback.prediction_start else "NO PREDICTION"
        pred_color = GREEN if playback.current_timestep >= playback.prediction_start else RED
        
        time_info = title_font.render(f"Time: {current_time}s | {status} | Speed: {playback.playback_speed:.1f}x", True, WHITE)
        screen.blit(time_info, (20, 22))
        
        pred_info = title_font.render(f"Status: {pred_status} | Selected Cell: ({playback.selected_x}, {playback.selected_y})", True, pred_color)
        screen.blit(pred_info, (20, 38))
        
        # 3행 레이아웃: Ground Truth, AI Prediction, Difference
        mini_panel_width = GRID_WIDTH * MINI_CELL_SIZE
        mini_panel_height = GRID_HEIGHT * MINI_CELL_SIZE
        
        start_x = 20
        start_y = 60
        
        # Variable info
        variable_info = [
            ("Fire State", "state"),
            ("Temperature", "°C"),
            ("Smoke", "od/m"),
            ("Visibility", "m"),
            ("CO", "ppm"),
            ("HCN", "ppm"),
            ("Air Vel", "m/s"),
            ("Thermal", "kW/m²"),
            ("Pressure", "Pa")
        ]
        
        # Row titles
        row_titles = ["GROUND TRUTH", "AI PREDICTION", "DIFFERENCE MAP"]
        row_colors = [YELLOW, CYAN, RED]
        
        # Draw 3 rows x 9 columns grid
        for row in range(3):  # 3 rows: GT, Pred, Diff
            # Row title
            row_title = title_font.render(row_titles[row], True, row_colors[row])
            screen.blit(row_title, (start_x - 15, start_y + row * (mini_panel_height + 60) + 20))
            
            for col in range(9):  # 9 variables
                var_idx = col
                var_name, var_unit = variable_info[var_idx]
                
                # Calculate position
                map_x = start_x + col * (mini_panel_width + 15)
                map_y = start_y + row * (mini_panel_height + 60) + 40
                
                # Variable title (only for first row)
                if row == 0:
                    var_title = font.render(f"{var_name}", True, WHITE)
                    title_rect = var_title.get_rect()
                    title_x = map_x + (mini_panel_width - title_rect.width) // 2
                    screen.blit(var_title, (title_x, map_y - 20))
                    
                    unit_title = font.render(f"({var_unit})", True, GRAY)
                    unit_rect = unit_title.get_rect()
                    unit_x = map_x + (mini_panel_width - unit_rect.width) // 2
                    screen.blit(unit_title, (unit_x, map_y - 10))
                
                # Calculate max difference for normalization
                max_diff = np.max(diff_frame[:, :, var_idx]) if np.any(diff_frame) else 1.0
                
                # Draw minimap
                for y in range(GRID_HEIGHT):
                    for x in range(GRID_WIDTH):
                        cell_rect = pygame.Rect(map_x + x * MINI_CELL_SIZE, map_y + y * MINI_CELL_SIZE, 
                                              MINI_CELL_SIZE, MINI_CELL_SIZE)
                        
                        if row == 0:  # Ground Truth
                            cell_data = gt_frame[y, x, :]
                            color = playback.get_cell_color(cell_data, var_idx)
                        elif row == 1:  # AI Prediction
                            cell_data = pred_frame[y, x, :]
                            color = playback.get_cell_color(cell_data, var_idx)
                        else:  # Difference
                            diff_value = diff_frame[y, x, var_idx]
                            color = playback.get_diff_color(diff_value, var_idx, max_diff)
                        
                        pygame.draw.rect(screen, color, cell_rect)
                        
                        # Highlight selected cell
                        if x == playback.selected_x and y == playback.selected_y:
                            pygame.draw.rect(screen, WHITE, cell_rect, 1)
                
                # Statistics (only for difference row)
                if row == 2:
                    var_names_short = ['fire', 'temperature', 'smoke', 'visibility', 'co', 'hcn', 'air_velocity', 'thermal_radiation', 'pressure']
                    current_var = var_names_short[var_idx]
                    
                    gt_min, gt_max, gt_avg = playback.get_statistics(gt_frame, current_var)
                    pred_min, pred_max, pred_avg = playback.get_statistics(pred_frame, current_var)
                    
                    # Show average error below diff map
                    avg_error = abs(gt_avg - pred_avg)
                    error_text = font.render(f"Err:{avg_error:.1f}", True, RED)
                    error_rect = error_text.get_rect()
                    error_x = map_x + (mini_panel_width - error_rect.width) // 2
                    screen.blit(error_text, (error_x, map_y + mini_panel_height + 2))
        
        # Selected cell detailed info panel (bottom)
        info_x = start_x
        info_y = start_y + 3 * (mini_panel_height + 60) + 70
        
        cell_title = title_font.render(f"SELECTED CELL ({playback.selected_x}, {playback.selected_y})", True, WHITE)
        screen.blit(cell_title, (info_x, info_y))
        
        gt_cell = gt_frame[playback.selected_y, playback.selected_x, :]
        pred_cell = pred_frame[playback.selected_y, playback.selected_x, :]
        diff_cell = diff_frame[playback.selected_y, playback.selected_x, :]
        
        # Show all 9 variables for selected cell
        info_y += 20
        for var_idx in range(9):
            var_name = variable_info[var_idx][0]
            gt_val = gt_cell[var_idx]
            pred_val = pred_cell[var_idx]
            diff_val = diff_cell[var_idx]
            
            # Calculate accuracy
            accuracy = (1 - min(1, diff_val / max(0.001, abs(gt_val)))) * 100 if gt_val != 0 else (100 if diff_val == 0 else 0)
            
            # Compact display
            cell_info = f"{var_name[:8]}: GT:{gt_val:.1f} AI:{pred_val:.1f} Err:{diff_val:.1f} ({accuracy:.0f}%)"
            
            # Color based on accuracy
            color = GREEN if accuracy > 90 else ORANGE if accuracy > 70 else RED
            
            info_surface = font.render(cell_info, True, color)
            screen.blit(info_surface, (info_x, info_y + var_idx * 12))
        
        # Controls (bottom right corner)
        controls_y = info_y + 9 * 12 + 20
        controls_title = title_font.render("CONTROLS", True, YELLOW)
        screen.blit(controls_title, (info_x, controls_y))
        
        controls = [
            "SPACE: Play/Pause",
            "↑↓←→: Move Selection",
            "CTRL+←→: Step Forward/Back", 
            "CTRL+↑↓: Speed Up/Down",
            "HOME/END: First/Last Frame",
            "R: Reset to Start"
        ]
        
        controls_y += 15
        for i, control in enumerate(controls):
            control_surface = font.render(control, True, WHITE)
            screen.blit(control_surface, (info_x, controls_y + i * 10))
        
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()


if __name__ == "__main__":
    main()