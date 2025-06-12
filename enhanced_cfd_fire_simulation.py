import numpy as np
import math

class EnhancedFireCell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
        # 기존 화재 속성들
        self.state = 0  # NORMAL
        self.temperature = 20.0
        self.fuel_load = 40.0
        self.smoke_density = 0.0
        
        # CFD 속성들 추가
        self.velocity_u = 0.0      # x방향 속도 (m/s)
        self.velocity_v = 0.0      # y방향 속도 (m/s)
        self.pressure = 101325.0   # 압력 (Pa)
        self.density = 1.225       # 공기 밀도 (kg/m³)
        
        # 이전 시간 스텝 값들 (시간 적분용)
        self.prev_velocity_u = 0.0
        self.prev_velocity_v = 0.0
        self.prev_temperature = 20.0
        self.prev_smoke_density = 0.0

class SimplifiedCFDFireSimulation:
    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        self.grid = [[EnhancedFireCell(x, y) for x in range(width)] for y in range(height)]
        
        # CFD 파라미터들
        self.dt = 1.0           # 시간 스텝 (초)
        self.dx = 1.0           # 격자 크기 (m)
        self.dy = 1.0           # 격자 크기 (m)
        
        # 물리 상수들
        self.gravity = 9.81     # 중력 가속도
        self.viscosity = 1.8e-5 # 공기 점성계수 (Pa·s)
        self.thermal_conductivity = 0.024  # 공기 열전도도 (W/m·K)
        self.specific_heat = 1005  # 공기 비열 (J/kg·K)
        
        # 수치해석 파라미터들
        self.relaxation_factor = 0.7  # Under-relaxation factor
        self.max_iterations = 10      # 압력 수정 반복 횟수
        
    def get_neighbors(self, x, y):
        """8방향 이웃 셀들 반환"""
        neighbors = []
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                neighbors.append(self.grid[ny][nx])
        return neighbors
    
    def calculate_momentum_equations(self):
        """운동량 방정식 풀이 (간단화된 Navier-Stokes)"""
        new_u = np.zeros((self.height, self.width))
        new_v = np.zeros((self.height, self.width))
        
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                cell = self.grid[y][x]
                
                # 이웃 셀들
                east = self.grid[y][x+1]
                west = self.grid[y][x-1] 
                north = self.grid[y-1][x]
                south = self.grid[y+1][x]
                
                # u-velocity (x방향) 운동량 방정식
                # ∂u/∂t + u∂u/∂x + v∂u/∂y = -1/ρ ∂p/∂x + ν∇²u + buoyancy_x
                
                # 대류항 (간단화: upwind scheme)
                if cell.velocity_u > 0:
                    dudx = (cell.velocity_u - west.velocity_u) / self.dx
                else:
                    dudx = (east.velocity_u - cell.velocity_u) / self.dx
                    
                if cell.velocity_v > 0:
                    dudy = (cell.velocity_u - south.velocity_u) / self.dy
                else:
                    dudy = (north.velocity_u - cell.velocity_u) / self.dy
                
                convection_u = cell.velocity_u * dudx + cell.velocity_v * dudy
                
                # 압력 구배
                pressure_grad_x = (east.pressure - west.pressure) / (2 * self.dx)
                
                # 점성 확산 (라플라시안)
                d2udx2 = (east.velocity_u - 2*cell.velocity_u + west.velocity_u) / (self.dx**2)
                d2udy2 = (north.velocity_u - 2*cell.velocity_u + south.velocity_u) / (self.dy**2)
                viscous_u = self.viscosity / cell.density * (d2udx2 + d2udy2)
                
                # 부력 효과 (화재로 인한)
                if cell.state >= 2:  # 점화 이상
                    temp_diff = cell.temperature - 20.0
                    buoyancy_u = 0.1 * temp_diff / (20.0 + 273.15) * self.gravity
                else:
                    buoyancy_u = 0.0
                
                # u-속도 업데이트
                du_dt = -convection_u - pressure_grad_x / cell.density + viscous_u + buoyancy_u
                new_u[y][x] = cell.velocity_u + self.dt * du_dt
                
                # v-velocity (y방향) 운동량 방정식
                # 대류항
                if cell.velocity_u > 0:
                    dvdx = (cell.velocity_v - west.velocity_v) / self.dx
                else:
                    dvdx = (east.velocity_v - cell.velocity_v) / self.dx
                    
                if cell.velocity_v > 0:
                    dvdy = (cell.velocity_v - south.velocity_v) / self.dy
                else:
                    dvdy = (north.velocity_v - cell.velocity_v) / self.dy
                
                convection_v = cell.velocity_u * dvdx + cell.velocity_v * dvdy
                
                # 압력 구배
                pressure_grad_y = (north.pressure - south.pressure) / (2 * self.dy)
                
                # 점성 확산
                d2vdx2 = (east.velocity_v - 2*cell.velocity_v + west.velocity_v) / (self.dx**2)
                d2vdy2 = (north.velocity_v - 2*cell.velocity_v + south.velocity_v) / (self.dy**2)
                viscous_v = self.viscosity / cell.density * (d2vdx2 + d2vdy2)
                
                # 부력 효과 (주로 수직 방향)
                if cell.state >= 2:  # 점화 이상
                    temp_diff = cell.temperature - 20.0
                    buoyancy_v = 0.3 * temp_diff / (20.0 + 273.15) * self.gravity
                else:
                    buoyancy_v = 0.0
                
                # v-속도 업데이트
                dv_dt = -convection_v - pressure_grad_y / cell.density + viscous_v + buoyancy_v
                new_v[y][x] = cell.velocity_v + self.dt * dv_dt
        
        # 속도 업데이트 (경계 조건 적용)
        for y in range(self.height):
            for x in range(self.width):
                if 1 <= x < self.width-1 and 1 <= y < self.height-1:
                    self.grid[y][x].velocity_u = new_u[y][x]
                    self.grid[y][x].velocity_v = new_v[y][x]
                else:
                    # 벽면 경계 조건 (no-slip)
                    self.grid[y][x].velocity_u = 0.0
                    self.grid[y][x].velocity_v = 0.0
    
    def solve_pressure_correction(self):
        """압력 수정 방정식 풀이 (간단한 SIMPLE 방법)"""
        for iteration in range(self.max_iterations):
            new_pressure = np.zeros((self.height, self.width))
            
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    cell = self.grid[y][x]
                    
                    # 이웃 셀들
                    east = self.grid[y][x+1]
                    west = self.grid[y][x-1]
                    north = self.grid[y-1][x]
                    south = self.grid[y+1][x]
                    
                    # 연속 방정식 잔차 계산
                    # ∂ρ/∂t + ∂(ρu)/∂x + ∂(ρv)/∂y = 0
                    mass_flux_x = (east.density * east.velocity_u - west.density * west.velocity_u) / (2 * self.dx)
                    mass_flux_y = (north.density * north.velocity_v - south.density * south.velocity_v) / (2 * self.dy)
                    
                    continuity_residual = mass_flux_x + mass_flux_y
                    
                    # 압력 수정 방정식 계수들
                    ap = 2.0 / (self.dx**2) + 2.0 / (self.dy**2)
                    ae = 1.0 / (self.dx**2)
                    aw = 1.0 / (self.dx**2)
                    an = 1.0 / (self.dy**2)
                    as_coef = 1.0 / (self.dy**2)
                    
                    # 압력 수정
                    pressure_correction = (ae * east.pressure + aw * west.pressure + 
                                         an * north.pressure + as_coef * south.pressure - 
                                         continuity_residual * self.dt) / ap
                    
                    new_pressure[y][x] = cell.pressure + self.relaxation_factor * (pressure_correction - cell.pressure)
            
            # 압력 업데이트
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    self.grid[y][x].pressure = new_pressure[y][x]
    
    def solve_energy_equation(self):
        """에너지 방정식 풀이 (온도 전달)"""
        new_temp = np.zeros((self.height, self.width))
        
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                cell = self.grid[y][x]
                
                # 이웃 셀들
                east = self.grid[y][x+1]
                west = self.grid[y][x-1]
                north = self.grid[y-1][x]
                south = self.grid[y+1][x]
                
                # 대류 열전달
                if cell.velocity_u > 0:
                    dTdx = (cell.temperature - west.temperature) / self.dx
                else:
                    dTdx = (east.temperature - cell.temperature) / self.dx
                    
                if cell.velocity_v > 0:
                    dTdy = (cell.temperature - south.temperature) / self.dy
                else:
                    dTdy = (north.temperature - cell.temperature) / self.dy
                
                convection_heat = cell.velocity_u * dTdx + cell.velocity_v * dTdy
                
                # 전도 열전달
                d2Tdx2 = (east.temperature - 2*cell.temperature + west.temperature) / (self.dx**2)
                d2Tdy2 = (north.temperature - 2*cell.temperature + south.temperature) / (self.dy**2)
                
                thermal_diffusivity = self.thermal_conductivity / (cell.density * self.specific_heat)
                conduction_heat = thermal_diffusivity * (d2Tdx2 + d2Tdy2)
                
                # 화재 열원
                if cell.state >= 3:  # 연소 상태
                    heat_source = cell.fuel_load * 50.0  # 연료 연소열
                else:
                    heat_source = 0.0
                
                # 온도 업데이트
                dT_dt = -convection_heat + conduction_heat + heat_source / (cell.density * self.specific_heat)
                new_temp[y][x] = cell.temperature + self.dt * dT_dt
        
        # 온도 업데이트
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                self.grid[y][x].temperature = max(20.0, new_temp[y][x])
    
    def solve_species_transport(self):
        """화학종 수송 방정식 (연기 확산)"""
        new_smoke = np.zeros((self.height, self.width))
        
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                cell = self.grid[y][x]
                
                # 이웃 셀들
                east = self.grid[y][x+1]
                west = self.grid[y][x-1]
                north = self.grid[y-1][x]
                south = self.grid[y+1][x]
                
                # 대류 수송
                if cell.velocity_u > 0:
                    dSdx = (cell.smoke_density - west.smoke_density) / self.dx
                else:
                    dSdx = (east.smoke_density - cell.smoke_density) / self.dx
                    
                if cell.velocity_v > 0:
                    dSdy = (cell.smoke_density - south.smoke_density) / self.dy
                else:
                    dSdy = (north.smoke_density - cell.smoke_density) / self.dy
                
                convection_smoke = cell.velocity_u * dSdx + cell.velocity_v * dSdy
                
                # 확산 수송
                smoke_diffusivity = 0.12  # m²/s
                d2Sdx2 = (east.smoke_density - 2*cell.smoke_density + west.smoke_density) / (self.dx**2)
                d2Sdy2 = (north.smoke_density - 2*cell.smoke_density + south.smoke_density) / (self.dy**2)
                diffusion_smoke = smoke_diffusivity * (d2Sdx2 + d2Sdy2)
                
                # 연기 생성
                if cell.state >= 3:  # 연소 상태
                    smoke_source = cell.fuel_load * 0.1
                else:
                    smoke_source = 0.0
                
                # 연기 농도 업데이트
                dS_dt = -convection_smoke + diffusion_smoke + smoke_source
                new_smoke[y][x] = max(0.0, cell.smoke_density + self.dt * dS_dt)
        
        # 연기 농도 업데이트
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                self.grid[y][x].smoke_density = new_smoke[y][x]
    
    def update_density(self):
        """이상기체 법칙으로 밀도 업데이트"""
        R = 287.0  # 공기 기체상수 (J/kg·K)
        
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                temp_kelvin = cell.temperature + 273.15
                cell.density = cell.pressure / (R * temp_kelvin)
    
    def update_cfd_step(self):
        """전체 CFD 시간 스텝 업데이트"""
        # 1. 운동량 방정식 풀이
        self.calculate_momentum_equations()
        
        # 2. 압력 수정
        self.solve_pressure_correction()
        
        # 3. 에너지 방정식 풀이  
        self.solve_energy_equation()
        
        # 4. 화학종 수송 방정식 풀이
        self.solve_species_transport()
        
        # 5. 밀도 업데이트
        self.update_density()
        
        # 6. 화재 상태 업데이트 (기존 화재 물리학)
        self.update_fire_physics()
    
    def update_fire_physics(self):
        """기존 화재 물리학 업데이트"""
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                
                # 간단한 화재 상태 전이
                if cell.state == 0 and cell.temperature > 150:  # NORMAL -> PREHEATING
                    cell.state = 1
                elif cell.state == 1 and cell.temperature > 400:  # PREHEATING -> IGNITION
                    cell.state = 2
                elif cell.state == 2:  # IGNITION -> BURNING
                    cell.state = 3
                elif cell.state == 3 and cell.fuel_load < 5:  # BURNING -> BURNED_OUT
                    cell.state = 5
                
                # 연료 소모
                if cell.state >= 3:
                    cell.fuel_load = max(0, cell.fuel_load - 0.1)

# 사용 예시
def test_cfd_simulation():
    sim = SimplifiedCFDFireSimulation(20, 20)
    
    # 점화점 설정
    sim.grid[11][18].state = 2  # IGNITION
    sim.grid[11][18].temperature = 400.0
    sim.grid[11][18].fuel_load = 60.0
    
    print("Enhanced CFD Fire Simulation Test")
    print("=" * 50)
    
    for timestep in range(10):
        sim.update_cfd_step()
        
        # 점화 지점 상태 출력
        cell = sim.grid[11][18]
        print(f"Step {timestep}: T={cell.temperature:.1f}°C, "
              f"u={cell.velocity_u:.2f}m/s, v={cell.velocity_v:.2f}m/s, "
              f"P={cell.pressure:.0f}Pa, ρ={cell.density:.3f}kg/m³")

if __name__ == "__main__":
    test_cfd_simulation() 