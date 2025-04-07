# maze_env.py
import numpy as np
from controller import Supervisor, Motor

class DroneRobot:
    def __init__(self):
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        
        # Initialize motors
        self.motors = [
            self.supervisor.getDevice("fl_motor"),
            self.supervisor.getDevice("fr_motor"),
            self.supervisor.getDevice("bl_motor"),
            self.supervisor.getDevice("br_motor")
        ]
        for motor in self.motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(1.0)
        
        # Initialize sensors
        self.gps = self.supervisor.getDevice("gps")
        self.imu = self.supervisor.getDevice("imu")
        self.gps.enable(self.timestep)
        self.imu.enable(self.timestep)
        
        # Control parameters
        self.k_vertical_thrust = 100
        self.TARGET_ALTITUDE = 0.5  # Single target altitude
        self.MAX_VELOCITY = 400
        
        # Get drone node and fields for reset
        self.drone_node = self.supervisor.getFromDef("QUADCOPTER")
        if self.drone_node is None:
            raise ValueError("Drone node 'QUADCOPTER' not found.")
        self.translation_field = self.drone_node.getField("translation")
        self.rotation_field = self.drone_node.getField("rotation")
        
        # Store initial state
        self.supervisor.step(self.timestep)
        self.original_position = np.array(self.translation_field.getSFVec3f())
        self.original_rotation = np.array(self.rotation_field.getSFRotation())

    def clamp(self, value, low, high):
        return max(low, min(value, high))

    def reset(self):
        """Reset drone to original position"""
        for motor in self.motors:
            motor.setVelocity(0.0)
        self.translation_field.setSFVec3f(self.original_position.tolist())
        self.rotation_field.setSFRotation(self.original_rotation.tolist())
        self.supervisor.step(self.timestep)
        return self._get_state()

    def _get_state(self):
        """Get current altitude"""
        _, _, altitude = self.gps.getValues()
        return altitude

    def simulate(self, alt_pid_coeffs, max_steps=5000):  # Fixed time period: 500 steps
        """Simulate drone with given altitude PID coefficients"""
        k_alt_p, k_alt_i, k_alt_d = alt_pid_coeffs
        
        self.reset()
        alt_integral = 0
        prev_alt_error = 0
        total_error = 0
        
        for step in range(max_steps):
            if self.supervisor.step(self.timestep) == -1:
                break
            
            altitude = self._get_state()
            alt_error = self.TARGET_ALTITUDE - altitude
            alt_integral += alt_error * (self.timestep / 1000.0)
            alt_integral = self.clamp(alt_integral, -50, 50)
            alt_error_derivative = (alt_error - prev_alt_error) / (self.timestep / 1000.0)
            prev_alt_error = alt_error
            
            # Calculate control input
            vertical_input = (k_alt_p * alt_error + 
                            k_alt_i * alt_integral + 
                            k_alt_d * alt_error_derivative)
            vertical_input = self.clamp(vertical_input, -20.0, 20.0)
            
            # Apply thrust
            base_thrust = self.k_vertical_thrust + vertical_input
            thrust = self.clamp(base_thrust, -self.MAX_VELOCITY, self.MAX_VELOCITY)
            
            self.motors[0].setVelocity(thrust) # fl
            self.motors[1].setVelocity(-thrust) # fr
            self.motors[2].setVelocity(-thrust) # bl
            self.motors[3].setVelocity(thrust) # br
            
            # Accumulate absolute error
            total_error += abs(alt_error)

        # Fitness is inverse of average error (plus small constant to avoid division by zero)
        avg_error = total_error / max_steps
        fitness = 1.0 / (avg_error + 0.001)
        return fitness
