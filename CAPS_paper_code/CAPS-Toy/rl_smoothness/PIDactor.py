import numpy as np

class PID:
    def __init__(self, Kp=0.2, Ki=0., Kd=0.):

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.clear()

    def clear(self):
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = np.array([0.0])

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 20.0

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain

    def setWindup(self, windup):
        """Integral windup, also known as integrator windup or reset windup,
        refers to the situation in a PID feedback controller where
        a large change in setpoint occurs (say a positive change)
        and the integral terms accumulates a significant error
        during the rise (windup), thus overshooting and continuing
        to increase as this accumulated error is unwound
        (offset by errors in the other direction).
        The specific problem is the excess overshooting.
        """
        self.windup_guard = windup

    def act(self, error, dt=0.02):
        delta_error = error - self.last_error
        
        self.PTerm = self.Kp * error
        self.ITerm += error * dt

        if (self.ITerm < -self.windup_guard):
            self.ITerm = -self.windup_guard
        elif (self.ITerm > self.windup_guard):
            self.ITerm = self.windup_guard

        self.DTerm = 0.0
        if dt > 0:
            self.DTerm = delta_error / dt

        # Remember last time and last error for next calculation
        self.last_error = error

        return self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)