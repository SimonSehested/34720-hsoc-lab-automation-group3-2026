# %%
import pyvisa as visa
from typing import Literal, Tuple
from abc import ABC, abstractmethod
import clr
import numpy as np
import threading
import time
from System import Decimal, Convert
from arduino_pm import ArduinoADC

# Must have downloaded and installed the Kinesis software from Thorlabs.
clr.AddReference(
    "C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.DeviceManagerCLI.dll"
)
clr.AddReference(
    "C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericMotorCLI.dll"
)
clr.AddReference(
    "C:\\Program Files\\Thorlabs\\Kinesis\\ThorLabs.MotionControl.PolarizerCLI.dll"
)
# Import the device manager and the polarizer. Ugly, but nothing to do since all of it is in .dll files.
from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.PolarizerCLI import *


class PolCon(ABC):
    @abstractmethod
    def set_position(self, paddle_num: int, position: float) -> None:
        pass

    @abstractmethod
    def get_position(self, paddle_num: int) -> float:
        pass

    @abstractmethod
    def wait(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def pref_optimize(
        self,
        pm: ArduinoADC,
        max_or_min: Literal["max", "min"],
        return_data: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[float, float]:
        pass

    def brute_force_optimize_single_paddle(
        self,
        pm: ArduinoADC,
        paddle_num: int,
        start_pos: float,
        end_pos: float,
        step_size: float,
        max_or_min: Literal["max", "min"] = "max",
    ) -> float:
        self.set_position(paddle_num, start_pos)
        self.wait()
        step_array = np.arange(start_pos, end_pos + step_size, step_size)
        position_array = np.zeros(len(step_array))
        voltage_array = np.zeros(len(step_array))
        for i, pos in enumerate(step_array):
            self.set_position(paddle_num, pos)
            self.wait()
            position_array[i] = self.get_position(paddle_num)
            voltage_array[i] = pm.get_voltage()
        voltage_array = self.correct_discontinuities(voltage_array)
        if max_or_min == "max":
            best_pos = position_array[np.argmax(voltage_array)]
        elif max_or_min == "min":
            best_pos = position_array[np.argmin(voltage_array)]
        print(f"Setting paddle {paddle_num} to {best_pos} for {max_or_min} voltage.")
        self.set_position(paddle_num, best_pos)
        self.wait()
        return position_array, voltage_array

    def brute_force_optimize(
        self,
        pm: ArduinoADC,
        start_pos: float,
        end_pos: float,
        step_size: float,
        return_data: bool = False,
        max_or_min: Literal["max", "min"] = "max",
    ) -> Tuple[float, float]:
        angle = np.zeros(self.num_paddles, dtype=object)
        voltage = np.zeros(self.num_paddles, dtype=object)
        for i in range(self.num_paddles):
            angle[i], voltage[i] = self.brute_force_optimize_single_paddle(
                pm, i + 1, start_pos, end_pos, step_size, max_or_min=max_or_min
            )
        if return_data:
            return angle, voltage
        else:
            pass

    @staticmethod
    def correct_discontinuities(voltage_values):
        diffs = np.diff(voltage_values)
        median_diff = np.median(np.abs(diffs))
        threshold = 10 * median_diff
        discontinuity_indices = np.where(np.abs(diffs) > threshold)[0] + 1
        for index in discontinuity_indices:
            voltage_values[index:] -= diffs[index - 1]
        return voltage_values


class ThorlabsMPC320(PolCon):
    def __init__(
        self,
        serial_no_idx=0,
        serial_no=None,
        limits=[0, 166],
        polling_rate=10,
        initial_velocity=50,
    ):
        print("Initializing Thorlabs MPC320...")
        self.polling_rate = polling_rate
        self.serial_no_idx = serial_no_idx
        self.serial_no = serial_no
        self.connect(initial_velocity=initial_velocity)
        self.monitoring = False
        self.limits = limits
        self.min_stepsize = 0.2
        self.num_paddles = 3
        print("Thorlabs MPC320 initialized.")

    def connect(self, initial_velocity=80):
        DeviceManagerCLI.BuildDeviceList()
        if self.serial_no is None:
            self.serial_no = DeviceManagerCLI.GetDeviceList()[self.serial_no_idx]
        else:
            self.serial_no = serial_no
        self.device = Polarizer.CreatePolarizer(self.serial_no)
        self.device.Connect(self.serial_no)
        self.start_polling()
        time.sleep(1)
        self.paddle1 = self.create_paddle(1)
        self.paddle2 = self.create_paddle(2)
        self.paddle3 = self.create_paddle(3)
        self.velocity = initial_velocity

    def reconnect(self):
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                self.close()
                time.sleep(1)
                self.connect()
                print("Reconnected successfully!")
                return
            except Exception as e:
                print(f"Reconnection attempt {attempt + 1} failed. Error: {e}")
        print("Failed to reconnect after multiple attempts.")

    def write(self, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(
                f"MPC {self.serial_no} disconnected. Attempting to reconnect. Error: {e}"
            )
            self.reconnect()

    def wait(self):
        # Only exists for compatibility with other devices
        pass

    def create_paddle(self, paddle_num):
        paddle_name = f"Paddle{paddle_num}"
        paddle = getattr(PolarizerPaddles, paddle_name)
        return paddle

    def get_paddle(self, paddle_num):
        return getattr(self, f"paddle{paddle_num}")

    def start_monitoring(self, paddle_num, pm, interval=1):
        positions = []
        voltage_values = []

        def monitor_position():
            while self.monitoring:
                position = self.get_position(paddle_num)
                positions.append(position)
                voltage_values.append(pm.get_voltage())
                time.sleep(interval / 1000)

        self.monitoring = True
        self.monitoring_thread = threading.Thread(target=monitor_position)
        self.monitoring_thread.start()
        return positions, voltage_values

    def stop_monitoring(self):
        self.monitoring = False
        if hasattr(self, "monitoring_thread"):
            self.monitoring_thread.join()

    def move_and_monitor(
        self, paddle_num, pm, start_pos=0, end_pos=165.9, interval=1, timeout=60000
    ):
        self.set_position(paddle_num, start_pos, timeout)
        pos, voltage = self.start_monitoring(paddle_num, pm, interval=interval)
        self.set_position(paddle_num, end_pos, timeout)
        self.stop_monitoring()
        return pos, voltage

    def brute_force_optimize_fast(
        self,
        pm,
        max_or_min="max",
        start_pos=0,
        end_pos=165.9,
        interval=1,
        return_data=False,
    ):
        """
        Optimize polarization using the continuous moving+monitoring method.
        """
        pos, voltage = [], []
        for i in range(3):
            pos_, voltage_ = self.move_and_monitor(
                i + 1, pm, start_pos=start_pos, end_pos=end_pos, interval=interval
            )
            pos.append(pos_)
            voltage_ = self.correct_discontinuities(voltage_)
            voltage.append(voltage_)
            if max_or_min == "max":
                print(
                    f"Setting paddle {i + 1} to {pos_[np.argmax(voltage_)]} for {max_or_min} voltage."
                )
                self.set_position(i + 1, pos_[np.argmax(voltage_)])
            elif max_or_min == "min":
                print(
                    f"Setting paddle {i + 1} to {pos_[np.argmin(voltage_)]} for {max_or_min} voltage."
                )
                self.set_position(i + 1, pos_[np.argmin(voltage_)])
        if return_data:
            return pos, voltage
        else:
            pass

    def pref_optimize(
        self,
        pm: ArduinoADC,
        max_or_min: Literal["max", "min"],
        start_pos: float = 0,
        end_pos: float = 165.9,
        interval: float = 1,
        return_data: bool = False,
    ) -> Tuple[float, float]:
        return self.brute_force_optimize_fast(
            pm,
            max_or_min=max_or_min,
            start_pos=start_pos,
            end_pos=end_pos,
            interval=interval,
            return_data=return_data,
        )

    def get_params(self):
        return self.write(self.device.GetPolParams)

    @property
    def velocity(self):
        return self.write(self.get_params().get_Velocity())

    @velocity.setter
    def velocity(self, new_velocity):
        params = self.get_params()
        self.write(params.set_Velocity, new_velocity)
        self.write(self.device.SetPolParams, params)

    def get_position(self, paddle_num):
        paddle = self.get_paddle(paddle_num)
        position = self.write(self.device.Position, paddle)
        return Convert.ToDouble(position)

    def set_position(self, paddle_num, new_pos, timeout=60000):
        paddle = self.get_paddle(paddle_num)
        new_pos = self.check_limits(new_pos)
        self.write(self.device.MoveTo, Decimal(new_pos), paddle, timeout)

    def move_relative(self, paddle_num, delta, timeout=60000):
        self.check_stepsize(delta)
        paddle = self.get_paddle(paddle_num)
        prev_pos = self.get_position(paddle_num)
        new_pos = prev_pos + delta
        if self.check_limits(new_pos) != new_pos:
            print("Did not move")
        self.write(self.device.MoveRelative, Decimal(delta), paddle, timeout)

    def check_limits(self, new_pos):
        if new_pos > self.limits[1] or new_pos < self.limits[0]:
            new_pos_init = new_pos
            new_pos = self.limits[0] if new_pos < self.limits[0] else self.limits[1]
            print(
                f"Warning: position {new_pos_init} is outside of limits {self.limits}. Setting to {new_pos}."
            )
        return new_pos

    def check_stepsize(self, stepsize):
        if np.abs(stepsize) < self.min_stepsize:
            raise ValueError(
                f"Stepsize {stepsize} is smaller than minimum stepsize {self.max_stepsize}"
            )

    @property
    def polling_rate(self):
        return self._polling_rate

    @polling_rate.setter
    def polling_rate(self, rate):
        self._polling_rate = rate

    def start_polling(self):
        self.write(self.device.StartPolling, self._polling_rate)
        time.sleep(1)
        self.write(self.device.EnableDevice)
        time.sleep(0.25)

    def stop_polling(self):
        self.write(self.device.StopPolling)

    def set_random_position(self, paddle_num, timeout=60000):
        self.set_position(paddle_num, np.random.uniform(*self.limits), timeout=timeout)

    def set_random_position_all(self, timeout=60000):
        for i in range(3):
            self.set_random_position(i + 1, timeout=timeout)

    def home(self, paddle_num, timeout=60000):
        paddle = self.get_paddle(paddle_num)
        self.write(self.device.Home, paddle, timeout)

    def close(self):
        self.stop_polling()
        self.write(self.device.Disconnect)
        self.write(self.device.Dispose)

    def operation_complete(self) -> Literal[0, 1]:
        return int(self.device.query("*OPC?"))


class Agilent11896A(PolCon):
    def __init__(self, GPIB_address=7, scan_rate=4):
        print("Initializing Agilent 11896A...")
        rm = visa.ResourceManager()
        self.device = rm.open_resource(f"GPIB0::{GPIB_address}::INSTR")
        self.device.timeout = 5000
        self.scan_rate = scan_rate
        self.num_paddles = 4
        self.set_brute_force_optimization_params()
        print("Agilent 11896A initialized.")

    def get_position(self, paddle_num: Literal[1, 2, 3, 4]) -> float:
        return float(self.device.query(f":PADD{paddle_num}:POS?"))

    def set_position(self, paddle_num: Literal[1, 2, 3, 4], position: float) -> None:
        self.device.write(f":PADD{paddle_num}:POS {position}")

    def start_scan(self) -> None:
        self.device.write(":INIT:IMM")

    def stop_scan(self) -> None:
        self.device.write(":ABOR")

    @property
    def scan_rate(self) -> float:
        return float(self.device.query(":SCAN:RATE?"))

    @scan_rate.setter
    def scan_rate(self, value: Literal[1, 2, 3, 4, 5, 6, 7, 8]) -> None:
        self.device.write(f":SCAN:RATE {value}")

    def wait(self) -> None:
        self.device.write("*WAI")

    def close(self) -> None:
        self.device.close()

    def operation_complete(self) -> Literal[0, 1]:
        return int(self.device.query("*OPC?"))

    def pref_optimize(
        self,
        pm: ArduinoADC,
        max_or_min: Literal["max", "min"],
        return_data: bool = False,
    ) -> Tuple[float, float]:
        return self.brute_force_optimize(
            pm,
            self.start_pos,
            self.end_pos,
            self.step_size,
            return_data=return_data,
            max_or_min=max_or_min,
        )

    def set_brute_force_optimization_params(
        self, start_pos: float = 0, end_pos: float = 1000, step_size: float = 20
    ) -> None:
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.step_size = step_size


def optimize_multiple_pol_cons(
    pm: ArduinoADC,
    *pol_cons: PolCon,
    max_or_min: Literal["max", "min"] = "max",
    tolerance: float = np.inf,
) -> None:
    """
    Optimize multiple polarization controllers simultaneously while checking for global optimization.
    BEWARE THAT OSA ANALOG OUT IS IN WEIRD VOLTAGE UNITS SO TOLERANCE IS NOT CORRECT ATM.
    """
    pow_arr = []

    # Initial optimization for all devices
    for pol_con in pol_cons:
        pol_con.pref_optimize(pm, max_or_min=max_or_min, return_data=True)
        pow_arr.append(pm.get_voltage())

    # Can set to inf to disable global optimization. Usually not necessary.
    if tolerance is not np.inf:
        while True:
            # Start from the first device and optimize
            pol_cons[0].pref_optimize(pm, max_or_min=max_or_min, return_data=True)
            new_power = pm.get_voltage()
            power_difference = abs(new_power - pow_arr[0])

            # If the power difference for the first device is within the tolerance, stop the function
            if power_difference <= tolerance:
                return

            # Update the power array for the first device
            pow_arr[0] = new_power

            # Optimize the rest of the devices
            for i in range(1, len(pol_cons)):
                pol_cons[i].pref_optimize(pm, max_or_min=max_or_min, return_data=True)
                new_power = pm.get_voltage()
                power_difference = abs(new_power - pow_arr[i])

                # If the power difference for any device is within the tolerance, stop the function
                if power_difference <= tolerance:
                    return

                # Update the power array for the current device
                pow_arr[i] = new_power


if __name__ == "__main__":
    #arduino = ArduinoADC("COM11")
    pol_con1 = ThorlabsMPC320()
    pol_con2 = ThorlabsMPC320(serial_no_idx=1)

    #pol_con2 = Agilent11896A(GPIB_address=22)
