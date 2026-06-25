#!/usr/bin/env python3
"""
SuperIO GPIO Controller for IPC
Group 7 only: GPIO70-73 (O1-O4) as output, GPIO74-77 (I1-I4) as input
Requires root privileges.
"""

import os

class PortIO:
    """Direct I/O port access via /dev/port"""
    
    def __init__(self):
        self._port = os.open("/dev/port", os.O_RDWR)
    
    def read(self, port):
        os.lseek(self._port, port, os.SEEK_SET)
        return os.read(self._port, 1)[0]
    
    def write(self, port, val):
        os.lseek(self._port, port, os.SEEK_SET)
        os.write(self._port, bytes([val & 0xFF]))
    
    def close(self):
        os.close(self._port)


class SuperIOGPIO:
    """SuperIO GPIO controller for Group 7"""
    
    INDEX_PORT = 0x2E
    DATA_PORT = 0x2F
    LDN_SEL_REG = 0x07
    GPIO_DEV_NUM = 0x07
    GPIO_BAR_MSB_REG = 0x62
    GPIO_BAR_LSB_REG = 0x63
    GPIO7X_DIR_REG = 0xCE
    
    def __init__(self):
        self._io = PortIO()
        self._gpio_bar = None
        self._init_directions()
    
    def _enter_config(self):
        self._io.write(self.INDEX_PORT, 0x87)
        self._io.write(self.INDEX_PORT, 0x01)
        self._io.write(self.INDEX_PORT, 0x55)
        self._io.write(self.INDEX_PORT, 0x55)
    
    def _exit_config(self):
        self._io.write(self.INDEX_PORT, 0x02)
        self._io.write(self.DATA_PORT, 0x02)
    
    def _select_dev(self, dev):
        self._io.write(self.INDEX_PORT, self.LDN_SEL_REG)
        self._io.write(self.DATA_PORT, dev)
    
    def _sio_read(self, reg):
        self._io.write(self.INDEX_PORT, reg)
        return self._io.read(self.DATA_PORT)
    
    def _sio_write(self, reg, val):
        self._io.write(self.INDEX_PORT, reg)
        self._io.write(self.DATA_PORT, val)
    
    def _get_gpio_bar(self):
        if self._gpio_bar is None:
            self._enter_config()
            self._select_dev(self.GPIO_DEV_NUM)
            msb = self._sio_read(self.GPIO_BAR_MSB_REG)
            lsb = self._sio_read(self.GPIO_BAR_LSB_REG)
            self._exit_config()
            self._gpio_bar = (msb << 8) | lsb
        return self._gpio_bar
    
    def _init_directions(self):
        """Set GPIO70-73 as output, GPIO74-77 as input"""
        self._enter_config()
        self._select_dev(self.GPIO_DEV_NUM)
        # Bit=1 means output, Bit=0 means input
        # Pins 0-3 output, pins 4-7 input -> 0x0F
        self._sio_write(self.GPIO7X_DIR_REG, 0x0F)
        self._exit_config()
    
    def write_output(self, pin, value):
        """
        Write to output pin (O1-O4)
        pin: 1-4 (physical O1-O4, corresponds to GPIO70-73)
        value: 0 or 1
        """
        if pin < 1 or pin > 4:
            raise ValueError("Output pin must be 1-4 (O1-O4)")
        bit = pin - 1
        reg = self._get_gpio_bar() + 6  # Group 7 offset
        current = self._io.read(reg)
        if value:
            self._io.write(reg, current | (1 << bit))
        else:
            self._io.write(reg, current & ~(1 << bit))
    
    def read_input(self, pin):
        """
        Read from input pin (I1-I4)
        pin: 1-4 (physical I1-I4, corresponds to GPIO74-77)
        Returns: 0 or 1
        """
        if pin < 1 or pin > 4:
            raise ValueError("Input pin must be 1-4 (I1-I4)")
        bit = pin + 3  # I1->bit4, I2->bit5, etc.
        reg = self._get_gpio_bar() + 6
        return (self._io.read(reg) >> bit) & 1
    
    def write_all_outputs(self, values):
        """
        Write all 4 outputs at once
        values: 4-bit value (bit0=O1, bit1=O2, bit2=O3, bit3=O4)
        """
        reg = self._get_gpio_bar() + 6
        current = self._io.read(reg) & 0xF0
        self._io.write(reg, current | (values & 0x0F))
    
    def read_all_inputs(self):
        """
        Read all 4 inputs at once
        Returns: 4-bit value (bit0=I1, bit1=I2, bit2=I3, bit3=I4)
        """
        reg = self._get_gpio_bar() + 6
        return (self._io.read(reg) >> 4) & 0x0F
    
    def close(self):
        self._io.close()


# === Test ===
if __name__ == "__main__":
    gpio = SuperIOGPIO()
    
    print("SuperIO GPIO Controller - Group 7")
    print(f"GPIO BAR: 0x{gpio._get_gpio_bar():04X}")
    print("=" * 40)
    
    # Read all inputs
    inputs = gpio.read_all_inputs()
    print(f"All inputs (I1-I4): {inputs:04b}")
    for i in range(1, 5):
        print(f"  I{i}: {gpio.read_input(i)}")
    
    # Test outputs
    print("\nToggling outputs O1-O4...")
    import time
    gpio.write_output(1, 0)
    time.sleep(3)
    gpio.write_output(1, 1)
    time.sleep(3)
    # gpio.write_output(1, 0)
    # time.sleep(3)

    # for pin in range(1, 5):
    #     print(f"  O{pin} = 1")
    #     gpio.write_output(pin, 1)
    #     time.sleep(0.5)
    #     gpio.write_output(pin, 0)
    
    print("\nAll outputs off")
    # gpio.write_all_outputs(0x00)
    gpio.close()
