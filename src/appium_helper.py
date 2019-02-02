import subprocess
from time import sleep
import os

class AppiumHelper(object):

    def get_device_capabilities():
        subprocess.run("adb devices", shell="True", check="True")
        os.environ['DEVICE_SERIAL_NO'] = AppiumHelper.get_output_for_command("adb get-serialno")
        os.environ['DEVICE_NAME'] = AppiumHelper.get_output_for_command("adb shell getprop ro.product.model")
        capabilities = {
            'platformName': 'Android',
            'udid': os.environ['DEVICE_SERIAL_NO'],
            'browserName': 'chrome',
            'deviceName': os.environ['DEVICE_NAME']
        }
        return capabilities

    def get_output_for_command(shell_command):
        return subprocess.Popen([shell_command], stdout=subprocess.PIPE, shell=True).communicate()[0].decode('ascii').strip()
