import requests

class FanLightController:
    def __init__(self, ip_address):
        self.base_url = f"http://{ip_address}"

    def set_fan_speed(self, speed):
        if 0 <= speed <= 255:
            response = requests.get(f"{self.base_url}/setspeed?speed={speed}")
            print(response.text)
        else:
            print("Speed must be between 0 and 255")

    def set_light(self, brightness):
        if 0 <= brightness <= 255:
            response = requests.get(f"{self.base_url}/setlight?brightness={brightness}")
            print(response.text)
        else:
            print("Brightness must be between 0 and 255")

# Usage example
if __name__ == "__main__":
    device_ip = "192.168.172.148"  # Replace with your ESP32's IP address
    controller = FanLightController(device_ip)

    while True:
        print("\nFan and Light Control Menu:")
        print("1. Set Fan Speed (0 to turn off, 1-255 to set speed)")
        print("2. Set Light Brightness (0 to turn off, 1-255 to set brightness)")
        print("3. Exit")

        choice = input("Enter your choice (1-3): ")

        if choice == "1":
            speed = int(input("Enter fan speed (0-255): "))
            controller.set_fan_speed(speed)
        elif choice == "2":
            brightness = int(input("Enter light brightness (0-255): "))
            controller.set_light(brightness)
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")