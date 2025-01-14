import requests
import time

ESP32_IP = "192.168.77.226"  # Replace with your ESP32's IP address
ESP32_URL = f"http://{ESP32_IP}/control"

def send_command(led_index, brightness):
    try:
        payload = {"led": led_index, "brightness": brightness}
        response = requests.post(ESP32_URL, json=payload)
        print(f"ESP32 response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with ESP32: {e}")

def main():
    print("ESP32 LED Control with PCA9685")
    led_channels = [0, 8, 15]
    
    while True:
        try:
            print("\nAvailable LED channels:", led_channels)
            led_index = int(input("Enter LED index (0, 1, or 2), or -1 to quit: "))
            if led_index == -1:
                print("Exiting...")
                break
            if led_index < 0 or led_index >= len(led_channels):
                print("Invalid LED index. Please enter 0, 1, or 2.")
                continue
            
            brightness = int(input("Enter LED brightness (0-4095, 0 is off, 4095 is full brightness): "))
            if brightness < 0 or brightness > 4095:
                print("Invalid brightness. Please enter a number between 0 and 4095.")
                continue
            
            send_command(led_index, brightness)
        except ValueError:
            print("Invalid input. Please enter numbers for LED index and brightness.")
        
        time.sleep(0.1)  # Small delay to prevent flooding the ESP32 with requests

if __name__ == '__main__':
    main()