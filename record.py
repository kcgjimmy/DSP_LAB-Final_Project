from Final_function import record_audio


if __name__ == "__main__":
    duration = 5  # Set the recording time in seconds
    gain = 5.0    # Adjust the gain as needed (increase for louder audio)

    # Start recording
    record_audio(duration,filename="output.wav", gain=5)