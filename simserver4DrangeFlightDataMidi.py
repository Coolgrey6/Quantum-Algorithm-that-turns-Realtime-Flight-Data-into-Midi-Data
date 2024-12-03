import mido
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
import numpy as np
import time
import random
import json
import websockets
import asyncio
import aiohttp
from pathlib import Path

class Quantum5DEncoder:
    def __init__(self, precision_qubits=3):
        self.precision = precision_qubits
        self.total_qubits = precision_qubits * 5
        self.q = QuantumRegister(self.total_qubits)
        self.c = ClassicalRegister(self.total_qubits)
        
    def encode_coordinate(self, value, start_qubit):
        self.qc = QuantumCircuit(self.q, self.c)
        binary_str = ''
        temp_val = value
        for i in range(self.precision):
            temp_val *= 2
            if temp_val >= 1:
                binary_str += '1'
                temp_val -= 1
            else:
                binary_str += '0'
                
        for i, bit in enumerate(binary_str):
            if bit == '1':
                self.qc.x(self.q[start_qubit + i])
    
    def encode_5d(self, lat, lon, elev, current_time, speed):
        # Adjust bounds check
        lat = max(min(lat, 90), -90)  # Clamp latitude between -90 and 90
        lon = max(min(lon, 180), -180)  # Clamp longitude between -180 and 180
        elev = max(min(elev, 8848), 0)  # Clamp elevation between 0 and 8848 meters
        current_time = max(min(current_time, 24), 0)  # Clamp time between 0 and 24 hours
        speed = max(min(speed, 1000), 0)  # Assuming max speed is around 1000 km/h, clamp speed

        lat_n = (lat + 90) / 180  
        lon_n = (lon + 180) / 360  
        elev_n = elev / 8848  
        time_n = current_time / 24  
        speed_n = speed / 1000  

        self.encode_coordinate(lat_n, 0)
        self.encode_coordinate(lon_n, self.precision)
        self.encode_coordinate(elev_n, 2 * self.precision)
        self.encode_coordinate(time_n, 3 * self.precision)
        self.encode_coordinate(speed_n, 4 * self.precision)
        
    def run(self, shots=1000):
        sv = Statevector.from_instruction(self.qc)
        probs = sv.probabilities_dict()
        return {state: int(prob * shots) for state, prob in probs.items() if prob > 0}

# Store connected WebSocket clients
connected = set()

# Register new WebSocket clients
async def register(websocket):
    connected.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        connected.remove(websocket)

# Fetch flight data from OpenSky Network, including speed
async def fetch_flight_data():
    usa_bbox = {"lamin": 25, "lamax": 49, "lomin": -125, "lomax": -66}
    api_url = f"https://opensky-network.org/api/states/all?lamin={usa_bbox['lamin']}&lamax={usa_bbox['lamax']}&lomin={usa_bbox['lomin']}&lomax={usa_bbox['lomax']}"
    
    async with aiohttp.ClientSession() as session:
        for attempt in range(3):
            try:
                async with session.get(api_url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'states' in data and data['states']:
                            return data['states']
                        else:
                            print("No flights found in the specified region.")
                            return []
                    elif response.status == 429:  # Rate limit reached
                        print("Rate limit hit, waiting before retry.")
                        await asyncio.sleep(2)
                    else:
                        print(f"API returned status code: {response.status}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                print(f"API error encountered: {e}")
                await asyncio.sleep(4 * (attempt + 1))
        print("Failed to fetch flight data after multiple attempts.")
        return []

# Quantum simulator with MIDI output using speed data
async def quantum_simulator():
    encoder = Quantum5DEncoder(precision_qubits=3)
    iteration = 0
    midi_out = None

    try:
        midi_out = mido.open_output('pythonport 2')  # Use the name of your virtual MIDI port here
    except OSError:
        print("No MIDI output device found. MIDI output will be disabled.")

    last_notes = {i: None for i in range(1, 11)}

    while True:
        try:
            flights = await fetch_flight_data()
            if not flights:
                print("No flight data available, waiting...")
                # Here, we'll keep playing the last notes if no new flights are available
                for channel in range(1, 11):
                    if last_notes[channel]:
                        note_on = mido.Message('note_on', channel=channel, note=last_notes[channel]['note'], velocity=last_notes[channel]['velocity'])
                        midi_out.send(note_on)
                await asyncio.sleep(2)  # Wait before retrying
                continue
            
            for channel, flight in enumerate(flights[:10], start=1):
                lat, lon, elev, current_time, speed = flight[6], flight[5], flight[13] if flight[13] is not None else 0, int(time.time() % 86400) / 3600, flight[9] if flight[9] is not None else 0
                flight_number = flight[1] if flight[1] else "Unknown"
                departure = flight[2] if flight[2] else "Unknown"
                destination = flight[11] if flight[11] else "Unknown"

                try:
                    # Encode without raising an exception due to bounds
                    encoder.encode_5d(lat, lon, elev, current_time, speed)
                    counts = encoder.run(shots=1000)

                    # Create data packet
                    data = {
                        'latitude': lat,
                        'longitude': lon,
                        'elevation': elev,
                        'time': current_time,
                        'flight_number': flight_number,
                        'departure': departure,
                        'destination': destination,
                        'timestamp': int(time.time() * 1000),
                        'iteration': iteration
                    }
                    
                    # Send to all connected clients
                    websockets_to_remove = set()
                    for websocket in connected:
                        try:
                            await websocket.send(json.dumps(data))
                        except websockets.exceptions.ConnectionClosed:
                            websockets_to_remove.add(websocket)
                    
                        # Clean up closed connections
                        connected.difference_update(websockets_to_remove)
                    
                    # Convert flight data into MIDI messages
                    if midi_out:
                        pitch = int((lat + 90) / 180 * 127)
                        velocity = int((lon + 180) / 360 * 127)
                        speed_factor = min(speed / 1000, 1)
                        adjusted_velocity = int(velocity * speed_factor)
                        
                        # Turn off previous note if there was one
                        if last_notes[channel]:
                            midi_out.send(mido.Message('note_off', channel=channel, note=last_notes[channel]['note'], velocity=last_notes[channel]['velocity']))
                        
                        note_on = mido.Message('note_on', channel=channel, note=pitch, velocity=adjusted_velocity)
                        midi_out.send(note_on)
                        
                        last_notes[channel] = {'note': pitch, 'velocity': adjusted_velocity}

                    # ... rest of your data sending code ...

                except Exception as e:
                    print(f"Flight {flight_number} on Channel {channel} - {e}. Skipping this flight.")
                    # Here we could decide to retry after a short delay if needed

            print(f"\nIteration {iteration}")
            print(f"Processed {min(10, len(flights))} flights")
            print(f"Connected clients: {len(connected)}")
            if midi_out:
                print("MIDI Data Sent for each flight on channels 1-10")

            await asyncio.sleep(4)  # Wait before the next fetch

        except Exception as e:
            print(f"Exception occurred in quantum simulator: {e}")
            await asyncio.sleep(1)

        iteration += 1

async def main():
    try:
        async with websockets.serve(register, "localhost", 8765):
            print("WebSocket server started on ws://localhost:8765")
            print(mido.get_output_names())
            await quantum_simulator()
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")

if __name__ == "__main__":
    print("Starting Quantum 5D WebSocket server for real-time flight tracking with MIDI output...")
    print(mido.get_output_names())
    asyncio.run(main())