import asyncio
import json
import websockets
from unity_environment import UnityEnvironment

sim = UnityEnvironment()

async def handle_client(websocket):
    print("Unity connected.")

    while True:
        try:
            msg = await websocket.recv()
        except websockets.ConnectionClosed:
            print("Unity disconnected.")
            break

        print("Received from Unity:", msg)

        try:
            packet = json.loads(msg)
        except:
            print("Invalid JSON from Unity.")
            continue

        response = None

        # -------- COMMAND ROUTER --------
        if packet.get("request") == "update":
            response = sim.update()

        elif packet.get("request") == "press":
            response = sim.do_press()

        elif packet.get("request") == "temperature":
            count = packet.get("count", 100)
            response = sim.temperature_result(count)

        else:
            response = { "error": "Unknown request" }

        # Serialize and send response
        json_packet = json.dumps(response)
        await websocket.send(json_packet)

        print("Sent response to Unity.")

async def main():
    print("Genesis WebSocket server listening at ws://localhost:8765")
    async with websockets.serve(handle_client, "localhost", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
