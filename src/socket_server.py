import asyncio 
import websockets 

import src.utils as utils
from src.config import CONFIG
from src.streamer import RewardsStreamer 

class WebsocketStreamingServer:
    def __init__(self, host, port, secret):
        self._secret = secret 
    
    async def _start(self, websocket, endpoint):
        print(f"=> Connected to {endpoint}")
        secret = await websocket.recv() 
        if secret == self._secret :
            ... 