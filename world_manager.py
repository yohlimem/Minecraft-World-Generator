import os
import subprocess
import time
from uu import Error
import requests
import shutil
from mcrcon import MCRcon
import amulet
import asyncio
from amulet.api.block import Block

class MinecraftWorldManager:
    def __init__(self, server_dir="minecraft_server"):
        self.server_dir = server_dir
        self.world_name = "world"
        self.world_path = os.path.join(server_dir, self.world_name)
        self.jar_path = os.path.join(server_dir, "server.jar")
        self.eula_path = os.path.join(server_dir, "eula.txt")
        self.props_path = os.path.join(server_dir, "server.properties")
        
        # RCON settings
        self.rcon_host = "localhost"
        self.rcon_port = 25575
        self.rcon_pass = "password123"

    def reset_world(self):
        """Deletes the existing 'world' folder to ensure a fresh generation."""
        if os.path.exists(self.world_path):
            print(f"Deleting old world at: {self.world_path}...")
            # We use a loop because sometimes Windows holds onto files briefly
            try:
                shutil.rmtree(self.world_path)
                print("Old world deleted.")
            except Exception as e:
                print(f"Warning: Could not delete world folder completely: {e}")

    def setup_server(self, seed=""):
        """Downloads server, accepts EULA, and configures seed."""
        if not os.path.exists(self.server_dir):
            os.makedirs(self.server_dir)

        # 1. Download Server JAR (1.21.1)
        url = "https://piston-data.mojang.com/v1/objects/59353fb40c36d304f2035d51e7d6e6baa98dc05c/server.jar"
        if not os.path.exists(self.jar_path):
            print("Downloading Server JAR...")
            response = requests.get(url)
            with open(self.jar_path, "wb") as f:
                f.write(response.content)

        # 2. Accept EULA
        with open(self.eula_path, "w") as f:
            f.write("eula=true")

        # 3. Configure Server (RCON + Seed)
        # We explicitly write the properties file to set the seed
        config = f"""
            enable-rcon=true
            rcon.password={self.rcon_pass}
            rcon.port={self.rcon_port}
            level-name={self.world_name}
            level-seed={seed}
        """
        with open(self.props_path, "w") as f:
            f.write(config)
        print(f"Server configured with Seed: '{seed}'")

    def generate_world(self, radius_in_blocks):
        """
        Starts the server and forces generation within a radius around 0,0.
        radius_in_blocks: The distance from 0,0 to generate (e.g., 100 blocks)
        """
        print("--- Step 1: Generating World ---")
        
        # Start Server
        cmd = ["java", "-Xmx2G", "-jar", "server.jar", "nogui"]
        process = subprocess.Popen(cmd, cwd=self.server_dir, stdin=subprocess.PIPE)
        
        print("Server starting... waiting 30s for boot...")
        time.sleep(30) 

        try:
            with MCRcon(self.rcon_host, self.rcon_pass, port=self.rcon_port) as mcr:
                print("Connected via RCON.")
                
                # Convert blocks to chunk coordinates (radius / 16)
                # We round up to ensure we cover the requested area
                chunk_radius = (radius_in_blocks // 16) + 1
                
                min_c = -chunk_radius
                max_c = chunk_radius
                
                # Define coordinates: ~ is relative (but forceload uses absolute)
                min_x, min_z = min_c * 16, min_c * 16
                max_x, max_z = max_c * 16, max_c * 16
                
                print(f"Forcing generation (Radius: {radius_in_blocks} blocks)")
                print(f"Chunk area: {min_c},{min_c} to {max_c},{max_c}")
                
                mcr.command(f"forceload add {min_x} {min_z} {max_x} {max_z}")
                
                # Wait heuristic: 0.1s per chunk is usually enough for modern CPUs
                total_chunks = (max_c - min_c) * (max_c - min_c)
                wait_time = max(15, total_chunks * 0.1) 
                
                print(f"Waiting {int(wait_time)} seconds for generation...")
                time.sleep(wait_time)
                
                # Cleanup commands
                mcr.command("forceload remove all")
                mcr.command("save-all")
                time.sleep(2) # Give it a moment to write to disk
                mcr.command("stop")
        except Exception as e:
            print(f"RCON Error: {e}")
            process.terminate()

        process.wait()
        print("Server stopped.")

    async def query_block(self, x, y, z):
        """
        Asynchronously query a specific block.
        Runs in a separate thread to avoid blocking the main loop.
        """
        def _blocking_query():
            try:
                level = amulet.load_level(self.world_path)
                block = level.get_block(x, y, z, "minecraft:overworld")
                level.close()
                return block.namespaced_name
            except Exception as e:
                return f"Error: {e}"

        # Offload the blocking IO to a thread
        return await asyncio.to_thread(_blocking_query)

    async def find_blocks(self, block_name, search_radius_chunks=2):
        """
        Asynchronously scans for blocks. 
        Critical for scanning large areas without freezing your application.
        """
        print(f"--- Querying for '{block_name}' (Async) ---")

        def _blocking_search():
            try:
                level = amulet.load_level(self.world_path)
            except Exception as e:
                print(f"Error loading world: {e}")
                return []

            found_locations = []
            
            # Iterate chunks
            for cx in range(-search_radius_chunks, search_radius_chunks + 1):
                for cz in range(-search_radius_chunks, search_radius_chunks + 1):
                    try:
                        chunk = level.get_chunk(cx, cz, "minecraft:overworld")
                        # Iterate blocks
                        for x in range(16):
                            for z in range(16):
                                # Full vertical scan (-64 to 320 for 1.21)
                                for y in range(-64, 320):
                                    try:
                                        block = chunk.get_block(x, y, z)
                                        if block_name in block.namespaced_name:
                                            global_x = (cx * 16) + x
                                            global_z = (cz * 16) + z
                                            found_locations.append((global_x, y, global_z))
                                    except:
                                        pass
                    except:
                        pass # Chunk not generated

            level.close()
            return found_locations

        # Offload the heavy search loop to a thread
        return await asyncio.to_thread(_blocking_search)


    async def get_block_matrix(self, center_x, center_y, center_z, size=5, only_visible=False):
        """
        Returns a Numpy 3D Matrix of block names.
        Optimized: Fetches data once, then performs raycasting in-memory.
        """
        print(f"--- Generating {size}x{size}x{size} Matrix (Numpy Optimized) ---")

        def _blocking_matrix():
            try:
                level = amulet.load_level(self.world_path)
            except Exception as e:
                print(f"Error: {e}")
                return np.array([])

            radius = size // 2
            
            # 1. Create empty Numpy Array (Size: N, N, N)
            # We use 'object' type to store arbitrary length strings
            raw_matrix = np.empty((size, size, size), dtype=object)

            # Define bounds relative to the loop
            # We map Global Coords -> Matrix Indices [0..size]
            min_x, max_x = center_x - radius, center_x + radius
            min_y, max_y = center_y - radius, center_y + radius
            min_z, max_z = center_z - radius, center_z + radius

            # 2. FILL MATRIX (The slow IO part)
            # We must fetch all blocks first so the raycaster can "see" walls
            print("Fetching blocks...")
            for gx in range(min_x, max_x + 1):
                for gy in range(min_y, max_y + 1):
                    for gz in range(min_z, max_z + 1):
                        try:
                            # Map global coord to local index
                            lx, ly, lz = gx - min_x, gy - min_y, gz - min_z
                            
                            if 0 <= lx < size and 0 <= ly < size and 0 <= lz < size:
                                block = level.get_block(gx, gy, gz, "minecraft:overworld")
                                raw_matrix[lx, ly, lz] = block.namespaced_name
                        except:
                            raw_matrix[lx, ly, lz] = "minecraft:air"

            level.close()

            # If we don't care about visibility, return the raw data now
            if not only_visible:
                return raw_matrix

            # 3. RAYCASTING (The fast Memory part)
            print("Calculating visibility...")
            
            # Create a copy for the output
            visible_matrix = raw_matrix.copy()
            
            # Helper: Is this block transparent?
            # We define a set for O(1) lookups
            transparent_blocks = {
                "minecraft:air", "minecraft:water", "minecraft:lava", 
                "minecraft:glass", "minecraft:grass", "minecraft:poppy",
                "minecraft:dandelion", "minecraft:torch"
            }
            
            # Local center index (e.g., 2,2,2 for a size 5 box)
            cx, cy, cz = radius, radius, radius

            # Iterate every voxel in the local matrix
            it = np.nditer(raw_matrix, flags=['multi_index', 'refs_ok'])
            for _ in it:
                tx, ty, tz = it.multi_index # Target Local Coords
                
                target_block = raw_matrix[tx, ty, tz]
                
                # Optimization: Air is always "visible" (don't process sky)
                if "air" in target_block:
                    continue
                    
                # Vector from Center -> Target
                vx, vy, vz = tx - cx, ty - cy, tz - cz
                dist = math.sqrt(vx**2 + vy**2 + vz**2)
                
                if dist == 0: continue # Center is always visible

                # Normalize direction
                step_x, step_y, step_z = vx/dist, vy/dist, vz/dist
                
                # Ray march from Center -> Target
                # Check every 0.5 units
                curr_d = 0.5
                is_occluded = False
                
                while curr_d < dist - 0.5:
                    # Current ray position (float)
                    rx = cx + (step_x * curr_d)
                    ry = cy + (step_y * curr_d)
                    rz = cz + (step_z * curr_d)
                    
                    # Round to nearest index
                    ix, iy, iz = int(round(rx)), int(round(ry)), int(round(rz))
                    
                    # Ensure we are inside the matrix bounds
                    if 0 <= ix < size and 0 <= iy < size and 0 <= iz < size:
                        block_at_ray = raw_matrix[ix, iy, iz]
                        
                        # If we hit a SOLID block, the target is hidden
                        if block_at_ray not in transparent_blocks:
                            is_occluded = True
                            break
                    
                    curr_d += 0.5

                if is_occluded:
                    visible_matrix[tx, ty, tz] = "undefined"

            return visible_matrix

        return await asyncio.to_thread(_blocking_matrix)

    async def break_block(self, x, y, z):
        """
        Breaks a block: Returns its name and immediately sets it to Air.
        Note: Works best when the server is offline.
        """
        print(f"--- Breaking block at {x}, {y}, {z} ---")

        def _blocking_break():
            try:
                # 1. Open the World
                level = amulet.load_level(self.world_path)
                
                # 2. Get the current block (The "Drop")
                original_block = level.get_block(x, y, z, "minecraft:overworld")
                block_name = original_block.namespaced_name
                
                # 3. Set it to Air (The "Break")
                air_block = amulet.api.block.Block("minecraft", "air")
                level.set_block(x, y, z, air_block, "minecraft:overworld")
                
                # 4. Save Changes to Disk
                level.save()
                
                level.close()
                return block_name
                
            except Exception as e:
                return f"Error: {e}"

        # Run in a separate thread to handle File I/O
        return await asyncio.to_thread(_blocking_break)
    
    async def get_adjacent_blocks(self, x, y, z):
        """
        Returns a dictionary of the 6 blocks directly touching the given position.
        Directions: Up, Down, North (-Z), South (+Z), East (+X), West (-X)
        """
        print(f"--- Scanning surroundings of {x}, {y}, {z} ---")

        def _blocking_scan():
            try:
                level = amulet.load_level(self.world_path)
            except Exception as e:
                return f"Error: {e}"

            # 1. Define the 6 neighbors (One-Hot style mapping)
            directions = {
                "up":    (0, 1, 0),
                "down":  (0, -1, 0),
                "north": (0, 0, -1), # Negative Z
                "south": (0, 0, 1),  # Positive Z
                "east":  (1, 0, 0),  # Positive X
                "west":  (-1, 0, 0)  # Negative X
            }
            
            results = {}

            # 2. Query all 6 positions
            for direction, (dx, dy, dz) in directions.items():
                target_x = x + dx
                target_y = y + dy
                target_z = z + dz
                
                try:
                    # We specify the dimension to ensure we query the Overworld
                    block = level.get_block(target_x, target_y, target_z, "minecraft:overworld")
                    results[direction] = block.namespaced_name
                except:
                    # If the chunk isn't generated or is void
                    results[direction] = "undefined"

            level.close()
            return results

        # Run in separate thread
        return await asyncio.to_thread(_blocking_scan)
# --- EXECUTION ---
if __name__ == "__main__":
    manager = MinecraftWorldManager()
    
    # 1. Synchronous Setup (Server generation is still blocking/sync)
    # We generally don't make the server generation async because the OS process blocks anyway
    manager.reset_world()
    manager.setup_server(seed="12345")
    manager.generate_world(radius_in_blocks=80)
    
    # 2. Asynchronous Querying
    async def main():
        print("\n--- Starting Async Operations ---")
        
        # Example: Run a query and a search at the same time?
        # Or just await them one by one:
        
        # Query single block
        block_name = await manager.query_block(0, 70, 0)
        print(f"Block at 0,70,0: {block_name}")
        
        # Search for ores
        print("Starting ore search...")
        diamonds = await manager.find_blocks("diamond_ore", search_radius_chunks=5)
        
        print(f"Found {len(diamonds)} diamond ore blocks.")
        if diamonds:
            print(f"First location: {diamonds[0]}")

    # Run the async loop
    asyncio.run(main())