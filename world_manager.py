import os
import subprocess
import time
import requests
import shutil
import math
from mcrcon import MCRcon
import amulet
from amulet.api.block import Block
import numpy as np
import asyncio
import matplotlib.pyplot as plt

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


    async def get_block_matrix(self, cx, cy, cz, size=5, only_visible=False):
        """
        Returns a 3D Matrix of block names.
        Uses Ray Marching (checking points along a line) for visibility.
        Simple, robust, and works with any namespace.
        """
        print(f"--- Matrix Scan ({size}x{size}x{size}) with Ray Marching ---")

        def _blocking():
            try:
                level = amulet.load_level(self.world_path)
            except: return np.array([])
            
            radius = size // 2
            # 1. Fetch Data
            raw = np.empty((size, size, size), dtype=object)
            min_x, max_x = cx - radius, cx + radius
            min_y, max_y = cy - radius, cy + radius
            min_z, max_z = cz - radius, cz + radius

            for gx in range(min_x, max_x + 1):
                for gy in range(min_y, max_y + 1):
                    for gz in range(min_z, max_z + 1):
                        lx, ly, lz = gx - min_x, gy - min_y, gz - min_z
                        if 0 <= lx < size and 0 <= ly < size and 0 <= lz < size:
                            try:
                                b = level.get_block(gx, gy, gz, "minecraft:overworld")
                                raw[lx, ly, lz] = b.namespaced_name
                            except: raw[lx, ly, lz] = "minecraft:air"
            level.close()

            if not only_visible: return raw

            # 2. Ray Marching (The Simple Way)
            visible = raw.copy()
            
            # Keywords to check transparency
            transparent_keywords = {
                "air", "water", "lava", "glass", "leaves", "grass", 
                "fern", "flower", "torch", "poppy", "dandelion", "kelp"
            }
            
            lcx, lcy, lcz = radius, radius, radius

            it = np.nditer(raw, flags=['multi_index', 'refs_ok'])
            for _ in it:
                tx, ty, tz = it.multi_index
                target_name = raw[tx, ty, tz]
                
                # Optimizations
                if "air" in target_name: continue
                if tx == lcx and ty == lcy and tz == lcz: continue

                # Vector from Center to Target
                vx, vy, vz = tx - lcx, ty - lcy, tz - lcz
                dist = math.sqrt(vx**2 + vy**2 + vz**2)
                
                if dist == 0: continue

                # Normalize Direction
                step_x, step_y, step_z = vx/dist, vy/dist, vz/dist
                
                # Walk along the line
                # We start at 0.6 to skip the center block itself
                # We stop at dist - 0.5 to stop right before the target block
                curr_d = 0.6
                occluded = False
                
                while curr_d < dist - 0.5:
                    # Get sample position
                    rx = lcx + (step_x * curr_d)
                    ry = lcy + (step_y * curr_d)
                    rz = lcz + (step_z * curr_d)
                    
                    # Round to nearest integer index
                    ix, iy, iz = int(round(rx)), int(round(ry)), int(round(rz))
                    
                    # Check bounds
                    if 0 <= ix < size and 0 <= iy < size and 0 <= iz < size:
                        block_hit = raw[ix, iy, iz]
                        
                        # Check transparency
                        is_transparent = False
                        for key in transparent_keywords:
                            if key in block_hit:
                                is_transparent = True
                                break
                        
                        if not is_transparent:
                            occluded = True
                            break
                    
                    # Step size: 0.5 ensures we don't skip over any 1x1 block
                    curr_d += 0.5

                if occluded:
                    visible[tx, ty, tz] = "undefined"

            return visible

        return await asyncio.to_thread(_blocking)

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
                level.set_version_block(x, y, z, "minecraft:overworld", ("java", (1,21,11)), air_block)
                
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
        
    async def drop_to_ground(self, x, y, z):
        """
        Simulates gravity: Lowers the Y coordinate until a solid block is found.
        Returns: The coordinate (x, new_y, z) directly ON TOP of the ground.
        """
        print(f"--- Dropping from {x}, {y}, {z} ---")

        def _blocking_drop():
            try:
                level = amulet.load_level(self.world_path)
                
                # Start at current Y and look down
                # World bottom is -64 in 1.21
                for current_y in range(int(y), -65, -1):
                    try:
                        # Check the block we are currently inside
                        block = level.get_block(x, current_y, z, "minecraft:overworld")
                        name = block.namespaced_name
                        
                        # List of things we fall through
                        non_solid = ["air", "water", "lava", "void"]
                        
                        # If the block is NOT air (it is solid), we hit the ground.
                        # The "safe spot" is the block immediately above this one.
                        if not any(ns in name for ns in non_solid):
                            level.close()
                            return (x, current_y + 1, z)
                            
                    except:
                        pass # Treat unload/errors as air/void and keep falling

                level.close()
                return (x, -64, z) # Hit bottom of the world (Void)
            
            except Exception as e:
                print(f"Error: {e}")
                return (x, y, z) # Return original if error

        return await asyncio.to_thread(_blocking_drop)
    def visualize_matrix(self, matrix):
        """
        Visualizes the 3D block matrix using Matplotlib voxels.
        Includes a red dot marking the center scan position.
        Must be called from the main thread (blocking).
        """
        print("--- Rendering 3D Visualization ---")
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not installed. Skipping visualization.")
            return
        
        # 1. Define Color Palette
        colors = {
            "minecraft:grass_block": (0.1, 0.8, 0.1, 1.0),
            "minecraft:dirt":        (0.6, 0.4, 0.2, 1.0),
            "minecraft:stone":       (0.5, 0.5, 0.5, 1.0),
            "minecraft:deepslate":   (0.3, 0.3, 0.3, 1.0),
            "minecraft:water":       (0.1, 0.1, 0.9, 0.4), # Transparent
            "minecraft:lava":        (1.0, 0.5, 0.0, 1.0),
            "minecraft:sand":        (0.9, 0.9, 0.6, 1.0),
            "minecraft:gravel":      (0.6, 0.6, 0.6, 1.0),
            # Ores highlight
            "minecraft:diamond_ore": (0.0, 1.0, 1.0, 1.0),
            "minecraft:gold_ore":    (1.0, 0.8, 0.0, 1.0),
            # Special
            "undefined":             (0.1, 0.1, 0.1, 0.1), # Faint shadow
            "default":               (0.8, 0.0, 0.8, 1.0)  # Magenta
        }

        # 2. Prepare Voxel Data
        size_x, size_y, size_z = matrix.shape
        filled = np.zeros(matrix.shape, dtype=bool)
        voxel_colors = np.zeros(matrix.shape + (4,), dtype=float)

        for x in range(size_x):
            for y in range(size_y):
                for z in range(size_z):
                    block_name = matrix[x, y, z]
                    if "air" in block_name: continue
                    
                    # Optional: hide undefined blocks completely
                    # if block_name == "undefined": continue

                    filled[x, y, z] = True
                    
                    if block_name in colors:
                        voxel_colors[x, y, z] = colors[block_name]
                    else:
                        # Keyword fallbacks
                        if "log" in block_name:   voxel_colors[x, y, z] = (0.4, 0.2, 0.0, 1.0)
                        elif "leaf" in block_name:voxel_colors[x, y, z] = (0.1, 0.6, 0.1, 0.5)
                        elif "ore" in block_name: voxel_colors[x, y, z] = (0.8, 0.8, 0.8, 1.0)
                        else:                     voxel_colors[x, y, z] = colors["default"]

        # 3. Plotting
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw the blocks
        ax.voxels(filled, facecolors=voxel_colors, edgecolors='k', linewidth=0.1, shade=True)
        
        # --- DRAW CENTER DOT ---
        # Calculate center index
        cx_idx, cy_idx, cz_idx = size_x // 2, size_y // 2, size_z // 2
        # Add 0.5 to put the dot in the middle of the central voxel block
        ax.scatter(cx_idx + 0.5, cy_idx + 0.5, cz_idx + 0.5,
                   color='red', s=200, marker='o', label='Scan Center', depthshade=False)
        ax.legend()
        # -----------------------

        # Formatting
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis (Height)')
        ax.set_zlabel('Z Axis')
        ax.set_title(f'Sigma Vision: {size_x}x{size_y}x{size_z} Scan')
        # Invert Z axis to match Minecraft's "North is negative Z" feel visually
        ax.invert_zaxis() 
        
        plt.show()
# --- EXECUTION ---
if __name__ == "__main__":
    manager = MinecraftWorldManager()
    
    # 1. Synchronous Setup (Server generation is still blocking/sync)
    # We generally don't make the server generation async because the OS process blocks anyway
    seed = "12345"
    # if seed
    #     manager.reset_world()
    #     manager.setup_server(seed="12345")
    #     manager.generate_world(radius_in_blocks=80)
    
    # 2. Asynchronous Querying
    async def main():
        # 1. Define Scan Parameters
        # A size of 7-9 is best for visualization. Too large gets laggy.
        size = 10
        center_x, center_y, center_z = 86, 91, -106
        
        # 2. Get the Matrix (Async)
        print("Scanning terrain...")
        await manager.break_block(87, 93, -109)
        await manager.break_block(87, 92, -109)
        await manager.break_block(87, 93, -110)
        matrix = await manager.get_block_matrix(
            center_x, center_y, center_z, 
            size=size, 
            only_visible=True # Set True to see the "Fog of War" (undefined blocks)
        )
        
        return matrix, (center_x, center_y, center_z)

    # Run the async loop
    matrix_data, center = asyncio.run(main())
    print(matrix_data)
    manager.visualize_matrix(matrix_data)