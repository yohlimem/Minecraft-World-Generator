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
import matplotlib.pyplot as plt
import heapq

# No asyncio needed anymore
# import asyncio 

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
        """Starts the server and forces generation within a radius around 0,0."""
        print("--- Step 1: Generating World ---")
        
        cmd = ["java", "-Xmx2G", "-jar", "server.jar", "nogui"]
        process = subprocess.Popen(cmd, cwd=self.server_dir, stdin=subprocess.PIPE)
        
        print("Server starting... waiting 30s for boot...")
        time.sleep(30) 

        try:
            with MCRcon(self.rcon_host, self.rcon_pass, port=self.rcon_port) as mcr:
                print("Connected via RCON.")
                chunk_radius = (radius_in_blocks // 16) + 1
                min_c, max_c = -chunk_radius, chunk_radius
                min_x, min_z = min_c * 16, min_c * 16
                max_x, max_z = max_c * 16, max_c * 16
                
                print(f"Forcing generation (Radius: {radius_in_blocks} blocks)")
                mcr.command(f"forceload add {min_x} {min_z} {max_x} {max_z}")
                
                total_chunks = (max_c - min_c) * (max_c - min_c)
                wait_time = max(15, total_chunks * 0.1) 
                
                print(f"Waiting {int(wait_time)} seconds for generation...")
                time.sleep(wait_time)
                
                mcr.command("forceload remove all")
                mcr.command("save-all")
                time.sleep(2)
                mcr.command("stop")
        except Exception as e:
            print(f"RCON Error: {e}")
            process.terminate()

        process.wait()
        print("Server stopped.")

    def query_block(self, x, y, z):
        """Synchronously query a specific block."""
        try:
            level = amulet.load_level(self.world_path)
            block = level.get_block(x, y, z, "minecraft:overworld")
            level.close()
            return block.namespaced_name
        except Exception as e:
            return f"Error: {e}"

    def find_blocks(self, block_name, search_radius_chunks=2):
        """Synchronously scans for blocks."""
        print(f"--- Querying for '{block_name}' (Sync) ---")
        try:
            level = amulet.load_level(self.world_path)
        except Exception as e:
            print(f"Error loading world: {e}")
            return []

        found_locations = []
        for cx in range(-search_radius_chunks, search_radius_chunks + 1):
            for cz in range(-search_radius_chunks, search_radius_chunks + 1):
                try:
                    chunk = level.get_chunk(cx, cz, "minecraft:overworld")
                    for x in range(16):
                        for z in range(16):
                            for y in range(-64, 320):
                                try:
                                    block = chunk.get_block(x, y, z)
                                    if block_name in block.namespaced_name:
                                        global_x = (cx * 16) + x
                                        global_z = (cz * 16) + z
                                        found_locations.append((global_x, y, global_z))
                                except: pass
                except: pass # Chunk not generated

        level.close()
        return found_locations

    def get_block_matrix(self, cx, cy, cz, size=5, only_visible=False):
        """
        Returns a 3D Matrix of block names.
        Synchronous Ray Marching.
        """
        print(f"--- Matrix Scan ({size}x{size}x{size}) with Ray Marching ---")
        
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
        
        transparent_keywords = {
            "air", "water", "lava", "glass", "leaves", "grass", 
            "fern", "flower", "torch", "poppy", "dandelion", "kelp"
        }
        
        lcx, lcy, lcz = radius, radius, radius

        it = np.nditer(raw, flags=['multi_index', 'refs_ok'])
        for _ in it:
            tx, ty, tz = it.multi_index
            target_name = raw[tx, ty, tz]
            
            if "air" in target_name: continue
            if tx == lcx and ty == lcy and tz == lcz: continue

            vx, vy, vz = tx - lcx, ty - lcy, tz - lcz
            dist = math.sqrt(vx**2 + vy**2 + vz**2)
            
            if dist == 0: continue

            step_x, step_y, step_z = vx/dist, vy/dist, vz/dist
            
            curr_d = 0.6
            occluded = False
            
            while curr_d < dist - 0.5:
                rx = lcx + (step_x * curr_d)
                ry = lcy + (step_y * curr_d)
                rz = lcz + (step_z * curr_d)
                
                ix, iy, iz = int(round(rx)), int(round(ry)), int(round(rz))
                
                if 0 <= ix < size and 0 <= iy < size and 0 <= iz < size:
                    block_hit = raw[ix, iy, iz]
                    is_transparent = False
                    for key in transparent_keywords:
                        if key in block_hit:
                            is_transparent = True
                            break
                    if not is_transparent:
                        occluded = True
                        break
                curr_d += 0.5

            if occluded:
                visible[tx, ty, tz] = "undefined"

        return visible

    def break_block(self, x, y, z):
        """Breaks a block: Returns its name and immediately sets it to Air."""
        print(f"--- Breaking block at {x}, {y}, {z} ---")
        try:
            level = amulet.load_level(self.world_path)
            original_block = level.get_block(x, y, z, "minecraft:overworld")
            block_name = original_block.namespaced_name
            
            air_block = amulet.api.block.Block("minecraft", "air")
            level.set_version_block(x, y, z, "minecraft:overworld", ("java", (1,21,11)), air_block)
            
            level.save()
            level.close()
            return block_name
        except Exception as e:
            return f"Error: {e}"

    def get_adjacent_blocks(self, x, y, z):
        """Returns a dictionary of the 6 blocks directly touching the given position."""
        print(f"--- Scanning surroundings of {x}, {y}, {z} ---")
        try:
            level = amulet.load_level(self.world_path)
            directions = {
                "up":    (0, 1, 0), "down":  (0, -1, 0),
                "north": (0, 0, -1), "south": (0, 0, 1),
                "east":  (1, 0, 0), "west":  (-1, 0, 0)
            }
            results = {}
            for direction, (dx, dy, dz) in directions.items():
                target_x, target_y, target_z = x + dx, y + dy, z + dz
                try:
                    block = level.get_block(target_x, target_y, target_z, "minecraft:overworld")
                    results[direction] = block.namespaced_name
                except:
                    results[direction] = "undefined"
            level.close()
            return results
        except Exception as e:
            return f"Error: {e}"

    def drop_to_ground(self, x, y, z):
        """Simulates gravity: Lowers the Y coordinate until a solid block is found."""
        print(f"--- Dropping from {x}, {y}, {z} ---")
        try:
            level = amulet.load_level(self.world_path)
            for current_y in range(int(y), -65, -1):
                try:
                    block = level.get_block(x, current_y, z, "minecraft:overworld")
                    name = block.namespaced_name
                    non_solid = ["air", "water", "lava", "void"]
                    if not any(ns in name for ns in non_solid):
                        level.close()
                        return (x, current_y + 1, z)
                except: pass 
            level.close()
            return (x, -64, z)
        except Exception as e:
            print(f"Error: {e}")
            return (x, y, z)

    def visualize_matrix(self, matrix):
        """Visualizes the 3D block matrix using Matplotlib voxels."""
        print("--- Rendering 3D Visualization ---")
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not installed.")
            return
        
        colors = {
            "minecraft:grass_block": (0.1, 0.8, 0.1, 1.0),
            "minecraft:dirt":        (0.6, 0.4, 0.2, 1.0),
            "minecraft:stone":       (0.5, 0.5, 0.5, 1.0),
            "minecraft:deepslate":   (0.3, 0.3, 0.3, 1.0),
            "minecraft:water":       (0.1, 0.1, 0.9, 0.4),
            "minecraft:lava":        (1.0, 0.5, 0.0, 1.0),
            "minecraft:sand":        (0.9, 0.9, 0.6, 1.0),
            "minecraft:gravel":      (0.6, 0.6, 0.6, 1.0),
            "minecraft:diamond_ore": (0.0, 1.0, 1.0, 1.0),
            "minecraft:gold_ore":    (1.0, 0.8, 0.0, 1.0),
            "undefined":             (0.1, 0.1, 0.1, 0.1),
            "default":               (0.8, 0.0, 0.8, 1.0)
        }

        size_x, size_y, size_z = matrix.shape
        filled = np.zeros(matrix.shape, dtype=bool)
        voxel_colors = np.zeros(matrix.shape + (4,), dtype=float)

        for x in range(size_x):
            for y in range(size_y):
                for z in range(size_z):
                    block_name = matrix[x, y, z]
                    if "air" in block_name: continue
                    filled[x, y, z] = True
                    
                    if block_name in colors:
                        voxel_colors[x, y, z] = colors[block_name]
                    else:
                        if "log" in block_name:   voxel_colors[x, y, z] = (0.4, 0.2, 0.0, 1.0)
                        elif "leaf" in block_name:voxel_colors[x, y, z] = (0.1, 0.6, 0.1, 0.5)
                        elif "ore" in block_name: voxel_colors[x, y, z] = (0.8, 0.8, 0.8, 1.0)
                        else:                     voxel_colors[x, y, z] = colors["default"]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(filled, facecolors=voxel_colors, edgecolors='k', linewidth=0.1, shade=True)
        
        cx_idx, cy_idx, cz_idx = size_x // 2, size_y // 2, size_z // 2
        ax.scatter(cx_idx + 0.5, cy_idx + 0.5, cz_idx + 0.5,
                   color='red', s=200, marker='o', label='Scan Center', depthshade=False)
        ax.legend()
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis (Height)')
        ax.set_zlabel('Z Axis')
        ax.set_title(f'Sigma Vision: {size_x}x{size_y}x{size_z} Scan')
        ax.invert_zaxis() 
        plt.show()
    
    def is_transparent(self, block_name):
        """
        Determines if a block is transparent (allows light/vision through).
        Args:
            block_name (str): The namespaced name (e.g., 'minecraft:glass')
        Returns:
            bool: True if transparent, False if solid/opaque.
        """
        # 1. Normalized check (ignore namespace prefix)
        # We split by ':' and take the last part to get just 'glass' or 'air'
        if ":" in block_name:
            name = block_name.split(":")[-1].lower()
        else:
            name = block_name.lower()

        # 2. Complete List of Transparent Keywords
        # If the block name contains ANY of these, it is transparent.
        transparent_keywords = {
            "air", "void", "cave_air",
            "water", "lava",
            "glass", "stained_glass", "pane",
            "leaves", "leaf",
            "grass", "fern", "bush", "shrub", "flower", "poppy", "dandelion", "rose", "tulip", "orchid", "allium", "bluet", "daisy", "lilac", "peony",
            "torch", "lantern", "lamp",
            "fence", "gate", "bars",
            "door", "trapdoor",
            "ladder", "vine", "scaffolding",
            "rail",
            "carpet",
            "snow", # snow layers
            "mushroom", "fungus",
            "sapling",
            "kelp", "seagrass", "lily_pad",
            "cobweb",
            "slime", "honey",
            "bamboo", "sugar_cane",
            "chain",
            "bed", # partial transparency
            "banner", "sign"
        }

        # 3. Check exact matches or partial matches
        # We check if any keyword is a substring of the block name
        for keyword in transparent_keywords:
            if keyword in name:
                return True
                
        return False
    
    def is_passable(self, block_name):
        """
        Determines if a player can physically walk through a block.
        UNLIKE transparency, Glass and Fences are NOT passable.
        """
        # Normalize name
        if ":" in block_name:
            name = block_name.split(":")[-1].lower()
        else:
            name = block_name.lower()

        # 1. The "Walk-Through" List
        # These are blocks that do not have collision boxes
        passable_keywords = {
            "air", "void", "cave_air",
            "water", "lava", # Technically fluid, but you can move through them
            "grass", "fern", "flower", "poppy", "dandelion", "rose", "tulip", "orchid",
            "torch", "redstone_torch", "soul_torch",
            "rail", "carpet", "snow", # Snow layers are passable usually
            "button", "lever", "tripwire", "string",
            "sapling", "kelp", "seagrass",
            "cobweb", # Slows you down, but passable
            "sign", "banner",
            "pressure_plate"
        }

        # 2. Check keywords
        for keyword in passable_keywords:
            if keyword in name:
                return True
        
        # 3. Special Case: Open Doors/Gates? 
        # (Too complex for simple string matching, assuming closed/solid for now)
        
        return False

    def check_path_clear(self, start_x, start_y, start_z, target_x, target_y, target_z):
        """
        Raytraces a straight line from Start to Target.
        Returns True if the path is clear of solid obstacles.
        """
        print(f"--- Checking Path: {start_x},{start_y},{start_z} -> {target_x},{target_y},{target_z} ---")
        
        try:
            level = amulet.load_level(self.world_path)
        except: return False

        # Vector Math
        dx = target_x - start_x
        dy = target_y - start_y
        dz = target_z - start_z
        
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        if distance == 0: 
            level.close()
            return True

        # Normalize direction
        step_x = dx / distance
        step_y = dy / distance
        step_z = dz / distance

        # Ray Marching Setup
        current_dist = 0.5 # Start slightly away from origin center
        
        is_clear = True
        
        while current_dist < distance:
            # Calculate sample point
            rx = start_x + (step_x * current_dist)
            ry = start_y + (step_y * current_dist)
            rz = start_z + (step_z * current_dist)
            
            # Get Block at this point
            ix, iy, iz = int(round(rx)), int(round(ry)), int(round(rz))
            
            try:
                # Optimized: Don't check if it's the start or target block exactly 
                # (We assume start is safe, and target is destination)
                if (ix == int(start_x) and iy == int(start_y) and iz == int(start_z)):
                    pass # Inside start block
                elif (ix == int(target_x) and iy == int(target_y) and iz == int(target_z)):
                    pass # Hit target block
                else:
                    block = level.get_block(ix, iy, iz, "minecraft:overworld")
                    name = block.namespaced_name
                    
                    if not self.is_passable(name):
                        print(f"Blocked by: {name} at {ix}, {iy}, {iz}")
                        is_clear = False
                        break
            except:
                pass # Void/Error assumed safe or ignore
            
            current_dist += 0.5 # Check every half block

        level.close()
        return is_clear
    
    def is_reachable_astar(self, start, target, max_nodes=2000):
        """
        Uses A* Pathfinding to check if a player can WALK to the target.
        Simulates gravity, jumping (1 block), and headroom (2 blocks).
        
        Args:
            start: tuple (x, y, z)
            target: tuple (x, y, z)
            max_nodes: Safety limit to prevent freezing on impossible paths.
        """
        print(f"--- A* Pathfinding: {start} -> {target} ---")
        
        # 1. Open Level ONCE (Optimization)
        try:
            level = amulet.load_level(self.world_path)
        except Exception as e:
            print(f"World Load Error: {e}")
            return False

        # Helper to read blocks quickly without reopening level
        def get_block_type(cx, cy, cz):
            try:
                b = level.get_block(cx, cy, cz, "minecraft:overworld")
                return b.namespaced_name
            except: return "minecraft:bedrock" # Treat error as solid wall

        # 2. Setup A* Structures
        # Priority Queue: (f_score, (x, y, z))
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        # Tracks where we came from (for path reconstruction if needed)
        came_from = {}
        
        # Cost from start to current node
        g_score = {start: 0}
        
        # Heuristic (Manhattan distance is fast for grid grids)
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

        nodes_checked = 0

        # 3. The Search Loop
        while open_set:
            if nodes_checked > max_nodes:
                print("A* Limit Reached (Path too complex or impossible)")
                level.close()
                return False

            # Pop node with lowest F score
            current_f, current = heapq.heappop(open_set)
            nodes_checked += 1

            if current == target:
                print(f"Path Found! Checked {nodes_checked} nodes.")
                level.close()
                return True

            cx, cy, cz = current
            
            # GENERATE MOVES: North, South, East, West
            neighbors = [
                (cx+1, cz), (cx-1, cz), (cx, cz+1), (cx, cz-1)
            ]

            for nx, nz in neighbors:
                # We check 3 possible vertical positions for every horizontal step:
                # 1. Flat (Walking same level)
                # 2. Jump (Stepping up 1 block)
                # 3. Drop (Stepping down 1 block)
                
                # We need to find which Y is valid
                target_y = None
                
                # Query relevant blocks column at neighbor
                # We need to know: Is head clear? Is feet clear? Is floor solid?
                
                # Check Flat Move (y)
                # Needs: Air at y, Air at y+1, Solid at y-1
                n_feet = get_block_type(nx, cy, nz)
                n_head = get_block_type(nx, cy+1, nz)
                n_floor = get_block_type(nx, cy-1, nz)
                
                if self.is_passable(n_feet) and self.is_passable(n_head) and not self.is_passable(n_floor):
                    target_y = cy
                
                # Check Jump Move (y+1)
                # Needs: Air at y+1, Air at y+2, Solid at y (The block we jump onto)
                # AND: We need headroom at current position to jump (cx, cy+2, cz)
                elif target_y is None: # Only check if flat failed
                    n_jump_feet = n_head # y+1
                    n_jump_head = get_block_type(nx, cy+2, nz) # y+2
                    # n_feet (y) is the block we are stepping onto, so it must be solid
                    
                    curr_headroom = get_block_type(cx, cy+2, cz)
                    
                    if (self.is_passable(n_jump_feet) and 
                        self.is_passable(n_jump_head) and 
                        not self.is_passable(n_feet) and 
                        self.is_passable(curr_headroom)):
                        target_y = cy + 1

                # Check Drop Move (y-1)
                # Needs: Air at y-1, Air at y, Solid at y-2
                elif target_y is None:
                    n_drop_feet = n_floor # y-1
                    n_drop_head = n_feet  # y
                    n_drop_floor = get_block_type(nx, cy-2, nz)
                    
                    if (self.is_passable(n_drop_feet) and 
                        self.is_passable(n_drop_head) and 
                        not self.is_passable(n_drop_floor)):
                        target_y = cy - 1

                # If we found a valid move
                if target_y is not None:
                    neighbor = (nx, target_y, nz)
                    
                    # Standard cost is 1 (walking)
                    new_g = g_score[current] + 1
                    
                    if neighbor not in g_score or new_g < g_score[neighbor]:
                        g_score[neighbor] = new_g
                        f = new_g + heuristic(neighbor, target)
                        heapq.heappush(open_set, (f, neighbor))
                        came_from[neighbor] = current

        level.close()
        return False

# --- EXECUTION ---
if __name__ == "__main__":
    manager = MinecraftWorldManager()
    
    # Optional: Setup
    # manager.reset_world()
    # manager.setup_server(seed="12345")
    # manager.generate_world(radius_in_blocks=80)
    
    # Define Scan Parameters
    size = 10
    center_x, center_y, center_z = 86, 91, -106
    
    print("Scanning terrain...")
    
    # Synchronous calls (No await)
    manager.break_block(87, 93, -109)
    manager.break_block(87, 92, -109)
    manager.break_block(87, 93, -110)
    
    matrix_data = manager.get_block_matrix(
        center_x, center_y, center_z, 
        size=size, 
        only_visible=True
    )
    
    print(matrix_data)
    manager.visualize_matrix(matrix_data)