import os
import subprocess
import time
import requests
import shutil
import math
import heapq
from typing import List, Tuple, Dict, Set, Optional, Any
from mcrcon import MCRcon # type: ignore
import amulet # type: ignore
from amulet.api.block import Block # type: ignore
import numpy as np
import matplotlib.pyplot as plt

class MinecraftWorldManager:
    def __init__(self, server_dir: str = "minecraft_server") -> None:
        self.server_dir: str = server_dir
        self.world_name: str = "world"
        self.world_path: str = os.path.join(server_dir, self.world_name)
        self.jar_path: str = os.path.join(server_dir, "server.jar")
        self.eula_path: str = os.path.join(server_dir, "eula.txt")
        self.props_path: str = os.path.join(server_dir, "server.properties")
        
        # RCON settings
        self.rcon_host: str = "localhost"
        self.rcon_port: int = 25575
        self.rcon_pass: str = "password123"

    def reset_world(self) -> None:
        """Deletes the existing 'world' folder to ensure a fresh generation."""
        if os.path.exists(self.world_path):
            print(f"Deleting old world at: {self.world_path}...")
            try:
                shutil.rmtree(self.world_path)
                print("Old world deleted.")
            except Exception as e:
                print(f"Warning: Could not delete world folder completely: {e}")

    def setup_server(self, seed: str = "") -> None:
        """Downloads server, accepts EULA, and configures seed."""
        if not os.path.exists(self.server_dir):
            os.makedirs(self.server_dir)

        # 1. Download Server JAR
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

    def generate_world(self, radius_in_blocks: int) -> None:
        """Starts the server and forces generation within a radius around 0,0."""
        print("--- Step 1: Generating World ---")
        
        cmd: List[str] = ["java", "-Xmx2G", "-jar", "server.jar", "nogui"]
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
                wait_time = max(15.0, total_chunks * 0.1) 
                
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

    def query_block(self, x: int, y: int, z: int) -> str:
        """Synchronously query a specific block."""
        try:
            level = amulet.load_level(self.world_path)
            block = level.get_block(x, y, z, "minecraft:overworld")
            level.close()
            return str(block.namespaced_name)
        except Exception as e:
            return f"Error: {e}"

    def find_blocks(self, block_name: str, search_radius_chunks: int = 2) -> List[Tuple[int, int, int]]:
        """Synchronously scans for blocks."""
        print(f"--- Querying for '{block_name}' (Sync) ---")
        try:
            level = amulet.load_level(self.world_path)
        except Exception as e:
            print(f"Error loading world: {e}")
            return []

        found_locations: List[Tuple[int, int, int]] = []
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
                except: pass

        level.close()
        return found_locations

    def is_transparent(self, block_name: str) -> bool:
        """Determines if a block allows vision through it."""
        if ":" in block_name:
            name = block_name.split(":")[-1].lower()
        else:
            name = block_name.lower()

        transparent_keywords: Set[str] = {
            "air", "void", "cave_air", "water", "lava",
            "glass", "stained_glass", "pane",
            "leaves", "leaf", "grass", "fern", "bush", "shrub", 
            "flower", "poppy", "dandelion", "rose", "tulip", 
            "torch", "lantern", "lamp", "fence", "gate", "bars",
            "door", "trapdoor", "ladder", "vine", "scaffolding", "rail",
            "carpet", "snow", "mushroom", "fungus", "sapling",
            "kelp", "seagrass", "lily_pad", "cobweb", "slime", "honey",
            "bamboo", "sugar_cane", "chain", "bed", "banner", "sign"
        }

        for keyword in transparent_keywords:
            if keyword in name:
                return True
        return False

    def is_passable(self, block_name: str) -> bool:
        """Determines if a player can physically walk through a block."""
        if ":" in block_name:
            name = block_name.split(":")[-1].lower()
        else:
            name = block_name.lower()

        passable_keywords: Set[str] = {
            "air", "void", "cave_air",
            "water", "lava",
            "grass", "fern", "flower", "poppy", "dandelion", "rose", "tulip", "orchid",
            "torch", "redstone_torch", "soul_torch",
            "rail", "carpet", "snow", 
            "button", "lever", "tripwire", "string",
            "sapling", "kelp", "seagrass",
            "cobweb", "sign", "banner", "pressure_plate"
        }

        for keyword in passable_keywords:
            if keyword in name:
                return True
        return False

    def get_block_matrix(self, cx: int, cy: int, cz: int, size: int = 5, only_visible: bool = False) -> np.ndarray:
        """Returns a 3D Matrix of block names."""
        print(f"--- Matrix Scan ({size}x{size}x{size}) with Ray Marching ---")
        
        try:
            level = amulet.load_level(self.world_path)
        except: return np.array([])
        
        radius = size // 2
        raw = np.empty((size, size, size), dtype=object)
        min_x, max_x = cx - radius, cx + radius
        min_y, max_y = cy - radius, cy + radius
        min_z, max_z = cz - radius, cz + radius

        # 1. Fetch Data
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

        # 2. Ray Marching
        visible = raw.copy()
        lcx, lcy, lcz = radius, radius, radius

        it = np.nditer(raw, flags=['multi_index', 'refs_ok'])
        for _ in it:
            tx, ty, tz = it.multi_index # type: ignore
            target_name: str = raw[tx, ty, tz] # type: ignore
            
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
                    block_hit: str = raw[ix, iy, iz] # type: ignore
                    if not self.is_transparent(block_hit):
                        occluded = True
                        break
                curr_d += 0.5

            if occluded:
                visible[tx, ty, tz] = "undefined"

        return visible

    def break_block(self, x: int, y: int, z: int) -> str:
        """Breaks a block: Returns its name and immediately sets it to Air."""
        print(f"--- Breaking block at {x}, {y}, {z} ---")
        try:
            level = amulet.load_level(self.world_path)
            original_block = level.get_block(x, y, z, "minecraft:overworld")
            block_name: str = original_block.namespaced_name
            
            air_block = Block("minecraft", "air")
            level.set_version_block(x, y, z, "minecraft:overworld", ("java", (1,21,11)), air_block)
            
            level.save()
            level.close()
            return block_name
        except Exception as e:
            return f"Error: {e}"

    def get_adjacent_blocks(self, x: int, y: int, z: int) -> Dict[str, str]:
        """Returns a dictionary of the 6 blocks directly touching the given position."""
        print(f"--- Scanning surroundings of {x}, {y}, {z} ---")
        try:
            level = amulet.load_level(self.world_path)
            directions = {
                "up":    (0, 1, 0), "down":  (0, -1, 0),
                "north": (0, 0, -1), "south": (0, 0, 1),
                "east":  (1, 0, 0), "west":  (-1, 0, 0)
            }
            results: Dict[str, str] = {}
            for direction, (dx, dy, dz) in directions.items():
                target_x, target_y, target_z = x + dx, y + dy, z + dz
                try:
                    block = level.get_block(target_x, target_y, target_z, "minecraft:overworld")
                    results[direction] = str(block.namespaced_name)
                except:
                    results[direction] = "undefined"
            level.close()
            return results
        except Exception as e:
            return {"error": str(e)}

    def drop_to_ground(self, x: int, y: int, z: int) -> Tuple[int, int, int]:
        """Simulates gravity: Lowers the Y coordinate until a solid block is found."""
        print(f"--- Dropping from {x}, {y}, {z} ---")
        try:
            level = amulet.load_level(self.world_path)
            # Iterate downwards from current y
            for current_y in range(int(y), -65, -1):
                try:
                    block = level.get_block(x, current_y, z, "minecraft:overworld")
                    name: str = block.namespaced_name
                    if not self.is_passable(name):
                        level.close()
                        return (x, current_y + 1, z)
                except: pass 
            level.close()
            return (x, -64, z)
        except Exception as e:
            print(f"Error: {e}")
            return (x, int(y), z)

    def check_path_clear(self, start_x: float, start_y: float, start_z: float, 
                         target_x: float, target_y: float, target_z: float) -> bool:
        """Raytraces a straight line from Start to Target."""
        print(f"--- Checking Path: {start_x},{start_y},{start_z} -> {target_x},{target_y},{target_z} ---")
        
        try:
            level = amulet.load_level(self.world_path)
        except: return False

        dx = target_x - start_x
        dy = target_y - start_y
        dz = target_z - start_z
        
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        if distance == 0: 
            level.close()
            return True

        step_x = dx / distance
        step_y = dy / distance
        step_z = dz / distance

        current_dist = 0.5 
        is_clear = True
        
        while current_dist < distance:
            rx = start_x + (step_x * current_dist)
            ry = start_y + (step_y * current_dist)
            rz = start_z + (step_z * current_dist)
            
            ix, iy, iz = int(round(rx)), int(round(ry)), int(round(rz))
            
            try:
                # Don't check exact start/end blocks to prevent self-collision
                if not ((ix == int(start_x) and iy == int(start_y) and iz == int(start_z)) or 
                        (ix == int(target_x) and iy == int(target_y) and iz == int(target_z))):
                    
                    block = level.get_block(ix, iy, iz, "minecraft:overworld")
                    name: str = block.namespaced_name
                    if not self.is_passable(name):
                        is_clear = False
                        break
            except: pass
            
            current_dist += 0.5

        level.close()
        return is_clear

    def is_reachable_astar(self, start: Tuple[int, int, int], 
                           target: Tuple[int, int, int], 
                           max_nodes: int = 2000) -> bool:
        """Uses A* Pathfinding to check if a player can WALK to the target."""
        print(f"--- A* Pathfinding: {start} -> {target} ---")
        
        try:
            level = amulet.load_level(self.world_path)
        except Exception as e:
            print(f"World Load Error: {e}")
            return False

        def get_block_type(cx: int, cy: int, cz: int) -> str:
            try:
                b = level.get_block(cx, cy, cz, "minecraft:overworld")
                return str(b.namespaced_name)
            except: return "minecraft:bedrock"

        # Priority Queue: (f_score, (x, y, z))
        open_set: List[Tuple[int, Tuple[int, int, int]]] = []
        heapq.heappush(open_set, (0, start))
        
        g_score: Dict[Tuple[int, int, int], int] = {start: 0}
        
        def heuristic(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

        nodes_checked = 0

        while open_set:
            if nodes_checked > max_nodes:
                print("A* Limit Reached")
                level.close()
                return False

            _, current = heapq.heappop(open_set)
            nodes_checked += 1

            if current == target:
                print(f"Path Found! Checked {nodes_checked} nodes.")
                level.close()
                return True

            cx, cy, cz = current
            neighbors = [(cx+1, cz), (cx-1, cz), (cx, cz+1), (cx, cz-1)]

            for nx, nz in neighbors:
                target_y: Optional[int] = None
                
                # Block queries
                n_feet = get_block_type(nx, cy, nz)
                n_head = get_block_type(nx, cy+1, nz)
                n_floor = get_block_type(nx, cy-1, nz)
                
                # 1. Flat Move
                if self.is_passable(n_feet) and self.is_passable(n_head) and not self.is_passable(n_floor):
                    target_y = cy
                
                # 2. Jump Move
                elif target_y is None:
                    n_jump_head = get_block_type(nx, cy+2, nz)
                    curr_headroom = get_block_type(cx, cy+2, cz)
                    
                    # n_head becomes feet, n_jump_head becomes head
                    if (self.is_passable(n_head) and self.is_passable(n_jump_head) and 
                        not self.is_passable(n_feet) and self.is_passable(curr_headroom)):
                        target_y = cy + 1

                # 3. Drop Move
                elif target_y is None:
                    n_drop_floor = get_block_type(nx, cy-2, nz)
                    # n_floor becomes feet, n_feet becomes head
                    if (self.is_passable(n_floor) and self.is_passable(n_feet) and 
                        not self.is_passable(n_drop_floor)):
                        target_y = cy - 1

                if target_y is not None:
                    neighbor = (nx, target_y, nz)
                    new_g = g_score[current] + 1
                    
                    if neighbor not in g_score or new_g < g_score[neighbor]:
                        g_score[neighbor] = new_g
                        f = new_g + heuristic(neighbor, target)
                        heapq.heappush(open_set, (f, neighbor))

        level.close()
        return False

    def visualize_matrix(self, matrix: np.ndarray) -> None:
        """Visualizes the 3D block matrix using Matplotlib voxels."""
        print("--- Rendering 3D Visualization ---")
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not installed.")
            return
        
        colors: Dict[str, Tuple[float, float, float, float]] = {
            "minecraft:grass_block": (0.1, 0.8, 0.1, 1.0),
            "minecraft:dirt":        (0.6, 0.4, 0.2, 1.0),
            "minecraft:stone":       (0.5, 0.5, 0.5, 1.0),
            "minecraft:water":       (0.1, 0.1, 0.9, 0.4),
            "minecraft:lava":        (1.0, 0.5, 0.0, 1.0),
            "undefined":             (0.1, 0.1, 0.1, 0.1),
            "default":               (0.8, 0.0, 0.8, 1.0)
        }

        size_x, size_y, size_z = matrix.shape
        filled = np.zeros(matrix.shape, dtype=bool)
        voxel_colors = np.zeros(matrix.shape + (4,), dtype=float)

        for x in range(size_x):
            for y in range(size_y):
                for z in range(size_z):
                    block_name: str = matrix[x, y, z] # type: ignore
                    if "air" in block_name: continue
                    filled[x, y, z] = True
                    
                    if block_name in colors:
                        voxel_colors[x, y, z] = colors[block_name]
                    else:
                        voxel_colors[x, y, z] = colors["default"]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(filled, facecolors=voxel_colors, edgecolors='k', linewidth=0.1, shade=True)
        plt.show()