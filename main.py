import numpy as np
from PIL import Image
import math
import zlib

def rgb_to_luminance(rgb):
    """Convert RGB to luminance value for sorting"""
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

def triangle_compression(image_path, triangle_size):
    # Load image
    img = Image.open(image_path)
    width, height = img.size
    pixels = np.array(img)

    # Calculate triangle dimensions
    tri_height = int(triangle_size * math.sqrt(3)/2)

    # Storage for all components
    triangles = []
    positions = []
    avg_rgbs = []
    orientation_bits = []

    # 1. Partition image into triangles
    for y in range(0, height, tri_height):
        for x in range(0, width, triangle_size):
            # Triangle pointing up (orientation 0)
            if y + tri_height <= height and x + triangle_size <= width:
                # Create mask for triangle area
                tri_mask = np.zeros((tri_height, triangle_size), dtype=bool)
                for dy in range(tri_height):
                    start = int(dy * (triangle_size/tri_height)/2)
                    end = triangle_size - start
                    tri_mask[dy, start:end] = True

                # Extract triangle pixels
                tri_area = pixels[y:y+tri_height, x:x+triangle_size]
                tri_pixels = tri_area[tri_mask]
                avg_rgb = np.mean(tri_pixels, axis=0)

                # Store data
                triangles.append(tri_pixels)
                avg_rgbs.append(avg_rgb)
                positions.append((x, y, 0))  # 0=up orientation

                # Triangle pointing down (orientation 1)
                if y + tri_height < height and x + triangle_size//2 < width:
                    tri_mask = np.zeros((tri_height, triangle_size), dtype=bool)
                    for dy in range(tri_height):
                        start = int((tri_height-dy) * (triangle_size/tri_height)/2)
                        end = triangle_size - start
                        tri_mask[dy, start:end] = True

                    tri_area = pixels[y:y+tri_height, x:x+triangle_size]
                    tri_pixels = tri_area[tri_mask]
                    avg_rgb = np.mean(tri_pixels, axis=0)

                    triangles.append(tri_pixels)
                    avg_rgbs.append(avg_rgb)
                    positions.append((x + triangle_size//2, y, 1))  # 1=down orientation

    # 2. Sort triangles by average RGB luminance
    sort_key = [rgb_to_luminance(rgb) for rgb in avg_rgbs]
    sort_indices = np.argsort(sort_key)

    # 3. Color normalization
    global_avg = np.mean(avg_rgbs, axis=0)
    color_shifts = [avg - global_avg for avg in avg_rgbs]

    # 4. Prepare compressed data components
    compressed_data = {
        'image_shape': (height, width, 3),
        'triangle_size': triangle_size,
        'sort_indices': sort_indices,
        'positions': positions,  # Original positions and orientations
        'global_avg': global_avg,
        'color_shifts': color_shifts,
        'adjusted_triangles': [np.clip(triangles[i] - color_shifts[i], 0, 255).astype(np.uint8)
                             for i in range(len(triangles))]
    }

    # 5. Compare compression sizes
    original_bytes = img.tobytes()
    original_compressed = len(zlib.compress(original_bytes))

    # Serialize our data efficiently
    serialized_data = (
        np.array([height, width, 3], dtype=np.uint16).tobytes() +
        np.array([triangle_size], dtype=np.uint8).tobytes() +
        np.array(sort_indices, dtype=np.uint16).tobytes() +
        np.array([pos[0] for pos in positions], dtype=np.uint16).tobytes() +  # x positions
        np.array([pos[1] for pos in positions], dtype=np.uint16).tobytes() +  # y positions
        np.array([pos[2] for pos in positions], dtype=np.uint8).tobytes() +   # orientations
        global_avg.astype(np.float16).tobytes() +
        np.array(color_shifts, dtype=np.float16).tobytes() +
        b''.join([t.tobytes() for t in compressed_data['adjusted_triangles']])
    )

    new_compressed = len(zlib.compress(serialized_data))

    return len(original_bytes), original_compressed, new_compressed
# Example usage
original_size, new_size, compressed_new = triangle_compression("img.png", 32)
print(original_size)
print(new_size)
print(compressed_new)
