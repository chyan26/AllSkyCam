import tkinter as tk
import math
import random

class HeadingVisualizer:
    """Class to visualize the heading direction in real-time using tkinter."""
    def __init__(self, root):
        self.heading = 0  # Initial heading value
        self.root = root
        self.root.title("Heading Visualizer")
        self.canvas = tk.Canvas(self.root, width=600, height=600, bg="white")
        self.canvas.pack()

        # Draw a circle to represent the compass
        self.center_x = 300
        self.center_y = 300
        self.radius = 250
        self.canvas.create_oval(
            self.center_x - self.radius, self.center_y - self.radius,
            self.center_x + self.radius, self.center_y + self.radius,
            outline="black"
        )

        # Draw the initial arrow
        self.arrow = self.canvas.create_line(
            self.center_x, self.center_y,
            self.center_x, self.center_y - self.radius,
            arrow=tk.LAST, fill="blue", width=10,arrowshape=(20, 25, 10) 
        )
        
        # Add heading text
        self.heading_text = self.canvas.create_text(150, 250, text="Heading: 0°")

    def update_heading(self, heading):
        """Update the heading direction and text."""
        self.heading = heading
        self._update_arrow()
        self._update_heading_text()

    def _update_arrow(self):
        """Update the arrow direction on the canvas."""
        angle_rad = math.radians(self.heading)
        end_x = self.center_x + self.radius * math.sin(angle_rad)
        end_y = self.center_y - self.radius * math.cos(angle_rad)
        self.canvas.coords(
            self.arrow,
            self.center_x, self.center_y,  # Start point
            end_x, end_y  # End point
        )
        
    def _update_heading_text(self):
        """Update the heading text display."""
        self.canvas.itemconfig(self.heading_text, text=f"Heading: {self.heading:.1f}°")
        
def animate_full_rotation(visualizer, heading):
    """
    Update visualizer with a single heading value.
    
    Args:
        visualizer: HeadingVisualizer instance to animate
        heading: Heading angle in degrees
    """
    visualizer.update_heading(heading)
    visualizer.update_heading_text(heading)


if __name__ == "__main__":
    visualizer = HeadingVisualizer()
    
    # Generate random headings (500 elements between 0 and 350)
    random_headings = [random.randint(0, 350) for _ in range(500)]
    
    # Set up animation loop outside the animate_full_rotation function
    def animation_loop(i=0):
        if i < len(random_headings):
            # Update single frame
            animate_full_rotation(visualizer, random_headings[i])
            # Schedule next update
            visualizer.root.after(100, lambda: animation_loop(i+1))
    
    # Start the animation loop
    animation_loop()
    
    # Start the main tkinter loop
    visualizer.start()