from manim import *
from manim4ai import LayoutManager  # Use your actual module name here
import numpy as np

class DummyScene(Scene):
    def construct(self): 
        pass


def demo_curved_trajectory():
    scene = DummyScene()
    # Initialize with more samples for smoother curve approximation
    layout = LayoutManager(max_objects=3, samples=20)

    # Create objects
    square = Square().shift(LEFT * 3)
    circle = Circle().shift(RIGHT * 3)
    triangle = Triangle().shift(UP * 2)


    def sr(m):
        return m.shift(RIGHT * 6)

    def mr(m):
        return m.move_to(LEFT * 3).rotate(PI)
    
    # Linear motion
    layout.place(scene, square, {
        "animation_order": 0,
        "animation_function": sr,
        "id": "square",
        "collision_group": "g1",
        "duration": 2
    })

    # Curved motion (arc)
    layout.place(scene, circle, {
        "animation_order": 0,
        "animation_function": mr,
        "id": "circle",
        "collision_group": "g2",
        "duration": 2
    })

    # Complex motion (scale, rotate and move)
    layout.place(scene, triangle, {
        "animation_order": 0,
        "animation_function": mr,
        "id": "triangle",
        "collision_group": "g3",
        "duration": 2
    })

    # Check for issues
    report = layout.static_check()
    if report:
        print("Static Check Report:\n", report)
    else:
        print("No issues detected.")

    # Render the animation
    layout.render()


if __name__ == "__main__":
    # Run the examples
    demo_curved_trajectory()
    # demo_complex_curves()