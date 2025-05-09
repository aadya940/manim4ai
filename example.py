from manim import *
from manim4ai import LayoutManager  # Use your actual module name here
import numpy as np

import time

def transformation_circle(m):
    return m.move_to(LEFT * 3).rotate(PI)

class DummyScene(Scene):
    def construct(self): 
        pass

scene = DummyScene()
layout = LayoutManager(max_objects=3, samples=20)

square = Square().shift(LEFT * 3)
circle = Circle().shift(RIGHT * 3)
triangle = Triangle().shift(UP * 2)

layout.place(scene, square, {
    "animation_order": 0,
    "animation_function": lambda m: m.shift(RIGHT * 6),
    "id": "square",
    "collision_group": "g1",
    "duration": 2
})

layout.place(scene, circle, {
    "animation_order": 0,
    "animation_function": transformation_circle,
    "id": "circle",
    "collision_group": "g2",
    "duration": 2
})

layout.place(scene, triangle, {
    "animation_order": 0,
    "animation_function": lambda m: m.scale(1.5).rotate(PI/2).move_to(DOWN * 2),
    "id": "triangle",
    "collision_group": "g3",
    "duration": 2
})

a = time.time()

print(layout.static_check())

b = time.time()

print(b-a)

# layout.render()