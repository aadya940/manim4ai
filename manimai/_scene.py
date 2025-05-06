from manim import *
import numpy as np
from shapely.geometry import box, Point, Polygon
from shapely.ops import unary_union

class LayoutManager:
    def __init__(self, max_objects=4) -> None:
        self.max_objects = max_objects
        self._info = {}  # Structure: {session: [(object, config, bbox)]}

    def place(self, session: Scene, object: Mobject, animation_config: dict):
        animation_order = animation_config.get("animation_order")
        animation_function = animation_config.get("animation_function")

        bbox = self._create_boundingbox(object)

        if session not in self._info:
            self._info[session] = {}

        if animation_order not in self._info[session]:
            self._info[session][animation_order] = []

        self._info[session][animation_order].append((object, animation_config, bbox))

    def static_check(self):
        FRAME_BOX = box(-7, -4, 7, 4)

        for session, orders in self._info.items():
            for order, group in orders.items():
                print(f"\nChecking Session '{session}', Animation Order {order}:")

                # Check object count
                if len(group) > self.max_objects:
                    print(f"  ⚠️ Too many objects ({len(group)} > {self.max_objects})")

                # Check intersections and screen bounds
                for i, (obj_i, config_i, bbox_i) in enumerate(group):
                    if not FRAME_BOX.contains(bbox_i):
                        print(f"  ⚠️ Object '{obj_i}' goes out of frame bounds")

                    for j in range(i + 1, len(group)):
                        obj_j, config_j, bbox_j = group[j]
                        if bbox_i.intersects(bbox_j):
                            print(f"  ⚠️ Overlap between '{obj_i}' and '{obj_j}'")

    def _spatial_info(self):
        # Optional: expose all object bounding boxes
        for session, orders in self._info.items():
            for order, group in orders.items():
                for obj, config, bbox in group:
                    print(f"Session={session}, Order={order}, Obj={obj}, BBox={bbox.bounds}")

    def _create_boundingbox(self, obj: Mobject):
        center = obj.get_center()
        width, height = obj.width, obj.height
        return box(center[0] - width/2, center[1] - height/2,
                   center[0] + width/2, center[1] + height/2)
