from manim import *
from shapely.geometry import box, Polygon, LineString
from manim.scene.scene import Scene
from manim import tempconfig
import os
import inspect
import numpy as np


class LayoutManager:
    """LayoutManager is a utility for managing and validating the layout of animated Mobjects in Manim scenes.

    It allows for the placement of objects in animation sequences, tracks their intended motion paths,
    and performs static checks to detect potential issues such as:

    - Exceeding the maximum number of objects per frame
    - Trajectories going outside the visible frame
    - Intersecting trajectories between objects (with optional collision groups to exempt certain overlaps)

    Features:
    - Supports curved trajectories through multi-point sampling
    - Collision detection between motion paths
    - Automatic rendering of animations
    """

    def __init__(self, max_objects=4, samples=10) -> None:
        """
        Initialize a new LayoutManager.

        Args:
            max_objects , int:
                Maximum number of objects allowed in a single animation order
            samples , int:
                Number of samples to take for trajectory approximation (higher = more accurate curves)
        """
        self.max_objects = max_objects
        self.samples = max(
            3, samples
        )  # Minimum 3 samples for better curve approximation
        self._info = {}  # Structure: {session: {order: [(object, config, path)]}}
        self.collision_groups = {}  # Structure: {group_id: [object_ids]}

    def place(self, session: Scene, object: Mobject, animation_config: dict):
        """
        Place an object in the scene with its animation configuration.

        Args:
            session (Scene):
                The Manim scene
            object , Mobject:
                The object to animate
            animation_config (dict):
                Configuration including:
                - animation_order:
                    Order in which animations should play
                - animation_function:
                    Function that transforms the object
                - collision_group (optional):
                    Group ID for collision exemption
                - id (optional):
                    Unique identifier for the object
                - duration (optional):
                    Animation duration in seconds
        """
        animation_order = animation_config.get("animation_order")
        animation_function = animation_config.get("animation_function")
        collision_group = animation_config.get("collision_group", None)
        object_id = animation_config.get("id", str(id(object)))

        # Register object in collision group if specified
        if collision_group:
            if collision_group not in self.collision_groups:
                self.collision_groups[collision_group] = []
            self.collision_groups[collision_group].append(object_id)

        # Compute trajectory using sampled points
        trajectory = self._compute_trajectory(object, animation_function)

        if session not in self._info:
            self._info[session] = {}

        if animation_order not in self._info[session]:
            self._info[session][animation_order] = []

        self._info[session][animation_order].append(
            (object, animation_config, trajectory, object_id)
        )

    def static_check(self) -> str:
        """
        Performs static checks on all placed objects and returns a string report.

        Returns:
            str:
                A report of potential layout issues, or an empty string if no issues found
        """
        FRAME_BOX = box(-7, -4, 7, 4)
        issues = []

        for session, orders in self._info.items():
            for order, group in orders.items():
                session_issues = []
                session_issues.append(
                    f"Checking Session '{session}', Animation Order {order}:"
                )

                # Check object count
                if len(group) > self.max_objects:
                    session_issues.append(
                        f"Too many objects ({len(group)} > {self.max_objects})"
                    )

                # Check intersections and screen bounds
                for i, (obj_i, config_i, traj_i, id_i) in enumerate(group):
                    if not self._path_within_bounds(traj_i, FRAME_BOX):
                        session_issues.append(
                            f" Object '{obj_i}' trajectory goes out of frame bounds"
                        )

                    for j in range(i + 1, len(group)):
                        obj_j, config_j, traj_j, id_j = group[j]

                        # Skip collision check if objects are in the same collision group
                        if self._in_same_collision_group(id_i, id_j):
                            continue

                        # First check if trajectories overlap at all (fast test)
                        if traj_i.intersects(traj_j):
                            # If trajectories intersect, check for time-based collisions
                            collision_times = self._detect_time_based_collisions(
                                obj_i, config_i, obj_j, config_j
                            )

                            if collision_times:
                                # Real collision detected
                                time_info = ", ".join(
                                    [f"t≈{t:.2f}" for t in collision_times]
                                )
                                session_issues.append(
                                    f"Objects '{obj_i}' and '{obj_j}' collide at normalized time {time_info}"
                                )

                if len(session_issues) > 1:  # More than just the header
                    issues.extend(session_issues)

        return "\n".join(issues) if issues else ""

    def _spatial_info(self):
        """Return spatial information about all managed objects for debugging"""
        info = []
        for session, orders in self._info.items():
            for order, group in orders.items():
                for obj, config, traj, obj_id in group:
                    info.append(
                        f"Session={session}, Order={order}, Obj={obj}, ID={obj_id}, Trajectory={traj}"
                    )
        return "\n".join(info)

    def _compute_trajectory(self, obj: Mobject, animation_function: callable):
        """
        Computes the trajectory of an object's bounding box by sampling the animation at multiple time points.

        Args:
            obj , Mobject:
                The object being animated
            animation_function (callable):
                Function that transforms the object

        Returns:
            LineString:
                A Shapely LineString representing the path of the object's bounding box
        """
        # If no animation function is provided, return a simple trajectory based on current position
        if animation_function is None:
            bbox = self._get_bounding_box(obj)
            return Polygon(
                [
                    [bbox[0][0], bbox[0][1]],  # top-left
                    [bbox[1][0], bbox[0][1]],  # top-right
                    [bbox[1][0], bbox[1][1]],  # bottom-right
                    [bbox[0][0], bbox[1][1]],  # bottom-left
                    [bbox[0][0], bbox[0][1]],  # close the polygon
                ]
            )

        # Sample the animation at multiple time points to capture curved motion
        points = []

        # Create a list to store the bounding box corners for each sample
        bbox_corners = []

        for t in np.linspace(0, 1, self.samples):
            # Create a temporary copy of the object for this time point
            temp_obj = obj.copy()

            # For t=0, use the original object's position
            if t == 0:
                bbox = self._get_bounding_box(temp_obj)
                bbox_corners.append(bbox)
                continue

            try:
                # Try to use partial_apply_function from manim if available
                # This is a more accurate way to get intermediate states for complex animations
                self._apply_animation_at_time(temp_obj, animation_function, t)

                # Get the bounding box at this time point
                bbox = self._get_bounding_box(temp_obj)
                bbox_corners.append(bbox)
            except Exception as e:
                # If partial application fails, fall back to linear interpolation
                if len(bbox_corners) == 0:
                    # Initial position (needed if t=0 failed)
                    start_bbox = self._get_bounding_box(obj)
                    bbox_corners.append(start_bbox)

                    # Final position
                    end_obj = obj.copy()
                    animation_function(end_obj)
                    end_bbox = self._get_bounding_box(end_obj)

                    # Linearly interpolate between start and end for all samples
                    for interp_t in np.linspace(0, 1, self.samples)[1:]:
                        interp_bbox = [
                            [
                                start_bbox[0][0] * (1 - interp_t)
                                + end_bbox[0][0] * interp_t,
                                start_bbox[0][1] * (1 - interp_t)
                                + end_bbox[0][1] * interp_t,
                            ],
                            [
                                start_bbox[1][0] * (1 - interp_t)
                                + end_bbox[1][0] * interp_t,
                                start_bbox[1][1] * (1 - interp_t)
                                + end_bbox[1][1] * interp_t,
                            ],
                        ]
                        bbox_corners.append(interp_bbox)
                    break

        # Convert the sequence of bounding boxes into a polygon path
        for bbox in bbox_corners:
            top_left = bbox[0]
            bottom_right = bbox[1]
            top_right = [bottom_right[0], top_left[1]]
            bottom_left = [top_left[0], bottom_right[1]]

            points.extend([top_left, top_right, bottom_right, bottom_left])

        # Create a LineString from all the sampled points to represent the trajectory
        if points:
            return LineString(points)
        else:
            # Fallback if no points were generated
            bbox = self._get_bounding_box(obj)
            return LineString([bbox[0], bbox[1]])

    def _apply_animation_at_time(self, obj, animation_function, t):
        """
        Applies an animation to an object at a specific time point t (0-1).
        This is a helper method to sample animation states at intermediate points.
        """
        # First, check if the animation_function returns a Manim Animation
        try:
            test_obj = obj.copy()
            anim = animation_function(test_obj)

            if hasattr(anim, "interpolate_mobject"):
                # It's an Animation object, so we can use interpolate_mobject
                anim.interpolate_mobject(t)
                return
        except:
            pass

        # If we get here, it's probably a transform function (like lambda m: m.shift(RIGHT))
        # For simple transforms, we can try a linear approximation
        if t == 1:
            # For t=1, just apply the full animation
            animation_function(obj)
        else:
            # Try to detect common Manim transforms and interpolate them
            try:
                # Create two copies - one with no transform, one with full transform
                start_obj = obj.copy()
                end_obj = obj.copy()
                animation_function(end_obj)

                # Interpolate position between start and end
                start_pos = start_obj.get_center()
                end_pos = end_obj.get_center()
                interp_pos = start_pos * (1 - t) + end_pos * t

                # Move the object to interpolated position
                obj.move_to(interp_pos)

                # Try to interpolate scaling
                start_scale = np.array(start_obj.get_height())
                end_scale = np.array(end_obj.get_height())
                if (
                    abs(start_scale - end_scale) > 0.01
                ):  # Only if there's a noticeable scale change
                    scale_factor = 1 + (end_scale / start_scale - 1) * t
                    obj.scale(scale_factor)

                # Try to interpolate rotation
                # (This is a simplified approximation)
                start_angle = (
                    start_obj.get_angle() if hasattr(start_obj, "get_angle") else 0
                )
                end_angle = end_obj.get_angle() if hasattr(end_obj, "get_angle") else 0
                if (
                    abs(end_angle - start_angle) > 0.01
                ):  # Only if there's a noticeable rotation
                    interp_angle = start_angle * (1 - t) + end_angle * t
                    obj.rotate(interp_angle - start_angle)

            except Exception as e:
                # If interpolation fails, fall back to direct application for t=1
                if t > 0.5:
                    animation_function(obj)

    def _get_bounding_box(self, obj: Mobject):
        """
        Returns the top-left and bottom-right corners of the bounding box for the given Mobject.

        Args:
            obj , Mobject:
                The Manim object

        Returns:
            tuple:
                ([min_x, max_y], [max_x, min_y]) - the top-left and bottom-right corners
        """
        points = obj.get_all_points()
        if len(points) == 0:
            return [0, 0], [0, 0]  # fallback if object has no geometry

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        top_left = [min_x, max_y]
        bottom_right = [max_x, min_y]

        return top_left, bottom_right

    def _path_within_bounds(self, path, frame_box: Polygon):
        """
        Check if a path stays within the frame boundaries.

        Args:
            path:
                Shapely geometry representing the object's path
            frame_box:
                Shapely box representing the visible frame

        Returns:
            bool: True if the path is completely within bounds
        """
        return frame_box.contains(path)

    def _in_same_collision_group(self, id_i, id_j):
        """
        Check if two objects belong to the same collision group.

        Args:
            id_i, id_j , str:
                Object identifiers

        Returns:
            bool:
                True if the objects are in the same collision group
        """
        for group_ids in self.collision_groups.values():
            if id_i in group_ids and id_j in group_ids:
                return True
        return False

    def _detect_time_based_collisions(self, obj_i, config_i, obj_j, config_j):
        """
        Detects if two objects actually collide during animation by checking
        their positions at multiple time points.

        Args:
            obj_i, obj_j:
                The two objects being checked
            config_i, config_j:
                Their animation configurations

        Returns:
            list:
                Time points (0-1) where collisions occur, or empty list if no collisions
        """
        # Get animation functions
        anim_func_i = config_i.get("animation_function")
        anim_func_j = config_j.get("animation_function")

        # If either object doesn't have an animation function, we can't detect time-based collisions
        if not anim_func_i or not anim_func_j:
            # Just check if they're currently overlapping
            return [0.0] if self._objects_overlap(obj_i, obj_j) else []

        # Set number of time points to check (more points = more accurate collision detection)
        check_points = max(
            self.samples, 10
        )  # At least 10 points for collision detection
        collision_times = []

        # Check collision at multiple time points
        for t in np.linspace(0, 1, check_points):
            # Make copies of objects to avoid modifying originals
            temp_i = obj_i.copy()
            temp_j = obj_j.copy()

            # Apply animations at time t
            self._apply_animation_at_time(temp_i, anim_func_i, t)
            self._apply_animation_at_time(temp_j, anim_func_j, t)

            # Check if objects overlap at this time point
            if self._objects_overlap(temp_i, temp_j):
                collision_times.append(t)

        return collision_times

    def _objects_overlap(self, obj_i, obj_j):
        """
        Checks if two Manim objects overlap by comparing their bounding boxes.

        Args:
            obj_i, obj_j:
                The two objects to check

        Returns:
            bool:
                True if the objects' bounding boxes overlap
        """
        # Get bounding boxes
        bbox_i_tl, bbox_i_br = self._get_bounding_box(obj_i)
        bbox_j_tl, bbox_j_br = self._get_bounding_box(obj_j)

        # Check for overlap (standard rectangle intersection test)
        # Two rectangles overlap if:

        # 1. One rectangle's left edge is to the left of the other's right edge, AND
        # 2. One rectangle's right edge is to the right of the other's left edge, AND
        # 3. One rectangle's top edge is above the other's bottom edge, AND
        # 4. One rectangle's bottom edge is below the other's top edge

        overlap_x = bbox_i_tl[0] <= bbox_j_br[0] and bbox_i_br[0] >= bbox_j_tl[0]
        overlap_y = bbox_i_tl[1] >= bbox_j_br[1] and bbox_i_br[1] <= bbox_j_tl[1]

        return overlap_x and overlap_y

    def render(self, output_file="scene.mp4", quality="medium_quality"):
        """
        Renders all placed objects and their animations as a Manim video.

        Args:
            output_file , str:
                The name of the output video file.
            quality , str:
                The quality of the video. Options are 'low_quality',
                'medium_quality', 'high_quality', 'production_quality'.

        Returns:
            str:
                Path to the generated video file.
        """
        # Get the caller's module name for the output directory
        frame = inspect.currentframe().f_back
        module_name = frame.f_globals["__name__"]
        if module_name == "__main__":
            # Get the file name without extension if it's the main module
            import sys

            module_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

        # Process all sessions
        for session, orders in self._info.items():
            # Create a temporary scene class for this session
            class GeneratedScene(session.__class__):
                def construct(self_scene):
                    # Process all animation orders (sorted)
                    for order in sorted(orders.keys()):
                        animations = []

                        # Get all animations for this order
                        for obj, config, _, _ in orders[order]:
                            # Get the animation function if available
                            anim_func = config.get("animation_function")
                            duration = config.get("duration", 1)

                            # If object isn't already in the scene, add it
                            if obj not in self_scene.mobjects:
                                self_scene.add(obj)

                            # Create animation based on the function
                            if anim_func:
                                from manim import Transform, Animation

                                try:
                                    # Check if the function already returns an Animation
                                    temp_obj = obj.copy()
                                    result = anim_func(temp_obj)

                                    if isinstance(result, Animation):
                                        # It returns an animation, use it directly
                                        animations.append(anim_func(obj))
                                    else:
                                        # It transforms the object, create a Transform animation
                                        temp_obj = obj.copy()
                                        anim_func(temp_obj)
                                        animations.append(
                                            Transform(obj, temp_obj, run_time=duration)
                                        )
                                except Exception as e:
                                    print(
                                        f"Warning: Could not create animation for {obj}: {e}"
                                    )
                                    # Try a different approach - just use the function directly
                                    try:
                                        animations.append(anim_func(obj))
                                    except Exception:
                                        print(f"Failed to animate {obj}, skipping.")

                        # Play all animations for this order group simultaneously
                        if animations:
                            self_scene.play(*animations)

            # Configure rendering with custom output directory
            render_config = {
                "output_file": output_file,
                "quality": quality,
                "module_name": module_name,
            }

            # Create an instance of the scene and render it
            scene = GeneratedScene()

            with tempconfig(render_config):
                scene.render()
                return True

        print("Warning: No sessions found to render.")
        return False
