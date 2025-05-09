from manim import *
from shapely.geometry import box, Polygon, LineString
from manim.scene.scene import Scene
from manim import tempconfig
import os
import inspect
import numpy as np
from functools import lru_cache


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

    # Class constant for frame boundaries
    FRAME_BOX = box(-7, -4, 7, 4)

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

        # Cache for bounding boxes to avoid recalculating for the same objects
        self._bbox_cache = {}

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

        # Use cached trajectory if possible, compute if needed
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
                    if not self._path_within_bounds(traj_i):
                        session_issues.append(
                            f" Object '{obj_i}' trajectory goes out of frame bounds"
                        )

                    # Early termination of inner loop if no more objects to check
                    if i == len(group) - 1:
                        continue

                    # Use collision lookup table approach for better performance with many objects
                    collision_candidates = []
                    for j in range(i + 1, len(group)):
                        obj_j, config_j, traj_j, id_j = group[j]

                        # Skip collision check if objects are in the same collision group
                        if self._in_same_collision_group(id_i, id_j):
                            continue

                        collision_candidates.append((j, obj_j, config_j, traj_j, id_j))

                    # Batch check collisions to reduce redundant calculations
                    for j, obj_j, config_j, traj_j, id_j in collision_candidates:
                        # First check if trajectories overlap at all (fast test)
                        if not traj_i.intersects(traj_j):
                            continue

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
        # Calculate a hash for caching
        obj_hash = id(obj)
        anim_hash = id(animation_function) if animation_function else 0
        cache_key = (obj_hash, anim_hash)

        # Return cached trajectory if exists
        cache_key = (obj_hash, anim_hash)

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

        # For efficiency, check if this animation is likely to be complex
        # (requires full sampling) or simple (can use simpler interpolation)
        is_complex_animation = self._is_complex_animation(animation_function)

        # Use fewer samples for simple animations
        sample_count = (
            self.samples if is_complex_animation else max(3, self.samples // 2)
        )

        # Sample the animation at multiple time points to capture curved motion
        bbox_corners = []

        # Pre-compute endpoints for efficiency
        start_bbox = self._get_bounding_box(obj)
        bbox_corners.append(start_bbox)

        # Compute end position (reused for interpolation if needed)
        end_obj = obj.copy()
        animation_function(end_obj)
        end_bbox = self._get_bounding_box(end_obj)

        # For simple animations, just do linear interpolation between start and end
        if not is_complex_animation:
            for t in np.linspace(0, 1, sample_count)[1:]:
                interp_bbox = [
                    [
                        start_bbox[0][0] * (1 - t) + end_bbox[0][0] * t,
                        start_bbox[0][1] * (1 - t) + end_bbox[0][1] * t,
                    ],
                    [
                        start_bbox[1][0] * (1 - t) + end_bbox[1][0] * t,
                        start_bbox[1][1] * (1 - t) + end_bbox[1][1] * t,
                    ],
                ]
                bbox_corners.append(interp_bbox)
        else:
            # For complex animations, sample at intermediate points
            for t in np.linspace(0, 1, sample_count)[
                1:-1
            ]:  # Skip first and last (already have them)
                # Create a temporary copy of the object for this time point
                temp_obj = obj.copy()

                try:
                    # Apply animation at this time point
                    self._apply_animation_at_time(temp_obj, animation_function, t)
                    bbox = self._get_bounding_box(temp_obj)
                    bbox_corners.append(bbox)
                except Exception:
                    # Fall back to linear interpolation for this point
                    interp_bbox = [
                        [
                            start_bbox[0][0] * (1 - t) + end_bbox[0][0] * t,
                            start_bbox[0][1] * (1 - t) + end_bbox[0][1] * t,
                        ],
                        [
                            start_bbox[1][0] * (1 - t) + end_bbox[1][0] * t,
                            start_bbox[1][1] * (1 - t) + end_bbox[1][1] * t,
                        ],
                    ]
                    bbox_corners.append(interp_bbox)

            # Add end point
            bbox_corners.append(end_bbox)

        # Convert the sequence of bounding boxes into a LineString path
        points = []
        for bbox in bbox_corners:
            top_left = bbox[0]
            bottom_right = bbox[1]
            top_right = [bottom_right[0], top_left[1]]
            bottom_left = [top_left[0], bottom_right[1]]

            # Simplified approach: just add corners of the box
            points.append(top_left)
            points.append(top_right)
            points.append(bottom_right)
            points.append(bottom_left)

        # Create a LineString from all the sampled points to represent the trajectory
        if points:
            return LineString(points)
        else:
            # Fallback if no points were generated
            bbox = self._get_bounding_box(obj)
            return LineString([bbox[0], bbox[1]])

    def _is_complex_animation(self, animation_function):
        """
        Determine if an animation is likely complex (needs full sampling) or simple.
        Simple animations can use linear interpolation for better performance.
        """
        if animation_function is None:
            return False

        # Check function name for common simple transformations
        func_name = (
            animation_function.__name__
            if hasattr(animation_function, "__name__")
            else str(animation_function)
        )
        simple_patterns = ["shift", "move_to", "scale", "rotate", "next_to", "linear"]

        for pattern in simple_patterns:
            if pattern in func_name.lower():
                return False

        # If lambda or complex animation, consider it complex
        return True

    def _apply_animation_at_time(self, obj, animation_function, t):
        """
        Applies an animation to an object at a specific time point t (0-1).
        This is a helper method to sample animation states at intermediate points.

        Optimized version with fast path for common animation types.
        """
        # Fast path for t=1 (full animation)
        if t == 1:
            animation_function(obj)
            return

        # Fast path for t=0 (no animation)
        if t == 0:
            return

        # Check for common animation types first
        try:
            # Extract animation name to detect common types
            func_name = (
                animation_function.__name__
                if hasattr(animation_function, "__name__")
                else str(animation_function)
            )

            # Fast path for shift animations
            if "shift" in func_name.lower():
                # Create reference object with full animation
                ref_obj = obj.copy()
                animation_function(ref_obj)

                # Calculate shift vector and apply partial shift
                start_pos = obj.get_center()
                end_pos = ref_obj.get_center()
                shift_vector = (end_pos - start_pos) * t
                obj.shift(shift_vector)
                return

            # Fast path for scale animations
            if "scale" in func_name.lower():
                # Apply partial scale
                # Extract scale factor if possible, otherwise use approximation
                end_obj = obj.copy()
                animation_function(end_obj)

                start_scale = obj.get_height()
                end_scale = end_obj.get_height()
                scale_factor = 1 + (end_scale / start_scale - 1) * t
                obj.scale(scale_factor)
                return

            # Try to use animation's interpolate_mobject if it's an Animation
            anim = animation_function(obj.copy())
            if hasattr(anim, "interpolate_mobject"):
                anim.interpolate_mobject(t)
                return
        except:
            pass

        # Fallback: create end state and interpolate
        try:
            end_obj = obj.copy()
            animation_function(end_obj)

            # Interpolate position
            start_pos = obj.get_center()
            end_pos = end_obj.get_center()
            interp_pos = start_pos * (1 - t) + end_pos * t
            obj.move_to(interp_pos)

            # Try to interpolate scaling if noticeable change
            start_scale = obj.get_height()
            end_scale = end_obj.get_height()
            if abs(start_scale - end_scale) > 0.01:
                scale_factor = 1 + (end_scale / start_scale - 1) * t
                obj.scale(scale_factor)

            # Try to interpolate rotation if applicable
            if hasattr(obj, "get_angle"):
                start_angle = obj.get_angle()
                end_angle = end_obj.get_angle()
                if abs(end_angle - start_angle) > 0.01:
                    interp_angle = start_angle * (1 - t) + end_angle * t
                    obj.rotate(interp_angle - start_angle)
        except:
            # Last resort: just apply full animation if t > 0.5
            if t > 0.5:
                animation_function(obj)

    @lru_cache(maxsize=128)
    def _get_bounding_box(self, obj: Mobject):
        """
        Returns the top-left and bottom-right corners of the bounding box for the given Mobject.
        Uses caching to avoid recalculating for the same object state.

        Args:
            obj , Mobject:
                The Manim object

        Returns:
            tuple:
                ([min_x, max_y], [max_x, min_y]) - the top-left and bottom-right corners
        """
        # Check if cached result exists
        obj_id = id(obj)
        if obj_id in self._bbox_cache:
            return self._bbox_cache[obj_id]

        # Calculate bounding box
        points = obj.get_all_points()
        if len(points) == 0:
            result = [0, 0], [0, 0]  # fallback if object has no geometry
            self._bbox_cache[obj_id] = result
            return result

        # Use numpy for faster min/max operations
        points_array = np.array(points)
        min_x, min_y = np.min(points_array, axis=0)[:2]  # Take only x,y
        max_x, max_y = np.max(points_array, axis=0)[:2]  # Take only x,y

        top_left = [min_x, max_y]
        bottom_right = [max_x, min_y]

        result = (top_left, bottom_right)
        self._bbox_cache[obj_id] = result
        return result

    def _path_within_bounds(self, path):
        """
        Check if a path stays within the frame boundaries.

        Args:
            path:
                Shapely geometry representing the object's path

        Returns:
            bool: True if the path is completely within bounds
        """
        return self.FRAME_BOX.contains(path)

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
        # Fast path: convert to lookup table the first time
        collision_lookup = getattr(self, "_collision_lookup", None)
        if collision_lookup is None:
            # Create lookup table mapping object ID to all its groups
            self._collision_lookup = {}
            for group_name, group_ids in self.collision_groups.items():
                for obj_id in group_ids:
                    if obj_id not in self._collision_lookup:
                        self._collision_lookup[obj_id] = set()
                    self._collision_lookup[obj_id].add(group_name)
            collision_lookup = self._collision_lookup

        # Check if objects share any group
        groups_i = collision_lookup.get(id_i, set())
        groups_j = collision_lookup.get(id_j, set())

        return bool(groups_i.intersection(groups_j))

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

        # Adaptive sampling - use fewer points for initial check, more for detailed analysis
        initial_points = min(5, self.samples)  # First pass with fewer points
        collision_times = []

        # First pass: check collision at initial resolution
        potential_collisions = False
        for t in np.linspace(0, 1, initial_points):
            # Make copies of objects to avoid modifying originals
            temp_i = obj_i.copy()
            temp_j = obj_j.copy()

            # Apply animations at time t
            self._apply_animation_at_time(temp_i, anim_func_i, t)
            self._apply_animation_at_time(temp_j, anim_func_j, t)

            # Check if objects overlap at this time point
            if self._objects_overlap(temp_i, temp_j):
                if initial_points == self.samples:
                    # If we're already at full resolution, record the collision
                    collision_times.append(t)
                else:
                    # Flag for second pass with higher resolution
                    potential_collisions = True
                    break

        # Second pass: higher resolution check only if needed
        if potential_collisions:
            check_points = max(self.samples, 10)  # Full resolution
            for t in np.linspace(0, 1, check_points):
                # Make copies of objects
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
