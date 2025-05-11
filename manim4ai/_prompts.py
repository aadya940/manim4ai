import textwrap

EXAMPLE_CODE = textwrap.dedent(
    """\
    
    EXAMPLE 1:

    from manim import *
    from manim4ai import LayoutManager  # Import the optimized LayoutManager
    import numpy as np

    # Define simple transformation functions - these get optimized fast-paths
    def move_to_right(m):
        return m.shift(RIGHT * 6)
        
    def transformation_circle(m):
        return m.move_to(LEFT * 3).rotate(PI)

    class DummyScene(Scene):
        def construct(self): 
            pass

    scene = DummyScene()
    # Use fewer samples for simple animations to improve performance
    layout = LayoutManager(max_objects=3, samples=10)  

    
    # Create and position objects before animation
    square = Square().shift(LEFT * 3)
    circle = Circle().shift(RIGHT * 3)
    triangle = Triangle().shift(UP * 2)

    # Place objects with well-defined configuration
    layout.place(scene, square, {
        "animation_order": 0,
        "animation_function": move_to_right,  # Using named function for clarity
        "id": "square",  
        "collision_group": "g1",
        "duration": 1.5
    })

    layout.place(scene, circle, {
        "animation_order": 0,
        "animation_function": transformation_circle,
        "id": "circle",
        "collision_group": "g2",
        "duration": 1.5
    })

    layout.place(scene, triangle, {
        "animation_order": 0,
        "animation_function": lambda m: m.scale(1.5).rotate(PI/2).move_to(DOWN * 2),
        "id": "triangle",
        "collision_group": "g3", 
        "duration": 1.5
    })


    EXAMPLE 2:

    from manim import *
    from manim4ai import LayoutManager
    import numpy as np

    # Define animation functions (simple, optimized)
    def rotate_shrink(m):
        return m.rotate(PI/2).scale(0.6).shift(LEFT * 2)

    def fade_out(m):
        return FadeOut(m)

    def shift_graph_up(m):
        return m.shift(UP * 1.5)

    class GraphAndParametricScene(Scene):
        def construct(self): 
            pass

    scene = GraphAndParametricScene()
    layout = LayoutManager(max_objects=3, samples=10)

    # Create Parametric Spiral
    spiral = ParametricFunction(
        lambda t: np.array([np.sin(2 * t) * t, np.cos(2 * t) * t, 0]),
        t_range=np.array([0, 2 * PI, 0.05]),
        color=BLUE
    ).scale(0.3).shift(LEFT * 4 + DOWN * 1)

    # Create Axes and Sine Graph
    axes = Axes(
        x_range=[-PI, PI, PI/2],
        y_range=[-1.5, 1.5, 1],
        axis_config={"color": WHITE}
    ).scale(0.7).shift(RIGHT * 3)

    sine_graph = axes.plot(lambda x: np.sin(x), color=YELLOW)

    # Layout placements
    layout.place(scene, spiral, {
        "animation_order": 0,
        "animation_function": rotate_shrink,
        "id": "spiral",
        "collision_group": "functions",
        "duration": 2.0
    })

    layout.place(scene, axes, {
        "animation_order": 0,
        "animation_function": shift_graph_up,
        "id": "axes",
        "collision_group": "graph",
        "duration": 2.0
    })

    layout.place(scene, sine_graph, {
        "animation_order": 0,
        "animation_function": lambda m: m.scale(1.1).shift(LEFT * 1),
        "id": "sine",
        "collision_group": "graph",
        "duration": 2.0
    })

    """
)

LIBRARIES = "- manim\n- manim4ai\n- numpy"

LAYOUT_DOC = textwrap.dedent(
    """\
    ## LayoutManager API Reference:
    
    ### LayoutManager.__init__(max_objects=4, samples=10)
    Initialize a new LayoutManager.
    
    Parameters:
    - max_objects (int): Maximum number of objects allowed in a single animation order
    - samples (int): Number of samples to take for trajectory approximation
      * Lower values (5-10): Faster performance, good for simple animations
      * Higher values (15-20): More accurate curve detection, better for complex paths
    
    ### LayoutManager.place(session, object, animation_config)
    Place an object in the scene with its animation configuration.
    
    Parameters:
    - session (Scene): The Manim scene instance
    - object (Mobject): The Manim object to animate
    - animation_config (dict): Configuration dictionary containing:
      * animation_order (int): Sequence number determining animation timing (required)
      * animation_function (callable): Function that transforms the object (required)
      * id (str): Unique identifier for the object (optional but recommended)
      * collision_group (str): Group ID for collision exemption (optional)
      * duration (float): Animation duration in seconds (optional, default=1)
    
    """
)


def system_prompt(topic: str) -> str:
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("Expected a non-empty string as topic.")

    return textwrap.dedent(
        f"""\
        You are an expert Manim animator tasked with generating a high-performance animation script on the following topic:

        {topic}

        You are allowed to use the following libraries in the animation code:
        {LIBRARIES}

        Here is an example on how to implement an animation:

        {EXAMPLE_CODE}

        {LAYOUT_DOC}

        ## Animation Guidelines:
        - Do not modify the `construct` method; it must remain empty.
        - Always use the `place` method of `LayoutManager` to animate objects.
        - Keep animation functions simple when possible - the layout manager optimizes simple animations (shift, scale, rotate) more efficiently than complex ones.
        - For animation_function:
          * Define with `def` or `lambda` to strictly accept only a single `Mobject` as input.
          * Can be a `manim.Animation` object like `FadeOut`
          * Pass as a parameter without calling it
          * Example: 
            # Take single `m` object with input without any other arguments.
            def move_to_right(m):
                return m.shift(RIGHT * 6) # Peform Animation.
        
            # Take single `m` object as input.
            def transformation_circle(m):
                return m.move_to(LEFT * 3).rotate(PI) # Perform Animation.
        
        - Group related objects in the same collision_group to exempt them from collision checks and improve performance.
        - Set appropriate samples (5-10 for simple animations, 15-20 for complex ones) to balance accuracy and performance.
        - All objects must remain fully visible within the screen bounds (-7 to 7 horizontally, -4 to 4 vertically).
        - Strictly use up to only 3 objects per animation order to avoid performance issues.
        - For better performance with many objects, batch them into sequential animation_orders rather than showing all at once.
        - Specify unique IDs for objects to help with debugging potential issues.
        - For complex scenes, use lower sample values (5-10).

        ## Performance Tips:
        - Use simple transformations when possible (shift, scale, rotate) which have optimized fast paths.
        - Group objects that are allowed to overlap into the same collision_group to reduce collision detection overhead.
        - Avoid unnecessary object copies - transform objects in place when possible.
        - For scenes with many objects, use sequential animation_orders to distribute computational load.
        - For complex paths, break them into simpler segments with multiple animation_orders.
        - Avoid creating more than 15-20 total objects across all animation orders to maintain good performance.
        
        Create a visually engaging, information-dense animation that effectively communicates the topic while following these guidelines for optimal performance.
        """
    )


def build_prompt(issues: str) -> str:
    if not isinstance(issues, str):
        raise ValueError("Issues must be a string.")

    return textwrap.dedent(
        f"""\
        The animation code you generated contains errors or performance issues that need to be addressed.

        ## Reported Issues:
        {issues.strip()}

        ## Troubleshooting Steps:
        1. If objects are moving outside visible bounds (-7 to 7 horizontally, -4 to 4 vertically):
           - Adjust starting positions to provide more room for movement
           - Reduce movement distances or scale objects smaller
           - Break complex movements into multiple smaller animation_orders

        2. If objects are colliding unintentionally:
           - Use different animation_orders to separate object movements in time
           - Adjust paths to avoid intersections
           - Place related objects in the same collision_group if overlap is acceptable

        3. For performance issues:
           - Reduce the number of samples (try 5-10 instead of 20+)
           - Use simpler animation functions that leverage optimized paths (shift, scale, rotate)
           - Reduce the number of objects per animation_order
           - Group objects that can overlap into the same collision_group

        4. For syntax errors:
           - Ensure all animation_functions accept exactly one Mobject parameter
           - Verify all required animation_config parameters are provided
           - Make sure objects are properly instantiated before placement

        Please regenerate the complete animation code with these corrections applied. The code should be ready to run without further modifications.

        Return only the corrected Python code without comments, explanations, or other text.
        """
    )
